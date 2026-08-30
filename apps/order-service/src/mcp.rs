//! MCP（Model Context Protocol）Server —— 向 shop-agent 暴露业务工具。
//!
//! 设计依据：`docs/architecture/order-service-mcp-design.md`
//! 选型依据：Phase 0 Spike 实测通过，`rmcp` 3.1.4（见实施计划 §附录 A）
//!
//! ## 关键约束
//!
//! 1. **工具名必须用连字符**：`rmcp` 默认以 Rust 方法名的蛇形暴露（如 `query_order`），
//!    而 shop-agent 的 action 名是 `query-order`，故每个 `#[tool]` 都要显式
//!    `name = "..."`。否则 shop-agent 侧需额外 alias 映射。
//!    → 由 `tests::` 中的用例强制校验，防止重构时退化。
//!
//! 2. **必须加 `#[tool_handler]`**：缺少它时服务正常启动、握手正常，但
//!    `tools/list` 静默返回空列表（编译期会警告 `tool_router is never read`）。
//!
//! 3. **不另写业务逻辑**：只做参数转换 + 调用 `crate::*_core` 函数，
//!    与 REST handler 同源（设计文档 §3.3）。
//!
//! 4. **本阶段只暴露只读工具**（Phase 1）。
//!    敏感写工具（`request-return` / `refund-confirm` / `check-balance`）在 Phase 3 接入。
//!    破坏性与管理类接口（`DELETE /orders/:id`、`PUT`、`POST /orders`）**永不暴露**。

use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars,
    tool, tool_handler, tool_router, ServerHandler,
};
use serde::Deserialize;
use sqlx::PgPool;

// ────────────────────────────────────────────────────────────────
// 请求参数（rmcp 从中生成 JSON Schema 作为 MCP inputSchema）
// ────────────────────────────────────────────────────────────────

/// 注意：`order_id` 在此**必须声明**——MCP 语义要求 server 完整描述所需参数。
/// shop-agent 侧会在本地把高后果字段从模型可见 schema 中剥离（硬强制），
/// 切除发生在消费侧而非本服务（见设计文档 §5.3「安全策略不外包」）。
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct QueryOrderRequest {
    #[schemars(description = "订单号")]
    pub order_id: String,
    #[schemars(description = "归属用户 ID；提供时做数据级权限校验，不符返回 403")]
    pub user_id: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetEvidenceRequest {
    #[schemars(description = "订单号")]
    pub order_id: String,
    #[schemars(description = "归属用户 ID；提供时做数据级权限校验")]
    pub user_id: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CouponInquiryRequest {
    #[schemars(description = "优惠券类型关键字（模糊匹配）；不传返回全部")]
    pub coupon_type: Option<String>,
}

// ────────────────────────────────────────────────────────────────
// MCP Server
// ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct OrderMcpServer {
    db: PgPool,
    tool_router: ToolRouter<Self>,
}

impl OrderMcpServer {
    pub fn new(db: PgPool) -> Self {
        Self {
            db,
            tool_router: Self::tool_router(),
        }
    }

    /// 已注册的工具名（供测试与运维检查使用）
    pub fn tool_names(&self) -> Vec<String> {
        self.tool_router
            .list_all()
            .into_iter()
            .map(|t| t.name.to_string())
            .collect()
    }
}

#[tool_router]
impl OrderMcpServer {
    /// 按订单号查询订单完整信息（金额、状态、商品、物流）
    #[tool(name = "query-order")]
    async fn query_order(&self, Parameters(req): Parameters<QueryOrderRequest>) -> String {
        match crate::query_order_core(&self.db, &req.order_id, req.user_id.as_deref()).await {
            Ok(order) => to_json_text(&order),
            Err(e) => error_text(&e),
        }
    }

    /// 查询售后举证摘要（买家举证 / 卖家举证 / 物流 / 金额），供纠纷协调使用
    #[tool(name = "get-evidence")]
    async fn get_evidence(&self, Parameters(req): Parameters<GetEvidenceRequest>) -> String {
        match crate::get_evidence_core(&self.db, &req.order_id, req.user_id.as_deref()).await {
            Ok(summary) => to_json_text(&summary),
            Err(e) => error_text(&e),
        }
    }

    /// 查询可用优惠券（可按类型过滤）
    #[tool(name = "coupon-inquiry")]
    async fn coupon_inquiry(&self, Parameters(req): Parameters<CouponInquiryRequest>) -> String {
        match crate::list_coupons_core(&self.db, req.coupon_type.as_deref()).await {
            Ok(v) => to_json_text(&v),
            Err(e) => error_text(&e),
        }
    }
}

// 关键：把 tool_router 挂到 ServerHandler 上。少了这个宏，tools/list 返回空。
#[tool_handler(router = self.tool_router)]
impl ServerHandler for OrderMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(
                "order-service：提供订单查询、售后举证摘要、优惠券查询。\
                 敏感写操作（退货/退款）暂未开放。",
            )
    }
}

// ────────────────────────────────────────────────────────────────
// 输出格式化
// ────────────────────────────────────────────────────────────────

fn to_json_text<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|e| format!("序列化失败: {e}"))
}

/// 错误转成可读文本。
///
/// 对齐项目原则：「服务不可用时返回诚实的错误，不编造数据」——
/// 查询失败时绝不返回占位数据，避免 Agent 基于假信息编造回答。
fn error_text(e: &crate::ApiError) -> String {
    serde_json::json!({
        "error": true,
        "message": e.to_string(),
    })
    .to_string()
}

// ────────────────────────────────────────────────────────────────
// 单元测试 —— 不需要真实数据库
//
// 用 connect_lazy 建池（不发起实际连接），仅验证协议层的正确性：
// 工具名、inputSchema、描述。这是 Phase 1 最容易出错的部分。
// ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造测试用 server。
    ///
    /// 注意：`connect_lazy` 虽不发起真实连接，但仍需 Tokio 上下文，
    /// 因此所有用例必须用 `#[tokio::test]` 而非 `#[test]`。
    fn test_server() -> OrderMcpServer {
        let pool = sqlx::postgres::PgPoolOptions::new()
            .connect_lazy("postgres://user:pass@localhost:5432/orders")
            .expect("connect_lazy 不应失败（它不发起连接）");
        OrderMcpServer::new(pool)
    }

    #[tokio::test]
    async fn tools_registered_with_kebab_case_names() {
        let names = test_server().tool_names();

        // 回归保护：rmcp 默认用蛇形方法名，必须靠 #[tool(name=...)] 覆盖为
        // shop-agent 的 action 名。若此处出现 query_order，说明 name 属性被漏掉。
        assert!(
            names.contains(&"query-order".to_string()),
            "缺少 query-order，实际: {names:?}"
        );
        assert!(
            names.contains(&"get-evidence".to_string()),
            "缺少 get-evidence，实际: {names:?}"
        );
        assert!(
            names.contains(&"coupon-inquiry".to_string()),
            "缺少 coupon-inquiry，实际: {names:?}"
        );

        // 绝不能出现蛇形名
        for n in &names {
            assert!(!n.contains('_'), "工具名应为连字符，实际: {n}");
        }
    }

    #[tokio::test]
    async fn only_readonly_tools_exposed() {
        let names = test_server().tool_names();

        // Phase 1 只暴露 3 个只读工具
        assert_eq!(names.len(), 3, "工具数量应为 3，实际: {names:?}");

        // 敏感/破坏性操作不得暴露
        for forbidden in [
            "request-return",
            "refund-confirm",
            "check-balance",
            "delete-order",
            "update-order",
            "create-order",
        ] {
            assert!(
                !names.iter().any(|n| n == forbidden),
                "不应暴露敏感/破坏性操作: {forbidden}"
            );
        }
    }

    #[tokio::test]
    async fn input_schema_declares_order_id_as_required() {
        let server = test_server();
        let tool = server
            .tool_router
            .list_all()
            .into_iter()
            .find(|t| t.name == "query-order")
            .expect("query-order 未注册");

        let schema = tool.schema_as_json_value();
        let props = schema.get("properties").expect("inputSchema 缺 properties");

        // 硬强制协同：order_id 必须在服务端 schema 中声明（设计文档 §5.3）——
        // 切除发生在 shop-agent 侧，服务端须保持契约完整。
        assert!(props.get("order_id").is_some(), "order_id 应出现在 schema 中");

        let required = schema
            .get("required")
            .and_then(|r| r.as_array())
            .expect("inputSchema 缺 required");
        let required: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(
            required.contains(&"order_id"),
            "order_id 应为必填，实际: {required:?}"
        );
    }

    #[tokio::test]
    async fn coupon_inquiry_schema_has_no_required_fields() {
        let server = test_server();
        let tool = server
            .tool_router
            .list_all()
            .into_iter()
            .find(|t| t.name == "coupon-inquiry")
            .expect("coupon-inquiry 未注册");

        let schema = tool.schema_as_json_value();
        // coupon_type 可选：不传应返回全部
        let required = schema.get("required").and_then(|r| r.as_array());
        assert!(
            required.map(|r| r.is_empty()).unwrap_or(true),
            "coupon_type 应为可选，实际 required: {required:?}"
        );
    }
}
