//! 订单服务（Rust + PostgreSQL）
//!
//! 提供最简单的 CRUD 接口 + 售后举证摘要接口。
//! shop-agent 的纠纷协调器通过 `GET /orders/:id/evidence` 获取双方举证数据，
//! 替代原先内联在 Python 侧的 Mock。
//!
//! 常用库：
//!   - axum         : HTTP 框架（路由 / JSON）
//!   - sqlx         : 异步 PostgreSQL 访问（runtime-tokio，纯 Rust 驱动，无需 libpq）
//!   - serde        : 序列化 / 反序列化

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::postgres::PgPoolOptions;
use sqlx::types::Json as SqlJson;
use sqlx::FromRow;
use sqlx::PgPool;
use std::error::Error;
use std::net::SocketAddr;

pub mod mcp;

#[derive(Clone)]
pub struct AppState {
    pub db: PgPool,
}

/// 统一的业务错误类型。
///
/// 存在意义：让 REST 与 MCP 两条路径**共用同一套业务逻辑**却各自表达错误——
/// REST 转成 HTTP 状态码，MCP 转成可读文本（见设计文档 §3.3「handler 复用」）。
#[derive(Debug)]
pub enum ApiError {
    NotFound,
    Forbidden,
    Internal(String),
}

impl From<ApiError> for StatusCode {
    fn from(e: ApiError) -> Self {
        match e {
            ApiError::NotFound => StatusCode::NOT_FOUND,
            ApiError::Forbidden => StatusCode::FORBIDDEN,
            ApiError::Internal(msg) => {
                eprintln!("internal error: {msg}");
                StatusCode::INTERNAL_SERVER_ERROR
            }
        }
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::NotFound => write!(f, "订单不存在"),
            ApiError::Forbidden => write!(f, "无权访问该订单"),
            ApiError::Internal(_) => write!(f, "服务内部错误"),
        }
    }
}

/// 订单行（与 `orders` 表一一对应，jsonb 列用 `SqlJson<Value>` 读写）。
#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct Order {
    order_id: String,
    order_amount_yuan: i64,
    buyer_evidence: SqlJson<Value>,
    seller_evidence: SqlJson<Value>,
    logistics: SqlJson<Value>,
    #[sqlx(default)]
    user_id: String,
}

// ── 售后举证摘要（与 Python 消费侧期望的字段完全一致）──────────────

#[derive(Debug, Serialize)]
struct BuyerEvidenceSummary {
    photo_count: usize,
    photo_descriptions: Vec<String>,
    complaint_time: String,
    complaint_note: String,
}

#[derive(Debug, Serialize)]
struct SellerEvidenceSummary {
    has_verification_video: bool,
    video_recorded_at: String,
    video_description: String,
    shipping_insured: bool,
    insured_amount_yuan: i64,
    seller_note: String,
}

#[derive(Debug, Serialize)]
struct LogisticsSummary {
    tracking_number: String,
    carrier: String,
    shipped_at: String,
    delivered_at: String,
    signed_by: String,
}

#[derive(Debug, Serialize)]
pub struct EvidenceSummary {
    order_id: String,
    order_amount_yuan: i64,
    buyer_evidence: BuyerEvidenceSummary,
    seller_evidence: SellerEvidenceSummary,
    logistics: LogisticsSummary,
}

/// 由存储的 jsonb 构造对外证据摘要（逻辑与原 Python Mock 一致）。
fn build_evidence(order: &Order) -> EvidenceSummary {
    let b = &order.buyer_evidence.0;
    let photos = b
        .get("photos")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let photo_count = photos.len();
    let photo_descriptions: Vec<String> = photos
        .iter()
        .filter_map(|p| p.get("description").and_then(|d| d.as_str()).map(str::to_string))
        .collect();
    let complaint_time = b
        .get("complaint_time")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let complaint_note = b
        .get("buyer_note")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let s = &order.seller_evidence.0;
    let video = s.get("verification_video");
    let has_video = video.is_some();
    let video_recorded_at = video
        .and_then(|v| v.get("recorded_at"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let video_description = video
        .and_then(|v| v.get("description"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let insurance = s.get("shipping_insurance");
    let shipping_insured = insurance
        .and_then(|v| v.get("insured"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let insured_amount_yuan = insurance
        .and_then(|v| v.get("insured_amount_yuan"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let seller_note = s
        .get("seller_note")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let l = &order.logistics.0;
    let logistics = LogisticsSummary {
        tracking_number: l.get("tracking_number").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        carrier: l.get("carrier").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        shipped_at: l.get("shipped_at").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        delivered_at: l.get("delivered_at").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        signed_by: l.get("signed_by").and_then(|v| v.as_str()).unwrap_or("").to_string(),
    };

    EvidenceSummary {
        order_id: order.order_id.clone(),
        order_amount_yuan: order.order_amount_yuan,
        buyer_evidence: BuyerEvidenceSummary {
            photo_count,
            photo_descriptions,
            complaint_time,
            complaint_note,
        },
        seller_evidence: SellerEvidenceSummary {
            has_verification_video: has_video,
            video_recorded_at,
            video_description,
            shipping_insured,
            insured_amount_yuan,
            seller_note,
        },
        logistics,
    }
}

// ── Handlers ───────────────────────────────────────────────────────

/// 健康检查。
///
/// 原先返回纯文本 `"ok"`；现扩展为 JSON 并附带 MCP 状态。
/// 调用方（docker-compose healthcheck、smoke_e2e）只判断 HTTP 200，
/// 因此改为 JSON 不影响既有消费方。
async fn health() -> Json<Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "order-service",
        "mcp": {
            "enabled": true,
            "endpoint": "/mcp",
            "tools": ["query-order", "get-evidence", "coupon-inquiry"],
        }
    }))
}

async fn list_orders(State(state): State<AppState>) -> Result<Json<Vec<Order>>, StatusCode> {
    let rows = sqlx::query_as::<_, Order>(
        "SELECT order_id, order_amount_yuan, buyer_evidence, seller_evidence, logistics FROM orders",
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| {
        eprintln!("list_orders error: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok(Json(rows))
}

/// 读接口的身份查询参数（用于数据级归属校验）
#[derive(Debug, Deserialize)]
struct UserIdQuery {
    #[serde(default)]
    user_id: Option<String>,
}

// ════════════════════════════════════════════════════════════════
// 核心业务逻辑（core）—— REST 与 MCP 共用
//
// 设计文档 §3.3：MCP tool 只做协议适配（参数转换），**不另写业务逻辑**，
// 与 REST handler 共用这些 core 函数，保证两条路径逻辑天然一致。
// ════════════════════════════════════════════════════════════════

/// 查询单个订单。
///
/// `user_id` 为 `Some` 时做归属校验（不存在→NotFound，不符→Forbidden）；
/// 为 `None` 时（兼容旧调用）仅按存在性返回。
pub async fn query_order_core(
    db: &PgPool,
    order_id: &str,
    user_id: Option<&str>,
) -> Result<Order, ApiError> {
    let row = sqlx::query_as::<_, Order>(
        "SELECT order_id, order_amount_yuan, buyer_evidence, seller_evidence, logistics, user_id \
         FROM orders WHERE order_id = $1",
    )
    .bind(order_id)
    .fetch_optional(db)
    .await
    .map_err(|e| ApiError::Internal(format!("query_order error: {e}")))?;

    let order = row.ok_or(ApiError::NotFound)?;

    if let Some(uid) = user_id {
        // 无 user_id 标注的订单（历史/公开数据）视为可读，避免误杀
        if order.user_id != uid && !order.user_id.is_empty() {
            return Err(ApiError::Forbidden);
        }
    }
    Ok(order)
}

/// 生成售后举证摘要（买家/卖家举证 + 物流 + 金额）。
pub async fn get_evidence_core(
    db: &PgPool,
    order_id: &str,
    user_id: Option<&str>,
) -> Result<EvidenceSummary, ApiError> {
    let order = query_order_core(db, order_id, user_id).await?;
    Ok(build_evidence(&order))
}

/// 查询优惠券列表，`coupon_type` 为 `None` 时返回全部。
pub async fn list_coupons_core(
    db: &PgPool,
    coupon_type: Option<&str>,
) -> Result<Value, ApiError> {
    let rows: Vec<(String, String, Option<f64>, Option<f64>, Option<f64>, Option<String>)> =
        match coupon_type {
            Some(ct) => sqlx::query_as(
                "SELECT name, type, threshold, discount, discount_rate, expire \
                 FROM coupons WHERE type LIKE '%' || $1 || '%'",
            )
            .bind(ct)
            .fetch_all(db)
            .await,
            None => sqlx::query_as(
                "SELECT name, type, threshold, discount, discount_rate, expire FROM coupons",
            )
            .fetch_all(db)
            .await,
        }
        .map_err(|e| ApiError::Internal(format!("list_coupons error: {e}")))?;

    let coupons: Vec<Value> = rows
        .into_iter()
        .map(|(name, ctype, threshold, discount, rate, expire)| {
            serde_json::json!({
                "name": name,
                "type": ctype,
                "threshold": threshold,
                "discount": discount,
                "discount_rate": rate,
                "expire": expire,
            })
        })
        .collect();
    let total = coupons.len();

    Ok(serde_json::json!({
        "message": "全部优惠券",
        "data": { "coupons": coupons, "total": total }
    }))
}

// ════════════════════════════════════════════════════════════════
// REST handler —— 薄包装，仅做 extractor 解析 + 错误码转换
// ════════════════════════════════════════════════════════════════

async fn get_order(
    State(state): State<AppState>,
    Path(order_id): Path<String>,
    Query(q): Query<UserIdQuery>,
) -> Result<Json<Order>, StatusCode> {
    query_order_core(&state.db, &order_id, q.user_id.as_deref())
        .await
        .map(Json)
        .map_err(Into::into)
}

async fn get_evidence(
    State(state): State<AppState>,
    Path(order_id): Path<String>,
    Query(q): Query<UserIdQuery>,
) -> Result<Json<EvidenceSummary>, StatusCode> {
    get_evidence_core(&state.db, &order_id, q.user_id.as_deref())
        .await
        .map(Json)
        .map_err(Into::into)
}

async fn create_order(
    State(state): State<AppState>,
    Json(order): Json<Order>,
) -> Result<(StatusCode, Json<Order>), StatusCode> {
    sqlx::query(
        "INSERT INTO orders (order_id, order_amount_yuan, buyer_evidence, seller_evidence, logistics) \
         VALUES ($1, $2, $3, $4, $5) \
         ON CONFLICT (order_id) DO UPDATE SET \
            order_amount_yuan = EXCLUDED.order_amount_yuan, \
            buyer_evidence = EXCLUDED.buyer_evidence, \
            seller_evidence = EXCLUDED.seller_evidence, \
            logistics = EXCLUDED.logistics",
    )
    .bind(&order.order_id)
    .bind(order.order_amount_yuan)
    .bind(&order.buyer_evidence)
    .bind(&order.seller_evidence)
    .bind(&order.logistics)
    .execute(&state.db)
    .await
    .map_err(|e| {
        eprintln!("create_order error: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok((StatusCode::CREATED, Json(order)))
}

async fn update_order(
    State(state): State<AppState>,
    Path(order_id): Path<String>,
    Json(order): Json<Order>,
) -> Result<Json<Order>, StatusCode> {
    let res = sqlx::query(
        "UPDATE orders SET order_amount_yuan = $2, buyer_evidence = $3, \
         seller_evidence = $4, logistics = $5 WHERE order_id = $1",
    )
    .bind(&order_id)
    .bind(order.order_amount_yuan)
    .bind(&order.buyer_evidence)
    .bind(&order.seller_evidence)
    .bind(&order.logistics)
    .execute(&state.db)
    .await
    .map_err(|e| {
        eprintln!("update_order error: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    if res.rows_affected() == 0 {
        return Err(StatusCode::NOT_FOUND);
    }
    let mut o = order;
    o.order_id = order_id;
    Ok(Json(o))
}

async fn delete_order(
    State(state): State<AppState>,
    Path(order_id): Path<String>,
) -> StatusCode {
    let res = sqlx::query("DELETE FROM orders WHERE order_id = $1")
        .bind(&order_id)
        .execute(&state.db)
        .await;
    match res {
        Ok(r) if r.rows_affected() > 0 => StatusCode::NO_CONTENT,
        _ => StatusCode::NOT_FOUND,
    }
}

// ── 业务 API（替代 shop-agent 内的本地 Mock：余额 / 优惠券 / 退货 / 退款确认）──

#[derive(Debug, Deserialize)]
struct ApiPayload {
    #[serde(default)]
    order_id: Option<String>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    coupon_type: Option<String>,
    #[serde(default)]
    refund_amount: Option<f64>,
    #[serde(default)]
    user_id: Option<String>, // 新增：调用方身份（由 shop-agent 从会话上下文注入），用于数据级归属校验
}

/// 查询账户余额 / 积分（默认客户端 'default'）
async fn account_balance(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    let row: (f64, i32, i32) = sqlx::query_as(
        "SELECT balance, points, coupons_count FROM balances WHERE client_id = $1",
    )
    .bind("default")
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        eprintln!("account_balance error: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?
    .unwrap_or((0.0, 0, 0));

    Ok(Json(serde_json::json!({
        "message": "账户查询成功",
        "data": {
            "balance": row.0,
            "points": row.1,
            "coupons_count": row.2
        }
    })))
}

/// 查询优惠券（可选按类型过滤）—— 薄包装，业务逻辑在 `list_coupons_core`
async fn list_coupons(
    State(state): State<AppState>,
    Json(payload): Json<ApiPayload>,
) -> Result<Json<Value>, StatusCode> {
    list_coupons_core(&state.db, payload.coupon_type.as_deref())
        .await
        .map(Json)
        .map_err(Into::into)
}

/// 校验订单存在且归属当前用户；不存在→404，归属不符→403。
///
/// 无 `user_id` 标注的订单（历史/公开数据）视为可读，避免误杀；
/// 但写操作 handler 应在调用前强校验 `user_id` 必填（见 `create_return`/`refund_confirm`）。
async fn verify_ownership(
    state: &AppState,
    order_id: &str,
    user_id: &str,
) -> Result<Order, StatusCode> {
    let row = sqlx::query_as::<_, Order>(
        "SELECT order_id, order_amount_yuan, buyer_evidence, seller_evidence, logistics, user_id \
         FROM orders WHERE order_id = $1",
    )
    .bind(order_id)
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        eprintln!("verify_ownership error: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    match row {
        None => Err(StatusCode::NOT_FOUND),
        Some(o) => {
            if o.user_id == user_id || o.user_id.is_empty() {
                Ok(o)
            } else {
                Err(StatusCode::FORBIDDEN) // 403：订单不属于当前用户
            }
        }
    }
}

/// 提交退货申请（真实写库，返回退货单号）
async fn create_return(
    State(state): State<AppState>,
    Json(payload): Json<ApiPayload>,
) -> Result<Json<Value>, StatusCode> {
    let order_id = match &payload.order_id {
        Some(id) => id.clone(),
        None => return Err(StatusCode::BAD_REQUEST), // 缺 order_id，拒绝落库
    };
    // 方案 A：写操作前强校验归属，缺失 user_id 或归属不符直接拒（非法请求不落库）
    let user_id = payload.user_id.clone().ok_or(StatusCode::BAD_REQUEST)?;
    if verify_ownership(&state, &order_id, &user_id).await.is_err() {
        return Err(StatusCode::FORBIDDEN);
    }

    let reason = payload.reason.clone().unwrap_or_else(|| "未说明".to_string());
    let suffix = if order_id.len() > 6 {
        &order_id[order_id.len() - 6..]
    } else {
        order_id.as_str()
    };
    let return_id = format!("RT{suffix}");
    let refund_amount = 299.00_f64; // 示例退款金额（真实场景应由订单金额计算）

    sqlx::query(
        "INSERT INTO returns (return_id, order_id, user_id, reason, status, refund_amount, expected_refund_time) \
         VALUES ($1, $2, $3, $4, '待审核', $5, '1-3个工作日')",
    )
    .bind(&return_id)
    .bind(&order_id)
    .bind(&user_id)
    .bind(&reason)
    .bind(refund_amount)
    .execute(&state.db)
    .await
    .map_err(|e| {
        eprintln!("create_return error: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(serde_json::json!({
        "message": "退货申请已提交",
        "data": {
            "return_id": return_id,
            "order_id": order_id,
            "reason": reason,
            "status": "待审核",
            "refund_amount": refund_amount,
            "expected_refund_time": "1-3个工作日"
        }
    })))
}

/// 记录退款确认请求（真实写库，待人工审批）
async fn refund_confirm(
    State(state): State<AppState>,
    Json(payload): Json<ApiPayload>,
) -> Result<Json<Value>, StatusCode> {
    let order_id = match &payload.order_id {
        Some(id) => id.clone(),
        None => return Err(StatusCode::BAD_REQUEST),
    };
    // 方案 A：写操作前强校验归属
    let user_id = payload.user_id.clone().ok_or(StatusCode::BAD_REQUEST)?;
    if verify_ownership(&state, &order_id, &user_id).await.is_err() {
        return Err(StatusCode::FORBIDDEN);
    }

    let reason = payload.reason.clone().unwrap_or_default();
    let refund_amount = payload.refund_amount.unwrap_or(0.0);

    sqlx::query(
        "INSERT INTO refunds (order_id, user_id, reason, refund_amount, status) \
         VALUES ($1, $2, $3, $4, 'PENDING_HUMAN_APPROVAL')",
    )
    .bind(&order_id)
    .bind(&user_id)
    .bind(&reason)
    .bind(refund_amount)
    .execute(&state.db)
    .await
    .map_err(|e| {
        eprintln!("refund_confirm error: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(serde_json::json!({
        "message": "退款申请需要人工审批确认",
        "data": {
            "order_id": order_id,
            "reason": reason,
            "refund_amount": refund_amount,
            "status": "PENDING_HUMAN_APPROVAL"
        }
    })))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let database_url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL 必须设置（postgres://user:pass@host:5432/db）");

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await
        .expect("无法连接 PostgreSQL，请检查 DATABASE_URL 与数据库可访问性");

    let state = AppState { db: pool };

    // MCP：与 REST 共存于同一 axum 进程，共享同一个 sqlx 连接池
    // （Phase 0 Spike V2 已验证可行；设计文档 §3.2）
    //
    // 类型标注不可省略：SessionManager 泛型参数（M）无法从 Default::default() 推断，
    // 需显式指定为 LocalSessionManager。
    let mcp_service: rmcp::transport::streamable_http_server::StreamableHttpService<
        mcp::OrderMcpServer,
        rmcp::transport::streamable_http_server::session::local::LocalSessionManager,
    > = rmcp::transport::streamable_http_server::StreamableHttpService::new(
        {
            let db = state.db.clone();
            move || Ok(mcp::OrderMcpServer::new(db.clone()))
        },
        Default::default(),
        Default::default(),
    );

    let app = Router::new()
        .route("/health", get(health))
        .route("/orders", get(list_orders).post(create_order))
        .route(
            "/orders/:order_id",
            get(get_order).put(update_order).delete(delete_order),
        )
        .route("/orders/:order_id/evidence", get(get_evidence))
        .route("/api/account/balance", post(account_balance))
        .route("/api/coupons/list", post(list_coupons))
        .route("/api/returns/create", post(create_return))
        .route("/api/refunds/confirm", post(refund_confirm))
        .nest_service("/mcp", mcp_service)
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("order-service listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
