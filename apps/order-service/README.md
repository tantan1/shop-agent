# 订单服务（order-service）

Rust + PostgreSQL 实现的轻量订单服务，作为 `shop-agent` 的**外部依赖**，提供售后举证等真实数据，替代原先内联在 `dispute_coordinator.py` 中的 Mock。

## 技术栈

- **axum** —— HTTP 框架（路由 / JSON）
- **sqlx**（runtime-tokio + postgres + json） —— 异步 PostgreSQL 访问，纯 Rust 驱动，运行时不依赖 libpq
- **rmcp 3.1.4** —— MCP Server（Streamable HTTP），向 shop-agent 暴露业务工具
- **PostgreSQL 16** —— 真实持久化的订单 / 举证数据

## MCP 工具（推荐集成方式）

`POST /mcp` —— MCP Streamable HTTP 端点，与 REST 接口共存于同一 axum 进程、共用同一个 sqlx 连接池。

shop-agent 侧通过 `MCP_CLIENT_SERVERS` 配置连接，用 `tools/list` 动态发现工具，替代硬编码接口映射。

| MCP tool | 对应能力 | 底层函数 |
|---|---|---|
| `query-order` | 按订单号查询订单完整信息 | `query_order_core` |
| `get-evidence` | 售后举证摘要（买家/卖家举证 + 物流 + 金额） | `get_evidence_core` |
| `coupon-inquiry` | 优惠券查询（可按类型过滤） | `list_coupons_core` |

**设计约束**：

- 工具名**必须**是连字符（用 `#[tool(name = "query-order")]` 显式指定）——
  `rmcp` 默认以 Rust 方法名的蛇形暴露，与 shop-agent 的 action 名不一致。
  已有单测锁定，防止重构退化。
- MCP tool **不另写业务逻辑**，只做参数转换 + 调用 `*_core` 函数，与 REST handler 同源。
- **当前只暴露只读工具**。敏感写操作（`request-return` / `refund-confirm` / `check-balance`）
  待后续阶段接入；破坏性与管理类接口（`DELETE` / `PUT` / `POST /orders`）**永不暴露**。
- `order_id` 在服务端 schema 中**必须声明**（契约完整性）。
  字段级硬强制由 shop-agent 侧在消费时剥离——安全策略不外包给外部服务。

相关文档：`docs/architecture/order-service-mcp-design.md`（设计）、
`docs/architecture/order-service-mcp-implementation-plan.md`（实施计划 + Phase 0 Spike 结论）

## REST 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET  | `/health` | 健康检查（JSON，含 mcp 状态） |
| GET  | `/orders` | 订单列表 |
| POST | `/orders` | 创建 / 更新订单（upsert） |
| GET  | `/orders/:order_id` | 单个订单完整数据 |
| PUT  | `/orders/:order_id` | 更新订单 |
| DELETE | `/orders/:order_id` | 删除订单 |
| GET  | `/orders/:order_id/evidence` | **售后举证摘要**（shop-agent 调用此接口） |
| POST | `/api/account/balance` | 账户余额 / 积分（替代原本地 Mock） |
| POST | `/api/coupons/list` | 优惠券列表（可选 `coupon_type` 过滤） |
| POST | `/api/returns/create` | 提交退货申请（真实写库） |
| POST | `/api/refunds/confirm` | 记录退款确认请求（真实写库，待人工审批） |

`/evidence` 返回的字段与历史 Mock 完全一致：

```json
{
  "order_id": "ORDER_WM20240601_001",
  "order_amount_yuan": 3299,
  "buyer_evidence": { "photo_count": 4, "photo_descriptions": [...], "complaint_time": "...", "complaint_note": "..." },
  "seller_evidence": { "has_verification_video": true, "video_recorded_at": "...", "video_description": "...", "shipping_insured": true, "insured_amount_yuan": 3000, "seller_note": "..." },
  "logistics": { "tracking_number": "SF123456789", "carrier": "顺丰速运", "shipped_at": "...", "delivered_at": "...", "signed_by": "..." }
}
```

## 本地运行

### 方式一：独立 docker compose（推荐）

```bash
cd apps/order-service
docker compose up --build
```

### 方式二：本地直接跑（需本机 Postgres）

```bash
export DATABASE_URL=postgres://postgres:postgres@localhost:5434/orders
cargo run
```

## 与 shop-agent 的集成

根 `docker-compose.yml` 中的 `order-service` **复用已有的 `pgvector` PostgreSQL 实例**（不再单独起 `order-postgres`），库名为 `orders`，由 `apps/order-service/init.sql` 挂载进 pgvector 的 initdb 目录在首次启动时自动建库建表。`shop-agent` 通过环境变量 `ORDER_SERVICE_URL=http://order-service:8080` 访问，纠纷协调器在收集事实时调用 `GET /orders/:id/evidence` 获取真实举证数据；工具层（`check-balance` / `coupon-inquiry` / `request-return` / `refund-confirm`）通过 `POST /api/*` 调用本服务的真实接口，不再返回内联假数据；服务不可用时返回诚实的错误提示，不编造证据或余额。独立运行见 `apps/order-service/docker-compose.yml`（同样改用 `pgvector` 镜像）。

内存占用：Rust 服务空闲约 10–15 MB，PostgreSQL 约 50–80 MB（已调小）。
