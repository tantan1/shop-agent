# 订单服务（order-service）

Rust + PostgreSQL 实现的轻量订单服务，作为 `shop-agent` 的**外部依赖**，提供售后举证等真实数据，替代原先内联在 `dispute_coordinator.py` 中的 Mock。

## 技术栈

- **axum** —— HTTP 框架（路由 / JSON）
- **sqlx**（runtime-tokio + postgres + json） —— 异步 PostgreSQL 访问，纯 Rust 驱动，运行时不依赖 libpq
- **PostgreSQL 16** —— 真实持久化的订单 / 举证数据

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET  | `/health` | 健康检查 |
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
