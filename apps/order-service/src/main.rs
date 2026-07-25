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
    extract::{Path, State},
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

#[derive(Clone)]
struct AppState {
    db: PgPool,
}

/// 订单行（与 `orders` 表一一对应，jsonb 列用 `SqlJson<Value>` 读写）。
#[derive(Debug, Serialize, Deserialize, FromRow)]
struct Order {
    order_id: String,
    order_amount_yuan: i64,
    buyer_evidence: SqlJson<Value>,
    seller_evidence: SqlJson<Value>,
    logistics: SqlJson<Value>,
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
struct EvidenceSummary {
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

async fn health() -> &'static str {
    "ok"
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

async fn get_order(
    State(state): State<AppState>,
    Path(order_id): Path<String>,
) -> Result<Json<Order>, StatusCode> {
    let row = sqlx::query_as::<_, Order>(
        "SELECT order_id, order_amount_yuan, buyer_evidence, seller_evidence, logistics \
         FROM orders WHERE order_id = $1",
    )
    .bind(&order_id)
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        eprintln!("get_order error: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    row.map(Json).ok_or(StatusCode::NOT_FOUND)
}

async fn get_evidence(
    State(state): State<AppState>,
    Path(order_id): Path<String>,
) -> Result<Json<EvidenceSummary>, StatusCode> {
    let row = sqlx::query_as::<_, Order>(
        "SELECT order_id, order_amount_yuan, buyer_evidence, seller_evidence, logistics \
         FROM orders WHERE order_id = $1",
    )
    .bind(&order_id)
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        eprintln!("get_evidence error: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    match row {
        Some(o) => Ok(Json(build_evidence(&o))),
        None => Err(StatusCode::NOT_FOUND),
    }
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

/// 查询优惠券（可选按类型过滤）
async fn list_coupons(
    State(state): State<AppState>,
    Json(payload): Json<ApiPayload>,
) -> Result<Json<Value>, StatusCode> {
    let rows: Vec<(String, String, Option<f64>, Option<f64>, Option<f64>, Option<String>)> = match &payload.coupon_type {
        Some(ct) => sqlx::query_as(
            "SELECT name, type, threshold, discount, discount_rate, expire \
             FROM coupons WHERE type LIKE '%' || $1 || '%'",
        )
        .bind(ct)
        .fetch_all(&state.db)
        .await,
        None => sqlx::query_as(
            "SELECT name, type, threshold, discount, discount_rate, expire FROM coupons",
        )
        .fetch_all(&state.db)
        .await,
    }
    .map_err(|e| {
        eprintln!("list_coupons error: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

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

    Ok(Json(serde_json::json!({
        "message": "全部优惠券",
        "data": { "coupons": coupons, "total": total }
    })))
}

/// 提交退货申请（真实写库，返回退货单号）
async fn create_return(
    State(state): State<AppState>,
    Json(payload): Json<ApiPayload>,
) -> Result<Json<Value>, StatusCode> {
    let order_id = payload.order_id.clone().unwrap_or_else(|| "未指定".to_string());
    let reason = payload.reason.clone().unwrap_or_else(|| "未说明".to_string());
    let suffix = if order_id.len() > 6 {
        &order_id[order_id.len() - 6..]
    } else {
        order_id.as_str()
    };
    let return_id = format!("RT{suffix}");
    let refund_amount = 299.00_f64; // 示例退款金额（真实场景应由订单金额计算）

    sqlx::query(
        "INSERT INTO returns (return_id, order_id, reason, status, refund_amount, expected_refund_time) \
         VALUES ($1, $2, $3, '待审核', $4, '1-3个工作日')",
    )
    .bind(&return_id)
    .bind(&order_id)
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
    let order_id = payload.order_id.clone().unwrap_or_default();
    let reason = payload.reason.clone().unwrap_or_default();
    let refund_amount = payload.refund_amount.unwrap_or(0.0);

    sqlx::query(
        "INSERT INTO refunds (order_id, reason, refund_amount, status) \
         VALUES ($1, $2, $3, 'PENDING_HUMAN_APPROVAL')",
    )
    .bind(&order_id)
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
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("order-service listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
