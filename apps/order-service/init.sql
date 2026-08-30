-- 订单服务库初始化（由 PostgreSQL 容器首次启动时自动执行）
-- 兼容复用 pgvector：pgvector 默认库为 shop_agent，此处确保 orders 库存在（幂等，可重复执行）
SELECT 'CREATE DATABASE orders'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'orders')\gexec
\c orders

CREATE TABLE IF NOT EXISTS orders (
    order_id           TEXT PRIMARY KEY,
    order_amount_yuan  BIGINT NOT NULL,
    buyer_evidence     JSONB NOT NULL,
    seller_evidence    JSONB NOT NULL,
    logistics          JSONB NOT NULL,
    user_id            TEXT NOT NULL DEFAULT 'default',   -- 归属用户（方案A：数据级权限基础）
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 注入原 Python Mock 中的示例举证数据，使迁移后行为一致
-- 注意：JSON 内不含单引号，可直接作为 SQL 字符串字面量
INSERT INTO orders (order_id, order_amount_yuan, buyer_evidence, seller_evidence, logistics, user_id)
VALUES (
    'ORDER_WM20240601_001',
    3299,
    '{
        "photos": [
            {"url": "https://cdn.shop.example/evidence/photos/buyer_001_01.jpg", "description": "洗衣机玻璃面板碎裂特写，裂纹从左上角延伸至右下角"},
            {"url": "https://cdn.shop.example/evidence/photos/buyer_001_02.jpg", "description": "外包装纸箱侧面有撞击凹陷痕迹"},
            {"url": "https://cdn.shop.example/evidence/photos/buyer_001_03.jpg", "description": "洗衣机整体外观，面板碎裂部位全景"},
            {"url": "https://cdn.shop.example/evidence/photos/buyer_001_04.jpg", "description": "物流面单特写，单号 SF123456789，收件人信息完整"}
        ],
        "complaint_time": "2024-06-03 14:22:00",
        "complaint_channel": "在线客服",
        "buyer_note": "收到货打开就发现玻璃面板碎了，外包装也有撞击痕迹"
    }'::jsonb,
    '{
        "verification_video": {
            "url": "https://cdn.shop.example/evidence/videos/seller_001.mp4",
            "duration_seconds": 58,
            "recorded_at": "2024-06-01 14:30:00",
            "description": "发货前验机视频：全程无剪辑，面板完好，通电测试正常",
            "key_frames": [
                "00:00-00:15: 完整外观展示，玻璃面板无任何裂纹",
                "00:16-00:35: 通电启动，显示屏正常亮起",
                "00:36-00:58: 各功能旋钮测试，进水排水正常"
            ]
        },
        "shipping_insurance": {"insured": true, "insured_amount_yuan": 3000, "insurance_company": "顺丰保价"},
        "seller_note": "发货前已录制完整验机视频，面板完好，快递运输中造成的破损应由快递公司负责"
    }'::jsonb,
    '{
        "tracking_number": "SF123456789",
        "carrier": "顺丰速运",
        "shipped_at": "2024-06-01 15:00:00",
        "delivered_at": "2024-06-03 12:15:00",
        "signed_by": "本人签收"
    }'::jsonb
)
ON CONFLICT (order_id) DO NOTHING;

-- ═══════════════════════════════════════════════════════════════════════
-- 业务 API 支撑表（替代 shop-agent 内的本地 Mock：余额 / 优惠券 / 退货 / 退款）
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS balances (
    client_id      TEXT PRIMARY KEY,
    balance        DOUBLE PRECISION NOT NULL DEFAULT 0,
    points         INTEGER NOT NULL DEFAULT 0,
    coupons_count  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS coupons (
    id           SERIAL PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    type         TEXT NOT NULL,
    threshold    DOUBLE PRECISION,
    discount     DOUBLE PRECISION,
    discount_rate DOUBLE PRECISION,
    expire       TEXT
);

CREATE TABLE IF NOT EXISTS returns (
    id          SERIAL PRIMARY KEY,
    return_id   TEXT NOT NULL,
    order_id    TEXT NOT NULL,
    user_id     TEXT NOT NULL DEFAULT 'default',
    reason      TEXT,
    status      TEXT NOT NULL DEFAULT '待审核',
    refund_amount DOUBLE PRECISION,
    expected_refund_time TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS refunds (
    id          SERIAL PRIMARY KEY,
    order_id    TEXT NOT NULL,
    user_id     TEXT NOT NULL DEFAULT 'default',
    reason      TEXT,
    refund_amount DOUBLE PRECISION,
    status      TEXT NOT NULL DEFAULT 'PENDING_HUMAN_APPROVAL',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 注入原 Python Mock 的示例数据，使迁移后行为一致
INSERT INTO balances (client_id, balance, points, coupons_count)
VALUES ('default', 520.00, 1280, 3)
ON CONFLICT (client_id) DO NOTHING;

INSERT INTO coupons (name, type, threshold, discount, expire) VALUES
    ('满200减30',        '满减券', 200, 30, '2026-06-30'),
    ('新用户满100减15',  '满减券', 100, 15, '2026-06-15'),
    ('全场9折',          '折扣券', NULL, NULL, '2026-06-10'),
    ('免运费券',          '运费券', NULL, NULL, '2026-06-20')
ON CONFLICT (name) DO NOTHING;

-- ═══════════════════════════════════════════════════════════════════════════════
-- Agent 执行状态 + 审计 + 审批表（Phase 1 PostgreSQL 化）
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS agent_executions (
    thread_id         TEXT NOT NULL PRIMARY KEY,
    status            TEXT NOT NULL DEFAULT 'running',
    current_node      TEXT,
    state_snapshot    JSONB NOT NULL,
    context           JSONB DEFAULT '{}',
    version           BIGINT NOT NULL DEFAULT 0,
    error_message     TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_agent_executions_status ON agent_executions(status);
CREATE INDEX IF NOT EXISTS idx_agent_executions_created_at ON agent_executions(created_at);

CREATE TABLE IF NOT EXISTS agent_events (
    event_id          BIGSERIAL,
    thread_id         TEXT NOT NULL,
    event_type        TEXT NOT NULL,
    node_name         TEXT,
    payload           JSONB NOT NULL,
    operator_id       TEXT,
    source            TEXT NOT NULL DEFAULT 'shop-agent',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
) PARTITION BY RANGE (created_at);

CREATE INDEX IF NOT EXISTS idx_agent_events_thread ON agent_events(thread_id);
CREATE INDEX IF NOT EXISTS idx_agent_events_type ON agent_events(event_type);
CREATE INDEX IF NOT EXISTS idx_agent_events_created_at ON agent_events(created_at);

CREATE TABLE IF NOT EXISTS agent_events_2026_08 PARTITION OF agent_events
    FOR VALUES FROM ('2026-08-01') TO ('2026-09-01');

CREATE TABLE IF NOT EXISTS agent_events_2026_09 PARTITION OF agent_events
    FOR VALUES FROM ('2026-09-01') TO ('2026-10-01');

CREATE TABLE IF NOT EXISTS agent_events_2026_10 PARTITION OF agent_events
    FOR VALUES FROM ('2026-10-01') TO ('2026-11-01');

CREATE TABLE IF NOT EXISTS human_approvals (
    approval_id           TEXT NOT NULL PRIMARY KEY,
    command_name          TEXT NOT NULL,
    action                TEXT NOT NULL,
    params                JSONB NOT NULL,
    params_masked         JSONB,
    conversation_id       TEXT NOT NULL,
    domain                TEXT NOT NULL DEFAULT 'ecommerce',
    user_id               TEXT,
    metadata              JSONB DEFAULT '{}',
    status                TEXT NOT NULL DEFAULT 'pending_approval',
    message               TEXT,
    undo_data             JSONB,
    graph_state_snapshot  JSONB,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at           TIMESTAMPTZ,
    resolved_by           TEXT,
    resolution_note       TEXT
);

CREATE INDEX IF NOT EXISTS idx_human_approvals_conversation ON human_approvals(conversation_id);
CREATE INDEX IF NOT EXISTS idx_human_approvals_status ON human_approvals(status);
CREATE INDEX IF NOT EXISTS idx_human_approvals_created_at ON human_approvals(created_at);
