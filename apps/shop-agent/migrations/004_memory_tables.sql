-- =============================================================================
-- Phase 2: L3 长期记忆基础版
-- 创建 memory_orders + memory_complaints 表
-- =============================================================================

-- 订单记忆表
CREATE TABLE IF NOT EXISTS memory_orders (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    order_id VARCHAR(64) NOT NULL,
    status VARCHAR(32),
    product_ids JSONB,
    total_amount DECIMAL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 订单记忆表索引
CREATE INDEX IF NOT EXISTS idx_memory_orders_user_id ON memory_orders(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_orders_created ON memory_orders(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_orders_order_id ON memory_orders(order_id);

-- 投诉记忆表
CREATE TABLE IF NOT EXISTS memory_complaints (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    complaint_id VARCHAR(64) NOT NULL,
    order_id VARCHAR(64),
    category VARCHAR(32),
    status VARCHAR(32),
    resolution TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    resolved_at TIMESTAMP
);

-- 投诉记忆表索引
CREATE INDEX IF NOT EXISTS idx_memory_complaints_user_id ON memory_complaints(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_complaints_created ON memory_complaints(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_complaints_status ON memory_complaints(status) WHERE status = '处理中';
