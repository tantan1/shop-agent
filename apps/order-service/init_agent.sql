-- Agent 执行状态 + 审计 + 审批 PostgreSQL 化（Phase 1 建表）

-- ═══════════════════════════════════════════════════════════════════════════════
-- 1. agent_executions（执行状态表）
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

-- ═══════════════════════════════════════════════════════════════════════════════
-- 2. agent_events（事件溯源表，按 created_at 时间分区）
-- ═══════════════════════════════════════════════════════════════════════════════
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

-- 分区：2026年8月
CREATE TABLE IF NOT EXISTS agent_events_2026_08 PARTITION OF agent_events
    FOR VALUES FROM ('2026-08-01') TO ('2026-09-01');

-- 分区：2026年9月
CREATE TABLE IF NOT EXISTS agent_events_2026_09 PARTITION OF agent_events
    FOR VALUES FROM ('2026-09-01') TO ('2026-10-01');

-- 分区：2026年10月
CREATE TABLE IF NOT EXISTS agent_events_2026_10 PARTITION OF agent_events
    FOR VALUES FROM ('2026-10-01') TO ('2026-11-01');

-- ═══════════════════════════════════════════════════════════════════════════════
-- 3. human_approvals（人在回路审批记录表）
-- ═══════════════════════════════════════════════════════════════════════════════
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
