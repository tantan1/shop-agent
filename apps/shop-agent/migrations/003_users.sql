-- =============================================================================
-- Phase 0: 轻量 User 模块 + 记忆架构前置
-- 创建 users + user_profiles 表
-- =============================================================================

-- 用户主表
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(64) PRIMARY KEY,
    user_type VARCHAR(32) NOT NULL DEFAULT 'anonymous',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- 用户画像表（记忆架构 Phase 2 使用）
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id VARCHAR(64) PRIMARY KEY REFERENCES users(id),
    preferences JSONB DEFAULT '{}',
    vip_level VARCHAR(32) DEFAULT 'normal',
    first_seen_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMP NOT NULL DEFAULT NOW(),
    total_orders INT DEFAULT 0,
    total_complaints INT DEFAULT 0,
    pending_issues JSONB DEFAULT '[]',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_users_last_seen ON users(last_seen_at);
CREATE INDEX IF NOT EXISTS idx_users_type ON users(user_type);
CREATE INDEX IF NOT EXISTS idx_user_profiles_last_seen ON user_profiles(last_seen_at);
