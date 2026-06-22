"""
基于角色的工具权限控制 —— 调用方应用角色（Mock 数据）

三层设计：
  1. 角色定义（Role）        — admin / operator / viewer
  2. Mock 调用方数据          — API Key → ClientInfo（角色 + 元数据）
  3. 权限检查函数             — 按角色判断工具是否可用

当前使用 mock 数据，后续可替换为数据库/Redis 查询。
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Optional


# ── 角色定义 ──────────────────────────────────────────────────────

class Role(str, Enum):
    """调用方角色枚举"""
    ADMIN = "admin"          # 管理端：所有工具可用
    OPERATOR = "operator"    # 运营端：不可执行退款
    VIEWER = "viewer"        # 只读端：仅查询类工具


# ── 角色 → 可用工具映射 ──────────────────────────────────────────

# 所有已注册的工具
_ALL_TOOLS: FrozenSet[str] = frozenset({
    "query-order",
    "check-shipping",
    "request-return",
    "check-balance",
    "coupon-inquiry",
    "knowledge_search",
})

# 只读工具（查询类，不产生副作用）
_READONLY_TOOLS: FrozenSet[str] = frozenset({
    "query-order",
    "check-shipping",
    "check-balance",
    "coupon-inquiry",
    "knowledge_search",
})

# 角色 → 可执行工具集合
ROLE_TOOL_PERMISSIONS: Dict[Role, FrozenSet[str]] = {
    Role.ADMIN:     _ALL_TOOLS,
    Role.OPERATOR:  _ALL_TOOLS - {"request-return"},
    Role.VIEWER:    _READONLY_TOOLS,
}


# ── Mock 调用方数据（API Key → 应用信息）────────────────────────

@dataclass(frozen=True)
class ClientInfo:
    """调用方应用上下文"""
    client_id: str
    role: Role
    client_name: str = ""
    api_key_prefix: str = ""  # 仅日志用，不存储完整 key

    @property
    def is_admin(self) -> bool:
        return self.role == Role.ADMIN


# Mock 调用方表：API Key → 应用信息
_MOCK_CLIENTS: Dict[str, ClientInfo] = {
    "ak_admin_2024": ClientInfo(
        client_id="order-service",
        role=Role.ADMIN,
        client_name="订单管理后台",
        api_key_prefix="ak_admin",
    ),
    "ak_operator_2024": ClientInfo(
        client_id="cs-console",
        role=Role.OPERATOR,
        client_name="客服工作台",
        api_key_prefix="ak_opera",
    ),
    "ak_viewer_2024": ClientInfo(
        client_id="analytics-dashboard",
        role=Role.VIEWER,
        client_name="数据分析看板",
        api_key_prefix="ak_viewe",
    ),
}


# ── 兼容旧版：将原来的 FIXED_API_KEY 也纳入 mock 数据 ─────────

def register_legacy_client(legacy_key: str) -> None:
    """将 .env 中的旧 FIXED_API_KEY 注册为 admin 调用方（向后兼容）。"""
    if legacy_key and legacy_key not in _MOCK_CLIENTS:
        _MOCK_CLIENTS[legacy_key] = ClientInfo(
            client_id="legacy-admin",
            role=Role.ADMIN,
            client_name="旧版调用方（FIXED_API_KEY）",
            api_key_prefix=legacy_key[:8] if len(legacy_key) >= 8 else legacy_key,
        )


# ── 权限检查函数 ──────────────────────────────────────────────────

def lookup_client(api_key: str) -> Optional[ClientInfo]:
    """根据 API Key 查找调用方（mock 查找）。

    Returns:
        ClientInfo 如果找到，否则 None
    """
    return _MOCK_CLIENTS.get(api_key)


def check_tool_permission(client: ClientInfo, tool_name: str) -> bool:
    """检查调用方是否有权限执行指定工具。

    Args:
        client: 调用方上下文
        tool_name: 工具名（如 "request-return"）

    Returns:
        True 如果允许执行
    """
    allowed = ROLE_TOOL_PERMISSIONS.get(client.role, frozenset())
    return tool_name in allowed


def get_client_accessible_tools(client: ClientInfo) -> FrozenSet[str]:
    """获取调用方可用的工具集合（用于工具注册时过滤）。"""
    return ROLE_TOOL_PERMISSIONS.get(client.role, frozenset())


# ── 请求级上下文（避免修改整个调用链）─────────────────────────────

_current_client: contextvars.ContextVar[Optional[ClientInfo]] = (
    contextvars.ContextVar("current_client", default=None)
)


def set_current_client(client: ClientInfo) -> None:
    """在当前请求上下文中设置调用方信息。"""
    _current_client.set(client)


def get_current_client() -> Optional[ClientInfo]:
    """获取当前请求上下文中的调用方信息。"""
    return _current_client.get(None)


def clear_current_client() -> None:
    """清除当前请求上下文中的调用方信息。"""
    _current_client.set(None)
