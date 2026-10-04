"""
基于角色的工具权限控制 —— 角色与调用方从配置文件加载（Ports & Adapters）。

三层设计：
  1. 角色定义（Role）        — admin / operator / viewer
  2. 调用方数据（ClientInfo）— 由 config/rbac.json 加载（API Key → 角色）
  3. 权限检查函数             — 按角色判断工具是否可用

外部能力（鉴权数据源）通过文件配置 stub 提供：
  - config/rbac.json 声明 角色→工具 映射 与 调用方→角色 映射
  - 代码统一经本模块接口读取，不直接依赖具体存储（DB/Redis 后续可替换）
  - 文件缺失/解析失败时回退到内置 DEFAULT_RBAC（保证开发可运行）
"""

from __future__ import annotations

import contextvars
import json
import os
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Optional

from src.ports import audit  # noqa: E402

# ── 角色定义 ──────────────────────────────────────────────────────


class Role(str, Enum):
    """调用方角色枚举"""

    ADMIN = "admin"  # 管理端：所有工具可用
    OPERATOR = "operator"  # 运营端：不可执行退款
    VIEWER = "viewer"  # 只读端：仅查询类工具


# ── RBAC 配置加载（文件配置 stub）───────────────────────────────

# 内置兜底配置（文件缺失时使用，仅开发态；生产应提供 config/rbac.json）
_DEFAULT_RBAC: Dict[str, object] = {
    "all_tools": [
        "query-order",
        "check-shipping",
        "request-return",
        "check-balance",
        "coupon-inquiry",
        "knowledge_search",
    ],
    "roles": {
        "admin": {"tools": ["*"]},
        "operator": {
            "tools": [
                "query-order",
                "check-shipping",
                "request-return",
                "check-balance",
                "coupon-inquiry",
                "knowledge_search",
            ]
        },
        "viewer": {
            "tools": [
                "query-order",
                "check-shipping",
                "check-balance",
                "coupon-inquiry",
                "knowledge_search",
            ]
        },
    },
    "clients": {
        "ak_admin_2024": {"client_id": "order-service", "role": "admin", "client_name": "订单管理后台"},
        "ak_operator_2024": {"client_id": "cs-console", "role": "operator", "client_name": "客服工作台"},
        "ak_viewer_2024": {"client_id": "analytics-dashboard", "role": "viewer", "client_name": "数据分析看板"},
    },
}

# 配置文件路径（相对于本文件：src/core/rbac.json）
_RBAC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rbac.json")


def _load_rbac() -> Dict[str, object]:
    """从 config/rbac.json 加载 RBAC；失败回退内置兜底。

    返回结构同 _DEFAULT_RBAC。生产环境若文件缺失会告警（不阻断启动）。
    """
    try:
        with open(_RBAC_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "roles" not in data:
            raise ValueError("rbac.json 缺少 roles 字段")
        return data
    except FileNotFoundError:
        # 开发态允许缺失，使用内置兜底
        return _DEFAULT_RBAC
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"加载 rbac.json 失败，回退内置配置：{e}", stacklevel=2)
        return _DEFAULT_RBAC


_RBAC = _load_rbac()
_ALL_TOOLS: FrozenSet[str] = frozenset(_RBAC.get("all_tools", []))
_ROLE_TOOLS: Dict[str, list] = {
    name: spec.get("tools", []) for name, spec in _RBAC.get("roles", {}).items()
}


def _expand_role_tools(role: str) -> FrozenSet[str]:
    """将角色的工具列表展开为 FrozenSet；'*' 表示全部工具。"""
    tools = _ROLE_TOOLS.get(role, [])
    if "*" in tools:
        return _ALL_TOOLS
    return frozenset(tools)


ROLE_TOOL_PERMISSIONS: Dict[Role, FrozenSet[str]] = {
    Role(name): _expand_role_tools(name) for name in _ROLE_TOOLS
}


# ── 调用方数据（从配置加载）────────────────────────────────────


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


def _build_clients() -> Dict[str, ClientInfo]:
    out: Dict[str, ClientInfo] = {}
    for api_key, spec in _RBAC.get("clients", {}).items():
        try:
            role = Role(spec["role"])
        except (KeyError, ValueError):
            continue
        out[api_key] = ClientInfo(
            client_id=spec.get("client_id", ""),
            role=role,
            client_name=spec.get("client_name", ""),
            api_key_prefix=api_key[:8] if len(api_key) >= 8 else api_key,
        )
    return out


_MOCK_CLIENTS: Dict[str, ClientInfo] = _build_clients()


# ── 兼容旧版：将原来的 FIXED_API_KEY 也纳入（向后兼容）─────────


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
    """根据 API Key 查找调用方（由配置文件驱动）。"""
    client = _MOCK_CLIENTS.get(api_key)
    # A 维度：审计"密钥读取"敏感操作（端口 stub，本地 JSONL）。
    # 成功解析即视为一次密钥使用（secret.read）；未知 key 视为鉴权失败。
    if client is None:
        audit.log(
            "auth.failed",
            principal=(api_key[:8] if len(api_key) >= 8 else api_key),
            action="lookup_client",
            decision="deny",
            detail={"reason": "unknown_api_key"},
        )
    else:
        audit.log(
            "secret.read",
            principal=client.client_id,
            action="lookup_client",
            decision="allow",
            detail={"role": client.role.value},
        )
    return client


def check_tool_permission(client: ClientInfo, tool_name: str) -> bool:
    """检查调用方是否有权限执行指定工具。

    admin 角色放行全部工具（含未来新增工具），避免新增工具被误拦截。
    """
    if client.role == Role.ADMIN:
        # A 维度：审计 admin 越权放行（端口 stub，本地 JSONL）
        audit.log(
            "authz.admin_allow",
            principal=client.client_id,
            action=tool_name,
            decision="allow",
            detail={"role": client.role.value},
        )
        return True
    allowed = ROLE_TOOL_PERMISSIONS.get(client.role, frozenset())
    return tool_name in allowed


def get_client_accessible_tools(client: ClientInfo) -> FrozenSet[str]:
    """获取调用方可用的工具集合（用于工具注册时过滤）。"""
    if client.role == Role.ADMIN:
        return _ALL_TOOLS
    return ROLE_TOOL_PERMISSIONS.get(client.role, frozenset())


# ── 请求级上下文（避免修改整个调用链）─────────────────────────────

_current_client: contextvars.ContextVar[Optional[ClientInfo]] = contextvars.ContextVar(
    "current_client", default=None
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
