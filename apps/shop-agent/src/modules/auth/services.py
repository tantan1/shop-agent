"""认证服务（对权限核心的薄封装）。

保持 `auth` 模块非空心壳：对外暴露可注入的认证能力，
底层委托 `src.core.permissions`（mock 调用方表驱动）。
"""
from __future__ import annotations

from typing import Optional

from src.core.permissions import ClientInfo, lookup_client
from src.ports import audit


class AuthService:
    """API Key 认证服务。"""

    def authenticate(self, api_key: str) -> Optional[ClientInfo]:
        """根据 API Key 解析调用方。

        未知 key 返回 None（不直接抛错，由依赖层决定 401/403 行为），
        并留下审计条目。
        """
        client = lookup_client(api_key)
        if client is None:
            audit.log(
                "auth.service_failed",
                principal=(api_key[:8] if len(api_key) >= 8 else api_key),
                action="authenticate",
                decision="deny",
                detail={"reason": "unknown_api_key"},
            )
        return client
