"""调用方仓储（对权限核心 mock 调用方表的薄封装）。"""
from __future__ import annotations

from typing import Optional

from src.core.permissions import ClientInfo, lookup_client


class ClientRepository:
    """API 调用方仓储。"""

    def find_by_api_key(self, api_key: str) -> Optional[ClientInfo]:
        return lookup_client(api_key)
