"""
A2A Webhook Service —— Webhook 订阅管理。

功能：
  - POST /a2a/webhooks          → 注册回调订阅
  - DELETE /a2a/webhooks/{id}   → 取消订阅

存储：内存 dict（可升级为 Redis）
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from src.shared.logger import APILogger
from src.modules.chat.schemas import (
    WebhookSubscriptionResponse,
)

logger = APILogger("a2a_webhook")


class A2AWebhookService:
    """A2A Webhook 订阅管理器（单例）。"""

    _instance: Optional["A2AWebhookService"] = None

    def __init__(self) -> None:
        self._subscriptions: Dict[str, WebhookSubscriptionResponse] = {}

    @classmethod
    def get_instance(cls) -> "A2AWebhookService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def subscribe(
        self,
        url: str,
        events: Optional[List[str]] = None,
        secret: Optional[str] = None,
        ttl_seconds: int = 86400,
    ) -> WebhookSubscriptionResponse:
        """注册 Webhook 订阅。

        Args:
            url:          回调 URL
            events:       订阅事件列表（默认 task.completed + task.failed）
            secret:       HMAC 签名密钥
            ttl_seconds:  有效期秒数（默认 24h）
        """
        sub_id = f"wh_{uuid.uuid4().hex[:12]}"
        now = self._now_iso()
        expires = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()

        sub = WebhookSubscriptionResponse(
            subscription_id=sub_id,
            url=url,
            events=events or ["task.completed", "task.failed"],
            created_at=now,
            expires_at=expires,
        )

        self._subscriptions[sub_id] = sub
        logger.info("Webhook 订阅已创建", subscription_id=sub_id, url=url, events=sub.events)
        return sub

    def unsubscribe(self, subscription_id: str) -> bool:
        """取消 Webhook 订阅。"""
        if subscription_id in self._subscriptions:
            self._subscriptions.pop(subscription_id)
            logger.info("Webhook 订阅已取消", subscription_id=subscription_id)
            return True
        return False

    def list_subscriptions(self) -> List[WebhookSubscriptionResponse]:
        """列出所有活跃订阅。"""
        now = datetime.now(timezone.utc)
        active = []
        expired = []
        for sub_id, sub in self._subscriptions.items():
            if sub.expires_at:
                expires = datetime.fromisoformat(sub.expires_at)
                if expires < now:
                    expired.append(sub_id)
                else:
                    active.append(sub)
            else:
                active.append(sub)

        # 清理过期订阅
        for eid in expired:
            self._subscriptions.pop(eid, None)

        return active

    def get_subscribers(self, event: str) -> List[WebhookSubscriptionResponse]:
        """获取订阅了指定事件的所有订阅者。"""
        subs = self.list_subscriptions()
        return [s for s in subs if event in s.events]


def get_a2a_webhook_service() -> A2AWebhookService:
    """获取 A2A Webhook 服务单例。"""
    return A2AWebhookService.get_instance()
