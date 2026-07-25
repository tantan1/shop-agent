"""
ReAct Agent 人在回路中断存储（用于退款确认后恢复执行）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.modules.chat.core.redis_cache_service import get_redis_cache_service
from src.shared.logger import APILogger

logger = APILogger("react_agent_interrupt")

_INTERRUPT_MEM: dict = {}
_INTERRUPT_KEY_PREFIX = "interrupt:"
_INTERRUPT_TTL = 3600


@dataclass
class InterruptContext:
    """人在回路中断上下文，替代散参传递。"""
    thread_id: str
    graph: Any
    config: dict
    conversation_id: str
    intent_steps: list
    domain: str
    order_id: str
    reason: str


def _store_interrupt(ctx: InterruptContext) -> None:
    """保存被中断的上下文，供后续 resume 使用（持久化到 Redis）。"""
    payload = {
        "conversation_id": ctx.conversation_id,
        "intent_steps": ctx.intent_steps,
        "domain": ctx.domain,
        "order_id": ctx.order_id,
        "reason": ctx.reason,
        "config": ctx.config,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        svc = get_redis_cache_service()
        if svc and svc.is_available:
            svc.set_json(f"{_INTERRUPT_KEY_PREFIX}{ctx.thread_id}", payload, ex=_INTERRUPT_TTL)
            return
    except Exception as e:
        logger.warning("中断上下文写入 Redis 失败，降级内存", thread_id=ctx.thread_id, error=str(e))
    _INTERRUPT_MEM[ctx.thread_id] = (
        ctx.graph,
        ctx.config,
        ctx.conversation_id,
        ctx.intent_steps,
        ctx.domain,
        ctx.order_id,
        ctx.reason,
    )


def _pop_interrupt(thread_id: str):
    """取出并删除中断上下文（优先 Redis，降级内存）。"""
    try:
        svc = get_redis_cache_service()
        if svc and svc.is_available:
            data = svc.get_json(f"{_INTERRUPT_KEY_PREFIX}{thread_id}")
            if data is not None:
                svc.delete_key(f"{_INTERRUPT_KEY_PREFIX}{thread_id}")
                return (
                    None,
                    data.get("config"),
                    data.get("conversation_id"),
                    data.get("intent_steps"),
                    data.get("domain"),
                    data.get("order_id"),
                    data.get("reason"),
                )
    except Exception as e:
        logger.warning("中断上下文读取 Redis 失败，降级内存", thread_id=thread_id, error=str(e))
    return _INTERRUPT_MEM.pop(thread_id, None)
