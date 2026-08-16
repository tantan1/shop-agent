"""远程 API 参数处理模块。"""
from __future__ import annotations

import re as _re
import time as _time

from src.modules.chat.schemas import ChatResponse
from src.shared.logger import APILogger

logger = APILogger("orchestrator_params")

_ORDER_REQUIRED_ACTIONS = {
    "check-shipping",
    "query-order",
    "request-return",
    "request-refund",
}
_ORDER_ID_PATTERNS = [
    _re.compile(r"(?:订单号|订单编号|单号|订单)\s*[:：]?\s*([A-Za-z0-9\-]{3,})"),
    _re.compile(r"(?:快递单号|物流单号|运单号)\s*[:：]?\s*([A-Za-z0-9\-]{6,})"),
    _re.compile(r"\b(\d{3,})\b"),
]


def _extract_order_id(text: str) -> str | None:
    if not text:
        return None
    for pat in _ORDER_ID_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1)
    return None


def _append_step(intent_steps: list, step_name: str, output_data: dict) -> None:
    intent_steps.append(
        {
            "step_name": step_name,
            "step_order": 1,
            "status": "success",
            "output_data": output_data,
        }
    )


def _try_compensate_from_current(orchestrator, request, intent_result, intent_steps, conversation_id) -> bool:
    current_msg = (request.message or "").strip()
    current_oid = _extract_order_id(current_msg) if current_msg else None
    if not current_oid:
        return False
    intent_result.params["order_id"] = current_oid
    _append_step(intent_steps, "参数补偿(当前轮)", {"from_current": True, "order_id": current_oid})
    logger.info(
        "当前轮参数补偿成功",
        action=intent_result.action,
        order_id=current_oid,
        conversation_id=conversation_id,
    )
    return True


def _try_compensate_from_history(orchestrator, request, intent_result, intent_steps, conversation_id) -> bool:
    history = orchestrator._redis_cache_service.get_chat_messages(conversation_id)
    current_msg = (request.message or "").strip()
    for msg in reversed(history):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if content.strip() == current_msg:
            continue
        hist_order = _extract_order_id(content)
        if hist_order:
            intent_result.params["order_id"] = hist_order
            _append_step(intent_steps, "参数补偿(历史)", {"from_history": True, "order_id": hist_order})
            logger.info(
                "历史补参成功",
                action=intent_result.action,
                order_id=hist_order,
                conversation_id=conversation_id,
            )
            return True
    return False


def _compensate_missing_order_id(orchestrator, request, intent_result, intent_steps, conversation_id) -> None:
    if (
        intent_result.action not in _ORDER_REQUIRED_ACTIONS
        or not intent_result.params
        or intent_result.params.get("order_id")
        or not orchestrator._redis_cache_service
        or not orchestrator._redis_cache_service.is_available
    ):
        return

    if _try_compensate_from_current(orchestrator, request, intent_result, intent_steps, conversation_id):
        return
    if _try_compensate_from_history(orchestrator, request, intent_result, intent_steps, conversation_id):
        return

    logger.info(
        "历史补参未命中",
        action=intent_result.action,
        history_len=len(orchestrator._redis_cache_service.get_chat_messages(conversation_id)),
        conversation_id=conversation_id,
    )


def _validate_required_params(intent_result, intent_steps, conversation_id: str = "") -> ChatResponse | None:
    """缺参兜底：需要订单号但当前轮+历史都无，直接反问。"""
    if intent_result.action not in _ORDER_REQUIRED_ACTIONS:
        return None

    has_order = bool(intent_result.params and intent_result.params.get("order_id"))
    has_tracking = bool(intent_result.params and intent_result.params.get("tracking_number"))
    if has_order or has_tracking:
        return None

    action = intent_result.action
    if "check-shipping" in action:
        missing_prompt = "请提供订单号或快递单号"
    else:
        missing_prompt = "请提供订单号"

    return ChatResponse(
        message=f"{missing_prompt}，以便为您查询相关信息。",
        conversation_id=conversation_id,
        steps=intent_steps
        + [
            {
                "step_name": "参数校验",
                "step_order": len(intent_steps),
                "status": "blocked",
                "output_data": {
                    "reason": "missing_order_id",
                    "action": intent_result.action,
                },
            }
        ],
        domain=intent_result.domain if hasattr(intent_result, "domain") else "ecommerce",
        status="need_more_info",
    )


async def _prepare_intent_params(orchestrator, request, intent_result, langfuse_handler, conversation_id):
    """参数抽取 + 缺参补偿 + 缺参兜底。

    Returns:
        (intent_result, intent_steps, blocked_response, t_params_ms)
    """
    intent_steps = [
        {
            "step_name": "意图识别",
            "step_order": 0,
            "status": "success",
            "output_data": intent_result.model_dump(),
        }
    ]

    t0 = _time.perf_counter()
    extracted_params = await orchestrator._intent_recognizer.extract_params(
        request.message, intent_result.action, langfuse_handler=langfuse_handler
    )
    t_params = (_time.perf_counter() - t0) * 1000
    if extracted_params:
        existing = intent_result.params or {}
        intent_result.params = {**extracted_params, **existing}
        _append_step(intent_steps, "参数抽取", {"extracted_params": extracted_params})

    _compensate_missing_order_id(orchestrator, request, intent_result, intent_steps, conversation_id)

    blocked = _validate_required_params(intent_result, intent_steps, conversation_id)
    if blocked:
        return intent_result, intent_steps, blocked, t_params

    return intent_result, intent_steps, None, t_params
