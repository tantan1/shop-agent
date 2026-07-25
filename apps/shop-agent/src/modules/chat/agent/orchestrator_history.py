"""对话历史持久化模块。"""
from __future__ import annotations

from src.modules.chat.config import chat_config
from src.shared.logger import APILogger

logger = APILogger("orchestrator_history")


def persist_turn(redis, conversation_id: str, user_message: str, assistant_message: str) -> None:
    """将一轮 user/assistant 消息写入 Redis 对话历史。

    仅供 remote_api 链路（ReAct / 直接 Tool / 缺参反问 / 纠纷协调）使用；
    RAG 链路由 GeneralAgentExecutor._add_to_history 自行持久化，不在此重复写入。

    历史是「缺参补偿」的数据来源：用户先说「订单号111」、后轮问「这个订单有物流吗」时，
    需要能从历史里翻出订单号，否则模型会在空参时幻觉出假订单号。

    任何异常都不得影响主链路，故整体 try/except 兜底。
    """
    if not redis or not redis.is_available:
        return
    try:
        max_turns = getattr(chat_config, "max_history_turns", 10)
        if user_message and user_message.strip():
            redis.add_chat_message(
                conversation_id,
                "user",
                user_message.strip()[:4096],
                max_turns=max_turns,
            )
        if assistant_message and assistant_message.strip():
            redis.add_chat_message(
                conversation_id,
                "assistant",
                assistant_message.strip()[:4096],
                max_turns=max_turns,
            )
    except Exception:
        logger.debug("写入对话历史失败，跳过", exc_info=True)
