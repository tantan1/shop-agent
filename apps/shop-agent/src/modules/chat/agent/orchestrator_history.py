"""对话历史持久化模块。"""
from __future__ import annotations

import asyncio
import json
import time

from src.modules.chat.config import chat_config
from src.modules.monitoring.metrics import (
    l2_save_duration_ms,
    l2_save_failure_total,
    l2_save_success_total,
    l2_save_triggered_total,
    l2_summary_tokens,
)
from src.shared.logger import APILogger

logger = APILogger("orchestrator_history")

L2_SAVE_INTERVAL = 5  # 每 5 轮保存一次 L2（低成本纯文本摘要，无差别触发）
LONG_CONVERSATION_THRESHOLD = getattr(
    chat_config, "long_conversation_threshold", 10
)  # L3 长对话兜底阈值：轮次超过该值才启动每 5 轮提取
L3_EXTRACT_FLAG_TTL = 30 * 86400  # L3 已提取标记 TTL（30 天）


def persist_turn(
    redis,
    conversation_id: str,
    user_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """将一轮 user/assistant 消息写入 Redis 对话历史，并触发 L2 保存。

     仅供 remote_api 链路（ReAct / 直接 Tool / 缺参反问 / 纠纷协调）使用；
     RAG 链路由 GeneralAgentExecutor._add_to_history 自行持久化，不在此重复写入。
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

        # 轮次计数器 + L2 触发
        turn_key = f"chat:turn_count:{conversation_id}"
        turn_number = redis._client.incr(turn_key)
        redis._client.expire(turn_key, 1 * 86400)
        if turn_number % L2_SAVE_INTERVAL == 0:
            asyncio.create_task(
                _trigger_l2_save(redis, conversation_id, turn_number, user_id)
            )
            # L3 独立触发：与 L2 解耦，各自异步、互不阻塞。
            # L3 是否真正提取由 MemoryExtractionTrigger.should_extract 决定
            # （对话结束 / 高价值意图 / 长对话每 5 轮兜底 / 关键词），
            # 短对话或普通轮次会快速返回 False，不调用 LLM，无额外成本。
            asyncio.create_task(
                _trigger_l3_extract(
                    redis, conversation_id, user_id, user_message, turn_number
                )
            )
    except Exception:
        logger.debug("写入对话历史失败，跳过", exc_info=True)


async def _trigger_l2_save(
    redis,
    conversation_id: str,
    turn_number: int,
    user_id: str,
) -> None:
    """触发 L2 短期记忆保存：增量摘要（基于 message seq 游标）+ Milvus 写入。

    去重机制：用 chat:l2_cursor:{conversation_id} 记录「最后已摘要到的消息 seq」，
    每次只摘要该 seq 之后的新增消息，天然避免重复与遗漏；不依赖布尔标记，
    也不受 L1 滚动截断影响（源数据来自 Redis 持久历史）。
    """
    start_time = time.perf_counter()
    l2_save_triggered_total.labels(trigger="turn").inc()
    try:
        from src.modules.chat.core.memory_service import ShortTermMemory

        cursor_key = f"chat:l2_cursor:{conversation_id}"
        cursor_raw = redis._client.get(cursor_key)
        cursor = int(cursor_raw.decode()) if cursor_raw else 0

        # 只读取 cursor 之后的新增消息（增量）
        new_messages = redis.get_chat_messages_since(
            conversation_id, since_seq=cursor, limit=200
        )
        if not new_messages:
            return

        short_term = ShortTermMemory()
        # 接入已实现的 ConversationSummarizer 生成真实摘要（替换原拼接占位）
        from src.modules.chat.agent.conversation_summarizer import ConversationSummarizer

        summarizer = ConversationSummarizer()
        msgs = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in new_messages
        ]
        summary = await summarizer.summarize_if_needed(msgs)
        key_entities = []

        # 生成 embedding（需要 embedding_service）
        try:
            from src.modules.chat.core.embedding_service import EmbeddingService

            emb_svc = EmbeddingService.get_instance()
            embedding = await emb_svc.embed_query(summary)
        except Exception as e:
            l2_save_failure_total.labels(trigger="turn", error="embedding_error").inc()
            logger.warning(f"L2 摘要 embedding 失败: {e}")
            return

        await short_term.store_summary(
            user_id=user_id,
            conversation_id=conversation_id,
            summary=summary,
            key_entities=key_entities,
            turn_number=turn_number,
            embedding=embedding,
        )
        # 推进游标到本次处理的最大 seq
        max_seq = max(int(m.get("seq", 0)) for m in new_messages)
        redis._client.setex(cursor_key, 7 * 86400, str(max_seq))
        l2_save_success_total.labels(trigger="turn").inc()
        duration_ms = (time.perf_counter() - start_time) * 1000
        l2_save_duration_ms.labels(trigger="turn").observe(duration_ms)
        l2_summary_tokens.observe(len(summary.split()))
    except Exception:
        l2_save_failure_total.labels(trigger="turn", error="unknown").inc()
        logger.debug("L2 保存失败，跳过", exc_info=True)


async def _trigger_l3_extract(
    redis,
    conversation_id: str,
    user_id: str,
    user_message: str,
    turn_number: int,
    force: bool = True,
) -> bool:
    """触发 L3 长期记忆提取（独立任务，与 L2 解耦）。

    从 Redis 持久历史取最近 N 轮对话作为提取窗口，交由 MemoryExtractionTrigger
    判断是否值得提取（长对话每 5 轮兜底 / 高价值意图 / 关键词 / 对话结束）。
    提取结果以「标签为键覆盖写」落库 PG + Milvus，窗口重叠不造成记忆重复。

    force=True 时跳过 should_extract 门控（用于每日兜底补提已结束/遗漏会话），
    通过把 is_ended 置 True 复用优先级 1（对话结束必须提取）。
    成功提取后写入 chat:l3_last_extract 标记，供每日兜底去重。
    返回是否实际执行了提取（成功或触发判定通过）。
    """
    try:
        from src.modules.chat.core.llm_service import LLMService
        from src.modules.chat.core.memory_extraction_trigger import (
            ExtractionContext,
            MemoryExtractionTrigger,
        )

        # 提取窗口：最近 20 轮（与 MemoryExtractor._format_history 一致）
        raw_history = redis.get_chat_messages_since(
            conversation_id, since_seq=0, limit=20
        )
        chat_history = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in raw_history
        ]
        if not chat_history:
            return True

        trigger = MemoryExtractionTrigger(LLMService.get_instance())
        ctx = ExtractionContext(
            user_id=user_id,
            conversation_id=conversation_id,
            chat_history=chat_history,
            user_message=user_message or "",
            is_ended=force,  # force 时强制判定通过
            turn_number=turn_number,
        )
        result = await trigger.try_extract(ctx)
        if result is not None:
            # 写入已提取标记：记录已提取到的轮次与 user_id（供每日兜底去重）
            flag_key = f"chat:l3_last_extract:{conversation_id}"
            try:
                redis._client.setex(
                    flag_key,
                    L3_EXTRACT_FLAG_TTL,
                    json.dumps(
                        {"turn": turn_number, "uid": user_id}, ensure_ascii=True
                    ),
                )
            except Exception:
                logger.debug("写入 L3 提取标记失败，忽略", exc_info=True)
            return True
        return True
    except Exception:
        logger.debug("L3 提取失败，跳过", exc_info=True)
        return True


async def run_l3_daily_backfill(redis) -> int:
    """每日兜底：扫描有进展但未提取 L3 的会话并强制补提。

    依据 chat:l3_last_extract 标记判断：标记缺失，或标记轮次 < 当前轮次
    （会话在标记之后又有新进展）时，强制（force）补提一次。
    避免对话结束/异常中断导致的 L3 遗漏。
    """
    if not getattr(redis, "is_available", True):
        return 0
    extracted = 0
    cursor = 0
    try:
        while True:
            cursor, keys = redis._client.scan(
                cursor, match="chat:turn_count:*", count=100
            )
            for key in keys:
                conv_id = key.split("chat:turn_count:")[1]
                try:
                    raw_turn = redis._client.get(f"chat:turn_count:{conv_id}")
                    turn_number = int(raw_turn.decode()) if raw_turn else 0
                except Exception:
                    continue
                if turn_number < 3:  # 过短对话不值得兜底
                    continue
                # 取已提取标记
                already = False
                try:
                    raw_flag = redis._client.get(
                        f"chat:l3_last_extract:{conv_id}"
                    )
                    if raw_flag:
                        flag = json.loads(raw_flag.decode())
                        if flag.get("turn", 0) >= turn_number:
                            already = True
                except Exception:
                    already = True
                if already:
                    continue
                # 取 user_id 与最近一段历史
                uid = "unknown"
                try:
                    raw_flag = redis._client.get(
                        f"chat:l3_last_extract:{conv_id}"
                    )
                    if raw_flag:
                        uid = json.loads(raw_flag.decode()).get("uid", "unknown")
                except Exception:
                    pass
                raw_history = redis.get_chat_messages_since(
                    conv_id, since_seq=0, limit=20
                )
                last_user = ""
                for m in reversed(raw_history):
                    if m.get("role") == "user":
                        last_user = m.get("content", "")
                        break
                if await _trigger_l3_extract(
                    redis, conv_id, uid, last_user, turn_number, force=True
                ):
                    extracted += 1
            if cursor == 0:
                break
    except Exception:
        logger.debug("L3 每日兜底扫描失败", exc_info=True)
    logger.info(f"L3 每日兜底完成，补提 {extracted} 个会话")
    return extracted
