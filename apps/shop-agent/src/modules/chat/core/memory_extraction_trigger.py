"""
记忆提取触发器：决定何时提取 L3 长期记忆
"""
from dataclasses import dataclass
from typing import Optional

from src.modules.chat.config import chat_config
from src.shared.logger import APILogger

logger = APILogger("memory_extraction_trigger")

# 长对话阈值：轮次超过该值才启动「每 5 轮」兜底提取（与 L2 的 5 轮常态区分开）
LONG_CONVERSATION_THRESHOLD = getattr(chat_config, "long_conversation_threshold", 10)


@dataclass
class ExtractionContext:
    """记忆提取上下文"""
    user_id: str
    conversation_id: str
    chat_history: list
    user_message: str
    last_intent: Optional[str] = None
    is_ended: bool = False
    turn_number: int = 0


class MemoryExtractionTrigger:
    """记忆提取触发器"""

    def __init__(self, llm_service):
        self._llm = llm_service
        self._extractor = None

    def _get_extractor(self):
        if self._extractor is None:
            from src.modules.chat.core.memory_extractor import MemoryExtractor
            self._extractor = MemoryExtractor(self._llm)
        return self._extractor

    async def try_extract(self, ctx: ExtractionContext) -> Optional[list]:
        """尝试提取记忆，返回提取的记忆列表或 None

        触发条件（优先级）：
        1. 对话结束（is_ended）
        2. 高价值意图完成（request-return / complaint / dispute）
        3. 长对话每 5 轮兜底（turn_number > LONG_CONVERSATION_THRESHOLD 且 %5==0）
        4. 关键词启发式（订单/退货/投诉/偏好等命中）
        """
        if not self.should_extract(ctx.is_ended, ctx.turn_number, ctx.last_intent, ctx.chat_history, ctx.user_message):
            return None

        try:
            extractor = self._get_extractor()
            memories = await extractor.extract(ctx.chat_history, ctx.user_message)
            if not memories:
                return None

            # 存储到 L3
            from src.modules.chat.core.embedding_service import EmbeddingService
            from src.modules.chat.core.memory_service import LongTermMemory
            from src.modules.chat.core.memory_milvus_service import MemoryBlockService

            emb_svc = EmbeddingService.get_instance()
            milvus = MemoryBlockService.get_instance()
            l3 = LongTermMemory(pg_session=None, milvus_service=milvus, embedding_service=emb_svc)
            await l3.update_from_conversation(ctx.user_id, memories)
            logger.info(
                "L3 记忆提取完成",
                user_id=ctx.user_id,
                conversation_id=ctx.conversation_id[:8],
                memory_count=len(memories),
            )
            return memories
        except Exception as e:
            logger.error(f"L3 记忆提取失败: {e}")
            return None

    def should_extract(
        self,
        is_ended: bool,
        turn_number: int,
        last_intent: Optional[str],
        chat_history: list,
        user_message: str,
    ) -> bool:
        """判断是否应该提取记忆（L3 与 L2 解耦：不无差别每 5 轮，仅在长对话/高价值/结束/关键词时触发）"""
        # 1. 对话结束（必须提取）
        if is_ended:
            return True

        # 2. 高价值意图
        if last_intent in {"request-return", "complaint", "dispute"}:
            return True

        # 3. 长对话每 5 轮兜底（轮次超过阈值才开始，避免把 5-10 轮常态误当"长对话"高频高成本提取）
        if (
            turn_number > LONG_CONVERSATION_THRESHOLD
            and turn_number > 0
            and turn_number % 5 == 0
        ):
            return True

        # 4. 关键词启发式（订单、退货、投诉、偏好等命中即提取）
        if self._has_important_info(user_message):
            return True

        return False

    def _has_important_info(self, text: str) -> bool:
        """简单规则判断是否包含重要信息"""
        keywords = ["订单", "退货", "投诉", "建议", "偏好", "尺码", "颜色", "品牌"]
        return any(kw in text for kw in keywords)
