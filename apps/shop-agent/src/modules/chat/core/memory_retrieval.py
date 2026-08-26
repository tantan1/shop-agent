"""
记忆检索融合（MRAG）：知识检索 + 记忆检索融合
"""
import time
from typing import Any, Dict, List, Optional

from src.modules.chat.core.memory_observability import MemoryObservability
from src.modules.chat.core.memory_service import LongTermMemory, ShortTermMemory
from src.modules.chat.core.milvus_service import MilvusService
from src.shared.logger import APILogger

logger = APILogger("memory_retrieval")
obs = MemoryObservability()


class RetrievalContext:
    """检索上下文：知识 + 记忆融合结果"""

    def __init__(
        self,
        documents: List[Dict[str, Any]],
        short_term_memories: List[Dict[str, Any]],
        long_term_memories: List[Dict[str, Any]],
    ):
        self.documents = documents
        self.short_term_memories = short_term_memories
        self.long_term_memories = long_term_memories

    def to_prompt_context(self) -> str:
        """生成 Prompt 注入文本"""
        parts = []
        if self.long_term_memories:
            parts.extend(self._format_profile())
        if self.short_term_memories:
            parts.extend(self._format_short_term())
        if self.long_term_memories:
            parts.extend(self._format_long_term())
        if self.documents:
            parts.extend(self._format_knowledge())
        result = "\n".join(parts)
        if result:
            obs.record_context_size("short_term", len(result))
        return result

    def _format_profile(self) -> List[str]:
        parts = ["【用户画像】"]
        profile = self.long_term_memories[0]
        parts.append(f"VIP等级: {profile.get('vip_level', 'normal')}")
        preferences = profile.get("preferences", {})
        if preferences:
            parts.append(f"偏好: {preferences}")
        parts.append("")
        return parts

    def _format_short_term(self) -> List[str]:
        parts = ["【近期对话摘要】（最近 7 天）"]
        for mem in self.short_term_memories[:3]:
            parts.append(f"- {mem.get('value', '')}")
        parts.append("")
        return parts

    def _format_long_term(self) -> List[str]:
        parts = ["【历史记忆】"]
        for mem in self.long_term_memories[:5]:
            if mem.get("block_type") != "profile":
                parts.append(f"- {mem.get('label', '')}: {mem.get('value', '')}")
        parts.append("")
        return parts

    def _format_knowledge(self) -> List[str]:
        parts = ["【知识库检索结果】"]
        for doc in self.documents[:5]:
            parts.append(f"- {doc.get('text', '')}")
        return parts


class MemoryRetrieval:
    """记忆检索融合服务"""

    def __init__(
        self,
        milvus_service: MilvusService,
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        embedding_service=None,
    ):
        self._milvus = milvus_service
        self._short_term = short_term
        self._long_term = long_term
        self._embedding = embedding_service

    async def retrieve(
        self,
        user_id: str,
        query: str,
        query_embedding: List[float],
    ) -> RetrievalContext:
        """融合检索：知识库 + 短期记忆 + 长期记忆

        Args:
            user_id: 用户 ID
            query: 用户查询
            query_embedding: 查询向量

        Returns:
            RetrievalContext 包含知识文档、短期记忆、长期记忆
        """
        # 1. 知识检索（现有 RAG）
        knowledge_docs = []
        try:
            raw_docs = self._milvus.hybrid_search(query_embedding, query, top_k=5)
            knowledge_docs = [
                {"text": doc.page_content, "metadata": doc.metadata}
                for doc in raw_docs
            ]
        except Exception as e:
            logger.warning(f"知识检索失败: {e}")

        # 2. 短期记忆检索（L2）
        short_term_memories = []
        if self._short_term and user_id:
            try:
                start = time.perf_counter()
                short_term_memories = await self._short_term.recall_recent(
                    user_id, query, query_embedding, top_k=3
                )
                duration = time.perf_counter() - start
                obs.record_recall("l2", duration, count=len(short_term_memories))
            except Exception as e:
                logger.warning(f"L2 记忆召回失败: {e}")

        # 3. 长期记忆检索（L3）
        long_term_memories = []
        if self._long_term and user_id:
            try:
                start = time.perf_counter()
                long_term_memories = await self._long_term.recall_relevant(
                    user_id, query, query_embedding, top_k=5
                )
                duration = time.perf_counter() - start
                obs.record_recall("l3", duration, count=len(long_term_memories))
            except Exception as e:
                logger.warning(f"L3 记忆召回失败: {e}")

        return RetrievalContext(
            documents=knowledge_docs,
            short_term_memories=short_term_memories,
            long_term_memories=long_term_memories,
        )
