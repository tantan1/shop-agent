"""分类层 —— 只回答「是什么意图 + 多自信」，不做任何执行决策。

每个分类器只负责一种判定来源，返回 ``IntentCandidate`` 或 ``None``
（``None`` = 本层不判定，交由下一层）。
路由（该不该走 RAG / 直接调工具 / 交 Agent）由 ``ExecutionRouter`` 负责，
不在本模块出现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

from src.shared.logger import APILogger

from .candidate import ClassifierSource, IntentCandidate
from .index_builder import (
    ExamplesProvider,
    ensure_intent_index_async,
    ensure_intent_index_sync,
)

logger = APILogger("intent_classifier")


class IntentClassifier(ABC):
    """分类器协议：输入用户消息，输出意图候选。"""

    source: ClassifierSource = "unknown"

    @abstractmethod
    async def classify(self, message: str) -> Optional[IntentCandidate]:
        """返回候选；``None`` 表示本层无法判定。"""
        raise NotImplementedError


class NegationRuleClassifier(IntentClassifier):
    """否定词 / 政策咨询过滤：命中即表示「无工具意图」，应走 RAG。

    词表由外部注入（后续移入 config/policy），本类不含业务常量。
    """

    source: ClassifierSource = "negation"

    def __init__(self, patterns: Sequence[str]):
        self._patterns: List[str] = [p for p in (patterns or []) if p]

    async def classify(self, message: str) -> Optional[IntentCandidate]:
        for p in self._patterns:
            if p in message:
                return IntentCandidate(
                    action=None, score=1.0, source=self.source, matched=p
                )
        return None


class FaissIntentClassifier(IntentClassifier):
    """FAISS 向量语义匹配。

    索引示例来自 ``examples_provider``（默认由调用方注入 SkillRegistry 抽取函数），
    不再从代码里读硬编码示例。
    """

    source: ClassifierSource = "faiss"

    def __init__(self, embedding_service, examples_provider: ExamplesProvider):
        self._embedding_service = embedding_service
        self._examples_provider = examples_provider

    async def classify(self, message: str) -> Optional[IntentCandidate]:
        idx = await ensure_intent_index_async(
            self._embedding_service, self._examples_provider
        )
        if idx is None or self._embedding_service is None:
            return None
        try:
            emb = self._embedding_service.get_embeddings()
            query_vec = await emb.aembed_query(message)
            matches = idx.search(query_vec, k=2)
        except Exception as e:
            logger.warning(f"FAISS 向量意图识别失败: {e}")
            return None

        if not matches:
            return None
        if len(matches) >= 2:
            logger.debug(
                f"FAISS向量匹配 top-2: ({matches[0].action},{matches[0].score:.3f}) "
                f"({matches[1].action},{matches[1].score:.3f})"
            )
        top = matches[0]
        return IntentCandidate(
            action=top.action,
            score=top.score,
            source=self.source,
            matched=top.matched,
        )

    def warmup_sync(self) -> None:
        """同步预热索引（lifespan 用，不依赖事件循环）。"""
        ensure_intent_index_sync(self._embedding_service, self._examples_provider)
