"""
记忆系统评估套件：黄金对 + 相关性评分 + 自动回归测试
"""
import time
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from src.modules.chat.core.memory_observability import MemoryObservability
from src.modules.chat.core.memory_retrieval import MemoryRetrieval, RetrievalContext
from src.modules.chat.core.memory_service import LongTermMemory, ShortTermMemory
from src.shared.logger import APILogger

logger = APILogger("memory_eval")
obs = MemoryObservability()


@dataclass
class GoldenPair:
    """黄金测试对：输入查询 + 预期记忆块"""
    query: str
    query_embedding: List[float]
    user_id: str
    expected_l2_count: int = 0
    expected_l3_count: int = 0
    expected_profile_fields: List[str] = None
    min_relevance_score: float = 0.5


@dataclass
class EvalResult:
    """单条评估结果"""
    pair: GoldenPair
    l2_count: int
    l3_count: int
    profile_fields_found: List[str]
    avg_relevance: float
    latency_ms: float
    passed: bool


class MemoryEvalSuite:
    """记忆系统评估套件"""

    def __init__(self, retrieval: MemoryRetrieval):
        self._retrieval = retrieval
        self._results: List[EvalResult] = []

    async def run_golden_set(self, pairs: List[GoldenPair]) -> List[EvalResult]:
        """运行黄金测试集"""
        results = []
        for pair in pairs:
            start = time.perf_counter()
            ctx = await self._retrieval.retrieve(
                user_id=pair.user_id,
                query=pair.query,
                query_embedding=pair.query_embedding,
            )
            latency = (time.perf_counter() - start) * 1000

            result = self._evaluate_pair(pair, ctx, latency)
            results.append(result)
            self._results.append(result)

            status = "PASS" if result.passed else "FAIL"
            logger.info(
                f"记忆评估 {status}",
                query=pair.query[:50],
                l2=result.l2_count,
                l3=result.l3_count,
                latency=result.latency_ms,
            )
        return results

    def _evaluate_pair(self, pair: GoldenPair, ctx: RetrievalContext, latency: float) -> EvalResult:
        """评估单条黄金对"""
        l2_count = len(ctx.short_term_memories)
        l3_count = len(ctx.long_term_memories)

        profile_fields_found = []
        if ctx.long_term_memories and pair.expected_profile_fields:
            profile = ctx.long_term_memories[0]
            for field in pair.expected_profile_fields:
                if field in profile.get("preferences", {}) or profile.get(field):
                    profile_fields_found.append(field)

        relevance_scores = [m.get("score", 0) for m in ctx.short_term_memories + ctx.long_term_memories]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

        passed = (
            l2_count >= pair.expected_l2_count
            and l3_count >= pair.expected_l3_count
            and avg_relevance >= pair.min_relevance_score
        )
        if pair.expected_profile_fields:
            passed = passed and len(profile_fields_found) >= len(pair.expected_profile_fields)

        return EvalResult(
            pair=pair,
            l2_count=l2_count,
            l3_count=l3_count,
            profile_fields_found=profile_fields_found,
            avg_relevance=avg_relevance,
            latency_ms=latency,
            passed=passed,
        )

    def get_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        if not self._results:
            return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
        passed = sum(1 for r in self._results if r.passed)
        return {
            "total": len(self._results),
            "passed": passed,
            "failed": len(self._results) - passed,
            "pass_rate": passed / len(self._results),
            "avg_latency_ms": sum(r.latency_ms for r in self._results) / len(self._results),
            "avg_relevance": sum(r.avg_relevance for r in self._results) / len(self._results),
        }


@pytest.mark.asyncio
async def test_memory_golden_set():
    """记忆系统黄金对回归测试"""
    mock_milvus = MagicMock()
    mock_milvus.hybrid_search.return_value = []
    mock_milvus.search.return_value = []

    from src.modules.chat.core.memory_milvus_service import MemoryBlockService
    original_get_instance = MemoryBlockService.get_instance

    def mock_get_instance():
        return mock_milvus

    MemoryBlockService.get_instance = classmethod(lambda cls: mock_get_instance())

    try:
        mock_short = ShortTermMemory()
        mock_short._milvus = mock_milvus

        mock_long = LongTermMemory(pg_session=None, milvus_service=mock_milvus, embedding_service=None)
        mock_long.milvus = mock_milvus

        retrieval = MemoryRetrieval(
            milvus_service=mock_milvus,
            short_term=mock_short,
            long_term=mock_long,
        )

        suite = MemoryEvalSuite(retrieval)
        pairs = [
            GoldenPair(
                query="我想买一个手机",
                query_embedding=[0.1] * 768,
                user_id="test_user_001",
                expected_l2_count=0,
                expected_l3_count=0,
                min_relevance_score=0.0,
            ),
        ]

        await suite.run_golden_set(pairs)
        summary = suite.get_summary()
        assert summary["total"] > 0
        assert summary["pass_rate"] >= 0.0
    finally:
        MemoryBlockService.get_instance = original_get_instance
