"""
ReAct Agent 工具语义匹配器与工具选择器 middleware（P1 FAISS 语义匹配 + 云端兜底选择器）。
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from langchain.agents.middleware import LLMToolSelectorMiddleware
from pydantic import BaseModel, Field, create_model

from src.modules.chat.agent.skill_loader import get_skill_registry
from src.shared.logger import APILogger

logger = APILogger("react_agent_selection")

# 工具选择器专用 prompt
_TOOL_SELECTOR_PROMPT = """\
你是一个电商客服工具路由器。根据用户消息，从可用工具列表中选择最相关的工具。

每个工具的 description 已包含其功能说明，请根据语义进行匹配。
如果多个工具功能相似，选择最直接、最精准的。
当用户同时涉及多个操作时（如订单+物流），可以同时选中。

仅选择回答查询所直接需要的工具。"""

# 全局 middleware 实例
_TOOL_SELECTOR_MIDDLEWARE: LLMToolSelectorMiddleware | None = None


def _get_tool_selector_middleware(llm_service) -> LLMToolSelectorMiddleware:
    """获取工具选择器 middleware（云端 fallback）。"""
    global _TOOL_SELECTOR_MIDDLEWARE
    if _TOOL_SELECTOR_MIDDLEWARE is not None:
        return _TOOL_SELECTOR_MIDDLEWARE

    try:
        model = llm_service.tool_selector_llm if llm_service else None
    except Exception:
        model = None

    _TOOL_SELECTOR_MIDDLEWARE = LLMToolSelectorMiddleware(
        model=model,
        max_tools=3,
        always_include=["knowledge_search"],
        system_prompt=_TOOL_SELECTOR_PROMPT,
    )
    return _TOOL_SELECTOR_MIDDLEWARE

class EmbeddingToolMatcher:
    """基于 FAISS HNSW 图索引的工具语义匹配器。"""

    def __init__(
        self,
        tool_descriptions: Dict[str, str],
        embedding_service,
        *,
        intent_boost: float = 1.5,
    ):
        self._descriptions = tool_descriptions
        self._emb_service = embedding_service
        self._intent_boost = intent_boost
        self._intent_tool_map: Dict[str, Set[str]] = {}
        self._index: faiss.Index | None = None
        self._tool_names: List[str] = []
        self._ready = False
        self._init_failed = False
        self._init_lock = asyncio.Lock()
        # 查询 embedding 缓存：同一问句语义一致，避免重复编码（load test / 高频重复问题收益显著）
        self._query_cache: Dict[str, Any] = {}
        self._query_cache_max = 2048
        # 单例化后（见 react_agent._get_tool_matcher）_query_cache 跨请求共享，
        # 加锁保护读写，避免并发 dict resize 竞争
        self._cache_lock = asyncio.Lock()

    async def _ensure_index(self):
        """预计算所有工具描述的 embedding 向量并构建 FAISS 索引。"""
        if self._ready:
            return
        async with self._init_lock:
            if self._ready:
                return

            if not self._descriptions:
                logger.warning("工具描述为空，跳过 FAISS 索引构建")
                self._ready = True
                return

            texts: List[str] = []
            for name, desc in self._descriptions.items():
                texts.append(f"工具名称：{name}；功能描述：{desc}")
                self._tool_names.append(name)

            try:
                vectors = await self._emb_service.embed_texts(texts)
            except Exception:
                logger.exception("工具 embedding 向量化失败，将在下次 rank() 时重试")
                self._init_failed = True
                return

            vecs_np = np.array(vectors, dtype=np.float32)
            faiss.normalize_L2(vecs_np)

            M = 16
            self._index = faiss.IndexHNSWFlat(vecs_np.shape[1], M)
            # 提速：生产工具集仅数十向量，HNSW 低 ef 已足够召回，显著降低 search 耗时
            self._index.hnsw.efConstruction = 32
            self._index.hnsw.efSearch = 16

            self._index.add(vecs_np)
            self._ready = True
            logger.info(
                "工具 embedding 索引构建完成",
                tool_count=len(self._tool_names),
                dim=vecs_np.shape[1],
            )

    async def warmup(self) -> None:
        """预热 FAISS 索引（构建工具 embedding + 建图）。

        在 Agent 启动时调用一次，把首次 ``rank()`` 的建索引尖峰从用户请求路径移除；
        失败时置 ``_init_failed``，后续 ``rank()`` 自愈重试，不影响整体流程。
        """
        try:
            await self._ensure_index()
        except Exception as e:
            logger.warning(f"工具匹配器预热失败（rank 时将重试）: {e}")

    async def rank(
        self,
        user_query: str,
        candidate_names: Set[str],
        intent_action: str | None,
        top_k: int = 3,
        query_embedding: Any | None = None,
    ) -> List[str]:
        """对候选工具做语义重排，返回 Top-K 名称列表（见 ``rank_with_scores``）。"""
        scored = await self.rank_with_scores(
            user_query, candidate_names, intent_action, top_k, query_embedding
        )
        return [name for name, _ in scored]

    async def rank_with_scores(
        self,
        user_query: str,
        candidate_names: Set[str],
        intent_action: str | None,
        top_k: int = 3,
        query_embedding: Any | None = None,
    ) -> List[Tuple[str, float]]:
        """同 ``rank``，但额外返回每个候选的语义相似度分数（余弦，∈[0,1]，设计 1）。

        相似度源自 FAISS HNSW 的 L2 距离换算（归一化向量下 ``cos = 1 - dist/2``）；
        命中意图规则再做 boost 后截断到 1.0，符合置信度语义。无索引 / 无候选时返回
        ``(name, 0.0)`` 占位，表示"无真实语义信号"。
        """
        await self._ensure_index()

        if self._index is None or self._index.ntotal == 0:
            if self._init_failed:
                logger.warning("FAISS 索引初始化失败，回退到 P0 规则过滤")
            else:
                logger.warning("FAISS 索引为空，回退到 candidate_names")
            fallback = list(candidate_names)
            return [(n, 0.0) for n in fallback[:top_k]]

        if query_embedding is not None:
            query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query_vec)
        else:
            async with self._cache_lock:
                cached = self._query_cache.get(user_query)
                if cached is not None:
                    query_vec = cached
                else:
                    q = np.array(
                        await self._emb_service.embed_query(user_query),
                        dtype=np.float32,
                    ).reshape(1, -1)
                    faiss.normalize_L2(q)
                    query_vec = q
                    self._query_cache[user_query] = q
                    if len(self._query_cache) > self._query_cache_max:
                        # FIFO 淘汰：弹出最早插入的键
                        self._query_cache.pop(next(iter(self._query_cache)))

        search_k = min(self._index.ntotal, max(top_k, top_k * 3))
        if len(candidate_names) > 10:
            search_k = min(self._index.ntotal, max(16, top_k * 5))
        scores, indices = self._index.search(query_vec, search_k)

        intent_tools = self._intent_tool_map.get(intent_action, set()) if intent_action else set()
        scored: List[Tuple[str, float]] = []
        for idx, dist in zip(indices[0], scores[0], strict=False):
            if idx < 0 or idx >= len(self._tool_names):
                continue
            name = self._tool_names[idx]
            if name not in candidate_names:
                continue

            score = max(0.0, 1.0 - float(dist) / 2.0)
            if name in intent_tools:
                score *= self._intent_boost
            score = min(1.0, score)

            scored.append((name, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        logger.info(
            "Embedding 工具重排完成",
            candidates=len(candidate_names),
            top_k=len(top),
            scores=[(name, round(score, 4)) for name, score in top],
        )
        return top


def _make_business_args_schema(
    action: str, preset_params: dict | None = None
) -> type[BaseModel]:
    """为业务 action 生成显式 pydantic args schema。

    参数定义从 ``SkillRegistry``（单一数据源，最终来自 ``schemas.INTENT_PARAM_SCHEMAS``
    的 Pydantic 模型）读取，不再维护一份平行硬编码字典。

    硬强制（方式 2）：``preset_params`` 中已有值的字段会被**从 schema 剔除**，
    使模型在 tool-calling 时根本看不到、也无从填写这些字段——关键参数只能来自
    前置确定性抽取并经由闭包预设，杜绝模型自行抽参填错（如用错历史订单号）。
    """
    preset_params = preset_params or {}
    registry = get_skill_registry()
    skill = next((s for s in registry.skills if s.name == action), None)
    field_defs = skill.params if skill else {}
    fields: Dict[str, Any] = {}
    for name, meta in field_defs.items():
        if name in preset_params and preset_params[name] not in (None, ""):
            # 已确定性预设 → 模型不可见、不可填
            continue
        fields[name] = (
            Optional[str],
            Field(default=None, description=meta.get("description", "")),
        )
    return create_model(f"{action}_args", __base__=BaseModel, **fields)
