"""FAISS 意图索引构建 —— 基础设施层，与业务策略解耦。

唯一数据源是 ``SkillRegistry``（各 SKILL.md frontmatter 的 ``examples``）。
代码里不再硬编码意图示例，改 SKILL.md 即生效，
消灭「改了 SKILL.md 但意图识别不跟随」的双份事实源问题。

索引在模块级缓存（与原 ``IntentRecognizer`` 类级共享语义一致），
测试可通过 ``reset_intent_index_cache()`` 重置。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import faiss
import numpy as np

from src.shared.logger import APILogger

logger = APILogger("intent_index")

# 示例提供者：Callable[[], Dict[intent, List[example]]]
ExamplesProvider = Callable[[], Dict[str, List[str]]]


@dataclass(frozen=True)
class IntentMatch:
    """单条索引命中结果。"""

    action: Optional[str]
    score: float
    matched: Optional[str] = None


@dataclass
class IntentIndex:
    """意图向量索引：``actions`` / ``examples`` 与 FAISS 行一一对应。"""

    index: Any
    actions: List[str]
    examples: List[str]
    dim: int

    def search(self, query_vec: Sequence[float], k: int = 2) -> List[IntentMatch]:
        """检索 top-k，返回按分数降序的命中列表。"""
        if self.index.ntotal <= 0:
            return []
        q = np.asarray([query_vec], dtype=np.float32)
        k = max(1, min(int(k), int(self.index.ntotal)))
        scores, indices = self.index.search(q, k)
        out: List[IntentMatch] = []
        for score, idx in zip(scores[0], indices[0]):
            i = int(idx)
            if i < 0 or i >= len(self.actions):
                continue
            out.append(
                IntentMatch(
                    action=self.actions[i],
                    score=float(score),
                    matched=self.examples[i],
                )
            )
        return out


# ── 示例抽取 ──

def flatten_examples(examples: Dict[str, List[str]]) -> tuple[List[str], List[str]]:
    """把 ``{intent: [examples]}`` 展开为与索引行对齐的两个列表。"""
    actions: List[str] = []
    phrases: List[str] = []
    for action, items in (examples or {}).items():
        for p in items or []:
            if not p:
                continue
            actions.append(str(action))
            phrases.append(str(p))
    return actions, phrases


def examples_from_registry(skill_registry) -> Dict[str, List[str]]:
    """从 SkillRegistry 抽取 ``{intent: [examples]}``（唯一数据源）。

    兼容 ``skills`` 为 List[SkillDef]（真实）或 Dict（测试 fake）。
    """
    if skill_registry is None:
        return {}
    skills = getattr(skill_registry, "skills", None)
    if not skills:
        return {}
    items = skills.values() if isinstance(skills, dict) else skills

    out: Dict[str, List[str]] = {}
    for s in items:
        name = getattr(s, "name", None)
        if not name:
            continue
        ex = [str(e).strip() for e in (getattr(s, "examples", None) or []) if str(e).strip()]
        if ex:
            out[str(name)] = ex
    return out


# ── 构建 ──

def build_index(vecs, actions: List[str], phrases: List[str]) -> Optional[IntentIndex]:
    if not actions or not phrases:
        return None
    m = np.asarray(vecs, dtype=np.float32)
    if m.ndim != 2 or m.shape[0] != len(actions):
        logger.warning(
            f"意图索引向量与标签数量不一致: vecs={m.shape[0] if m.ndim == 2 else '?'}, labels={len(actions)}"
        )
        return None
    dim = int(m.shape[1])
    index = faiss.IndexFlatIP(dim)  # BGE 输出已 L2 归一化 → 内积 = 余弦
    index.add(np.ascontiguousarray(m))
    return IntentIndex(index=index, actions=list(actions), examples=list(phrases), dim=dim)


# ── 全局缓存 ──

_INDEX_CACHE: Optional[IntentIndex] = None


def get_cached_index() -> Optional[IntentIndex]:
    return _INDEX_CACHE


def reset_intent_index_cache() -> None:
    """清空全局索引缓存（仅供测试）。"""
    global _INDEX_CACHE
    _INDEX_CACHE = None


def _log_built(idx: IntentIndex, examples: Dict[str, List[str]]) -> None:
    logger.info(
        "FAISS 意图索引构建完成",
        dim=idx.dim,
        intents=len(examples),
        total_vectors=len(idx.actions),
    )


async def ensure_intent_index_async(
    embedding_service, examples_provider: ExamplesProvider
) -> Optional[IntentIndex]:
    """异步构建（或复用）意图索引。"""
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
    if not embedding_service:
        logger.warning("Embedding 服务未初始化，跳过 FAISS 意图索引构建")
        return None
    try:
        examples = examples_provider() or {}
        actions, phrases = flatten_examples(examples)
        if not phrases:
            logger.warning("意图示例为空：SkillRegistry 未提供任何 examples")
            return None
        emb = embedding_service.get_embeddings()
        vecs = await emb.aembed_documents(phrases)
        idx = build_index(vecs, actions, phrases)
        if idx is None:
            return None
        _log_built(idx, examples)
        _INDEX_CACHE = idx
        return idx
    except Exception as e:
        logger.warning(f"FAISS 意图索引构建失败: {e}")
        return None


def ensure_intent_index_sync(
    embedding_service, examples_provider: ExamplesProvider
) -> Optional[IntentIndex]:
    """同步构建（或复用）意图索引 —— 用于 lifespan 预热，不依赖事件循环。"""
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
    if not embedding_service:
        logger.warning("Embedding 服务未初始化，跳过 FAISS 意图索引构建")
        return None
    try:
        examples = examples_provider() or {}
        actions, phrases = flatten_examples(examples)
        if not phrases:
            logger.warning("意图示例为空：SkillRegistry 未提供任何 examples")
            return None
        emb = embedding_service.get_embeddings()
        vecs = emb.embed_documents(phrases)
        idx = build_index(vecs, actions, phrases)
        if idx is None:
            return None
        _log_built(idx, examples)
        _INDEX_CACHE = idx
        return idx
    except Exception as e:
        logger.warning(f"FAISS 意图索引构建失败: {e}")
        return None
