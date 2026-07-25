"""语义缓存存储（08 批次2/2b）。

本批用**进程内**存储（dict：key→(answer, ts)）+ 本地词频向量 + 余弦相似度（E2 乙）。
真实语义 embedding 留 ``embed`` 钩子不接模型（接口签名不变，未来换真模型只改内部）。

命中返回缓存答案，proxy 据此短路不进模型（cache_hits_total+1，不 record）。

设计约束：
- 零依赖：不引入 embedding 模型/外部调用（防 SPOF）。embed 仅做词频向量占位。
- D2：``gateway_cache_hits_total`` 注册在**本模块级**，避免 2b 三路争抢 metrics.py。
- async 友好：本类方法均为内存操作，无阻塞 IO；reload 走 policy 懒触发，不在热路径同步读盘。
- B3：cache_store 仅在 egress 治理链全部通过后调用（调用方保证），故命中即返回合规内容。
"""
from __future__ import annotations

import math
import time
from typing import Any

from gateway.cache.pii_gate import gate as pii_gate
from gateway.cache.policy import policy as cache_policy
from gateway.config import settings
from gateway.metrics import register_counter

# D2：缓存命中计数（区分缓存短路 vs 真实消耗，接 03）。
# 注册在 cache/store.py 模块级（标签 {tenant, bucket}）。
cache_hits_total = register_counter(
    "gateway_cache_hits_total",
    "cache hit short-circuits by tenant and bucket",
    ("tenant", "bucket"),
)

# 缓存写入计数（含被 PII 门禁拦截的写操作，供可观测）。
cache_writes_total = register_counter(
    "gateway_cache_writes_total",
    "cache write attempts by tenant and bucket",
    ("tenant", "bucket"),
)
# PII 门禁拦截写入计数（硬约束可观测）。
cache_pii_blocked_total = register_counter(
    "gateway_cache_pii_blocked_total",
    "cache writes blocked by PII gate by tenant and bucket",
    ("tenant", "bucket"),
)


def embed(text: str) -> dict[str, float]:
    """词频向量（E2 乙）：归一化后分词词频向量。真实语义 embedding 留钩子。

    接口签名稳定：未来换真模型只改本函数内部，调用方不变。
    """
    from gateway.cache.policy import _normalize

    tokens = _normalize(text).split()
    vec: dict[str, float] = {}
    for tok in tokens:
        vec[tok] = vec.get(tok, 0.0) + 1.0
    # L2 归一化（余弦分母），空向量返回空 dict
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm == 0:
        return {}
    return {k: v / norm for k, v in vec.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """余弦相似度（基于已 L2 归一化向量，点积即余弦）。"""
    if not a or not b:
        return 0.0
    # 在较小向量上迭代，对较大向量取交集（稀疏向量点积）。
    if len(a) <= len(b):
        small, large = a, b
    else:
        small, large = b, a
    return sum(v * large.get(k, 0.0) for k, v in small.items())


class SemanticCache:
    """进程内语义缓存（词频向量 + 余弦 + TTL + 业务分片）。"""

    def __init__(self) -> None:
        # key: 归一化向量哈希 → (answer, ts, bucket, tenant, model)
        self._store: dict[str, tuple[str, float, str, str, str]] = {}
        self._by_raw: dict[str, str] = {}  # raw_norm_key → vec_key 反向索引（便于清理）

    def _vec_key(self, vec: dict[str, float]) -> str:
        # 用词频向量构造稳定 key：排序后拼 token:weight
        parts = sorted(f"{k}={v:.6f}" for k, v in vec.items())
        return "|".join(parts)

    def lookup(
        self,
        prompt: str,
        bucket: str,
        tenant: str = "default",
        model: str = "",
    ) -> str | None:
        """近似检索：返回命中答案或 None。命中即视为短路（调用方不进模型）。

        硬约束（代码强制）：CACHE_ENABLED=false / 分片 enabled=false → 直接不命中。
        """
        if not settings.cache_enabled:
            return None
        bp = cache_policy.resolve(bucket)
        if not bp.enabled:
            return None
        vec = embed(prompt)
        if not vec:
            return None
        vkey = self._vec_key(vec)
        # 精确/近似匹配：遍历候选（进程内规模可接受）
        best: tuple[float, str] | None = None
        for key, entry in self._store.items():
            answer, ts, ebucket, etenant, emodel = entry
            if ebucket != bucket:
                continue
            # TTL 失效（写操作不命中由写入时保证；此处仅检查 TTL）
            if time.time() - ts > bp.ttl:
                continue
            sim = cosine(vec, self._vec_of(key))
            if sim >= bp.threshold:
                if best is None or sim > best[0]:
                    best = (sim, answer)
        if best is not None:
            cache_hits_total.inc(labels={"tenant": tenant, "bucket": bucket})
            return best[1]
        return None

    def _vec_of(self, key: str) -> dict[str, float]:
        """从存储 key 反解向量（key 即 vec 序列化，无精度损失）。"""
        out: dict[str, float] = {}
        for part in key.split("|"):
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                continue
        return out

    def store(
        self,
        prompt: str,
        answer: str,
        bucket: str,
        tenant: str = "default",
        model: str = "",
    ) -> bool:
        """写入缓存（经 PII 门禁 + 写操作跳过 + TTL）。

        返回 True 表示写入成功；False 表示被硬约束/分片策略拦截。

        B3 前提由调用方保证：本方法只在 egress 治理链全部通过后调用，
        故写入的 answer 已是治理后干净内容（已过 judge_egress + guardrails + 脱敏）。
        """
        cache_writes_total.inc(labels={"tenant": tenant, "bucket": bucket})
        # 硬约束层（代码强制，不参与配置优先级）
        if not settings.cache_enabled:
            return False
        bp = cache_policy.resolve(bucket)
        if not bp.enabled:
            return False
        if bp.skip_write:
            # 写操作分片：永不命中、永不写入（硬约束，配置不可覆盖）
            return False
        # PII 硬门禁：整条不写（调 07 引擎）
        try:
            if not pii_gate.check(prompt) or not pii_gate.check(answer):
                cache_pii_blocked_total.inc(labels={"tenant": tenant, "bucket": bucket})
                return False
        except Exception:
            # PII 门禁异常：保守拒绝写入（不存不确定内容）
            cache_pii_blocked_total.inc(labels={"tenant": tenant, "bucket": bucket})
            return False
        vec = embed(prompt)
        if not vec:
            return False
        vkey = self._vec_key(vec)
        self._store[vkey] = (answer, time.time(), bucket, tenant, model)
        return True

    def clear(self) -> None:
        """清空（仅供测试）。"""
        self._store.clear()
        self._by_raw.clear()


# 模块级单例
cache = SemanticCache()
