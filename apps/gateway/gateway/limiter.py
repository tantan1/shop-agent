"""集中限流与预算熔断（03 §2 三层治理：计量 → 限流/配额 → 预算熔断）。

进程内令牌桶（token bucket），按 tenant 隔离 + 全局桶两层（引 03 §4）。
本批次为单实例演示实现；多实例分布式桶（redis）与真实 token 计数在批次3 接。
判断均在「建连前」执行，与出向 fail 开关同层，不破坏无旁路不变量。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

from .config import settings
from .types import Verdict


@dataclass
class _Bucket:
    """令牌桶：容量 capacity，当前令牌 tokens，refill 速率（令牌/秒）。"""

    capacity: int
    tokens: float
    refill: float
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        if self.updated_at == 0.0:
            self.updated_at = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        elapsed = now - self.updated_at
        self.updated_at = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


_global_bucket: _Bucket | None = None
_tenant_buckets: Dict[str, _Bucket] = {}


def _global() -> _Bucket:
    global _global_bucket
    if _global_bucket is None:
        _global_bucket = _Bucket(
            capacity=settings.rate_limit_burst,
            tokens=settings.rate_limit_burst,
            refill=settings.rate_limit_global_rps,
        )
    return _global_bucket


def _tenant(tenant: str) -> _Bucket:
    b = _tenant_buckets.get(tenant)
    if b is None:
        b = _Bucket(
            capacity=settings.rate_limit_burst,
            tokens=settings.rate_limit_burst,
            refill=settings.rate_limit_tenant_rps,
        )
        _tenant_buckets[tenant] = b
    return b


def check(tenant: str) -> Verdict:
    """建连前限流检查：全局桶 + 单租户桶任一不足即 deny（429）。

    返回 Verdict.deny(reason="rate limited") 供调用方转 429 + Retry-After。
    """
    if not _global().allow():
        return Verdict.deny("rate limited (global)")
    if not _tenant(tenant).allow():
        return Verdict.deny("rate limited (tenant)")
    return Verdict.allow()
