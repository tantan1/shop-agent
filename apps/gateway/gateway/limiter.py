"""集中限流与预算熔断（03 §2 三层治理：计量 → 限流/配额 → 预算熔断）。

令牌桶（token bucket），按 tenant 隔离 + 全局桶两层（引 03 §4）。
判断均在「建连前」执行，与出向 fail 开关同层，不破坏无旁路不变量。

分布式支持（批次3）：
  - 若 settings.redis_url 配置，桶状态存 Redis，用 Lua 脚本原子更新，
    保证多副本部署时限流凭证仍可证伪（不会因每副本各自计数而放大配额）。
  - 未配置 redis_url 时优雅降级为进程内桶（单实例/本地开发/单测兼容）。
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from .config import settings
from .types import Verdict


# ── 令牌桶接口 ──────────────────────────────────────────────────────────────
class _TokenBucket(ABC):
    """令牌桶抽象：capacity 容量，refill 速率（令牌/秒）。"""

    @abstractmethod
    def allow(self) -> bool:
        """原子地补充令牌并尝试消费 1 个，成功返回 True。"""
        ...


# ── 进程内实现（单实例 / 降级） ──────────────────────────────────────────────
@dataclass
class _LocalBucket(_TokenBucket):
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


# ── Redis 实现（多副本分布式，Lua 原子令牌桶） ──────────────────────────────
# KEYS[1]=bucket key, ARGV[1]=capacity, ARGV[2]=refill(令牌/秒), ARGV[3]=now(秒)
_REDIS_LUA = """
local cap = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local data = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil then
  tokens = cap
  ts = now
end
tokens = math.min(cap, tokens + (now - ts) * refill)
local allowed = 0
if tokens >= 1.0 then
  tokens = tokens - 1.0
  allowed = 1
end
redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', KEYS[1], 3600)
return allowed
"""


class _RedisBucket(_TokenBucket):
    """Redis 令牌桶：用 Lua 脚本在服务器端原子完成补充+消费，避免多副本竞态。"""

    def __init__(self, redis, key: str, capacity: int, refill: float) -> None:
        self._redis = redis
        self._key = key
        self._capacity = capacity
        self._refill = refill
        self._script = redis.register_script(_REDIS_LUA)

    def allow(self) -> bool:
        allowed = self._script(
            keys=[self._key],
            args=[self._capacity, self._refill, time.time()],
        )
        # redis-py 返回 int（Lua number 转 bytes/int 视版本）
        return int(allowed) == 1


# ── 工厂：按配置选择实现 ─────────────────────────────────────────────────────
_redis_client = None
_redis_initialized = False


def _get_redis():
    """懒加载 Redis 客户端；首次失败时置空并永久降级本地（避免每请求重试）。"""
    global _redis_client, _redis_initialized
    if _redis_initialized:
        return _redis_client
    _redis_initialized = True
    if not settings.redis_url:
        return None
    try:
        import redis  # 延迟导入，无依赖时不报错
        _redis_client = redis.from_url(settings.redis_url, decode_responses=False)
        _redis_client.ping()
        return _redis_client
    except Exception:
        _redis_client = None
        return None


def _make_bucket(key: str, capacity: int, refill: float) -> _TokenBucket:
    r = _get_redis()
    if r is not None:
        return _RedisBucket(r, f"gw:limiter:{key}", capacity, refill)
    return _LocalBucket(capacity=capacity, tokens=float(capacity), refill=refill)


# ── 两层桶管理（接口不变） ───────────────────────────────────────────────────
_global_bucket: Optional[_TokenBucket] = None
_tenant_buckets: Dict[str, _TokenBucket] = {}


def _global() -> _TokenBucket:
    global _global_bucket
    if _global_bucket is None:
        _global_bucket = _make_bucket("global", settings.rate_limit_burst, settings.rate_limit_global_rps)
    return _global_bucket


def _tenant(tenant: str) -> _TokenBucket:
    b = _tenant_buckets.get(tenant)
    if b is None:
        b = _make_bucket(f"tenant:{tenant}", settings.rate_limit_burst, settings.rate_limit_tenant_rps)
        _tenant_buckets[tenant] = b
    return b


def check(tenant: str) -> Verdict:
    """建连前限流检查：全局桶 + 单租户桶任一不足即 deny（429）。

    返回 Verdict.deny(reason="rate limited") 供调用方转 429 + Retry-After。
    多副本部署下（配置 redis_url）桶状态跨实例共享，凭证仍可证伪。
    """
    if not _global().allow():
        return Verdict.deny("rate limited (global)")
    if not _tenant(tenant).allow():
        return Verdict.deny("rate limited (tenant)")
    return Verdict.allow()
