"""
Redis 滑动窗口速率限制器（请求次数 + Token 消耗双维度），含内存降级。

用法：
    from src.core.rate_limiter import RateLimiter, get_rate_limiter

    limiter = get_rate_limiter()

    # ── 请求次数限流 ──
    # 方式1：FastAPI 中间件（全局，排除 health/docs 等）
    app.middleware("http")(limiter.middleware)

    # 方式2：依赖注入
    @router.post("/agent/chat", dependencies=[Depends(limiter.dependency(30))])
    async def agent_chat(...): ...

    # ── Token 消耗限流 ──
    # 预检（LLM 调用前）
    allowed, remaining, reset = limiter.check_tokens(
        key="user_ip:endpoint", estimated_tokens=1500,
        max_tokens=100000, window_seconds=60
    )
    # 上报（LLM 调用后，回调/on_llm_end 中调用）
    limiter.report_tokens(key="user_ip:endpoint", actual_tokens=1280, window_seconds=60)
"""

import time
import threading
from typing import Optional, Dict, Tuple

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.shared.logger import APILogger

logger = APILogger("rate_limiter")

# 排除速率限制的路径（健康检查、文档、监控指标）
_EXCLUDED_PATHS = ("/health", "/docs", "/redoc", "/openapi.json", "/metrics", "/monitoring/metrics")


class RateLimiter:
    """Redis 为主、内存降级的滑动窗口速率限制器。

    优先使用 Redis（INCR + EXPIRE），Redis 不可用时回退到本地内存字典。
    响应自动注入 X-RateLimit-* 标头。
    """

    # ── 内存降级（单进程）─────────────────────────────────────────────
    _memory_store: Dict[str, Tuple[int, float]] = {}  # key -> (count, window_start)
    _memory_lock = threading.Lock()

    def __init__(self):
        self._redis = None  # 懒加载，首次请求时初始化

    # ── Redis 懒加载 ──────────────────────────────────────────────────

    def _ensure_redis(self) -> bool:
        if self._redis is not None:
            return True
        try:
            import redis as redis_lib
            from src.modules.chat.config import chat_config

            config = chat_config
            host = getattr(config, "redis_host", None) or "localhost"
            port = getattr(config, "redis_port", None) or 6379
            password = getattr(config, "redis_password", None) or None

            self._redis = redis_lib.Redis(
                host=host, port=port, password=password, db=0,
                decode_responses=True,
                socket_connect_timeout=3, socket_timeout=3,
            )
            self._redis.ping()
            logger.info("速率限制器 Redis 连接成功")
            return True
        except Exception:
            logger.warning("速率限制器 Redis 不可用，回落到内存降级（仅限单进程）")
            self._redis = None
            return False

    # ── 核心逻辑 ──────────────────────────────────────────────────────

    def _check_redis(self, key: str, max_requests: int, window_seconds: int
                     ) -> Tuple[bool, int, int]:
        """Redis 滑动窗口检查。返回 (allowed, remaining, reset_seconds)。"""
        now_ms = int(time.time() * 1000)
        window_ms = window_seconds * 1000
        cutoff = now_ms - window_ms

        pipe = self._redis.pipeline()
        # 移除窗口外的旧条目
        pipe.zremrangebyscore(key, 0, cutoff)
        # 统计窗口内请求数
        pipe.zcard(key)
        # 添加当前请求时间戳
        pipe.zadd(key, {str(now_ms): now_ms})
        # 设置 key 过期（窗口 + 1s 冗余）
        pipe.expire(key, window_seconds + 1)
        results = pipe.execute()

        current_count = results[1]  # zcard（添加前的数量）
        allowed = current_count < max_requests

        # 计算剩余配额和重置时间
        remaining = max(0, max_requests - current_count - 1)
        oldest = self._redis.zrange(key, 0, 0, withscores=True)
        if oldest:
            reset_seconds = max(0, int((oldest[0][1] + window_ms - now_ms) / 1000))
        else:
            reset_seconds = window_seconds

        return allowed, remaining, reset_seconds

    def _check_memory(self, key: str, max_requests: int, window_seconds: int
                      ) -> Tuple[bool, int, int]:
        """本地内存滑动窗口检查（Redis 不可用时降级）。"""
        now = time.time()
        with self._memory_lock:
            count, window_start = self._memory_store.get(key, (0, now))
            if now - window_start > window_seconds:
                # 窗口过期，重置
                count = 1
                window_start = now
            else:
                count += 1
            self._memory_store[key] = (count, window_start)

        allowed = count <= max_requests
        remaining = max(0, max_requests - count)
        reset_seconds = max(0, int(window_start + window_seconds - now))
        return allowed, remaining, reset_seconds

    def check(self, key: str, max_requests: int = 30, window_seconds: int = 60
              ) -> Tuple[bool, int, int]:
        """检查是否允许请求。返回 (allowed, remaining, reset_seconds)。

        Args:
            key: 速率限制 key（通常是 client_ip:endpoint）
            max_requests: 时间窗口内最大请求数
            window_seconds: 滑动窗口长度（秒）
        """
        redis_key = f"rate_limit:{key}"
        try:
            if not self._ensure_redis():
                return self._check_memory(redis_key, max_requests, window_seconds)
            return self._check_redis(redis_key, max_requests, window_seconds)
        except Exception:
            logger.debug("Redis 速率限制异常，回退内存: %s", key[:50])
            return self._check_memory(redis_key, max_requests, window_seconds)

    # ── FastAPI 中间件 ────────────────────────────────────────────────

    async def middleware(self, request: Request, call_next) -> Response:
        """全局速率限制中间件（排除 health/docs/monitoring 路径）。"""
        path = request.url.path.rstrip("/")
        if path in _EXCLUDED_PATHS or path.startswith("/docs") or path.startswith("/redoc"):
            return await call_next(request)

        # 按 IP + 路径做 key（可按需求调整为 user_id 或 api_key）
        client_ip = request.client.host if request.client else "unknown"
        key = f"{client_ip}:{path}"
        # 全局默认限制：30 req / 60s
        allowed, remaining, reset_seconds = self.check(key, max_requests=30, window_seconds=60)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = "30"
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_seconds)
        return response

    # ── 依赖注入 ──────────────────────────────────────────────────────

    def dependency(self, max_requests: int = 30, window_seconds: int = 60):
        """FastAPI Depends 工厂：创建端点级速率限制依赖。

        用法：
            @router.post("/agent/chat", dependencies=[Depends(limiter.dependency(30))])
            async def agent_chat(...): ...
        """
        async def _limiter(request: Request):
            client_ip = request.client.host if request.client else "unknown"
            path = request.url.path.rstrip("/")
            key = f"{client_ip}:{path}"
            allowed, remaining, reset_seconds = self.check(
                key, max_requests=max_requests, window_seconds=window_seconds
            )
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "请求过于频繁，请稍后再试",
                        "retry_after": reset_seconds,
                    },
                    headers={
                        "Retry-After": str(reset_seconds),
                        "X-RateLimit-Limit": str(max_requests),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(reset_seconds),
                    },
                )
        return _limiter


    # ── Token 消耗限流 ──────────────────────────────────────────────

    def _check_tokens_redis(self, key: str, estimated_tokens: int,
                            max_tokens: int, window_seconds: int
                            ) -> Tuple[bool, int, int]:
        """Redis ZSET 滑动窗口检查 token 是否超限。
        返回 (allowed, remaining_tokens, reset_seconds)。"""
        now_ms = int(time.time() * 1000)
        window_ms = window_seconds * 1000
        cutoff = now_ms - window_ms

        pipe = self._redis.pipeline()
        # 清除窗口外的旧条目
        pipe.zremrangebyscore(key, 0, cutoff)
        # 累加窗口内所有 token 数（member 格式: "ts:actual_tokens"）
        pipe.zrange(key, 0, -1, withscores=False)
        results = pipe.execute()

        # 计算窗口内已消耗 token（member 格式: "{ts}:{tokens}"）
        used_tokens = 0
        for member in results[1]:
            try:
                _, tokens = member.rsplit(":", 1)
                used_tokens += int(tokens)
            except (ValueError, AttributeError):
                pass

        # 检查是否超限
        projected = used_tokens + estimated_tokens
        allowed = projected <= max_tokens
        remaining = max(0, max_tokens - projected)

        if allowed:
            # 预占 estimated_tokens（后续可用 report_tokens 修正为真实值）
            pipe2 = self._redis.pipeline()
            pipe2.zadd(key, {f"{now_ms}:{estimated_tokens}": now_ms})
            pipe2.expire(key, window_seconds + 1)
            pipe2.execute()

        # 计算重置时间
        oldest = self._redis.zrange(key, 0, 0, withscores=True)
        if oldest:
            reset_seconds = max(0, int((oldest[0][1] + window_ms - now_ms) / 1000))
        else:
            reset_seconds = window_seconds

        return allowed, remaining, reset_seconds

    def _check_tokens_memory(self, key: str, estimated_tokens: int,
                             max_tokens: int, window_seconds: int
                             ) -> Tuple[bool, int, int]:
        """内存降级 token 限流（单进程）。"""
        now = time.time()
        with self._memory_lock:
            record = self._memory_store.get(key)
            if record is None or now - record[1] > window_seconds:
                # 首个 token：(累计 token 数, 窗口起始时间)
                used_tokens = 0
                window_start = now
            else:
                used_tokens, window_start = record

            projected = used_tokens + estimated_tokens
            allowed = projected <= max_tokens
            remaining = max(0, max_tokens - projected)

            if allowed:
                self._memory_store[key] = (projected, window_start)

        reset_seconds = max(0, int(window_start + window_seconds - now))
        return allowed, remaining, reset_seconds

    def check_tokens(self, key: str, estimated_tokens: int,
                     max_tokens: int = 100000, window_seconds: int = 60
                     ) -> Tuple[bool, int, int]:
        """Token 消耗预检：检查本次预估 token 是否会导致超限。

        应在 LLM 调用之前调用，用预估 token 数做检查。

        Args:
            key: 限流 key（如 client_ip:endpoint 或 user_id:endpoint）
            estimated_tokens: 本次请求预估的 token 数
            max_tokens: 时间窗口内最大允许 token 数（默认 100K/min）
            window_seconds: 滑动窗口长度（秒，默认 60）

        Returns:
            (allowed, remaining_tokens, reset_seconds)
        """
        redis_key = f"token_limit:{key}"
        try:
            if not self._ensure_redis():
                return self._check_tokens_memory(
                    redis_key, estimated_tokens, max_tokens, window_seconds
                )
            return self._check_tokens_redis(
                redis_key, estimated_tokens, max_tokens, window_seconds
            )
        except Exception:
            logger.debug("Redis token 限流异常，回退内存: %s", key[:50])
            return self._check_tokens_memory(
                redis_key, estimated_tokens, max_tokens, window_seconds
            )

    def report_tokens(self, key: str, actual_tokens: int,
                      estimated_tokens: int = 0, window_seconds: int = 60):
        """Token 消耗上报：LLM 调用完成后，用真实 token 数修正预估。

        应在 on_llm_end 回调中调用，用 API 返回的真实 token 数修正 Redis 计数。
        采用"移除预估 + 写入真实"策略，避免偏差累积。

        Args:
            key: 限流 key（需与 check_tokens 一致）
            actual_tokens: 实际消耗的 token 数（通常为 total_tokens）
            estimated_tokens: 之前预估值（用于移除，默认 0 则跳过移除）
            window_seconds: 窗口长度（需与 check_tokens 一致）
        """
        redis_key = f"token_limit:{key}"
        try:
            if not self._ensure_redis():
                # 内存模式：直接修正累计值
                with self._memory_lock:
                    record = self._memory_store.get(redis_key)
                    if record:
                        used, ws = record
                        corrected = used - estimated_tokens + actual_tokens
                        self._memory_store[redis_key] = (max(0, corrected), ws)
                return

            now_ms = int(time.time() * 1000)
            pipe = self._redis.pipeline()
            if estimated_tokens > 0:
                # 移除预占条目
                pipe.zrem(redis_key, f"*:{estimated_tokens}")
            # 写入真实值
            pipe.zadd(redis_key, {f"{now_ms}:{actual_tokens}": now_ms})
            pipe.expire(redis_key, window_seconds + 1)
            pipe.execute()
        except Exception:
            logger.debug("Token 上报失败: %s", key[:50])

    # ── Token 限流依赖注入 ─────────────────────────────────────────

    def token_dependency(self, estimated_tokens: int = 0,
                         max_tokens: int = 100000, window_seconds: int = 60):
        """FastAPI Depends 工厂：创建 token 消耗限流依赖。

        用法：
            @router.post("/agent/chat", dependencies=[
                Depends(limiter.token_dependency(max_tokens=100000))
            ])
            async def agent_chat(...): ...

        注意：estimated_tokens 需要调用方在请求体中传入或由 middleware 预估。
        """
        async def _token_limiter(request: Request):
            client_ip = request.client.host if request.client else "unknown"
            path = request.url.path.rstrip("/")
            key = f"{client_ip}:{path}"

            # 如果没有传入预估 token，跳过 token 限流（仅做请求次数限流）
            est = estimated_tokens
            if est <= 0:
                return

            allowed, remaining, reset_seconds = self.check_tokens(
                key, estimated_tokens=est,
                max_tokens=max_tokens, window_seconds=window_seconds,
            )
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Token 消耗超限，请稍后再试",
                        "retry_after": reset_seconds,
                        "limit_type": "token",
                    },
                    headers={
                        "Retry-After": str(reset_seconds),
                        "X-Token-Limit-Max": str(max_tokens),
                        "X-Token-Limit-Remaining": str(remaining),
                        "X-Token-Limit-Reset": str(reset_seconds),
                    },
                )
        return _token_limiter


# 全局单例（供 main.py 和 routers.py 共用）
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """获取全局 RateLimiter 单例。"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
