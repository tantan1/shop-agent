"""失控循环防护（scope-05 最小版）。

在网关层检测同一 tenant 短时间内重复触发**相同请求**的失控循环（Agent 自我循环调用），
超阈值即熔断返回 429，防止无限烧钱/烧配额。

边界（最小版，见 scope-05）：
- 仅 tenant 级、基于请求指纹：fp = hash(tenant + model + 归一化 prompt)。
- 不深入 Agent 链路追踪，只防「同一请求被重复打爆」。
- 滑动时间窗：窗口内同指纹计数超 LOOP_GUARD_MAX 即 deny，过期自动清理。
- 与限流同层：建连前执行，fail 开关同层，不破坏无旁路不变量。
- 进程内状态（多实例一致性留后续，接 redis）。

指标：gateway_loop_guarded_total{tenant} 随熔断累加。
"""
from __future__ import annotations

import hashlib
import time

from .config import settings
from .types import Verdict

# 指纹 -> 命中时间戳列表（滑动窗口）
_fingerprints: dict[str, list[float]] = {}


def _fp(tenant: str, model: str, prompt: bytes) -> str:
    """归一化指纹：tenant + model + 去空白 prompt 的 sha1。"""
    norm = (prompt or b"").decode("utf-8", "ignore")
    norm = " ".join(norm.split())  # 折叠空白，避免格式差造成的指纹漂移
    h = hashlib.sha1(f"{tenant}|{model}|{norm}".encode("utf-8")).hexdigest()
    return h


def check(tenant: str, model: str, prompt: bytes) -> Verdict:
    """建连前失控循环检查：同指纹窗口内超阈值即 deny（429）。

    返回 Verdict.deny(reason="loop guarded") 供调用方转 429。
    """
    max_hits = settings.loop_guard_max
    window = settings.loop_guard_window_sec
    if max_hits <= 0:
        return Verdict.allow()  # 开关关闭

    now = time.monotonic()
    key = _fp(tenant, model, prompt)
    hits = _fingerprints.get(key)
    if hits is None:
        hits = []
        _fingerprints[key] = hits
    # 清理窗口外旧命中
    hits[:] = [t for t in hits if now - t <= window]
    hits.append(now)

    if len(hits) > max_hits:
        from .metrics import get_metric

        m = get_metric("gateway_loop_guarded_total")
        if m is not None:
            m.inc(labels={"tenant": tenant})
        return Verdict.deny("loop guarded")
    return Verdict.allow()


def reset() -> None:
    """清空循环检测状态（仅供测试）。"""
    _fingerprints.clear()
