"""指标埋点端口（stub 实现：进程内聚合，可导出到 /metrics）。

G 维度修复：成本与资源用量需可被 metrics 采集（token/¥ 计数器）。
当前 stub 用进程内 dict 聚合；生产可替换为 Prometheus / OTel Meter。
调用方只依赖 `metrics.record_cost(...)` / `metrics.incr(...)`。
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

_counters: Dict[str, float] = {}
_gauges: Dict[str, float] = {}
_lock = threading.Lock()


def incr(name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
    """累加计数器（如请求数、错误数）。tags 当前仅作标注，stub 阶段忽略维度。"""
    key = name
    with _lock:
        _counters[key] = _counters.get(key, 0.0) + value


def record_cost(model: str, tokens: int = 0, amount: float = 0.0) -> None:
    """记录模型调用成本（token 数与折算金额）。

    Args:
        model: 模型名
        tokens: 本次消耗 token 数
        amount: 折算金额（¥），由调用方按单价计算
    """
    incr(f"cost.tokens.{model}", tokens)
    incr("cost.tokens.total", tokens)
    incr("cost.amount.total", amount)


def snapshot() -> Dict[str, float]:
    """导出当前聚合指标（供 /metrics 或调试使用）。"""
    with _lock:
        return dict(_counters)
