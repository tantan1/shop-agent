"""Loki LogQL 临时查询封装（RCA 被叫醒后才查、查完即弃）。

设计约束（04 篇 §5）：
- 日志走「应用主动推送 → Loki 集中 → Alertmanager 判定 → webhook 叫醒 agent」。
- agent **平时零日志存储**，只在被告警叫醒后用 LogQL 临时查询相关日志原文做 RCA，
  查完即弃，绝不持久化原始日志流（呼应 §9 存储边界）。
- 直接走 httpx 调 Loki HTTP API（/loki/api/v1/query_range），不引入新依赖。
"""

from __future__ import annotations

import logging
import os
import time

import httpx

logger = logging.getLogger("monitoring_agent.loki")

LOKI_URL = os.getenv("LOKI_URL", "http://loki:3100").rstrip("/")
_QUERY_TIMEOUT = float(os.getenv("LOKI_QUERY_TIMEOUT_SEC", "10"))
_LOOKBACK_SEC = int(os.getenv("LOKI_LOOKBACK_SEC", "900"))  # 默认回看 15 分钟


class LokiUnavailable(RuntimeError):
    """Loki 不可达或查询失败（非致命：RCA 降级为纯指标/拓扑归因）。 """


def query_logs(logql: str, lookback_sec: int = _LOOKBACK_SEC, limit: int = 200) -> list[dict]:
    """执行 LogQL 范围查询，返回日志条目列表（含 ts / line / labels）。

    失败抛 :class:`LokiUnavailable`，由调用方决定是否降级。
    """
    end = int(time.time() * 1000_000_000)  # 纳秒
    start = end - lookback_sec * 1_000_000_000
    try:
        with httpx.Client(timeout=_QUERY_TIMEOUT) as c:
            r = c.get(
                f"{LOKI_URL}/loki/api/v1/query_range",
                params={
                    "query": logql,
                    "start": start,
                    "end": end,
                    "limit": limit,
                    "direction": "backward",
                },
            )
            r.raise_for_status()
            data = r.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Loki 查询失败 logql=%s err=%s", logql, exc)
        raise LokiUnavailable(str(exc)) from exc

    if data.get("status") != "success":
        raise LokiUnavailable(f"非 success 响应: {data.get('status')}")

    streams = data.get("data", {}).get("result", [])
    out: list[dict] = []
    for stream in streams:
        labels = stream.get("stream", {})
        for ts, line in stream.get("values", []):
            out.append({"ts_ns": ts, "line": line, "labels": labels})
    # 按时间倒序返回，最新在前
    out.sort(key=lambda x: x["ts_ns"], reverse=True)
    return out


async def query_logs_async(logql: str, lookback_sec: int = _LOOKBACK_SEC, limit: int = 200) -> list[dict]:
    """异步版 query_logs，供 RCA 并行查询使用。"""
    end = int(time.time() * 1000_000_000)  # 纳秒
    start = end - lookback_sec * 1_000_000_000
    try:
        async with httpx.AsyncClient(timeout=_QUERY_TIMEOUT) as c:
            r = await c.get(
                f"{LOKI_URL}/loki/api/v1/query_range",
                params={
                    "query": logql,
                    "start": start,
                    "end": end,
                    "limit": limit,
                    "direction": "backward",
                },
            )
            r.raise_for_status()
            data = r.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Loki 异步查询失败 logql=%s err=%s", logql, exc)
        raise LokiUnavailable(str(exc)) from exc

    if data.get("status") != "success":
        raise LokiUnavailable(f"非 success 响应: {data.get('status')}")

    streams = data.get("data", {}).get("result", [])
    out: list[dict] = []
    for stream in streams:
        labels = stream.get("stream", {})
        for ts, line in stream.get("values", []):
            out.append({"ts_ns": ts, "line": line, "labels": labels})
    out.sort(key=lambda x: x["ts_ns"], reverse=True)
    return out


def error_logs_for(service: str, lookback_sec: int = _LOOKBACK_SEC) -> list[dict]:
    """RCA 常用：拉取某服务近窗内的 error 级日志原文。

    应用侧已用 structlog 的 JSONRenderer 输出（04 篇 §5），经 OTel 管道：
    - 字段 ``service`` 被提升为 resource ``service.name`` → Loki **label** ``service_name``
      （清单见 infra/k8s/middleware/otel-agent.yaml 的 transform processor）；
    - 字段 ``level`` 是 JSON attribute，非 label，需用 ``| json`` 管道在 LogQL 里提取过滤。

    故 LogQL 为 ``{service_name="<svc>"} | json level="error"``。
    日志原文可能含 PII —— 调用方负有出站脱敏责任（alerts 模块入站已脱敏，
    此处原文仅用于 agent 内部推断，不落库、不外发原始行）。
    """
    logql = f'{{service_name="{service}"}} | json level="error"'
    return query_logs(logql, lookback_sec=lookback_sec)
