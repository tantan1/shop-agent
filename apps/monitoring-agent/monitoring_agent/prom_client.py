"""Prometheus 即时查询封装（无状态分析层的只读数据源）。

monitoring-agent 平时是「被叫醒才工作」的无状态分析层（见 04 篇 §5/§9）：
不轮询、不持久化原始指标流。这里仅在 RCA 被触发时，临时拉取
Prometheus 即时查询快照做根因关联，查完即弃。

直接走 httpx 调 Prometheus HTTP API（/api/v1/query），**不引入新依赖**
（prometheus-client 是客户端库，仅用于暴露指标，不能发查询），
也不新增 SPOF —— Prometheus server 挂了不影响 agent 自身存活。
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

logger = logging.getLogger("monitoring_agent.prom")

PROM_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090").rstrip("/")
_QUERY_TIMEOUT = float(os.getenv("PROM_QUERY_TIMEOUT_SEC", "5"))


class PrometheusUnavailable(RuntimeError):
    """Prometheus 不可达或查询失败（非致命：RCA 降级为纯拓扑归因）。"""


def query(expr: str) -> list[dict]:
    """执行 PromQL 即时查询，返回 result 列表（空查询/失败返回空列表）。

    失败不抛异常：RCA 是分析层，单数据源缺失不应让整条归因链路崩溃，
    由调用方决定降级策略。
    """
    try:
        with httpx.Client(timeout=_QUERY_TIMEOUT) as c:
            r = c.get(
                f"{PROM_URL}/api/v1/query",
                params={"query": expr},
            )
            r.raise_for_status()
            data = r.json()
    except Exception as exc:  # noqa: BLE001 - 只读数据源失败需降级而非崩溃
        logger.warning("Prometheus 查询失败 expr=%s err=%s", expr, exc)
        raise PrometheusUnavailable(str(exc)) from exc

    if data.get("status") != "success":
        raise PrometheusUnavailable(f"非 success 响应: {data.get('status')}")
    return data.get("data", {}).get("result", [])


def query_value(expr: str) -> float | None:
    """取即时查询的第一个标量值；无样本/非法值返回 None。

    Prometheus 在无样本时会返回 ``NaN`` 字符串（如 ``"NaN"``），``float("NaN")``
    解析成功但 ``json.dumps`` 无法序列化（ValueError: Out of range float values），
    会导致 RCA 接口 500。这里把 NaN/Inf 也归一为 None，确保可被 JSON 编码。
    """
    results = query(expr)
    if not results:
        return None
    try:
        value = float(results[0]["value"][1])
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    if value != value or value in (float("inf"), float("-inf")):  # NaN 或 Inf
        return None
    return value


async def query_value_async(expr: str) -> float | None:
    """异步版 query_value，供 RCA 并行快照使用。"""
    try:
        async with httpx.AsyncClient(timeout=_QUERY_TIMEOUT) as c:
            r = await c.get(
                f"{PROM_URL}/api/v1/query",
                params={"query": expr},
            )
            r.raise_for_status()
            data = r.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Prometheus 异步查询失败 expr=%s err=%s", expr, exc)
        raise PrometheusUnavailable(str(exc)) from exc

    if data.get("status") != "success":
        raise PrometheusUnavailable(f"非 success 响应: {data.get('status')}")
    results = data.get("data", {}).get("result", [])
    if not results:
        return None
    try:
        value = float(results[0]["value"][1])
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def scrape_targets() -> list[dict]:
    """查询 /api/v1/targets，返回 activeTargets 列表（服务发现来源，方案 A）。

    Prometheus 的 target 即「正在被抓取的服务实例」——这是比手工清单更接近
    真实运行状态的动态服务库存。失败抛 PrometheusUnavailable，由调用方回退
    到静态清单（k8s 外/未接 Prometheus 的服务仍需静态注册）。
    """
    try:
        with httpx.Client(timeout=_QUERY_TIMEOUT) as c:
            r = c.get(f"{PROM_URL}/api/v1/targets")
            r.raise_for_status()
            data = r.json()
    except Exception as exc:  # noqa: BLE001 - 只读数据源失败需降级而非崩溃
        logger.warning("Prometheus targets 查询失败 err=%s", exc)
        raise PrometheusUnavailable(str(exc)) from exc

    if data.get("status") != "success":
        raise PrometheusUnavailable(f"非 success 响应: {data.get('status')}")
    return data.get("data", {}).get("activeTargets", [])


# ── gateway golden signal 快照（数据接口见 09b §一）──────────────────────
# 这些指标由 09 批次在网关 /metrics 暴露，Prometheus 经 gateway job 抓取。

GATEWAY_UP = 'up{job="gateway"}'
P99_LATENCY = (
    "histogram_quantile(0.99, "
    'sum(rate(gateway_request_latency_seconds_bucket{job="gateway"}[5m])) by (le))'
)
REQ_RATE = 'sum(rate(gateway_requests_total{job="gateway"}[5m]))'
TOKEN_RATE = 'sum(rate(gateway_tokens_total{job="gateway"}[5m]))'
LOOP_RATE = 'sum(rate(gateway_loop_guarded_total{job="gateway"}[5m]))'
SAFETY_RATE = (
    'sum(rate(gateway_injection_denied_total{job="gateway"}[5m])) + '
    'sum(rate(gateway_governance_error_total{job="gateway"}[5m]))'
)


def gateway_snapshot() -> dict[str, float | None]:
    """拉取网关 golden signal 快照，供 RCA 做指标侧关联。

    任意单指标查询失败不影响其余：逐项捕获，缺失项记 None。
    Prometheus 整体不可达时返回 ``{"_unavailable": True}``，调用方可据此
    区分「快照为空」与「数据源缺失」，避免误判为 gateway 在线。
    """
    snapshot: dict[str, float | None] = {}
    specs = {
        "gateway_up": GATEWAY_UP,
        "p99_latency_s": P99_LATENCY,
        "req_rate": REQ_RATE,
        "token_rate": TOKEN_RATE,
        "loop_rate": LOOP_RATE,
        "safety_rate": SAFETY_RATE,
    }
    first_err: Exception | None = None
    for key, expr in specs.items():
        try:
            snapshot[key] = query_value(expr)
        except PrometheusUnavailable as exc:
            snapshot[key] = None
            first_err = first_err or exc
    if first_err is not None:
        snapshot["_unavailable"] = True
    return snapshot


async def gateway_snapshot_async() -> dict[str, float | None]:
    """异步并行版 gateway_snapshot，供 RCA 分析使用，减少总延迟。"""
    specs = {
        "gateway_up": GATEWAY_UP,
        "p99_latency_s": P99_LATENCY,
        "req_rate": REQ_RATE,
        "token_rate": TOKEN_RATE,
        "loop_rate": LOOP_RATE,
        "safety_rate": SAFETY_RATE,
    }
    tasks = {key: asyncio.create_task(query_value_async(expr)) for key, expr in specs.items()}
    snapshot: dict[str, float | None] = {}
    first_err: Exception | None = None
    for key, task in tasks.items():
        try:
            snapshot[key] = await task
        except PrometheusUnavailable as exc:
            snapshot[key] = None
            first_err = first_err or exc
    if first_err is not None:
        snapshot["_unavailable"] = True
    return snapshot
