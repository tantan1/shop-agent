"""SkyWalking OAP 拓扑查询封装（方案 C：真实调用依赖边来源）。

与 prom_client/loki_client 同为「被叫醒才工作」的只读数据源：
- monitoring-agent 平时不轮询，仅在构建依赖拓扑时查询 OAP。
- 直接走 httpx 调 OAP GraphQL 端点（/graphql），不引入新依赖。
- OAP 不可达时抛 :class:`SkyWalkingUnavailable`，调用方降级到
  静态依赖清单（STATIC_DEPENDENCIES）+ 拓扑健康矩阵，不阻断巡检。

接口对应（v10 OAP query-protocol）：
- getGlobalTopology(duration, layer) → Topology{nodes, calls}
  nodes: [{id, name, type}]   —— 服务节点（含接入 agent 的进程）
  calls: [{source, target}]   —— 真实调用边（source -> target 即依赖方向）
"""

from __future__ import annotations

import logging
import os
import time

import httpx

logger = logging.getLogger("monitoring_agent.skywalking")

SKYWALKING_URL = os.getenv("SKYWALKING_URL", "http://skywalking-oap:12800").rstrip("/")
_QUERY_TIMEOUT = float(os.getenv("SKYWALKING_QUERY_TIMEOUT_SEC", "5"))
_LOOKBACK_MIN = int(os.getenv("SKYWALKING_LOOKBACK_MIN", "30"))  # 默认回看 30 分钟


class SkyWalkingUnavailable(RuntimeError):
    """OAP 不可达或拓扑查询失败（非致命：RCA 降级为静态依赖 + 拓扑矩阵归因）。"""


def _graphql_payload(duration_min: int) -> dict:
    """构造 getGlobalTopology 的 GraphQL 请求体。

    duration 格式：MINUTE step 下为 ``yyyy-MM-dd HHmm``（RFC3339 见 query-protocol）。
    """
    end = int(time.time())
    end_t = time.localtime(end)
    start_t = time.localtime(end - duration_min * 60)
    fmt = "%Y-%m-%d %H%M"
    query = (
        "query { getGlobalTopology(duration: {start: \"%(start)s\", end: \"%(end)s\", "
        'step: MINUTE}) { nodes { id name type } calls { source target } } }'
    ) % {"start": time.strftime(fmt, start_t), "end": time.strftime(fmt, end_t)}
    return {"query": query}


def fetch_topology(duration_min: int = _LOOKBACK_MIN) -> dict:
    """查询 OAP 全局拓扑，返回 {nodes: [{id, name, type}], calls: [{source, target}]}。

    失败抛 :class:`SkyWalkingUnavailable`（RCA 据此降级，不影响 agent 存活）。
    """
    try:
        with httpx.Client(timeout=_QUERY_TIMEOUT) as c:
            r = c.post(f"{SKYWALKING_URL}/graphql", json=_graphql_payload(duration_min))
            r.raise_for_status()
            data = r.json()
    except Exception as exc:  # noqa: BLE001 - 只读数据源失败需降级而非崩溃
        logger.warning("SkyWalking 拓扑查询失败 err=%s", exc)
        raise SkyWalkingUnavailable(str(exc)) from exc

    if "errors" in data:
        raise SkyWalkingUnavailable(f"OAP GraphQL 错误: {data['errors']}")
    topo = data.get("data", {}).get("getGlobalTopology") or {}
    return {
        "nodes": topo.get("nodes", []),
        "calls": topo.get("calls", []),
    }


def dependency_edges(duration_min: int = _LOOKBACK_MIN) -> list[dict[str, str]]:
    """把 OAP 拓扑转成依赖边列表：[{source: 服务名, target: 服务名}, ...]。

    服务名取 node.name（更可读），调用边按 source -> target 方向保留，
    即 target 依赖 source（source 被 target 调用）。
    """
    topo = fetch_topology(duration_min)
    name_by_id = {n.get("id"): n.get("name") for n in topo.get("nodes", []) if n.get("id")}
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for c in topo.get("calls", []):
        src = name_by_id.get(c.get("source"))
        tgt = name_by_id.get(c.get("target"))
        if not src or not tgt or src == tgt:
            continue
        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)
        edges.append({"source": src, "target": tgt})
    return edges
