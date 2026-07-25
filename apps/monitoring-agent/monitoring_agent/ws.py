"""WebSocket 实时推送通道（demo 实时告警）。

monitoring-agent 完成 RCA 后，经 broadcast() 把告警 + 处置建议主动推给所有
已连接的 demo 页面（替代前端轮询 /status）。连接池为进程内单例，断线自动剔除。

安全约束（04/06/07 跨平面）：推送前对 root_cause / recommendations 经 alerts.redact
脱敏，PII 不越平面。WebSocket 端点本身不暴露写能力（只读推送），无需 Bearer 鉴权；
demo 经 Ingress 80 暴露，与 /ingest/* 写入口隔离。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import WebSocket

from .alerts import redact

logger = logging.getLogger("monitoring_agent")

# 进程内 WebSocket 连接池（每个元素是已 accept 的 WebSocket）。
_CONNECTIONS: set[WebSocket] = set()
_lock = asyncio.Lock()


async def register(ws: WebSocket) -> None:
    """把新连接加入池（连接建立时调用）。"""
    async with _lock:
        _CONNECTIONS.add(ws)
    logger.info("WebSocket 客户端接入，当前连接数=%d", len(_CONNECTIONS))


async def unregister(ws: WebSocket) -> None:
    """连接断开/异常时从池移除。"""
    async with _lock:
        _CONNECTIONS.discard(ws)
    logger.info("WebSocket 客户端断开，当前连接数=%d", len(_CONNECTIONS))


async def broadcast(message: dict) -> None:
    """向所有已连接 demo 推送一条消息（JSON 序列化）。

    单个客户端发送失败不影响其他客户端（逐条 try）。断线连接在此处统一清理。
    """
    payload = json.dumps(message, ensure_ascii=False, default=str)
    async with _lock:
        targets = list(_CONNECTIONS)
    dead: list[WebSocket] = []
    for ws in targets:
        try:
            await ws.send_text(payload)
        except Exception as exc:  # 客户端已断但未触发 disconnect 事件
            logger.warning("WebSocket 推送失败，标记断开: %s", exc)
            dead.append(ws)
    if dead:
        async with _lock:
            for ws in dead:
                _CONNECTIONS.discard(ws)


def build_alert_message(rca: "object") -> dict:
    """把 RCA 结果封装为实时告警推送消息（已脱敏）。

    rca 为 RcaResult 实例，提取 severity / root_cause / affected / recommendations，
    以及结构化修复动作 ``remediation``（供前端确认按钮调用 /demo/remediate）。
    """
    recs = [redact(r) for r in (rca.recommendations or [])]
    return {
        "type": "alert",
        "ts": int(time.time()),
        "severity": rca.severity,
        "source": getattr(rca, "source", None),
        "root_cause": redact(rca.root_cause or ""),
        "affected": list(rca.affected or []),
        "recommendations": recs,
        "remediation": getattr(rca, "remediation", None),
        "used_llm": bool(getattr(rca, "used_llm", False)),
    }


async def send_topology(ws: WebSocket, topology: dict) -> None:
    """向单连接推送一次拓扑健康矩阵（/status 的实时镜像）。"""
    try:
        await ws.send_text(json.dumps(
            {"type": "topology", "ts": int(time.time()), **topology},
            ensure_ascii=False, default=str,
        ))
    except Exception as exc:
        logger.warning("WebSocket 拓扑推送失败: %s", exc)
