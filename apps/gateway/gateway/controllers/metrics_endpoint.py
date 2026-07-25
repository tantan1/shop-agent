"""计量 exposition 端点（Prometheus 文本格式，批次3 接真实计数）。

注意：本文件只是 HTTP 入口薄壳，真正的计量逻辑在业务模块 gateway/metrics.py
（record / estimate_tokens / render）。命名为 metrics_endpoint 以避免与业务模块 metrics.py 撞名。
"""
from __future__ import annotations

from fastapi import APIRouter, Response

from gateway.metrics import render

router = APIRouter()


@router.get("/metrics")
async def metrics():
    return Response(render(), media_type="text/plain; version=0.0.4")
