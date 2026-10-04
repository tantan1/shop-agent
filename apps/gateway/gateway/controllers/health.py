"""健康探活端点（旁路，不进入计量与检测链路，01 §2 拓扑不变量）。"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    # 旁路：不进入计量与检测链路，monitoring-agent 探此口、不过 /v1、不污染 LLM 计量
    return {"status": "ok"}


@router.get("/v1/health")
async def health_v1():
    # shop-agent 探活拼接为 gateway:8001/v1/health；proxy 的 catch-all /v1/{path}
    # 会把它当 LLM 请求转发而返 503，故在此提供精确健康端点返回 200。
    return {"status": "ok"}


@router.get("/v1/health/liveliness")
async def health_v1_liveliness():
    return {"status": "ok"}
