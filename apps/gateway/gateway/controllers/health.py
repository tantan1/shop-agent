"""健康探活端点（旁路，不进入计量与检测链路，01 §2 拓扑不变量）。"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    # 旁路：不进入计量与检测链路，monitoring-agent 探此口、不过 /v1、不污染 LLM 计量
    return {"status": "ok"}
