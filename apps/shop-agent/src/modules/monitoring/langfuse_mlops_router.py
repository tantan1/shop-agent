"""MLOps 训练回流触发端点。

提供一键触发「从标注平台拉取 correct_tool → 导出 → 校验 → 训练回流」：
    POST {API_V1_PREFIX}/mlops/trigger-training

数据源（当前为 Langfuse 的 trace/score）属于实现细节，不进入资源路径——
历史路径曾为 /mlops/tasks/export（PostgreSQL 期）与
/mlops/langfuse/trigger-training（Langfuse 集成期），数据源更换时 URL 不应随之变化。
"""
from fastapi import APIRouter

from src.modules.monitoring.langfuse_mlops import trigger_training

router = APIRouter(prefix="/mlops", tags=["mlops"])


@router.post("/trigger-training")
async def trigger_training_endpoint():
    return await trigger_training()
