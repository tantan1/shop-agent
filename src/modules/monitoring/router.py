"""
监控相关路由
提供 Prometheus 指标暴露端点
"""
from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from src.core.config import config
from src.modules.auth.dependencies import require_admin

router = APIRouter(prefix="/monitoring", tags=["监控"])

# 指标端点路径常量
METRICS_ENDPOINT = "/metrics"

# 是否启用 Metrics 端点认证 (从环境变量读取，默认 false 方便 Prometheus 抓取)
METRICS_AUTH_ENABLED = getattr(config, 'METRICS_AUTH_ENABLED', False)


@router.get(METRICS_ENDPOINT, response_class=PlainTextResponse, include_in_schema=False)
async def get_metrics():
    """
    Prometheus 指标暴露端点 (无需认证，方便 Prometheus 抓取)
    返回 Prometheus 格式的指标数据
    """
    return PlainTextResponse(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )


@router.get("/stats", dependencies=[Depends(require_admin)])
async def get_application_stats():
    """
    获取应用统计信息 (仅管理员)
    返回业务指标汇总
    """
    from src.modules.monitoring.metrics import (
        api_call_counter,
        agent_conversation_counter,
        exception_counter,
        active_users
    )
    
    # 从 Prometheus 注册表收集指标快照
    snapshot = {}
    for collector in REGISTRY.collect():
        for sample in collector.samples:
            snapshot[sample.name] = sample.value
    
    return {
        "status": "ok",
        "metrics_endpoint": "/api/v1/monitoring/metrics",
        "prometheus_scrape_config": {
            "job_name": "shop-agent",
            "metrics_path": "/api/v1/monitoring/metrics",
            "static_configs": [{"targets": ["shop-agent:8000"]}]
        },
        "snapshot": snapshot
    }
