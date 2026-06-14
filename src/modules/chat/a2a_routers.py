"""
A2A (Agent-to-Agent) 路由 —— 所有 A2A 协议端点。

端点概览：
  P0: 异步任务
    POST   /a2a/tasks/send              — 提交异步任务
    GET    /a2a/tasks/{task_id}         — 查询任务状态/结果
    POST   /a2a/tasks/{task_id}/cancel  — 取消任务
    GET    /a2a/tasks                   — 列出任务

  P1: Webhook 订阅
    POST   /a2a/webhooks                — 注册回调
    DELETE /a2a/webhooks/{subscription_id} — 取消订阅

  P2: Conversation 上下文共享
    GET    /a2a/conversations           — 列出对话
    GET    /a2a/conversations/{conversation_id}/messages — 历史消息

  P3: A2A 专用健康检查
    GET    /a2a/health                  — 依赖状态报告
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import JSONResponse

from src.modules.auth.dependencies import verify_api_key
from src.shared.responses import success_response, error_response

from src.modules.chat.schemas import (
    A2ATaskRequest,
    A2ATaskStatusResponse,
    A2ATaskListResponse,
    WebhookSubscriptionRequest,
    WebhookSubscriptionResponse,
    A2AConversationSummary,
    A2AConversationListResponse,
    A2AHealthResponse,
)
from src.modules.chat.core.a2a_task_service import get_a2a_task_service
from src.modules.chat.core.a2a_webhook_service import get_a2a_webhook_service

router = APIRouter(prefix="/a2a", tags=["A2A Agent-to-Agent"])

# ── 服务启动时间（用于 uptime 计算） ──
_START_TIME = time.time()


# =============================================================================
# P0: A2A 异步任务 API
# =============================================================================

@router.post("/tasks/send", summary="提交异步 Agent 任务（A2A）")
async def a2a_send_task(
    request: A2ATaskRequest,
    req: Request,
    _: None = Depends(verify_api_key),
):
    """提交异步任务，立即返回 task_id。外部系统通过 GET /a2a/tasks/{task_id} 轮询结果。

    请求示例:
    ```json
    {
        "message": "帮我查一下订单 ORD-2024-001 的物流状态",
        "domain": "ecommerce",
        "conversation_id": "ext_conv_001",
        "callback_url": "https://your-system.com/webhooks/shop-agent"
    }
    ```

    响应示例:
    ```json
    {
        "success": true,
        "code": 200,
        "data": {
            "task_id": "task_a1b2c3d4e5f6a7b8",
            "status": "pending",
            "created_at": "2026-06-25T03:00:00.000Z",
            "conversation_id": "ext_conv_001",
            "domain": "ecommerce"
        }
    }
    ```
    """
    service = get_a2a_task_service()

    task = service.create_task(
        message=request.message,
        domain=request.domain,
        conversation_id=request.conversation_id,
        skill_id=request.skill_id,
        context=request.context,
        callback_url=request.callback_url,
    )

    # 后台异步执行（不阻塞 HTTP 响应）
    asyncio.create_task(
        service.run_task(
            task=task,
            message=request.message,
            domain=request.domain,
            conversation_id=request.conversation_id,
            skill_id=request.skill_id,
            context=request.context,
            callback_url=request.callback_url,
        )
    )

    return success_response(data=task.model_dump())


@router.get("/tasks/{task_id}", summary="查询任务状态（A2A）")
async def a2a_get_task(
    task_id: str,
    _: None = Depends(verify_api_key),
):
    """查询异步任务的状态和结果。

    状态说明:
    - pending:   排队等待执行
    - running:   正在执行
    - completed: 执行成功（result 字段含回复文本）
    - failed:    执行失败（error 字段含错误信息）
    - cancelled: 已取消

    请求示例:
    ```
    GET /a2a/tasks/task_a1b2c3d4e5f6a7b8
    ```

    响应示例 (completed):
    ```json
    {
        "success": true,
        "code": 200,
        "data": {
            "task_id": "task_a1b2c3d4e5f6a7b8",
            "status": "completed",
            "result": "您的订单 ORD-2024-001 当前物流状态为...",
            "created_at": "2026-06-25T03:00:00.000Z",
            "started_at": "2026-06-25T03:00:01.000Z",
            "completed_at": "2026-06-25T03:00:05.234Z",
            "conversation_id": "ext_conv_001",
            "domain": "ecommerce"
        }
    }
    ```
    """
    service = get_a2a_task_service()
    task = service.get_task(task_id)
    if task is None:
        return JSONResponse(
            status_code=404,
            content=error_response(message=f"任务 {task_id} 不存在", code=404),
        )
    return success_response(data=task.model_dump())


@router.post("/tasks/{task_id}/cancel", summary="取消任务（A2A）")
async def a2a_cancel_task(
    task_id: str,
    _: None = Depends(verify_api_key),
):
    """取消一个 pending 或 running 状态的任务。

    已终态的任务（completed/failed/cancelled）无法取消。

    请求示例:
    ```
    POST /a2a/tasks/task_a1b2c3d4e5f6a7b8/cancel
    ```
    """
    service = get_a2a_task_service()
    ok, msg = service.cancel_task(task_id)

    if not ok:
        status_code = 404 if "不存在" in msg else 409
        return JSONResponse(
            status_code=status_code,
            content=error_response(message=msg, code=status_code),
        )

    task = service.get_task(task_id)
    return success_response(data=task.model_dump(), message=msg)


@router.get("/tasks", summary="列出所有任务（A2A）")
async def a2a_list_tasks(
    limit: int = Query(default=50, ge=1, le=200, description="每页数量"),
    offset: int = Query(default=0, ge=0, description="偏移量"),
    _: None = Depends(verify_api_key),
):
    """列出所有任务（最新优先，支持分页）。

    请求示例:
    ```
    GET /a2a/tasks?limit=20&offset=0
    ```
    """
    service = get_a2a_task_service()
    result = service.list_tasks(limit=limit, offset=offset)
    return success_response(data=result.model_dump())


# =============================================================================
# P1: Webhook 订阅
# =============================================================================

@router.post("/webhooks", summary="注册 Webhook 回调（A2A）")
async def a2a_subscribe_webhook(
    request: WebhookSubscriptionRequest,
    _: None = Depends(verify_api_key),
):
    """注册 Webhook 回调订阅，任务完成后自动推送通知。

    请求示例:
    ```json
    {
        "url": "https://your-system.com/webhooks/shop-agent",
        "events": ["task.completed", "task.failed"],
        "secret": "my-hmac-secret-key",
        "ttl_seconds": 86400
    }
    ```

    响应示例:
    ```json
    {
        "success": true,
        "code": 200,
        "data": {
            "subscription_id": "wh_a1b2c3d4e5f6",
            "url": "https://your-system.com/webhooks/shop-agent",
            "events": ["task.completed", "task.failed"],
            "created_at": "2026-06-25T03:00:00.000Z",
            "expires_at": "2026-06-26T03:00:00.000Z"
        }
    }
    ```
    """
    service = get_a2a_webhook_service()
    sub = service.subscribe(
        url=request.url,
        events=request.events,
        secret=request.secret,
        ttl_seconds=request.ttl_seconds or 86400,
    )
    return success_response(data=sub.model_dump())


@router.delete("/webhooks/{subscription_id}", summary="取消 Webhook 订阅（A2A）")
async def a2a_unsubscribe_webhook(
    subscription_id: str,
    _: None = Depends(verify_api_key),
):
    """取消 Webhook 订阅。

    请求示例:
    ```
    DELETE /a2a/webhooks/wh_a1b2c3d4e5f6
    ```
    """
    service = get_a2a_webhook_service()
    ok = service.unsubscribe(subscription_id)
    if not ok:
        return JSONResponse(
            status_code=404,
            content=error_response(message=f"订阅 {subscription_id} 不存在", code=404),
        )
    return success_response(data={"subscription_id": subscription_id, "deleted": True})


# =============================================================================
# P2: Conversation 上下文共享
# =============================================================================

# 简易内存存储（可升级为 DB）
_conversations_store: Dict[str, Dict[str, Any]] = {}


@router.get("/conversations", summary="列出对话（A2A）")
async def a2a_list_conversations(
    limit: int = Query(default=50, ge=1, le=200, description="每页数量"),
    offset: int = Query(default=0, ge=0, description="偏移量"),
    domain: str = Query(default="ecommerce", description="领域筛选"),
    _: None = Depends(verify_api_key),
):
    """列出活跃对话（供多 Agent 协作时的上下文发现）。

    请求示例:
    ```
    GET /a2a/conversations?domain=ecommerce&limit=20
    ```
    """
    all_convs = sorted(
        _conversations_store.values(),
        key=lambda c: c.get("last_active_at", ""),
        reverse=True,
    )
    if domain:
        all_convs = [c for c in all_convs if c.get("domain") == domain]

    total = len(all_convs)
    page = all_convs[offset : offset + limit]

    summaries = [
        A2AConversationSummary(
            conversation_id=c["conversation_id"],
            message_count=c.get("message_count", 0),
            created_at=c.get("created_at", ""),
            last_active_at=c.get("last_active_at", ""),
            domain=c.get("domain", "ecommerce"),
            status=c.get("status", "active"),
        )
        for c in page
    ]

    return success_response(
        data=A2AConversationListResponse(total=total, conversations=summaries).model_dump()
    )


@router.get("/conversations/{conversation_id}/messages", summary="获取对话历史（A2A）")
async def a2a_get_conversation_messages(
    conversation_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    _: None = Depends(verify_api_key),
):
    """获取指定对话的历史消息（供其他 Agent 读取上下文）。

    请求示例:
    ```
    GET /a2a/conversations/ext_conv_001/messages?limit=20
    ```
    """
    conv = _conversations_store.get(conversation_id)
    if not conv:
        return JSONResponse(
            status_code=404,
            content=error_response(message=f"对话 {conversation_id} 不存在", code=404),
        )

    messages = conv.get("messages", [])
    return success_response(data={
        "conversation_id": conversation_id,
        "total": len(messages),
        "messages": messages[-limit:],  # 最近 N 条
    })


def register_conversation_event(
    conversation_id: str,
    user_message: str,
    assistant_message: str,
    domain: str = "ecommerce",
) -> None:
    """注册对话事件（供 ChatAgentService 回调）。"""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    conv = _conversations_store.get(conversation_id)

    if not conv:
        conv = {
            "conversation_id": conversation_id,
            "domain": domain,
            "created_at": now,
            "message_count": 0,
            "messages": [],
            "status": "active",
        }
        _conversations_store[conversation_id] = conv

    conv["messages"].append({"role": "user", "content": user_message, "timestamp": now})
    conv["messages"].append({"role": "assistant", "content": assistant_message, "timestamp": now})
    conv["message_count"] = len(conv["messages"])
    conv["last_active_at"] = now


# =============================================================================
# P3: A2A 专用健康检查
# =============================================================================

@router.get("/health", summary="A2A 专用健康检查", include_in_schema=True)
async def a2a_health():
    """返回 Agent 依赖状态，供上游系统做就绪探测。

    响应示例:
    ```json
    {
        "status": "healthy",
        "agent_name": "Shop-Agent Orchestrator",
        "version": "1.0.0",
        "uptime_seconds": 12345.6,
        "dependencies": {
            "llm": "healthy",
            "embedding": "healthy",
            "vector_db": "healthy",
            "redis": "unavailable",
            "mcp_server": "disabled"
        },
        "skills_count": 5,
        "mcp_enabled": false
    }
    ```
    """
    dependencies: Dict[str, str] = {}

    # LLM
    try:
        from src.core.config import config as core_config
        if getattr(core_config, "TONGYI_API_KEY", None):
            dependencies["llm"] = "healthy"
        else:
            dependencies["llm"] = "unconfigured"
    except Exception:
        dependencies["llm"] = "unknown"

    # Vector DB (Milvus)
    try:
        from src.modules.chat.core.milvus_service import MilvusService
        milvus = MilvusService.get_instance()
        if milvus.is_connected():
            dependencies["vector_db"] = "healthy"
        else:
            dependencies["vector_db"] = "disconnected"
    except Exception:
        dependencies["vector_db"] = "unknown"

    # Redis
    try:
        from src.core.rate_limiter import get_rate_limiter
        rl = get_rate_limiter()
        if hasattr(rl, "_redis") and rl._redis is not None:
            dependencies["redis"] = "healthy"
        else:
            dependencies["redis"] = "unavailable"
    except Exception:
        dependencies["redis"] = "unknown"

    # MCP Server
    from src.core.config import config as core_config
    mcp_enabled = getattr(core_config, "MCP_ENABLED", False)
    dependencies["mcp_server"] = "enabled" if mcp_enabled else "disabled"

    # Skills
    try:
        from src.modules.chat.agent.skill_loader import get_skill_registry
        registry = get_skill_registry()
        skills_count = len(registry.skills)
    except Exception:
        skills_count = 0

    # 整体健康判定
    unhealthy_deps = [k for k, v in dependencies.items() if v in ("disconnected", "unavailable") and k != "redis"]
    if unhealthy_deps:
        status = "degraded" if dependencies.get("vector_db") != "disconnected" else "unhealthy"
    else:
        status = "healthy"

    return success_response(data=A2AHealthResponse(
        status=status,
        agent_name="Shop-Agent Orchestrator",
        version="1.0.0",
        uptime_seconds=round(time.time() - _START_TIME, 1),
        dependencies=dependencies,
        skills_count=skills_count,
        mcp_enabled=mcp_enabled,
    ).model_dump())
