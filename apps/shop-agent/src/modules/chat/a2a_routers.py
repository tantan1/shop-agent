"""
A2A (Agent-to-Agent) 路由 —— 所有 A2A 协议端点。

端点概览：
  P0: 异步任务（单入口 —— 不按业务类型拆端点，按 skill_id 硬路由 + 意图识别兜底）
    POST   /a2a/tasks/send              — 提交异步任务
    GET    /a2a/tasks/{task_id}         — 查询任务状态/结果
    POST   /a2a/tasks/{task_id}/cancel  — 取消任务
    POST   /a2a/tasks/{task_id}/input   — 向 input-required 任务补充输入/人工审批
    GET    /a2a/tasks                   — 列出任务

  路由约定：对端先 GET /.well-known/agent-card.json 做能力发现（skills[].id / examples），
  提交时传 skill_id 走确定性硬路由（跳过意图识别）；不传则由服务端 P0/P1/P2 识别兜底。

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
from typing import Any, Dict

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from src.modules.auth.dependencies import verify_api_key
from src.modules.chat.core.a2a_task_service import get_a2a_task_service
from src.modules.chat.core.a2a_webhook_service import get_a2a_webhook_service
from src.modules.chat.schemas import (
    A2AConversationListResponse,
    A2AConversationSummary,
    A2AHealthResponse,
    A2ATaskInputRequest,
    A2ATaskRequest,
    WebhookSubscriptionRequest,
)
from src.shared.logger import APILogger
from src.shared.responses import error_response, success_response

router = APIRouter(prefix="/a2a", tags=["A2A Agent-to-Agent"])

logger = APILogger("a2a_router")

# ── 服务启动时间（用于 uptime 计算） ──
_START_TIME = time.time()


def _validate_skill_routing(
    skill_id: str | None, context: dict | None
) -> tuple[int, str] | None:
    """校验 A2A 硬路由请求 —— 失败即 fail-closed，返回 (状态码, 错误说明)。

    两道闸：
      1. skill_id 必须存在于 SkillRegistry（否则对端可能指向一个不存在的/私有的能力）；
      2. context 的字段必须落在该 skill 的参数契约内，未在契约中的字段一律拒绝
         —— 而不是静默丢弃（静默丢弃会让对端以为参数生效了）。

    若该 skill 没有参数契约（params 为空，无法可依），则跳过第 2 道闸并记录告警，
    避免因为缺元数据把所有调用都挡在门外。
    """
    if not skill_id:
        return None

    try:
        from src.modules.chat.agent.skill_loader import get_skill_registry

        registry = get_skill_registry()
    except Exception as e:  # pragma: no cover - 注册表不可用属于基础设施故障
        logger.error("SkillRegistry 不可用，无法校验 skill_id", skill_id=skill_id, error=str(e))
        return 503, "能力注册表暂不可用，无法校验 skill_id，请稍后重试"

    skill = next((s for s in registry.skills if s.name == skill_id), None)
    if skill is None:
        available = sorted({s.name for s in registry.skills})
        return (
            400,
            f"未知 skill_id '{skill_id}'；请从 /.well-known/agent-card.json 的 "
            f"skills[].id 中选择。当前可用：{', '.join(available) or '(无)'}",
        )

    if not context:
        return None

    declared = set(skill.params or {})
    if not declared:
        logger.warning(
            "skill 无参数契约，跳过 context 严格校验",
            skill_id=skill_id,
            context_keys=sorted(context),
        )
        return None

    unknown = sorted(set(context) - declared)
    if unknown:
        return (
            422,
            f"context 含未声明字段 {unknown}；skill '{skill_id}' 接受的参数为："
            f"{', '.join(sorted(declared))}",
        )
    return None


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

    路由方式二选一：
      - 传 `skill_id`：硬路由，跳过意图识别，action 锁定为该 skill（推荐，最稳）。
        skill_id 必须在 Agent Card 的 skills[].id 内，否则 400。
      - 不传 `skill_id`：由服务端 P0/P1/P2 意图识别兜底。

    传了 skill_id 时，`context` 会按该 skill 的参数契约校验（未声明字段 422），
    校验通过后作为确定性参数注入，模型不再从 message 里重新抽取。

    请求示例（硬路由）:
    ```json
    {
        "message": "帮我查一下这个订单的状态",
        "skill_id": "query-order",
        "context": {"order_id": "WB202405270001"},
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
            "domain": "ecommerce",
            "skill_id": "query-order"
        }
    }
    ```
    """
    invalid = _validate_skill_routing(request.skill_id, request.context)
    if invalid:
        status_code, detail = invalid
        logger.warning(
            "A2A 任务路由校验失败",
            skill_id=request.skill_id,
            status_code=status_code,
            detail=detail,
        )
        return JSONResponse(
            status_code=status_code, content=error_response(message=detail, code=status_code)
        )

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
    - pending:        排队等待执行
    - running:        正在执行
    - completed:      执行成功（result 含回复文本，artifacts 含结构化产物）
    - failed:         执行失败（error 字段含错误信息）
    - cancelled:      已取消
    - input-required: 已暂停，等待外部补充输入或人工审批
                      （message 说明缺什么，interrupt_data 含恢复上下文；
                        对端通过 POST /a2a/tasks/{task_id}/input 恢复）

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
            "result": "您的订单 WB202405270001 当前物流状态为...",
            "created_at": "2026-06-25T03:00:00.000Z",
            "started_at": "2026-06-25T03:00:01.000Z",
            "completed_at": "2026-06-25T03:00:05.234Z",
            "conversation_id": "ext_conv_001",
            "domain": "ecommerce",
            "skill_id": "query-order",
            "artifacts": [
                {"name": "result", "mime_type": "text/plain", "parts": [{"kind": "text", "text": "..."}]},
                {"name": "trace", "mime_type": "application/json", "parts": [{"kind": "data", "data": {}}]}
            ]
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


@router.post("/tasks/{task_id}/input", summary="向暂停任务补充输入（A2A）")
async def a2a_provide_task_input(
    task_id: str,
    request: A2ATaskInputRequest,
    _: None = Depends(verify_api_key),
):
    """恢复处于 `input-required` 状态的任务。

    两类暂停都走这里：
      - 高后果动作（如退款）等待人工审批 —— confirm=true 批准 / false 拒绝；
      - 信息不足以继续执行 —— 拒绝后由对端带齐信息重新提交新任务。

    请求示例:
    ```json
    {"confirm": true, "remark": "已核对订单，同意退款"}
    ```

    响应示例:
    ```json
    {
        "success": true,
        "code": 200,
        "data": {"task_id": "task_a1b2c3d4e5f6a7b8", "status": "completed", "result": "退款已受理..."}
    }
    ```
    """
    service = get_a2a_task_service()
    ok, msg, task = await service.resume_task(
        task_id=task_id,
        confirm=request.confirm,
        remark=request.remark,
    )

    if not ok:
        status_code = 404 if "不存在" in msg else 409
        if "恢复执行失败" in msg:
            status_code = 500
        return JSONResponse(
            status_code=status_code, content=error_response(message=msg, code=status_code)
        )

    return success_response(data=task.model_dump() if task else None, message=msg)


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
    return success_response(
        data={
            "conversation_id": conversation_id,
            "total": len(messages),
            "messages": messages[-limit:],  # 最近 N 条
        }
    )


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
    unhealthy_deps = [
        k for k, v in dependencies.items() if v in ("disconnected", "unavailable") and k != "redis"
    ]
    if unhealthy_deps:
        status = "degraded" if dependencies.get("vector_db") != "disconnected" else "unhealthy"
    else:
        status = "healthy"

    return success_response(
        data=A2AHealthResponse(
            status=status,
            agent_name="Shop-Agent Orchestrator",
            version="1.0.0",
            uptime_seconds=round(time.time() - _START_TIME, 1),
            dependencies=dependencies,
            skills_count=skills_count,
            mcp_enabled=mcp_enabled,
        ).model_dump()
    )
