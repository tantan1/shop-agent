"""远程 API 意图处理模块。"""
from __future__ import annotations

import asyncio
import copy
import time as _time

from src.modules.chat.agent.dispute_coordinator import should_use_dispute_coordinator
from src.modules.chat.agent.orchestrator_history import persist_turn
from src.modules.chat.agent.orchestrator_params import (
    _prepare_intent_params,
)
from src.modules.chat.schemas import ChatResponse
from src.shared.logger import APILogger

logger = APILogger("orchestrator_remote")


async def handle_remote_intent(orchestrator, ctx) -> ChatResponse:
    """remote_api 链路入口：执行业务流程，并在返回前统一持久化本轮对话历史。

    真实逻辑在 ``_handle_remote_intent_inner``。这里用薄包装统一收口历史写入，
    避免在内部四条 return 分支（缺参反问 / 锁冲突 / 纠纷协调 / ReAct / 直接Tool）
    逐个重复埋点、漏埋。
    """
    response = await handle_remote_intent_inner(orchestrator, ctx)
    conversation_id = ctx.request.conversation_id or f"conv_{int(_time.time())}"
    persist_turn(
        orchestrator._redis_cache_service,
        conversation_id,
        getattr(ctx.request, "user_id", "eval-bot"),
        ctx.request.message,
        response.message if response else "",
    )
    return response


async def handle_remote_intent_inner(orchestrator, ctx) -> ChatResponse:
    """处理 call_remote_api 意图：参数抽取 → 门控 → Tool/ReAct"""
    conversation_id = ctx.request.conversation_id or f"conv_{int(_time.time())}"
    t_handler_start = _time.perf_counter()

    intent_result, intent_steps, blocked, t_params = await _prepare_intent_params(
        orchestrator, ctx.request, ctx.intent_result, ctx.langfuse_handler, conversation_id
    )

    # 设计 5（选项 C）多意图盲区捕获：在「缺参 blocked 早退」之前捕获，否则 blocked 的多意图请求
    # （如本例「退货退款 + 领优惠」因 request-return 缺订单号被 blocked）永远走不到 execute_direct_tool_flow
    # 里的捕获点，导致漏标。react 路径未 blocked 时由 react_agent._mlops_capture_tool_select 单独捕获；
    # 这里覆盖 direct_tool 路径（含未 blocked），以及「react 模式但被缺参 blocked 早退」的分支
    # （该分支 execute_react_flow 不会被进入，故 react 的捕获也落空）。best-effort，异常一律吞掉。
    if intent_result.mode != "react" or blocked:
        try:
            from src.modules.monitoring.langfuse_mlops import capture_multi_intent_select

            all_names = []
            svc = getattr(orchestrator, "_tool_service", None)
            if svc is not None and hasattr(svc, "tools"):
                all_names = [getattr(t, "name", "") for t in svc.tools]
            capture_multi_intent_select(
                ctx.request.message, all_names, [intent_result.action] if intent_result.action else [], source="p0_rule"
            )
        except Exception:  # noqa: BLE001
            pass

    if blocked:
        return blocked

    # 浅拷贝：替换 intent_result / intent_steps 而不污染调用方持有的 ctx。
    # 必须浅拷贝——嵌套的 langfuse_handler 持有 OTel span 引用，深拷贝会切链到另一条 trace。
    flow_ctx = copy.copy(ctx)
    flow_ctx.intent_result = intent_result
    flow_ctx.intent_steps = intent_steps

    dispute_response = await try_dispute_flow(orchestrator, flow_ctx, t_handler_start, t_params)
    if dispute_response:
        return dispute_response

    # 路由读取 plan.mode（对齐 yaml_flow 的 node.type），不再直接判 complexity
    if intent_result.mode == "react":
        return await execute_react_flow(orchestrator, flow_ctx, t_handler_start, t_params)

    return await execute_direct_tool_flow(orchestrator, flow_ctx, t_handler_start, t_params)


async def try_dispute_flow(orchestrator, ctx, t_handler_start: float, t_params: float) -> ChatResponse | None:
    """尝试走纠纷协调流程（含分布式锁）。"""
    if not should_use_dispute_coordinator(
        message=ctx.request.message,
        emotion_result=ctx.emotion_result,
        intent_action=ctx.intent_result.action,
    ):
        return None

    order_id = ctx.intent_result.params.get("order_id") if ctx.intent_result.params else None
    conversation_id = ctx.request.conversation_id or ""
    lock_name = f"dispute:{conversation_id}:{order_id or 'unknown'}"
    lock_ttl = 300
    lock_token: str | None = None
    lock_renewer: "asyncio.Task[None] | None" = None

    if orchestrator._redis_cache_service and orchestrator._redis_cache_service.is_available:
        lock_token = await orchestrator._redis_cache_service.acquire_lock(lock_name, ttl_seconds=lock_ttl)
        if lock_token is not None:
            lock_renewer = asyncio.create_task(
                orchestrator._renew_lock_loop(lock_name, lock_token, lock_ttl)
            )
        if lock_token is None:
            logger.warning(
                "纠纷协调流程已在执行中，跳过重复触发",
                conversation_id=conversation_id,
                order_id=order_id,
                lock_name=lock_name,
            )
            return ChatResponse(
                message="您的问题正在处理中，请稍候片刻，我们正在为您核实相关信息…",
                conversation_id=conversation_id,
                steps=ctx.intent_steps
                + [
                    {
                        "step_name": "纠纷协调-分布式锁",
                        "step_order": len(ctx.intent_steps),
                        "status": "skipped",
                        "output_data": {
                            "reason": "lock_busy",
                            "lock_name": lock_name,
                            "detail": "同一纠纷协调任务正在执行中",
                        },
                    }
                ],
                documents_used=[],
                safety_passed=True,
                stream_available=True,
                domain=ctx.domain,
                status="processing",
            )

    try:
        logger.info(
            "触发纠纷协调流程",
            action=ctx.intent_result.action,
            emotion=ctx.emotion_result.level.name if ctx.emotion_result else "unknown",
            params=ctx.intent_result.params,
            lock_name=lock_name,
            lock_acquired=lock_token is not None,
        )
        t0 = _time.perf_counter()
        dispute_coordinator = orchestrator._dispute_coordinator
        response = await dispute_coordinator.resolve(
            request=ctx.request,
            emotion_result=ctx.emotion_result,
            conversation_id=conversation_id,
            domain=ctx.domain,
            intent_steps=ctx.intent_steps,
            order_id=order_id,
            langfuse_handler=ctx.langfuse_handler,
        )
        t_dispute = (_time.perf_counter() - t0) * 1000
        t_total = (_time.perf_counter() - t_handler_start) * 1000
        logger.debug(
            "Agent编排耗时统计 [remote_api→dispute]",
            duration_total_ms=round(t_total, 1),
            duration_params_ms=round(t_params, 1),
            duration_dispute_ms=round(t_dispute, 1),
            action=ctx.intent_result.action,
        )
        return response
    finally:
        if lock_renewer is not None:
            lock_renewer.cancel()
            try:
                await lock_renewer
            except (asyncio.CancelledError, Exception):
                pass
        if lock_token and orchestrator._redis_cache_service:
            await orchestrator._redis_cache_service.release_lock(lock_name, lock_token)


async def execute_react_flow(orchestrator, ctx, t_handler_start: float, t_params: float) -> ChatResponse:
    """ReAct 复杂意图处理。"""
    ir = ctx.intent_result
    complexity_label = "multi_step" if ir.mode == "react" else "simple"
    logger.info(
        "意图命中但需要ReAct",
        action=ir.action,
        mode=ir.mode,
        complexity=complexity_label,
        reason=ir.plan.reason if ir.plan else None,
        params=ir.params,
    )
    t0 = _time.perf_counter()
    response = await orchestrator._chat_with_react_agent(ctx)
    t_react = (_time.perf_counter() - t0) * 1000
    t_total = (_time.perf_counter() - t_handler_start) * 1000
    logger.debug(
        "Agent编排耗时统计 [remote_api→ReAct]",
        duration_total_ms=round(t_total, 1),
        duration_params_ms=round(t_params, 1),
        duration_react_ms=round(t_react, 1),
        action=ctx.intent_result.action,
    )
    return response


async def execute_direct_tool_flow(orchestrator, ctx, t_handler_start: float, t_params: float) -> ChatResponse:
    """直接 Tool 调用（简单意图）。"""
    from src.modules.chat.core.content_filter import ContentFilterService
    from src.modules.monitoring.metrics import (
        tool_select_exit_total,
        tool_select_stage_total,
    )

    # 成本漏斗对齐：direct_tool 即「P0 规则直接命中」——意图路由确定性映射到单工具，
    # 等价于漏斗在 P0 提前结束。记录 exit_total / stage_total，使成本漏斗面板对全量流量有数据
    # （此前漏斗只覆盖 react 路径，direct_tool 流量不进漏斗导致面板长期 no data）。
    if ctx.intent_result.action:
        tool_select_exit_total.labels(stage="p0_rule", stop_condition="rule_hit").inc()
        tool_select_stage_total.labels(stage="p0_rule", outcome="hit").inc()

    t0 = _time.perf_counter()
    # 早停直调路径此前不经过 LangChain 工具回调，trace 里看不到调用了哪个工具。
    # 这里补一个 OTel span（挂在根 trace 下），span 名即工具名，并按 Langfuse OTEL
    # 约定写 input.value / output.value；导出前仍经 PII 脱敏处理器过滤。
    # 多意图盲区捕获已前移至 handle_remote_intent_inner（缺参 blocked 早退之前），此处不再重复。
    import json as _json

    from opentelemetry import trace as _otel_trace

    action = ctx.intent_result.action
    params = ctx.intent_result.params or {}
    _tracer = _otel_trace.get_tracer("orchestrator_remote")
    with _tracer.start_as_current_span(f"tool:{action}") as _tool_span:
        try:
            _tool_span.set_attribute(
                "input.value", _json.dumps(params, ensure_ascii=False, default=str)
            )
            _tool_span.set_attribute("input.mime_type", "application/json")
        except Exception:  # noqa: BLE001
            pass
        try:
            tool_response = await orchestrator._tool_service.dispatch(action, params)
        except Exception as tool_err:  # noqa: BLE001
            # 补齐失败回灌缺口：早停直调路径此前 dispatch 抛异常不会进复核队列，
            # 人工标注台因此看不到该失败样本。best-effort 记录后仍按原行为向上抛出。
            try:
                from src.modules.chat.agent.react_agent import _mlops_capture_exec_failed

                await _mlops_capture_exec_failed(
                    user_query=ctx.request.message,
                    tool_name=action,
                    selection_source="p0_rule",
                    error=str(tool_err),
                    conversation_id=ctx.request.conversation_id or "",
                )
            except Exception:  # noqa: BLE001
                pass
            raise
        _tool_span.set_attribute("output.value", tool_response or "")
        _tool_span.set_attribute("output.mime_type", "text/plain")
    t_tool = (_time.perf_counter() - t0) * 1000

    output_filter_safe = True
    cf = ContentFilterService.get_instance()
    output_check = cf.filter_output(tool_response, ctx.domain)
    if not output_check.is_safe:
        output_filter_safe = False
        logger.warning(
            "Direct Tool 输出安全检查未通过",
            domain=ctx.domain,
            risk_categories=output_check.risk_categories,
        )
        tool_response = output_check.filtered_text or "抱歉，当前无法处理您的请求，请稍后重试。"

    step_index = len(ctx.intent_steps)
    response = ChatResponse(
        message=tool_response,
        conversation_id=ctx.request.conversation_id or "",
        steps=ctx.intent_steps
        + [
            {
                "step_name": "Tool调用(直接)",
                "step_order": step_index,
                "status": "success",
                "output_data": {"action": ctx.intent_result.action, "params": ctx.intent_result.params},
            }
        ],
        documents_used=[],
        safety_passed=output_filter_safe,
        stream_available=True,
        domain=ctx.domain,
    )

    t_total = (_time.perf_counter() - t_handler_start) * 1000
    logger.debug(
        "Agent编排耗时统计 [remote_api→direct_tool]",
        duration_total_ms=round(t_total, 1),
        duration_params_ms=round(t_params, 1),
        duration_tool_ms=round(t_tool, 1),
        action=ctx.intent_result.action,
    )
    logger.log_business_event(
        "电商Agent Tool直接调用",
        success=True,
        domain=ctx.domain,
        action=ctx.intent_result.action,
        params=list(ctx.intent_result.params.keys()) if ctx.intent_result.params else [],
        conversation_id=ctx.request.conversation_id or "",
        message_length=len(ctx.request.message),
        response_length=len(tool_response),
    )
    return response
