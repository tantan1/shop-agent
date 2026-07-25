"""LLM 流量代理端点（01 §2 三道检测点位串联）。

职责：仅做 HTTP 入口装配，把请求依次交给 ①入向注入闸（含③工具参数）→ 治理面钩子 → 路由 →
fail 开关 / 限流 / 预算 → 转发（含 02 §4 故障转移重试）→ ②出向 judge_egress；自身不含业务分支判断。

注入三道防线编号（scope-10 §1 为唯一权威）：①入向 prompt ②出向 judge_egress ③工具参数。
fail 开关属 01 §5 降级机制，**不占用三道防线编号**。

响应 metadata 按 02 篇要求带回证据头：
  X-Upstream-Model : 实际命中的后端（fallback 时即回退后端）
  X-Fallback       : true 表示本次经过故障转移
"""
from __future__ import annotations

import hashlib
import json
import time

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from gateway.config import settings
from gateway.hooks.egress import decide
from gateway.hooks.governance import governance_error_total, hooks
from gateway.hooks.injection import gate, injection_denied_total
from gateway.limiter import check as limit_check
from gateway.litellm_router import llm_router
from gateway.loopguard import check as loop_check
from gateway.metrics import (
    estimate_tokens,
    mark_budget_exceeded,
    mark_rate_limited,
    record,
    register_counter,
    request_latency_seconds,
    tenant_tokens,
)
from gateway.router import route
from gateway.tracing import trace_llm
from gateway.types import GovernanceError, GovernanceStage, GatewayMode, VerdictReview

# 05 失控循环熔断计数（建连前决策，与限流同层）
loop_guarded_total = register_counter(
    "gateway_loop_guarded_total", "requests rejected by loop guard", ("tenant",)
)

router = APIRouter()


def _review_header(reviews: list[VerdictReview]) -> str | None:
    """把 human_review 标记拼成响应头值（多标记逗号分隔）。E3 三通道之一。"""
    if not reviews:
        return None
    return ",".join(r.header_value() for r in reviews)


async def _scan_guardrails(stage: str, texts: list[str]) -> tuple[str | None, list[VerdictReview]]:
    """对一批文本做 06 合规护栏扫描（基于未脱敏原文）。

    返回 (deny_reason, reviews)：deny_reason 非空表示硬违规需拦截；
    reviews 为模糊边界标记列表（不阻断，仅用于注入响应头 + 已由 governance 打点/日志）。
    C1：guardrails_check 内部异常会向上抛 GovernanceError，由调用方按 fail_mode 分流。
    """
    deny_reason: str | None = None
    reviews: list[VerdictReview] = []
    for text in texts:
        if not text:
            continue
        verdict = await hooks.guardrails_check(stage, text)
        if not verdict.allowed:
            # 首个硬违规即拒绝（取 reason 作拦截依据；reason 不含原文）
            if deny_reason is None:
                deny_reason = verdict.reason
            continue
        if verdict.review is not None:
            reviews.append(verdict.review)
    return deny_reason, reviews


def _cache_hit_response(cached_answer: str, model: str) -> Response:
    """命中缓存：返回非流式固定 chat.completion（C3，已确认）。

    - 即便原请求 ``stream=true`` 也返回标准 ``chat.completion``（不模拟 SSE）。
    - usage 归零（零真实消耗）；证据头 ``X-Cache: HIT`` + ``X-Upstream-Model``(写入时模型)
      + ``X-Fallback: false``。
    - B3：命中不重跑 egress 治理链（缓存内容写入时已过 judge_egress+guardrails+脱敏）。
    """
    body = {
        "id": "cache-" + hashlib.md5(cached_answer.encode("utf-8")).hexdigest()[:12],
        "object": "chat.completion",
        "created": int(__import__("time").time()),
        "model": model or "",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": cached_answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return Response(
        content=json.dumps(body).encode(),
        status_code=200,
        media_type="application/json",
        headers={
            "X-Cache": "HIT",
            "X-Upstream-Model": model or "",
            "X-Fallback": "false",
        },
    )


def _extract_answer(body_obj: dict) -> str | None:
    """从 OpenAI 兼容 chat.completion 响应取出 assistant 文本（供 cache_store）。"""
    if not isinstance(body_obj, dict):
        return None
    choices = body_obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    for ch in choices:
        if not isinstance(ch, dict):
            continue
        msg = ch.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"]
    return None


def _extract_stream_answer(chunks: list[dict]) -> str | None:
    """从流式 chunk 列表拼接 delta.content（供 cache_store，流式场景）。"""
    parts: list[str] = []
    for chunk_obj in chunks:
        if not isinstance(chunk_obj, dict):
            continue
        for ch in chunk_obj.get("choices", []) or []:
            if not isinstance(ch, dict):
                continue
            delta = ch.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                parts.append(delta["content"])
    return "".join(parts) if parts else None


def _tenant_of(request: Request) -> str:
    # 批次1：优先 X-Tenant 头，否则按 Authorization API Key 映射，都没有则 default
    tenant = request.headers.get("X-Tenant")
    if tenant:
        return tenant
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        mapped = settings.tenant_of_key(auth[7:].strip())
        if mapped:
            return mapped
    return "default"


@router.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(path: str, request: Request) -> Response:
    """HTTP 入口：计时包装，把请求交给 _proxy_core，统一上报延迟 golden signal（09）。"""
    start = time.perf_counter()
    resp = await _proxy_core(path, request)
    request_latency_seconds.observe(
        time.perf_counter() - start,
        labels={
            "tenant": getattr(request.state, "tenant", "default"),
            "model": getattr(request.state, "model", "unknown"),
            "mode": settings.fail_mode.value,
            "status": str(resp.status_code),
        },
    )
    return resp


async def _proxy_core(path: str, request: Request) -> Response:
    raw = await request.body()
    model = None
    if request.method == "POST":
        try:
            model = json.loads(raw or b"{}").get("model")
        except Exception:
            pass
    tenant = _tenant_of(request)
    # 09：把 tenant/model 暂存到 request.state，供外层 proxy 计时打点（不改动核心逻辑分支）。
    request.state.tenant = tenant
    request.state.model = model or "unknown"

    # ① 入向注入第一道闸 + ③ 工具参数检测（批次2/10 实现 RegexInjectionGate）
    # D4：check 已定型为 async，调用方一律 await。命中即拦，先于脱敏执行（顺序不变量）。
    payload = json.loads(raw or b"{}") if raw else {}
    # 记录原始请求的 stream 标志：脱敏 hook 可能重建 payload 并丢弃该字段，
    # 故在 ingress 治理前从原始 body 取值（is_stream 决策用）。
    req_stream = bool(payload.get("stream")) if isinstance(payload, dict) else False
    verdict = await gate.check(payload)
    if not verdict.allowed:
        # 降级（scope-10 §3/§5）：closed→拦截 403；open→放行+强告警打点（不静默）。
        injection_denied_total.inc(
            labels={"stage": GovernanceStage.INGRESS.value, "mode": settings.fail_mode.value}
        )
        if settings.fail_mode == GatewayMode.FAIL_OPEN:
            # open：放行原文 + 强告警（日志/metrics 承载，不含原文）
            pass
        else:
            return Response(
                content=json.dumps({"error": "blocked by injection gate", "reason": verdict.reason}).encode(),
                status_code=403,
                media_type="application/json",
            )
    elif verdict.flagged:
        # 出向（非 user 角色）不可信命中：不阻断业务（deny 会自伤正常请求），
        # 仅打点告警（不含原文，E3 纪律），由上游 Agent 做隔离/人在回路确认。
        # 细粒度 {role, rule} 打点已在 gate.check 内完成，此处补一级 ingress 总览计数。
        injection_denied_total.inc(
            labels={"stage": GovernanceStage.INGRESS.value, "mode": settings.fail_mode.value}
        )

    # 06 合规护栏（B2 顺序：注入 → guardrails → 脱敏）：基于未脱敏原文判定。
    # 收集 ingress 侧 human_review 标记，最终注入响应头（E3 不阻断）。
    reviews: list[VerdictReview] = []
    ingress_texts: list[str] = []
    for msg in payload.get("messages", []) if isinstance(payload, dict) else []:
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            ingress_texts.append(msg["content"])
    try:
        deny_reason, ingress_reviews = await _scan_guardrails(GovernanceStage.INGRESS.value, ingress_texts)
        reviews.extend(ingress_reviews)
    except GovernanceError as e:
        # C1：guardrails 异常按 fail_mode 分流（细则同 scope-10）
        governance_error_total.inc(
            labels={"stage": e.stage, "mode": settings.fail_mode.value}
        )
        if settings.fail_mode == GatewayMode.FAIL_OPEN:
            pass  # open：放行（脱敏仍走，下方 run 执行）
        else:
            return Response(
                content=json.dumps({"error": "governance unavailable"}).encode(),
                status_code=502,
                media_type="application/json",
            )
    else:
        if deny_reason is not None:
            injection_denied_total.inc(
                labels={"stage": GovernanceStage.INGRESS.value, "mode": settings.fail_mode.value}
            )
            if settings.fail_mode == GatewayMode.FAIL_OPEN:
                pass  # open：放行 + 强告警
            else:
                return Response(
                    content=json.dumps({"error": "blocked by guardrails", "reason": deny_reason}).encode(),
                    status_code=400,
                    media_type="application/json",
                )

    # 治理面钩子（脱敏，批次2/07；缓存查询见下方 cache_lookup 命中短路）
    # D4：治理钩子统一 async，调用方一律 await
    payload = await hooks.run("ingress", payload)

    # 08 语义缓存查询（B2 顺序：注入 → guardrails → 脱敏 → cache_lookup → 模型）
    # 命中即短路返回缓存答案（不进模型、不 record、cache_hits+1），且不重跑 egress 治理链（B3）。
    # C1：cache_lookup 内部异常按 fail_mode 分流（fail-closed 转 502；fail-open 放行进模型）。
    try:
        cached = await hooks.cache_lookup(GovernanceStage.INGRESS.value, payload)
    except GovernanceError as e:
        governance_error_total.inc(
            labels={"stage": e.stage, "mode": settings.fail_mode.value}
        )
        if settings.fail_mode == GatewayMode.FAIL_OPEN:
            cached = None  # open：查不到就进模型
        else:
            return Response(
                content=json.dumps({"error": "governance unavailable"}).encode(),
                status_code=502,
                media_type="application/json",
            )
    else:
        cached = cached if cached is not None else None
    if cached is not None:
        # 命中短路：不进模型、不 record（成本账区分缓存短路 vs 真实消耗，08 §5）
        # 注意：upstream_model 此时尚未赋值（赋值在下方路由决策），命中即代表"用本次
        # 请求 model 之前答过"，故回显请求原始 model（cache_store 写入面按同 model 写）。
        return _cache_hit_response(cached, model or "")

    # 路由决策（仅网关侧：模型名改写 + fallback 归属声明，实际调度交 LiteLLM Router）
    decision = route(model, tenant)
    # 发往 Router 的模型名：本地任务改写为 qwen3-unified，云端保持原样
    send_model = decision.upstream_model or model or ""
    # payload 此时已脱敏（ingress hooks.run 已执行），从中抽取 messages 与透传参数
    messages = payload.get("messages", []) if isinstance(payload, dict) else []
    forward_kwargs = {
        k: v for k, v in payload.items() if k not in ("model", "messages", "stream")
    } if isinstance(payload, dict) else {}
    # stream 由网关决定（is_stream），不转发业务 body 里的 stream 字段，
    # 避免调用 acompletion/astream 时与显式 stream= 形参冲突（重复参数 TypeError）。
    # is_stream 基于原始请求的 stream 标志（req_stream，脱敏前已记录），不依赖脱敏后 payload。
    is_stream = req_stream

    # fail-closed/open 开关（01 §5 降级，非注入防线）+ 限流 + 预算（建连前决策，流式/非流式统一）
    # decide(model) 判断目标 model 是否在 Router model_list 注册：未注册 closed 拒绝（无旁路不变量）
    if not decide(decision.model or send_model):
        return Response(
            content=json.dumps({"error": "gateway fail-closed: model not registered"}).encode(),
            status_code=503,
            media_type="application/json",
        )

    limit_verdict = limit_check(tenant)
    if not limit_verdict.allowed:
        mark_rate_limited(tenant)
        return Response(
            content=json.dumps({"error": "rate limited"}).encode(),
            status_code=429,
            headers={"Retry-After": "1"},
            media_type="application/json",
        )

    est = estimate_tokens(raw)
    record(tenant, est)
    if settings.budget_tenant_tokens > 0 and tenant_tokens(tenant) > settings.budget_tenant_tokens:
        mark_budget_exceeded(tenant)
        return Response(
            content=json.dumps({"error": "budget exceeded"}).encode(),
            status_code=429,
            media_type="application/json",
        )

    # 05 失控循环防护（建连前，与限流同层，无旁路不变量）
    loop_verdict = loop_check(tenant, model or "unknown", raw)
    if not loop_verdict.allowed:
        loop_guarded_total.inc(labels={"tenant": tenant})
        return Response(
            content=json.dumps({"error": "loop guarded"}).encode(),
            status_code=429,
            headers={"Retry-After": "1"},
            media_type="application/json",
        )

    # 02 §4 故障转移由 LiteLLM Router 接管（model_list 多 deployment + num_retries）。
    # 网关不再手写 fallback_chain 遍历；Router 调用失败统一抛 GovernanceError，按 fail_mode 分流。
    upstream_model = send_model
    used_fallback = False
    if is_stream:
        # 流式：Router 返回 async generator，逐块透过治理链后再外发
        try:
            upstream_stream = llm_router.astream(send_model, messages, **forward_kwargs)
        except GovernanceError as e:
            governance_error_total.inc(
                labels={"stage": e.stage, "mode": settings.fail_mode.value}
            )
            if settings.fail_mode == GatewayMode.FAIL_OPEN:
                pass  # open：无上游可放，仍返回 502 占位（裸跑无回退）
            return Response(
                content=json.dumps({"error": "all backends unavailable", "model": upstream_model}).encode(),
                status_code=503,
                media_type="application/json",
                headers={"X-Fallback": "false"},
            )
    else:
        # 非流式：Router 返回 (dict, actual_model, used_fallback)
        try:
            resp_obj, actual_model, used_fallback = await llm_router.acompletion(
                send_model, messages, **forward_kwargs
            )
        except GovernanceError as e:
            # C1：Router 调用失败（含重试耗尽）按 fail_mode 分流（fail-closed→503）
            governance_error_total.inc(
                labels={"stage": e.stage, "mode": settings.fail_mode.value}
            )
            trace_llm(tenant, model or "unknown", est, ok=False)
            if settings.fail_mode == GatewayMode.FAIL_OPEN:
                pass  # open：无上游可放，仍返 503（裸跑无回退）
            return Response(
                content=json.dumps({"error": "all backends unavailable", "model": upstream_model}).encode(),
                status_code=503,
                media_type="application/json",
                headers={"X-Fallback": "false"},
            )
        # 04 trace：模型调用成功
        trace_llm(tenant, model or "unknown", est, ok=True)
        upstream_model = actual_model or send_model

    # 响应 metadata 证据头（02 篇：后端对业务无感，但留痕可供调用方按需读取）
    resp_headers: dict[str, str] = {}
    resp_headers["X-Upstream-Model"] = upstream_model
    resp_headers["X-Fallback"] = "true" if used_fallback else "false"

    # 出向治理钩子（非流式：整块；流式：逐块）
    # 编排顺序不变量（B2 / scope-10 §3）：先 judge_egress（基于未脱敏原文）→ 再脱敏外发。
    if is_stream:


        async def _stream():
            # Router 流式返回 async generator，逐块产出 (chunk_dict, actual_model)。
            # 每块已解析，直接过治理链；fail-closed 下命中即截断 + 终止标记（C2）。
            aborted = False  # fail-closed 下已补发终止标记，避免重复
            stream_reviews: list[VerdictReview] = []  # 流式 review 累积（E3 响应头通道的流式等价）
            stream_chunks: list[dict] = []  # 累积已放行 chunk 供 cache_store（B3 前提：已通过治理）
            try:
                async for chunk_obj, _actual in upstream_stream:
                    if not isinstance(chunk_obj, dict):
                        # 非 dict chunk：按 open 原样透传 + 打点
                        governance_error_total.inc(
                            labels={"stage": GovernanceStage.STREAM.value, "mode": settings.fail_mode.value}
                        )
                        yield b"data: " + json.dumps(chunk_obj).encode() + b"\n\n"
                        continue
                    # ② 出向 judge_egress：对未脱敏原文逐块检测（fail-closed 默认拦）
                    try:
                        verdict = await hooks.judge_egress(chunk_obj)
                        if not verdict.allowed:
                            injection_denied_total.inc(
                                labels={"stage": GovernanceStage.STREAM.value, "mode": settings.fail_mode.value}
                            )
                            if settings.fail_mode == GatewayMode.FAIL_OPEN:
                                # open：放行原文 + 强告警（仍脱敏外发）
                                yield b"data: " + json.dumps(await hooks.run_stream("egress", chunk_obj)).encode() + b"\n\n"
                                continue
                            # closed：截断 + 终止标记（已发 chunk 不回滚，C2 语义）
                            if not aborted:
                                yield b'data: {"choices":[{"finish_reason":"content_policy_violation"}]}\n\n'
                                yield b"data: [DONE]\n\n"
                                aborted = True
                            return
                        # B2 顺序：judge_egress（已通过）→ 06 guardrails（基于原文）→ 脱敏外发
                        stream_texts: list[str] = []
                        for ch in chunk_obj.get("choices", []) if isinstance(chunk_obj, dict) else []:
                            if isinstance(ch, dict) and isinstance(ch.get("delta"), dict):
                                c = ch["delta"].get("content")
                                if isinstance(c, str):
                                    stream_texts.append(c)
                        deny_reason, s_reviews = await _scan_guardrails(
                            GovernanceStage.STREAM.value, stream_texts
                        )
                        stream_reviews.extend(s_reviews)
                        if deny_reason is not None:
                            injection_denied_total.inc(
                                labels={"stage": GovernanceStage.STREAM.value, "mode": settings.fail_mode.value}
                            )
                            if settings.fail_mode == GatewayMode.FAIL_OPEN:
                                yield b"data: " + json.dumps(await hooks.run_stream("egress", chunk_obj)).encode() + b"\n\n"
                                continue
                            if not aborted:
                                yield b'data: {"choices":[{"finish_reason":"content_policy_violation"}]}\n\n'
                                yield b"data: [DONE]\n\n"
                                aborted = True
                            return
                        # 放行内容经脱敏外发
                        chunk_obj = await hooks.run_stream("egress", chunk_obj)
                        stream_chunks.append(chunk_obj)
                        yield b"data: " + json.dumps(chunk_obj).encode() + b"\n\n"
                    except GovernanceError as e:
                        governance_error_total.inc(
                            labels={"stage": e.stage, "mode": settings.fail_mode.value}
                        )
                        if settings.fail_mode == GatewayMode.FAIL_OPEN:
                            yield b"data: " + json.dumps(chunk_obj).encode() + b"\n\n"
                            continue
                        if not aborted:
                            yield b'data: {"choices":[{"finish_reason":"governance_error"}]}\n\n'
                            yield b"data: [DONE]\n\n"
                            aborted = True
                        return
            finally:
                if not aborted:
                    yield b"data: [DONE]\n\n"
            # 08 流式 cache_store：所有 chunk 已通过 egress 治理链（B3 前提），末尾写入。
            if stream_chunks and not aborted:
                answer = _extract_stream_answer(stream_chunks)
                if answer:
                    try:
                        await hooks.cache_store(GovernanceStage.STREAM.value, payload, answer)
                    except GovernanceError:
                        pass
            if stream_reviews:
                hdr = _review_header(stream_reviews)
                if hdr:
                    yield f": X-Guardrails-Review: {hdr}\n\n".encode()

        return StreamingResponse(
            _stream(),
            status_code=200,
            headers=resp_headers,
        )

    # 非流式：② judge_egress 整块检测（原文）→ 脱敏外发
    # Router 成功返回即 200，body 为已解析 dict（resp_obj）
    body_obj = resp_obj
    _status = 200

    try:
        verdict = await hooks.judge_egress(body_obj)
        if not verdict.allowed:
            injection_denied_total.inc(
                labels={"stage": GovernanceStage.EGRESS.value, "mode": settings.fail_mode.value}
            )
            if settings.fail_mode == GatewayMode.FAIL_OPEN:
                body_obj = await hooks.run("egress", body_obj)
                return Response(
                    content=json.dumps(body_obj).encode(),
                    status_code=_status,
                    headers=resp_headers,
                )
            return Response(
                content=json.dumps(
                    {"error": "blocked by egress judge", "reason": verdict.reason}
                ).encode(),
                status_code=400,
                media_type="application/json",
            )
        egress_texts: list[str] = []
        for ch in body_obj.get("choices", []) if isinstance(body_obj, dict) else []:
            if isinstance(ch, dict) and isinstance(ch.get("message"), dict):
                c = ch["message"].get("content")
                if isinstance(c, str):
                    egress_texts.append(c)
        deny_reason, egress_reviews = await _scan_guardrails(GovernanceStage.EGRESS.value, egress_texts)
        reviews.extend(egress_reviews)
        if deny_reason is not None:
            injection_denied_total.inc(
                labels={"stage": GovernanceStage.EGRESS.value, "mode": settings.fail_mode.value}
            )
            if settings.fail_mode == GatewayMode.FAIL_OPEN:
                body_obj = await hooks.run("egress", body_obj)
                return Response(
                    content=json.dumps(body_obj).encode(),
                    status_code=_status,
                    headers=resp_headers,
                )
            return Response(
                content=json.dumps(
                    {"error": "blocked by guardrails", "reason": deny_reason}
                ).encode(),
                status_code=400,
                media_type="application/json",
            )
        body_obj = await hooks.run("egress", body_obj)
        body = json.dumps(body_obj).encode()
        _egress_answer = _extract_answer(body_obj)
        if _egress_answer:
            try:
                await hooks.cache_store(GovernanceStage.EGRESS.value, payload, _egress_answer)
            except GovernanceError as e:
                governance_error_total.inc(
                    labels={"stage": e.stage, "mode": settings.fail_mode.value}
                )
                if settings.fail_mode == GatewayMode.FAIL_OPEN:
                    pass
                else:
                    return Response(
                        content=json.dumps({"error": "governance unavailable"}).encode(),
                        status_code=502,
                        media_type="application/json",
                    )
    except GovernanceError as e:
        governance_error_total.inc(
            labels={"stage": e.stage, "mode": settings.fail_mode.value}
        )
        if settings.fail_mode == GatewayMode.FAIL_OPEN:
            return Response(content=body, status_code=_status, headers=resp_headers)
        return Response(
            content=json.dumps({"error": "governance unavailable"}).encode(),
            status_code=502,
            media_type="application/json",
        )
    review_header = _review_header(reviews)
    if review_header:
        resp_headers["X-Guardrails-Review"] = review_header
    return Response(
        content=body,
        status_code=_status,
        headers=resp_headers,
    )
