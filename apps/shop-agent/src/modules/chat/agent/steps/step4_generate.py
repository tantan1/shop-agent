"""步骤4：回答生成（Token 预算守卫 + 输出过滤 + 质量评估）。"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from src.modules.chat.agent.prompts import WARNING_TEMPLATES, PromptTemplateManager
from src.modules.chat.agent.schemas import AgentStepResult, SafetyCheckResult
from src.modules.chat.agent.steps.base import AgentContext, BaseStep
from src.modules.chat.core.content_filter import ContentFilterService
from src.shared.logger import APILogger

logger = APILogger("step4_generate")


@dataclass
class GenerateSuccessContext:
    """回答生成成功上下文，封装 _build_success_result 所需参数。"""
    ctx: AgentContext
    step_config: Any
    response: str
    rag_context: str
    duration: int
    quality_evaluation: Dict[str, Any]


@dataclass
class PromptBuildContext:
    """Prompt 构建上下文，封装 _build_and_fit_prompt 所需参数。"""
    ctx: AgentContext
    step_config: Any
    template: str
    graph_context: str
    rag_context: str
    safety_check_result: str
    safety_reminder: str
    chat_history_str: str
    current_time: str


class GenerateStep(BaseStep):
    """构建 prompt → Token 预算守卫 → LLM 生成 → 输出过滤 → 质量评估。"""

    step_name = "回答生成"
    step_order = 4

    async def execute(self, ctx: AgentContext, safety_result: SafetyCheckResult, rag_context: str, graph_context: str = "") -> Tuple[AgentStepResult, str, Dict[str, Any]]:
        step_config = getattr(ctx.config, "step4", None)
        if not step_config or not step_config.enabled:
            return (
                AgentStepResult(
                    step_name=step_config.name if step_config else "回答生成",
                    step_order=self.step_order,
                    status="skipped",
                ),
                "抱歉，暂无法生成回答。",
                {"is_solved": False, "quality_score": 0},
            )

        start_time = time.time()

        try:
            # GrowthBook 实验覆盖：prompt_template_key（实验级切换生成 prompt 模板）
            template = PromptTemplateManager.get(
                ctx.domain, ctx.get_override("prompt_template_key", step_config.prompt_template_key)
            )
            chat_history_str = await self._prepare_chat_history(ctx, template)

            safety_reminder = self._build_safety_reminder(safety_result)
            safety_check_result = self._build_safety_check_result(safety_result)
            current_time = time.strftime("%Y年%m月%d日 %H:%M")
            prompt_content, rag_context = self._build_and_fit_prompt(
                PromptBuildContext(
                    ctx=ctx,
                    step_config=step_config,
                    template=template,
                    graph_context=graph_context,
                    rag_context=rag_context,
                    safety_check_result=safety_check_result,
                    safety_reminder=safety_reminder,
                    chat_history_str=chat_history_str,
                    current_time=current_time,
                )
            )

            messages = [{"role": "system", "content": prompt_content}]
            # 实验覆盖（Phase 4）：llm_model / llm_temperature / llm_max_tokens。
            # 仅当 override 非 None 才透传；None 沿用服务默认。
            # 注：chat_qwen 的基础默认 temperature=0.3 保持（零回归），
            #     llm_temperature 覆盖时再覆盖该值。
            model = ctx.get_override("llm_model", None)
            temperature = ctx.get_override("llm_temperature", None)
            max_tokens = ctx.get_override("llm_max_tokens", None)
            llm_kwargs: dict = {}
            if model is not None:
                llm_kwargs["model"] = model
            if temperature is not None:
                llm_kwargs["temperature"] = temperature
            if max_tokens is not None:
                llm_kwargs["max_tokens"] = max_tokens
            response = await ctx.llm_service.chat_qwen(
                messages, temperature=0.3, langfuse_handler=ctx.langfuse_handler, **llm_kwargs
            )

            response, quality_evaluation, output_filter_safe = self._filter_and_evaluate(ctx, response, rag_context)

            await self._store_chat_history(ctx, ctx.request.message, response)

            duration = int((time.time() - start_time) * 1000)
            logger.info(f"[{ctx.domain}] {step_config.name}完成", response_length=len(response))

            return self._build_success_result(
                GenerateSuccessContext(
                    ctx=ctx,
                    step_config=step_config,
                    response=response,
                    rag_context=rag_context,
                    duration=duration,
                    quality_evaluation=quality_evaluation,
                )
            )

        except Exception as e:
            duration = int((time.time() - start_time) * 1000)
            logger.error(f"[{ctx.domain}] {step_config.name}失败: {str(e)}")
            return (
                AgentStepResult(
                    step_name=step_config.name,
                    step_order=self.step_order,
                    status="failed",
                    error_message=str(e),
                    duration_ms=duration,
                ),
                "抱歉，服务暂时繁忙，请稍后重试。",
                {"is_solved": False, "quality_score": 0},
            )

    async def _prepare_chat_history(self, ctx: AgentContext, template: str) -> str:
        """准备对话历史摘要。"""
        needs_chat_history = "{chat_history}" in template
        if not (needs_chat_history and ctx.redis_cache_service and ctx.redis_cache_service.is_available):
            return ""

        history = ctx.redis_cache_service.get_chat_messages(
            ctx.conversation_id, max_turns=ctx.config.max_history_turns
        )
        if not history:
            return ""

        from src.modules.chat.agent.conversation_summarizer import ConversationSummarizer
        summarizer = ConversationSummarizer(llm_service=ctx.llm_service)
        max_history_tokens = getattr(ctx.config, "max_history_tokens", 2000)
        return await summarizer.summarize_if_needed(
            messages=history[-ctx.config.max_history_turns:],
            max_tokens=max_history_tokens,
        )

    def _build_and_fit_prompt(
        self,
        build_ctx: PromptBuildContext,
    ) -> Tuple[str, str]:
        """构建 prompt 并在超出 token 预算时缩减 RAG 上下文。"""
        memory_context = getattr(build_ctx.ctx, "memory_context", "") or ""
        def _format_prompt(rag: str) -> str:
            memory_ctx = ""
            if memory_context:
                memory_ctx = f"\n# 用户长期记忆\n{memory_context}\n"
            return build_ctx.template.format(
                current_time=build_ctx.current_time,
                graph_context=build_ctx.graph_context or "（无相关商品关系数据）",
                rag_context=rag,
                user_question=build_ctx.ctx.request.message,
                safety_check_result=build_ctx.safety_check_result,
                safety_reminder=build_ctx.safety_reminder,
                chat_history=build_ctx.chat_history_str,
                product_info=rag,
                knowledge_base=rag,
                context=rag,
                category="",
                memory_context=memory_ctx,
            )

        rag_context = build_ctx.rag_context
        prompt_content = _format_prompt(rag_context)

        # Token 预算守卫
        MAX_PROMPT_TOKENS = getattr(build_ctx.ctx.config, "max_generation_prompt_tokens", 6800)
        estimator_tk = _get_token_estimator()
        prompt_tokens = estimator_tk.estimate(prompt_content)

        if prompt_tokens > MAX_PROMPT_TOKENS:
            logger.warning(
                f"[{build_ctx.ctx.domain}] Prompt token {prompt_tokens} 超出预算"
                f" {MAX_PROMPT_TOKENS}，逐次缩减RAG上下文",
                prompt_tokens=prompt_tokens,
                rag_chars=len(build_ctx.rag_context),
            )
            fitted = False
            rag_docs = build_ctx.rag_context.split("\n\n")
            for n in range(len(rag_docs), 0, -1):
                reduced_rag = "\n\n".join(rag_docs[:n])
                test_prompt = _format_prompt(reduced_rag)
                test_tokens = estimator_tk.estimate(test_prompt)
                if test_tokens <= MAX_PROMPT_TOKENS:
                    prompt_content = test_prompt
                    rag_context = reduced_rag
                    logger.info(
                        f"[{build_ctx.ctx.domain}] RAG上下文缩减成功",
                        docs_before=len(rag_docs),
                        docs_after=n,
                        tokens_before=prompt_tokens,
                        tokens_after=test_tokens,
                    )
                    fitted = True
                    break

            if not fitted:
                fallback_rag = "（RAG 检索结果过长已省略，请基于通用知识回答）"
                prompt_content = _format_prompt(fallback_rag)
                rag_context = fallback_rag
                logger.warning(
                    f"[{build_ctx.ctx.domain}] RAG上下文全部丢弃，降为无检索回答",
                    final_prompt_tokens=estimator_tk.estimate(prompt_content),
                )

        return prompt_content, rag_context

    def _filter_and_evaluate(
        self, ctx: AgentContext, response: str, rag_context: str
    ) -> Tuple[str, Dict[str, Any], bool]:
        """输出安全过滤 + 质量评估。"""
        output_filter_safe = True
        # TODO(GB-SDK): PipelineOverrides.content_filter_enabled 待接入（实验级覆盖输出安全过滤开关）
        if ctx.config.content_filter_enabled:
            cf = ContentFilterService.get_instance()
            output_check = cf.filter_output(response, ctx.domain)
            if not output_check.is_safe:
                output_filter_safe = False
                logger.warning(
                    f"[{ctx.domain}] 输出内容安全检查未通过",
                    risk_categories=output_check.risk_categories,
                    original_length=len(response),
                )
                if output_check.filtered_text:
                    response = output_check.filtered_text
                else:
                    response = WARNING_TEMPLATES.get("content_filtered", "").format(
                        risk_categories=", ".join(output_check.risk_categories),
                        warning_message="请重新描述您的问题",
                    )

        quality_evaluation = _quick_evaluate_answer_quality(response, rag_context)
        quality_evaluation["output_filter_safe"] = output_filter_safe
        return response, quality_evaluation, output_filter_safe

    async def _store_chat_history(self, ctx: AgentContext, request_message: str, response: str):
        """存储用户和助手消息到 Redis 缓存，并触发 L2 保存。"""
        if not (ctx.redis_cache_service and ctx.redis_cache_service.is_available):
            return
        try:
            # add_chat_message 是同步方法（返回 bool），不能用 await，
            # 否则抛 "object bool can't be used in 'await' expression"，导致对话历史静默丢失。
            ctx.redis_cache_service.add_chat_message(
                ctx.conversation_id, "user", request_message[:4096],
                max_turns=ctx.config.max_history_turns, expire_days=1,
            )
            ctx.redis_cache_service.add_chat_message(
                ctx.conversation_id, "assistant", response[:4096],
                max_turns=ctx.config.max_history_turns, expire_days=1,
            )

            # 轮次计数器 + L2 触发
            turn_key = f"chat:turn_count:{ctx.conversation_id}"
            turn_number = ctx.redis_cache_service._client.incr(turn_key)
            ctx.redis_cache_service._client.expire(turn_key, 1 * 86400)
            if turn_number % 5 == 0:
                import asyncio
                asyncio.create_task(
                    self._trigger_l2_save(ctx, turn_number)
                )
        except Exception as e:
            logger.warning(f"[{ctx.domain}] 存储缓存失败: {str(e)[:100]}")

    async def _trigger_l2_save(self, ctx: AgentContext, turn_number: int) -> None:
        """触发 L2 短期记忆保存"""
        import time

        from src.modules.monitoring.metrics import (
            l2_save_duration_ms,
            l2_save_failure_total,
            l2_save_success_total,
            l2_save_triggered_total,
            l2_summary_tokens,
        )

        start_time = time.perf_counter()
        l2_save_triggered_total.labels(trigger="turn").inc()
        try:
            from src.modules.chat.core.memory_service import ShortTermMemory

            redis = ctx.redis_cache_service
            dedup_key = f"chat:l2_saved:{ctx.conversation_id}"
            if redis._client.exists(dedup_key):
                return

            short_term = ShortTermMemory()
            history = redis.get_chat_messages(ctx.conversation_id, max_turns=50)
            if not history:
                return

            summary = "; ".join([m.get("content", "") for m in history[-10:]])
            key_entities = []

            try:
                from src.modules.chat.core.embedding_service import EmbeddingService
                emb_svc = EmbeddingService.get_instance()
                embedding = await emb_svc.embed_query(summary)
            except Exception as e:
                l2_save_failure_total.labels(trigger="turn", error="embedding_error").inc()
                logger.warning(f"L2 摘要 embedding 失败: {e}")
                return

            await short_term.store_summary(
                user_id=ctx.user_id,
                conversation_id=ctx.conversation_id,
                summary=summary,
                key_entities=key_entities,
                turn_number=turn_number,
                embedding=embedding,
            )
            redis._client.setex(dedup_key, 7 * 86400, "1")
            l2_save_success_total.labels(trigger="turn").inc()
            duration_ms = (time.perf_counter() - start_time) * 1000
            l2_save_duration_ms.labels(trigger="turn").observe(duration_ms)
            l2_summary_tokens.observe(len(summary.split()))
        except Exception:
            l2_save_failure_total.labels(trigger="turn", error="unknown").inc()
            logger.debug("L2 保存失败，跳过", exc_info=True)

    def _build_success_result(
        self,
        success_ctx: GenerateSuccessContext,
    ) -> Tuple[AgentStepResult, str, Dict[str, Any]]:
        """构建成功执行结果。"""
        return (
            AgentStepResult(
                step_name=success_ctx.step_config.name,
                step_order=self.step_order,
                input_data={
                    "user_question": success_ctx.ctx.request.message,
                    "context_length": len(success_ctx.rag_context),
                },
                output_data={"response": success_ctx.response, **success_ctx.quality_evaluation},
                status="success",
                duration_ms=success_ctx.duration,
            ),
            success_ctx.response,
            success_ctx.quality_evaluation,
        )

    def _build_safety_reminder(self, safety_result: SafetyCheckResult) -> str:
        if not safety_result.is_safe:
            return WARNING_TEMPLATES.get("default", "").format(
                risk_categories=", ".join(safety_result.risk_categories),
                warning_message=safety_result.warning_message or "请咨询专业人员",
            )
        return "问题已通过审查。"

    def _build_safety_check_result(self, safety_result: SafetyCheckResult) -> str:
        result = f"风险等级: {safety_result.risk_level}"
        if safety_result.risk_categories:
            result += f"\n涉及内容: {', '.join(safety_result.risk_categories)}"
        return result


def _get_token_estimator():
    from src.core.token_estimator import get_token_estimator
    return get_token_estimator()


def _quick_evaluate_answer_quality(response: str, rag_context: str) -> Dict[str, Any]:
    """快速评估答案质量。"""
    reasons = []
    quality_score = 5

    has_rag_context = rag_context and rag_context != "暂无相关检索结果"
    if has_rag_context:
        reasons.append("有检索结果支撑")
        quality_score += 1
    else:
        reasons.append("无检索结果")
        quality_score -= 2

    response_len = len(response)
    if response_len >= 50:
        reasons.append(f"长度适中({response_len}字)")
        quality_score += 1
    elif response_len >= 20:
        quality_score -= 1
    else:
        quality_score -= 2

    from src.modules.chat.config import chat_config
    low_quality_patterns = getattr(chat_config, "low_quality_patterns", [])
    has_low_quality = any(pattern in response for pattern in low_quality_patterns)
    if has_low_quality:
        for pattern in low_quality_patterns:
            if pattern in response:
                reasons.append(f"包含低质量模式: {pattern}")
                quality_score -= 2
                break

    quality_score = max(0, min(10, quality_score))
    is_solved = quality_score >= 6 and not has_low_quality

    return {
        "is_solved": is_solved,
        "quality_score": round(quality_score, 1),
        "eval_reason": "; ".join(reasons),
    }
