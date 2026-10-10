"""步骤1：问题理解/改写。"""
from __future__ import annotations

from src.modules.chat.agent.prompts import PromptTemplateManager
from src.modules.chat.agent.schemas import AgentStepResult, QuestionRewriteResult
from src.modules.chat.agent.steps.base import AgentContext, BaseStep
from src.ports.pii import redact as redact_pii
from src.shared.logger import APILogger

logger = APILogger("step1_understand")


class UnderstandStep(BaseStep):
    """使用 LLM 理解并改写用户问题，生成多角度检索查询。"""

    step_name = "问题理解"
    step_order = 1

    async def execute(self, ctx: AgentContext) -> AgentStepResult:
        step_config = getattr(ctx.config, "step1", None)
        if not step_config or not step_config.enabled:
            return self._skip_result(ctx)

        import time
        start_time = time.time()

        try:
            # GrowthBook 实验覆盖：prompt_template_key（实验级切换改写 prompt 模板）
            template = PromptTemplateManager.get(
                ctx.domain, ctx.get_override("prompt_template_key", step_config.prompt_template_key)
            )

            # 多轮融合：step1 开启时把历史折进改写 prompt（0 新增调用，改写与指代一体）
            history_block = self._build_history_block(ctx)
            messages = [
                {"role": "system", "content": template},
                {"role": "user", "content": f"用户问题：{ctx.request.message}{history_block}"},
            ]

            # 实验覆盖（Phase 4）：llm_model / llm_temperature / llm_max_tokens。
            # 仅当 override 非 None 才透传给底层 LLM；None 表示沿用服务默认，零回归。
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
            response = await ctx.llm_service.chat_step1(
                messages, langfuse_handler=ctx.langfuse_handler, **llm_kwargs
            )
            queries = [q.strip() for q in response.split("\n") if q.strip()]

            # 多轮融合：改写产物作为检索 query（红线 §4：仅影响 RAG 检索，不改原文）
            if queries:
                ctx.retrieval_query = queries[0]

            result = QuestionRewriteResult(
                original_question=ctx.request.message,
                rewritten_queries=queries or [ctx.request.message],
                keywords=list(set([w for q in queries for w in q.split()])),
            )

            duration = int((time.time() - start_time) * 1000)
            logger.info(
                f"[{ctx.domain}] {step_config.name}完成",
                message_length=len(ctx.request.message),
            )

            return AgentStepResult(
                step_name=step_config.name,
                step_order=self.step_order,
                input_data={"user_question": ctx.request.message},
                output_data=result.model_dump(),
                status="success",
                duration_ms=duration,
            )

        except Exception as e:
            duration = int((time.time() - start_time) * 1000)
            logger.error(f"[{ctx.domain}] {step_config.name}失败: {str(e)}")
            return AgentStepResult(
                step_name=step_config.name,
                step_order=self.step_order,
                input_data={"user_question": ctx.request.message},
                status="failed",
                error_message=str(e),
                duration_ms=duration,
            )

    @staticmethod
    def _build_history_block(ctx: AgentContext) -> str:
        """读取最近对话历史，格式化并脱敏，供 step1 改写消歧；无历史/不可用返回空。"""
        redis = getattr(ctx, "redis_cache_service", None)
        if not redis or not getattr(redis, "is_available", False) or not ctx.conversation_id:
            return ""
        try:
            hist = redis.get_chat_messages(ctx.conversation_id, max_turns=3)
        except Exception:
            return ""
        if not hist:
            return ""
        lines = [
            f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')}"
            for m in hist if isinstance(m, dict)
        ]
        if not lines:
            return ""
        return (
            "\n\n参考对话历史（仅用于消解代词/省略，如\"它/这个\"指代，请勿编造）:\n"
            + redact_pii("\n".join(lines))
        )
