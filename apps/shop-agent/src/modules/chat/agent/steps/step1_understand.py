"""步骤1：问题理解/改写。"""
from __future__ import annotations

from src.modules.chat.agent.prompts import PromptTemplateManager
from src.modules.chat.agent.schemas import AgentStepResult, QuestionRewriteResult
from src.modules.chat.agent.steps.base import AgentContext, BaseStep
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
            template = PromptTemplateManager.get(ctx.domain, step_config.prompt_template_key)

            messages = [
                {"role": "system", "content": template},
                {"role": "user", "content": f"用户问题：{ctx.request.message}"},
            ]

            response = await ctx.llm_service.chat_qwen(
                messages, langfuse_handler=ctx.langfuse_handler
            )
            queries = [q.strip() for q in response.split("\n") if q.strip()]

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
