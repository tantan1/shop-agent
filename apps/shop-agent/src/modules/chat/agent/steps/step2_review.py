"""步骤2：内容审查/安全检查。"""
from __future__ import annotations

from typing import Tuple

from src.modules.chat.agent.prompts import PromptTemplateManager
from src.modules.chat.agent.schemas import AgentStepResult, SafetyCheckResult
from src.modules.chat.agent.steps.base import AgentContext, BaseStep
from src.shared.logger import APILogger

logger = APILogger("step2_review")


class ReviewStep(BaseStep):
    """分级安全审查：本地小模型 → 云端 LLM 复核。"""

    step_name = "内容审查"
    step_order = 2

    async def execute(self, ctx: AgentContext) -> Tuple[AgentStepResult, SafetyCheckResult]:
        step_config = getattr(ctx.config, "step2", None)
        if not step_config or not step_config.enabled:
            default_result = SafetyCheckResult(
                is_safe=True, risk_level="low", risk_categories=[], can_proceed=True
            )
            return (
                AgentStepResult(
                    step_name=step_config.name if step_config else "内容审查",
                    step_order=self.step_order,
                    input_data={"user_question": ctx.request.message},
                    output_data=default_result.model_dump(),
                    status="skipped",
                ),
                default_result,
            )

        import time
        start_time = time.time()

        try:
            template = PromptTemplateManager.get(ctx.domain, step_config.prompt_template_key)
            messages = [
                {"role": "system", "content": template},
                {"role": "user", "content": f"用户问题：{ctx.request.message}"},
            ]

            if step_config.output_format == "json" and step_config.response_schema:
                from src.modules.chat.agent.schemas import get_structured_schema
                schema_class = get_structured_schema(step_config.response_schema)
                if schema_class:
                    try:
                        structured_result = await ctx.llm_service.chat_qwen_structured(
                            messages,
                            schema_class,
                            temperature=0.0,
                            langfuse_handler=ctx.langfuse_handler,
                        )
                        if hasattr(structured_result, "model_dump"):
                            safety_data = structured_result.model_dump()
                        else:
                            safety_data = structured_result
                        logger.info(f"[{ctx.domain}] {step_config.name}使用结构化输出")
                    except Exception as e:
                        logger.warning(
                            f"[{ctx.domain}] {step_config.name}结构化输出失败，降级到JSON解析: {str(e)[:100]}"
                        )
                        safety_data = None
            else:
                safety_data = None

            if safety_data is None:
                response = await ctx.llm_service.chat_qwen(
                    messages,
                    langfuse_handler=ctx.langfuse_handler,
                )
                safety_data = _parse_json_from_llm(response)

            if safety_data is None:
                sensitive_keywords = getattr(ctx.config, "sensitive_keywords", ["诊断", "处方", "胸痛"])
                detected = [kw for kw in sensitive_keywords if kw in ctx.request.message]
                safety_data = {
                    "is_safe": False if detected else True,
                    "risk_level": "high" if detected else "low",
                    "risk_categories": detected or ["解析失败"],
                }

            safety_result = _build_safety_result(safety_data)
            duration = int((time.time() - start_time) * 1000)

            logger.info(
                f"[{ctx.domain}] {step_config.name}完成",
                is_safe=safety_result.is_safe,
            )

            return (
                AgentStepResult(
                    step_name=step_config.name,
                    step_order=self.step_order,
                    input_data={"user_question": ctx.request.message},
                    output_data=safety_result.model_dump(),
                    status="success",
                    duration_ms=duration,
                ),
                safety_result,
            )

        except Exception as e:
            duration = int((time.time() - start_time) * 1000)
            logger.error(f"[{ctx.domain}] {step_config.name}失败: {str(e)}")

            safety_result = SafetyCheckResult(
                is_safe=False,
                risk_level="high",
                risk_categories=["服务异常"],
                can_proceed=False,
            )

            return (
                AgentStepResult(
                    step_name=step_config.name,
                    step_order=self.step_order,
                    input_data={"user_question": ctx.request.message},
                    status="failed",
                    error_message=str(e),
                    duration_ms=duration,
                ),
                safety_result,
            )


def _parse_json_from_llm(text: str) -> dict:
    """从 LLM 返回文本中提取 JSON 对象。"""
    json_str = text
    if "```json" in text:
        json_str = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        json_str = text.split("```")[1].split("```")[0]
    import json
    try:
        return json.loads(json_str)
    except Exception:
        return None


def _build_safety_result(data: dict) -> SafetyCheckResult:
    """将安全审查数据统一转换为 SafetyCheckResult。"""
    if "compliant" in data:
        is_safe = data.get("compliant", True)
        issue = data.get("issue", "")
        return SafetyCheckResult(
            is_safe=is_safe,
            risk_level="high" if not is_safe else "low",
            risk_categories=[issue] if not is_safe and issue else [],
            can_proceed=is_safe,
        )
    is_safe = data.get("is_safe", True)
    return SafetyCheckResult(
        is_safe=is_safe,
        risk_level=data.get("risk_level", "low"),
        risk_categories=data.get("risk_categories", []),
        warning_message=data.get("warning_message"),
        can_proceed=is_safe or data.get("risk_level", "low") != "high",
    )
