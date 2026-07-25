"""Agent 执行管线（Pipeline Pattern）。

将 step1-step4 串行为有序管道，每步独立可测、可替换、可跳过。
GeneralAgentExecutor 保留 Langfuse/缓存/响应构建等横切逻辑，业务步骤委托给 Pipeline。
"""
from __future__ import annotations

from typing import List

from src.modules.chat.agent.schemas import AgentStepResult, ChatResponse
from src.modules.chat.agent.steps.base import AgentContext
from src.modules.chat.agent.steps.step1_understand import UnderstandStep
from src.modules.chat.agent.steps.step2_review import ReviewStep
from src.modules.chat.agent.steps.step3_retrieve import RetrieveStep
from src.modules.chat.agent.steps.step4_generate import GenerateStep
from src.shared.logger import APILogger

logger = APILogger("agent_pipeline")


class Pipeline:
    """通用 Agent 执行管线。"""

    def __init__(self, ctx: AgentContext):
        self.ctx = ctx
        self.steps = [
            UnderstandStep(),
            ReviewStep(),
            RetrieveStep(),
            GenerateStep(),
        ]

    async def run(self) -> tuple[ChatResponse, List[AgentStepResult]]:
        """执行完整管线，返回最终响应和步骤列表。"""
        steps_results: List[AgentStepResult] = []
        documents_used: List[str] = []

        # 步骤1：问题理解
        step1_result = await self.steps[0].execute(self.ctx)
        steps_results.append(step1_result)
        queries = step1_result.output_data.get("rewritten_queries", []) if step1_result.output_data else []

        # 检测元描述并回退
        _META_DESCRIPTION_PREFIXES = (
            "问题类型：", "核心需求：", "关键实体：", "问题分类：",
            "问题类型:", "核心需求:", "关键实体:", "问题分类:",
        )
        _has_meta_queries = any(q.startswith(_META_DESCRIPTION_PREFIXES) for q in queries)
        if _has_meta_queries:
            logger.warning(
                f"[{self.ctx.domain}] 步骤1 输出了元描述而非检索词，已丢弃",
                meta_queries=queries,
            )
            queries = [self.ctx.request.message]
        else:
            if self.ctx.request.message not in queries:
                queries.append(self.ctx.request.message)

        # 步骤2：安全审查
        step2_result, safety_result = await self.steps[1].execute(self.ctx)
        steps_results.append(step2_result)

        if not safety_result.can_proceed:
            warning_response = self._build_fallback_response(safety_result)
            return ChatResponse(
                message=warning_response,
                conversation_id=self.ctx.conversation_id,
                steps=[s.model_dump() for s in steps_results],
                documents_used=[],
                safety_passed=False,
                stream_available=True,
            ), steps_results

        # 步骤3：知识检索（与图查询并行）
        step3_result, documents = await self.steps[2].execute(self.ctx)
        steps_results.append(step3_result)
        documents_used = [doc["content"][:_MAX_DOC_CHARS] for doc in documents[:5]]

        rag_context = "\n\n".join(documents_used)
        if not rag_context:
            if self.ctx.graph_context:
                rag_context = "（商品详情检索无结果，请参考下方商品关系图数据进行推荐）"
            else:
                rag_context = "暂无相关检索结果"

        # 步骤4：回答生成
        step4_result, response, quality_evaluation = await self.steps[3].execute(
            self.ctx, safety_result=safety_result, rag_context=rag_context, graph_context=self.ctx.graph_context
        )
        steps_results.append(step4_result)

        # 缓存高质量答案
        if step4_result.status == "success" and quality_evaluation.get("is_solved"):
            await self._store_to_cache(response)

        output_filter_safe = quality_evaluation.get("output_filter_safe", True)
        final_safety_passed = safety_result.is_safe and output_filter_safe

        return ChatResponse(
            message=response,
            conversation_id=self.ctx.conversation_id,
            steps=[s.model_dump() for s in steps_results],
            documents_used=documents_used,
            safety_passed=final_safety_passed,
            stream_available=True,
            cache_hit=False,
        ), steps_results

    def _build_fallback_response(self, safety_result) -> str:
        from src.modules.chat.agent.prompts import WARNING_TEMPLATES
        if not safety_result.is_safe:
            return WARNING_TEMPLATES.get("default", "").format(
                risk_categories=", ".join(safety_result.risk_categories) or "敏感内容",
                warning_message=safety_result.warning_message or "建议咨询专业人员",
            )
        return "抱歉，暂无法处理您的请求。"

    async def _store_to_cache(self, response: str) -> None:
        if not self.ctx.redis_cache_service or not self.ctx.redis_cache_service.is_available:
            return
        try:
            await self.ctx.redis_cache_service.store_conversation(
                conversation_id=self.ctx.conversation_id,
                question=self.ctx.request.message,
                answer=response,
                embedding=self.ctx.question_embedding,
            )
        except Exception as e:
            logger.warning(f"[{self.ctx.domain}] 存储缓存失败: {str(e)[:100]}")


_MAX_DOC_CHARS = 800
