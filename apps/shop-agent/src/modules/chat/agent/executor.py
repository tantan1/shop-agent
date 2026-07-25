"""
通用Agent执行器
支持多领域配置，通过domain参数切换不同业务场景

检索说明：
    - 使用向量数据库原生混合检索（Dense + Sparse BM25），兼容 Milvus 与 PgVector
    - 无需手动实现 BM25 和 RRF 融合

架构：执行器负责 Langfuse/缓存/响应构建等横切逻辑，业务步骤委托给 Pipeline。
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from src.modules.chat.agent.pipeline import Pipeline
from src.modules.chat.agent.schemas import AgentStepResult
from src.modules.chat.agent.steps.base import AgentContext
from src.modules.chat.core.embedding_service import EmbeddingService
from src.modules.chat.core.graph_service import NebulaGraphService
from src.modules.chat.core.llm_service import LLMService
from src.modules.chat.core.redis_cache_service import RedisCacheService
from src.modules.monitoring.langchain_callback import get_prometheus_callback
from src.modules.monitoring.langfuse_callback import create_langfuse_handler

if TYPE_CHECKING:
    from src.modules.chat.core.milvus_service import MilvusService
    from src.modules.chat.core.pgvector_service import PgVectorService
    from src.modules.chat.schemas import ChatRequest, ChatResponse

    VectorStoreService = Union[MilvusService, PgVectorService]

from src.modules.chat.agent.prompts import GUIDANCE_TEMPLATES, WARNING_TEMPLATES
from src.modules.chat.config import AgentConfig, get_agent_config
from src.shared.logger import APILogger

logger = APILogger("general_agent_executor")

# 类常量定义
_MAX_DOC_CHARS = 800


class GeneralAgentExecutor:
    """通用 Agent 执行器（Pipeline 编排器）。"""

    def __init__(
        self,
        *,
        domain: str = "ecommerce",
        agent_config: Optional[AgentConfig] = None,
        llm_service: Optional[LLMService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        milvus_service: Optional[VectorStoreService] = None,
        redis_cache_service: Optional[RedisCacheService] = None,
    ):
        self.domain = domain
        self.config = agent_config or get_agent_config(domain)

        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.milvus_service = milvus_service
        self.redis_cache_service = redis_cache_service

    @property
    def agent_name(self) -> str:
        return self.config.name

    @property
    def agent_description(self) -> str:
        return self.config.description

    async def _get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        if self.redis_cache_service and self.redis_cache_service.is_available:
            return self.redis_cache_service.get_chat_messages(
                conversation_id, max_turns=self.config.max_history_turns
            )
        return []

    async def _add_to_history(self, conversation_id: str, role: str, content: str):
        if self.redis_cache_service and self.redis_cache_service.is_available:
            content_clean = content.strip()[:4096]
            self.redis_cache_service.add_chat_message(
                conversation_id, role, content_clean,
                max_turns=self.config.max_history_turns, expire_days=1,
            )

    async def _query_graph_context(self, user_question: str) -> str:
        import os
        enabled = os.getenv("NEBULA_GRAPH_ENABLED", "true").lower() == "true"
        if not enabled:
            return ""
        try:
            graph = NebulaGraphService.get_instance()
            context = await graph.query_and_build_context(user_question)
            if context:
                logger.info(
                    f"[{self.domain}] 图查询命中",
                    graph_context_len=len(context),
                )
            return context
        except Exception as e:
            logger.debug(f"[{self.domain}] 图查询兜底跳过: {str(e)[:80]}")
            return ""

    async def _generate_fallback_response(self, user_question: str, safety_result) -> str:
        if not safety_result.is_safe:
            return WARNING_TEMPLATES.get("default", "").format(
                risk_categories=", ".join(safety_result.risk_categories) or "敏感内容",
                warning_message=safety_result.warning_message or "建议咨询专业人员",
            )
        return GUIDANCE_TEMPLATES.get("default", "").format(user_question=user_question)

    async def execute(
        self,
        request: ChatRequest,
        langfuse_handler=None,
    ) -> ChatResponse:
        """执行完整的 Agent 流程（Langfuse → Pipeline → 缓存 → 响应构建）。"""
        callback = get_prometheus_callback()
        executor_start_time = time.time()
        conversation_id = request.conversation_id or f"conv_{int(time.time())}"

        # Langfuse v4.x: 外部传入 handler 时复用，否则内部创建
        if langfuse_handler is None:
            result = create_langfuse_handler(
                session_id=conversation_id,
                tags=[request.domain, "agent-chat"],
                trace_name=f"{request.domain}-agent-chat",
                metadata={"domain": request.domain},
            )
            if result:
                self._langfuse_handler, self._langfuse_ctx = result
                self._langfuse_ctx.__enter__()
            else:
                self._langfuse_handler = None
                self._langfuse_ctx = None
        else:
            self._langfuse_handler = langfuse_handler
            self._langfuse_ctx = None

        steps: List[AgentStepResult] = []
        documents_used: List[str] = []
        question_embedding: Optional[List[float]] = None

        try:
            # 预计算向量（供 step3 复用，避免重复调 embedding API）
            if self.embedding_service:
                question_embedding = await self.embedding_service.embed_query(request.message)

            # 构建 Pipeline 上下文并执行
            ctx = AgentContext(
                request=request,
                domain=self.domain,
                config=self.config,
                llm_service=self.llm_service,
                embedding_service=self.embedding_service,
                milvus_service=self.milvus_service,
                redis_cache_service=self.redis_cache_service,
                langfuse_handler=self._langfuse_handler,
                conversation_id=conversation_id,
                question_embedding=question_embedding,
            )

            # 图查询与 Pipeline 并行执行
            graph_task = self._query_graph_context(request.message)
            pipeline = Pipeline(ctx)
            response, steps_results = await pipeline.run()
            graph_context = await graph_task

            # 合并图上下文到响应
            if graph_context and response:
                response.graph_context = graph_context  # type: ignore

            steps = [s.model_dump() for s in steps_results]
            documents_used = response.documents_used or []

            # 记录业务事件
            logger.log_business_event(
                f"{self.agent_name}对话",
                success=True,
                domain=self.domain,
                conversation_id=conversation_id,
                quality_score=response.metadata.get("quality_score") if hasattr(response, 'metadata') else None,
                is_solved=response.metadata.get("is_solved") if hasattr(response, 'metadata') else None,
            )

            executor_duration = time.time() - executor_start_time
            callback.on_chain_end(
                outputs={"status": "success", "steps": len(steps)},
                run_id=f"executor_{conversation_id}",
            )
            logger.info(
                f"[{self.domain}] 执行器完成",
                duration_ms=int(executor_duration * 1000),
                steps=len(steps),
            )

            return response

        except Exception as e:
            logger.error(f"[{self.domain}] Agent执行失败: {str(e)}")
            logger.log_business_event(
                f"{self.agent_name}对话",
                success=False,
                domain=self.domain,
                conversation_id=conversation_id,
                error=str(e),
            )
            callback.on_chain_error(error=e, run_id=f"executor_{conversation_id}")

            return ChatResponse(
                message="抱歉，服务暂时繁忙，请稍后重试。",
                conversation_id=conversation_id,
                steps=steps,
                documents_used=documents_used,
                safety_passed=False,
                stream_available=True,
            )
        finally:
            if self._langfuse_ctx:
                self._langfuse_ctx.__exit__(None, None, None)


# 向后兼容的类型别名
HospitalAgentExecutor = GeneralAgentExecutor
