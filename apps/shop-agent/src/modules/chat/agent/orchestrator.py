"""
Agent 编排器
负责意图识别 → 路由分发 → ReAct Agent / 直接 Tool / RAG 流程
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.core.config import config
from src.core.token_estimator import get_token_estimator
from src.modules.chat.agent.dispute_coordinator import (
    DisputeCoordinator,
    should_use_dispute_coordinator,
)
from src.modules.chat.agent.orchestrator_history import persist_turn
from src.modules.chat.agent.orchestrator_params import (
    _prepare_intent_params,
)
from src.modules.chat.agent.orchestrator_rag import chat_rag, generate_response, search_similar_documents
from src.modules.chat.agent.orchestrator_remote import (
    execute_direct_tool_flow,
    execute_react_flow,
    handle_remote_intent,
    try_dispute_flow,
)
from src.modules.chat.agent.react_agent import ReActRunContext
from src.modules.chat.config import chat_config
from src.modules.chat.core.sentiment_service import (
    EmotionResult,
    SentimentService,
)
from src.modules.chat.core.synonym_normalizer import InputNormalizer
from src.modules.chat.schemas import (
    ChatRequest,
    ChatResponse,
    IntentResult,
)
from src.shared.logger import APILogger

if TYPE_CHECKING:
    from src.modules.chat.core.embedding_service import EmbeddingService
    from src.modules.chat.core.intent_recognizer import IntentRecognizer
    from src.modules.chat.core.llm_service import LLMService
    from src.modules.chat.core.milvus_service import MilvusService
    from src.modules.chat.core.pgvector_service import PgVectorService
    from src.modules.chat.core.redis_cache_service import RedisCacheService
    from src.modules.chat.core.tool_registry import ToolService

    VectorStoreService = Union[MilvusService, PgVectorService]

logger = APILogger("agent_orchestrator")


@dataclass
class AgentRoutingContext:
    """编排器路由上下文，封装 remote_api / RAG 公共参数。"""

    request: ChatRequest
    intent_result: IntentResult
    domain: str
    langfuse_handler: Any = None
    emotion_result: Any = None
    input_truncated: bool = False
    intent_steps: list | None = None


class AgentOrchestrator:
    """Agent 编排器，将 ChatAgentService 中的业务流程抽离到此处"""

    def __init__(
        self,
        *,
        llm_service: LLMService,
        embedding_service: EmbeddingService,
        milvus_service: VectorStoreService,
        intent_recognizer: IntentRecognizer,
        tool_service: ToolService,
        redis_cache_service: RedisCacheService | None = None,
    ):
        self._llm_service: LLMService = llm_service
        self._embedding_service: EmbeddingService = embedding_service
        self._milvus_service: VectorStoreService = milvus_service
        self._intent_recognizer: IntentRecognizer = intent_recognizer
        self._tool_service: ToolService = tool_service
        self._redis_cache_service: RedisCacheService | None = redis_cache_service

        self._input_normalizer = InputNormalizer(llm_service=llm_service)
        self._sentiment_service: SentimentService | None = None

    def _ensure_sentiment_service(self) -> SentimentService:
        """懒初始化情绪检测服务"""
        if self._sentiment_service is None:
            from src.modules.chat.core.local_model_service import LocalModelService

            self._sentiment_service = SentimentService(
                local_model=LocalModelService.get_instance(),
                llm=self._llm_service,
            )
        return self._sentiment_service

    def _ensure_dispute_coordinator(self) -> DisputeCoordinator:
        """懒初始化纠纷协调器"""
        if not hasattr(self, "_dispute_coordinator") or self._dispute_coordinator is None:
            self._dispute_coordinator = DisputeCoordinator(
                llm=self._llm_service,
                tool_service=self._tool_service,
            )
        return self._dispute_coordinator

    @property
    def embedding(self):
        return self._embedding_service.get_embeddings()

    @property
    def embeddings(self):
        return self.embedding

    @property
    def milvus(self):
        return self._milvus_service

    @property
    def llm(self):
        return self._llm_service

    def _get_normalize_enabled(self) -> bool:
        return getattr(chat_config, "synonym_normalize_enabled", True)

    async def _normalize_input(self, message: str, domain: str) -> str:
        if not self._get_normalize_enabled():
            return message
        try:
            return await self._input_normalizer.normalize(message, domain)
        except Exception as e:
            logger.warning(f"同义词归一化异常，回退原文本: {str(e)[:100]}")
            return message

    async def _preprocess_request(self, request: ChatRequest, domain: str) -> dict:
        normalized_message = await self._normalize_input(request.message, domain)

        t_norm_start = _time.perf_counter()
        max_input_tokens = config.MAX_USER_MESSAGE_TOKENS
        truncation_strategy = config.TRUNCATION_STRATEGY
        token_estimator = get_token_estimator()
        truncated_text, orig_tokens, trunc_tokens = token_estimator.truncate_to_tokens(
            normalized_message,
            max_tokens=max_input_tokens,
            strategy=truncation_strategy,
        )
        t_norm = (_time.perf_counter() - t_norm_start) * 1000

        _was_truncated = orig_tokens > trunc_tokens
        if _was_truncated:
            request = request.model_copy(update={"message": truncated_text})
            logger.info(
                "用户输入 token 超预算，已智能截断",
                strategy=truncation_strategy,
                original_tokens=orig_tokens,
                truncated_tokens=trunc_tokens,
                max_tokens=max_input_tokens,
                preview=truncated_text[:60],
            )
        normalized_message = request.message

        emotion_result: EmotionResult | None = None
        try:
            sentiment_svc = self._ensure_sentiment_service()
            emotion_result = await sentiment_svc.detect(
                request.message,
                session_id=request.conversation_id or "",
                skip_cloud=True,
            )
        except Exception:
            logger.debug("情绪检测异常，跳过", exc_info=True)

        conversation_id = request.conversation_id or f"conv_{int(_time.time())}"
        if emotion_result and emotion_result.is_emergency:
            response = ChatResponse(
                message=(
                    "非常抱歉给您带来了不好的体验，我们已经将您的问题升级给高级专员处理，"
                    "专员将尽快通过电话或在线客服与您联系。如有紧急问题，"
                    "请拨打客服热线 400-XXX-XXXX。"
                ),
                conversation_id=conversation_id,
                steps=[{
                    "step_name": "情绪检测-舆情升级",
                    "step_order": 0,
                    "status": "escalated",
                    "output_data": {
                        "level": emotion_result.level.name,
                        "keywords": emotion_result.keywords,
                    },
                }],
                documents_used=[],
                safety_passed=True,
                stream_available=True,
                domain=domain,
                status="escalated",
                input_truncated=_was_truncated,
                input_original_tokens=orig_tokens if _was_truncated else None,
                input_truncated_tokens=trunc_tokens if _was_truncated else None,
            )
            return {
                "escalated": True,
                "response": response,
                "intent_result": None,
                "truncated": _was_truncated,
                "orig_tokens": orig_tokens,
                "trunc_tokens": trunc_tokens,
                "emotion_result": emotion_result,
                "intent_steps": [],
                "t_norm": t_norm,
                "path": "escalation",
            }

        intent_result = await self._intent_recognizer.recognize(
            normalized_message, langfuse_handler=getattr(self, "_langfuse_handler", None)
        )
        intent_steps = [{
            "step_name": "意图识别",
            "step_order": 0,
            "status": "success",
            "output_data": intent_result.model_dump(),
        }]

        return {
            "escalated": False,
            "response": None,
            "intent_result": intent_result,
            "truncated": _was_truncated,
            "orig_tokens": orig_tokens,
            "trunc_tokens": trunc_tokens,
            "emotion_result": emotion_result,
            "intent_steps": intent_steps,
            "t_norm": t_norm,
            "path": "normal",
        }

    async def _route_intent(self, ctx: AgentRoutingContext) -> ChatResponse:
        if ctx.intent_result.intent == "call_remote_api" and ctx.intent_result.action:
            return await handle_remote_intent(self, ctx)
        return await self._chat_with_rag_agent(
            ctx.request, ctx.domain,
            langfuse_handler=ctx.langfuse_handler,
            input_truncated=ctx.input_truncated,
        )

    def _inject_response_metadata(
        self, response: ChatResponse, truncated: bool,
        orig_tokens: int, trunc_tokens: int, experiment_assignment=None,
    ) -> ChatResponse:
        if truncated:
            response.input_truncated = True
            response.input_original_tokens = orig_tokens
            response.input_truncated_tokens = trunc_tokens
        if experiment_assignment:
            response.experiment_group = experiment_assignment.variant_name
        return response

    async def _chat_with_react_agent(self, ctx: AgentRoutingContext) -> ChatResponse:
        from src.modules.chat.agent.react_agent import ReActAgent

        react = ReActAgent(
            llm_service=self._llm_service,
            tool_service=self._tool_service,
            embedding_service=self._embedding_service,
            milvus_service=self._milvus_service,
            emotion_result=ctx.emotion_result,
            input_truncated=ctx.input_truncated,
        )
        return await react.run(ReActRunContext(
            request=ctx.request,
            intent_result=ctx.intent_result,
            conversation_id=ctx.request.conversation_id or "",
            domain=ctx.domain,
            intent_steps=ctx.intent_steps or [],
            langfuse_handler=ctx.langfuse_handler,
        ))

    async def chat_with_agent(self, request: ChatRequest, experiment_assignment=None) -> ChatResponse:
        from src.modules.monitoring.langfuse_callback import create_langfuse_handler

        domain = getattr(request, "domain", "ecommerce")
        conversation_id = request.conversation_id or f"conv_{int(_time.time())}"
        t_overall_start = _time.perf_counter()

        exp_tags = [domain, "agent-orchestrator"]
        exp_metadata = {"domain": domain, "type": "agent-orchestrator"}
        if experiment_assignment is not None:
            exp_tags.extend(experiment_assignment.to_tags())
            exp_metadata["experiment"] = experiment_assignment.to_metadata()

        result = create_langfuse_handler(
            session_id=conversation_id,
            tags=exp_tags,
            trace_name=f"{domain}-orchestrator",
            metadata=exp_metadata,
        )
        langfuse_handler = None
        langfuse_ctx = None
        if result:
            langfuse_handler, langfuse_ctx = result
            langfuse_ctx.__enter__()

        try:
            preprocess_result = await self._preprocess_request(request, domain)
            if preprocess_result.get("escalated"):
                return preprocess_result["response"]

            intent_result = preprocess_result["intent_result"]
            _was_truncated = preprocess_result["truncated"]
            orig_tokens = preprocess_result["orig_tokens"]
            trunc_tokens = preprocess_result["trunc_tokens"]
            emotion_result = preprocess_result["emotion_result"]

            t0 = _time.perf_counter()
            routing_ctx = AgentRoutingContext(
                request=request,
                intent_result=intent_result,
                domain=domain,
                langfuse_handler=langfuse_handler,
                emotion_result=emotion_result,
                input_truncated=_was_truncated,
                intent_steps=preprocess_result.get("intent_steps", []),
            )
            response = await self._route_intent(routing_ctx)
            t_intent = (_time.perf_counter() - t0) * 1000

            response = self._inject_response_metadata(
                response, _was_truncated, orig_tokens, trunc_tokens, experiment_assignment
            )

            t_overall = (_time.perf_counter() - t_overall_start) * 1000
            logger.debug(
                "Agent编排耗时统计 [整体]",
                duration_total_ms=round(t_overall, 1),
                duration_norm_ms=round(preprocess_result.get("t_norm", 0), 1),
                duration_intent_ms=round(t_intent, 1),
                path=preprocess_result.get("path", "unknown"),
            )
            return response
        finally:
            if langfuse_ctx:
                langfuse_ctx.__exit__(None, None, None)

    async def _chat_with_rag_agent(
        self, request: ChatRequest, domain: str,
        langfuse_handler=None, input_truncated: bool = False,
    ) -> ChatResponse:
        from src.modules.chat.agent.executor import GeneralAgentExecutor

        executor = GeneralAgentExecutor(
            domain=domain,
            llm_service=self._llm_service,
            embedding_service=self._embedding_service,
            milvus_service=self._milvus_service,
            redis_cache_service=self._redis_cache_service,
        )
        if input_truncated:
            request = request.model_copy(
                update={
                    "message": (
                        "[系统提示：用户原始输入较长，已被自动精简，部分细节可能丢失。"
                        "如果回复时发现缺少关键信息（如订单号、手机号等），请主动询问用户补充。]\n\n"
                        + request.message
                    )
                }
            )
        response = await executor.execute(request, langfuse_handler=langfuse_handler)
        response.domain = domain
        logger.log_business_event(
            f"{executor.agent_name}对话",
            success=True,
            domain=domain,
            conversation_id=response.conversation_id,
            message_length=len(request.message),
            response_length=len(response.message),
            safety_passed=response.safety_passed,
        )
        return response
