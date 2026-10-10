"""
Agent 编排器
负责意图识别 → 路由分发 → ReAct Agent / 直接 Tool / RAG 流程
"""

from __future__ import annotations

import time as _time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generator

from src.core.config import config
from src.core.token_estimator import get_token_estimator
from src.modules.chat.agent.dispute_coordinator import DisputeCoordinator
from src.modules.chat.agent.orchestrator_remote import handle_remote_intent
from src.modules.chat.agent.react_agent import ReActRunContext
from src.modules.chat.config import chat_config
from src.modules.chat.core.sentiment_service import (
    EmotionResult,
    SentimentService,
)
from src.modules.chat.core.intent.candidate import ExecutionPlan
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
    user_id: str = ""
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
        self._sentiment_service = SentimentService()
        self._dispute_coordinator = DisputeCoordinator(
            llm=self._llm_service,
            tool_service=self._tool_service,
        )

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
        # TODO(GB-SDK): PipelineOverrides.synonym_normalize_enabled 待接入（实验级覆盖同义词归一化开关）
        return getattr(chat_config, "synonym_normalize_enabled", True)

    async def _normalize_input(self, message: str, domain: str) -> str:
        if not self._get_normalize_enabled():
            return message
        try:
            return await self._input_normalizer.normalize(message, domain)
        except Exception as e:
            logger.warning(f"同义词归一化异常，回退原文本: {str(e)[:100]}")
            return message

    async def _normalize_and_truncate(self, request: ChatRequest, domain: str) -> dict:
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
        return {
            "request": request,
            "orig_tokens": orig_tokens,
            "trunc_tokens": trunc_tokens,
            "was_truncated": _was_truncated,
            "t_norm": t_norm,
        }

    async def _detect_emotion(self, request: ChatRequest) -> tuple[EmotionResult | None, ChatResponse | None]:
        emotion_result: EmotionResult | None = None
        try:
            emotion_result = await self._sentiment_service.detect(
                request.message,
                session_id=request.conversation_id or "",
                skip_cloud=True,
            )
        except Exception:
            logger.debug("情绪检测异常，跳过", exc_info=True)

        if emotion_result and emotion_result.is_emergency:
            conversation_id = request.conversation_id or f"conv_{int(_time.time())}"
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
                domain=getattr(request, "domain", "ecommerce"),
                status="escalated",
            )
            return emotion_result, response
        return emotion_result, None

    async def _resolve_intent(self, request: ChatRequest) -> tuple[IntentResult, list]:
        forced_skill = getattr(request, "skill_id", None)
        if forced_skill:
            intent_result = IntentResult(
                plan=ExecutionPlan(
                    mode="direct_tool",
                    skill=forced_skill,
                    confidence=1.0,
                    reason=f"调用方硬路由指定 skill={forced_skill}，跳过意图识别",
                ),
                action=forced_skill,
                params=getattr(request, "context", None) or None,
            )
            intent_steps = [{
                "step_name": "意图识别（skill_id 硬路由）",
                "step_order": 0,
                "status": "success",
                "output_data": intent_result.model_dump(),
            }]
            logger.info(
                "A2A/MCP 硬路由生效，跳过意图识别",
                skill_id=forced_skill,
                conversation_id=request.conversation_id,
            )
            return intent_result, intent_steps

        # TODO(GB-SDK): PipelineOverrides.intent_recognition_mode 待接入
        #               （当前 _intent_recognizer.recognize 不接收 mode 参数，无 local/llm 显式分流入口）
        intent_result = await self._intent_recognizer.recognize(
            request.message,
            langfuse_handler=getattr(self, "_langfuse_handler", None),
        )
        intent_steps = [{
            "step_name": "意图识别",
            "step_order": 0,
            "status": "success",
            "output_data": intent_result.model_dump(),
        }]
        return intent_result, intent_steps

    async def _preprocess_request(self, request: ChatRequest, domain: str) -> dict:
        ctx = await self._normalize_and_truncate(request, domain)
        emotion_result, emergency_response = await self._detect_emotion(ctx["request"])
        if emergency_response:
            return {
                "escalated": True,
                "response": emergency_response,
                "intent_result": None,
                "truncated": ctx["was_truncated"],
                "orig_tokens": ctx["orig_tokens"],
                "trunc_tokens": ctx["trunc_tokens"],
                "emotion_result": emotion_result,
                "intent_steps": [],
                "t_norm": ctx["t_norm"],
                "path": "escalation",
            }

        intent_result, intent_steps = await self._resolve_intent(ctx["request"])
        return {
            "escalated": False,
            "response": None,
            "intent_result": intent_result,
            "truncated": ctx["was_truncated"],
            "orig_tokens": ctx["orig_tokens"],
            "trunc_tokens": ctx["trunc_tokens"],
            "emotion_result": emotion_result,
            "intent_steps": intent_steps,
            "t_norm": ctx["t_norm"],
            "path": "normal",
        }


    async def _route_intent(self, ctx: AgentRoutingContext, experiment_assignment=None) -> ChatResponse:
        # 路由只读 plan.mode（对齐 yaml_flow node.type），不再读 deprecated 的
        # intent / complexity。rag_pipeline → RAG；direct_tool / react → 远程 API
        # 链路（其内部再按 mode 分流到 Tool 调用或 ReAct）。
        if ctx.intent_result.mode in ("direct_tool", "react") and ctx.intent_result.action:
            return await handle_remote_intent(self, ctx)
        return await self._chat_with_rag_agent(
            ctx.request, ctx.domain, ctx.user_id,
            langfuse_handler=ctx.langfuse_handler,
            input_truncated=ctx.input_truncated,
            experiment_assignment=experiment_assignment,
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
        from src.modules.chat.agent.postgres_approval_store import PostgresApprovalStore
        from src.modules.chat.agent.postgres_execution_store import PostgresExecutionStore
        from src.modules.chat.agent.react_agent import ReActAgent
        from src.shared.database import get_async_session

        async with get_async_session() as db:
            execution_store = PostgresExecutionStore(db)
            approval_store = PostgresApprovalStore(db)
            react = ReActAgent(
                llm_service=self._llm_service,
                tool_service=self._tool_service,
                embedding_service=self._embedding_service,
                milvus_service=self._milvus_service,
                emotion_result=ctx.emotion_result,
                input_truncated=ctx.input_truncated,
                approval_store=approval_store,
            )
            return await react.run(ReActRunContext(
                request=ctx.request,
                intent_result=ctx.intent_result,
                conversation_id=ctx.request.conversation_id or "",
                user_id=ctx.user_id,
                domain=ctx.domain,
                intent_steps=ctx.intent_steps or [],
                langfuse_handler=ctx.langfuse_handler,
            ))

    @staticmethod
    @contextmanager
    def _langfuse_span(handler: Any, ctx: Any) -> Generator[Any, None, None]:
        if ctx is not None:
            ctx.__enter__()
        try:
            yield handler
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)

    async def chat_with_agent(self, request: ChatRequest, experiment_assignment=None) -> ChatResponse:
        from src.modules.monitoring.langfuse_callback import create_langfuse_handler

        domain = getattr(request, "domain", "ecommerce")
        user_id = getattr(request, "user_id", "") or f"anon_{int(_time.time())}"
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

        chat_status = "error"
        try:
            with self._langfuse_span(langfuse_handler, langfuse_ctx):
                # Langfuse OTEL 语义约定：根 span 上的 input.value / output.value 会被
                # 映射为 trace 顶层 input/output。FastAPI 自动根 span 只带 http.* 属性，
                # 不补这两个属性时 Langfuse 显示 "didn't receive an input or output"。
                # 内容会在导出前经 PiiRedactionSpanProcessor 自动脱敏。
                from opentelemetry import trace as _otel_trace

                _root_span = _otel_trace.get_current_span()
                _root_span.set_attribute("input.value", request.message or "")
                _root_span.set_attribute("input.mime_type", "text/plain")

                preprocess_result = await self._preprocess_request(request, domain)
                if preprocess_result.get("escalated"):
                    _root_span.set_attribute(
                        "output.value",
                        getattr(preprocess_result["response"], "message", "") or "",
                    )
                    _root_span.set_attribute("output.mime_type", "text/plain")
                    chat_status = "success"
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
                    user_id=user_id,
                    langfuse_handler=langfuse_handler,
                    emotion_result=emotion_result,
                    input_truncated=_was_truncated,
                    intent_steps=preprocess_result.get("intent_steps", []),
                )
                response = await self._route_intent(routing_ctx, experiment_assignment=experiment_assignment)
                t_intent = (_time.perf_counter() - t0) * 1000

                response = self._inject_response_metadata(
                    response, _was_truncated, orig_tokens, trunc_tokens, experiment_assignment
                )

                # 异步触发 L3 记忆提取（不阻塞响应）
                try:
                    from src.modules.chat.core.memory_extraction_trigger import ExtractionContext, MemoryExtractionTrigger
                    trigger = MemoryExtractionTrigger(self._llm_service)
                    import asyncio
                    task = trigger.try_extract(
                        ExtractionContext(
                            user_id=user_id,
                            conversation_id=conversation_id,
                            chat_history=[],
                            user_message=request.message,
                            last_intent=intent_result.action,
                            is_ended=True,
                            turn_number=getattr(request, "turn_number", 0),
                        )
                    )

                    async def _safe_memory_extract(coro):
                        try:
                            # 后台 L3 记忆提取：超时与异常均隔离，不阻塞/拖垮主请求
                            await asyncio.wait_for(coro, timeout=config.AGENT_TIMEOUT)
                        except Exception as ex:  # noqa: BLE001
                            logger.warning("L3 记忆提取失败（已忽略）", error=str(ex))

                    asyncio.create_task(_safe_memory_extract(task))
                except Exception:
                    pass

                t_overall = (_time.perf_counter() - t_overall_start) * 1000
                logger.debug(
                    "Agent编排耗时统计 [整体]",
                    duration_total_ms=round(t_overall, 1),
                    duration_norm_ms=round(preprocess_result.get("t_norm", 0), 1),
                    duration_intent_ms=round(t_intent, 1),
                    path=preprocess_result.get("path", "unknown"),
                )
                _root_span.set_attribute(
                    "output.value", getattr(response, "message", "") or ""
                )
                _root_span.set_attribute("output.mime_type", "text/plain")
                chat_status = "success"
                return response
        finally:
            # 无论 success / error / escalated 都计时并计数（Phase 5 护栏数据源）。
            elapsed_ms = (_time.perf_counter() - t_overall_start) * 1000
            try:
                from src.modules.monitoring.metrics import (
                    agent_chat_counter,
                    agent_chat_duration_ms,
                )
                agent_chat_duration_ms.observe(elapsed_ms)
                agent_chat_counter.labels(status=chat_status).inc()
            except Exception:  # noqa: BLE001
                pass

    async def _chat_with_rag_agent(
        self, request: ChatRequest, domain: str, user_id: str = "",
        langfuse_handler=None, input_truncated: bool = False,
        experiment_assignment=None,
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
        response = await executor.execute(
            request, langfuse_handler=langfuse_handler, user_id=user_id,
            experiment_assignment=experiment_assignment,
        )
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
