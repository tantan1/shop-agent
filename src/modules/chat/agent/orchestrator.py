"""
Agent 编排器
负责意图识别 → 路由分发 → ReAct Agent / 直接 Tool / RAG 流程
"""
from __future__ import annotations

import time as _time
from typing import TYPE_CHECKING, List

from langchain_core.documents import Document

from src.modules.chat.schemas import (
    ChatQueryRequest,
    ChatQueryResponse,
    ChatRequest,
    ChatResponse,
    IntentResult,
)
from src.modules.chat.core.content_filter import ContentFilterService
from src.modules.chat.core.synonym_normalizer import InputNormalizer
from src.modules.chat.core.sentiment_service import (
    SentimentService,
    EmotionResult,
)
from src.modules.chat.agent.dispute_coordinator import (
    DisputeCoordinator,
    should_use_dispute_coordinator,
)
from src.modules.chat.config import chat_config
from src.shared.exceptions import ValidationException
from src.shared.logger import APILogger
from src.core.token_estimator import get_token_estimator
from src.core.config import config

if TYPE_CHECKING:
    from src.modules.chat.core.llm_service import LLMService
    from src.modules.chat.core.embedding_service import EmbeddingService
    from src.modules.chat.core.milvus_service import MilvusService
    from src.modules.chat.core.intent_recognizer import IntentRecognizer
    from src.modules.chat.core.tool_registry import ToolService
    from src.modules.chat.core.redis_cache_service import RedisCacheService

logger = APILogger("agent_orchestrator")


class AgentOrchestrator:
    """Agent 编排器，将 ChatAgentService 中的业务流程抽离到此处"""

    def __init__(
        self,
        *,
        llm_service: LLMService,
        embedding_service: EmbeddingService,
        milvus_service: MilvusService,
        intent_recognizer: IntentRecognizer,
        tool_service: ToolService,
        redis_cache_service: RedisCacheService | None = None,
    ):
        self._llm_service: LLMService = llm_service
        self._embedding_service: EmbeddingService = embedding_service
        self._milvus_service: MilvusService = milvus_service
        self._intent_recognizer: IntentRecognizer = intent_recognizer
        self._tool_service: ToolService = tool_service
        self._redis_cache_service: RedisCacheService | None = redis_cache_service

        # 输入同义词归一化（L1+L2 默认开启，L3 按配置关闭）
        self._input_normalizer = InputNormalizer(llm_service=llm_service)

        # 情绪检测服务（级联 L1→L2→L3）
        self._sentiment_service: SentimentService | None = None

    # ═════════════════════════════════════════════════════════════════
    # 情绪检测（懒初始化）
    # ═════════════════════════════════════════════════════════════════

    def _ensure_sentiment_service(self) -> SentimentService:
        """懒初始化情绪检测服务（需要 local_model 引用，首次调用时注入）。"""
        if self._sentiment_service is None:
            from src.modules.chat.core.local_model_service import LocalModelService
            self._sentiment_service = SentimentService(
                local_model=LocalModelService.get_instance(),
                llm=self._llm_service,
            )
        return self._sentiment_service

    # ═════════════════════════════════════════════════════════════════
    # 纠纷协调器（懒初始化）
    # ═════════════════════════════════════════════════════════════════

    def _ensure_dispute_coordinator(self) -> DisputeCoordinator:
        """懒初始化纠纷协调器（首次纠纷触发时创建）。"""
        if not hasattr(self, '_dispute_coordinator') or self._dispute_coordinator is None:
            self._dispute_coordinator = DisputeCoordinator(
                llm=self._llm_service,
                tool_service=self._tool_service,
            )
        return self._dispute_coordinator

    # =========================================================================
    # 基础属性（从 ChatAgentService 迁移）
    # =========================================================================

    @property
    def embedding(self):
        """获取嵌入服务实例"""
        return self._embedding_service.get_embeddings()

    @property
    def embeddings(self):
        """获取嵌入服务（兼容旧代码）"""
        return self.embedding

    @property
    def milvus(self):
        return self._milvus_service

    @property
    def llm(self):
        return self._llm_service

    # =========================================================================
    # 输入归一化
    # =========================================================================

    def _get_normalize_enabled(self) -> bool:
        """检查是否启用同义词归一化"""
        return getattr(chat_config, 'synonym_normalize_enabled', True)

    async def _normalize_input(self, message: str, domain: str) -> str:
        """对用户输入执行同义词归一化（L1+L2，L3按配置）"""
        if not self._get_normalize_enabled():
            return message
        try:
            return await self._input_normalizer.normalize(message, domain)
        except Exception as e:
            logger.warning(
                f"同义词归一化异常，回退原文本: {str(e)[:100]}"
            )
            return message  # 归一化失败不阻塞流程

    # =========================================================================
    # 旧版 RAG 聊天流程
    # =========================================================================

    async def _search_similar_documents(self, query: str, top_k: int = 3) -> List[Document]:
        """搜索相似的文档"""
        try:
            query_embedding = await self.embeddings.aembed_query(query)
            documents = self.milvus.search_similar(query_embedding, top_k)
            return documents
        except Exception as e:
            import traceback
            logger.error(f"Document search failed: {str(e)}\n{traceback.format_exc()}")
            raise ValidationException("文档搜索失败", str(e))

    async def _generate_response(
        self, query: str, documents: List[Document], langfuse_handler=None
    ) -> str:
        """基于检索到的文档生成回答"""
        try:
            context = "\n\n".join([doc.page_content for doc in documents[:3]]) or "暂无相关信息"

            from datetime import datetime
            current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")

            from src.modules.chat.agent.prompts import PromptTemplateManager
            template = PromptTemplateManager.get("ecommerce", "ecommerce_step4_generate")

            if template:
                prompt_content = template.format(
                    graph_context="",
                    rag_context=context,
                    user_question=query,
                    current_time=current_time,
                    safety_check_result="风险等级: low",
                    safety_reminder="",
                    chat_history="",
                    product_info=context,
                    knowledge_base=context,
                    context=context,
                    category="",
                )
            else:
                prompt_content = f"基于以下信息回答用户问题。\n\n知识库：\n{context}\n\n问题：{query}\n\n回答："

            response = await self.llm.chat_qwen_with_prompt(
                prompt=prompt_content,
                system_prompt="你是电商公司的官方助手",
                langfuse_handler=langfuse_handler,
            )
            return response
        except Exception as e:
            import traceback
            logger.error(f"Response generation failed: {str(e)}\n{traceback.format_exc()}")
            return "抱歉，我暂时无法回答这个问题。"

    async def chat_rag(self, request: ChatQueryRequest) -> ChatQueryResponse:
        """RAG 聊天接口（旧版，简单检索→生成）"""
        from src.modules.monitoring.langfuse_callback import create_langfuse_handler

        conversation_id = getattr(request, "conversation_id", None) or "rag-default"
        domain = getattr(request, "domain", "ecommerce")

        # Langfuse v4.x: 每请求创建 handler + propagate_attributes 上下文
        result = create_langfuse_handler(
            session_id=conversation_id,
            tags=[domain, "chat-rag"],
            trace_name=f"{domain}-rag",
            metadata={"domain": domain, "type": "rag"},
        )
        langfuse_handler = None
        langfuse_ctx = None
        if result:
            langfuse_handler, langfuse_ctx = result
            langfuse_ctx.__enter__()

        try:
            # 1. 输入归一化 + 搜索相似文档
            search_query = await self._normalize_input(request.message, domain)
            similar_docs = await self._search_similar_documents(search_query)

            # 2. 生成回答（使用原始消息保留完整语义）
            response_text = await self._generate_response(
                request.message, similar_docs, langfuse_handler=langfuse_handler
            )

            # 3. 输出内容安全过滤（规则引擎，零 LLM 成本）
            cf = ContentFilterService.get_instance()
            output_check = cf.filter_output(response_text, domain)
            if not output_check.is_safe:
                logger.warning(
                    "RAG 输出安全检查未通过",
                    domain=domain,
                    risk_categories=output_check.risk_categories,
                )
                if output_check.filtered_text:
                    response_text = output_check.filtered_text
                else:
                    response_text = "抱歉，当前无法处理您的请求，请稍后重试。"

            # 4. 构建响应
            response = ChatQueryResponse(
                message=response_text,
                relevant_documents=[doc.page_content for doc in similar_docs],
                document_count=len(similar_docs),
            )

            logger.log_business_event(
                "RAG聊天查询",
                success=True,
                query=request.message,
                document_count=len(similar_docs),
                response_length=len(response_text),
            )
            return response
        except (ValidationException,):
            raise
        except Exception as e:
            logger.log_business_event(
                "RAG聊天查询",
                success=False,
                error=str(e),
                query=request.message,
            )
            raise ValidationException("聊天查询失败", str(e))
        finally:
            if langfuse_ctx:
                langfuse_ctx.__exit__(None, None, None)

    # =========================================================================
    # Agent 编排 —— ReAct 路径
    # =========================================================================

    async def _chat_with_react_agent(
        self,
        request: ChatRequest,
        intent_result: IntentResult,
        conversation_id: str,
        domain: str,
        intent_steps: list,
        langfuse_handler=None,
        emotion_result: EmotionResult | None = None,
        input_truncated: bool = False,
    ) -> ChatResponse:
        """
        将意图命中的复杂查询交给 ReAct Agent 处理。

        Agent 拿到意图上下文后自主决定：
        1. 是否需要先查 RAG（政策/规则类）
        2. 调用对应的 business tool（查订单/物流/退货/余额/优惠券）
        3. 将工具结果与 RAG 结果融合成最终回复
        """
        from src.modules.chat.agent.react_agent import ReActAgent

        react = ReActAgent(
            llm_service=self._llm_service,
            tool_service=self._tool_service,
            embedding_service=self._embedding_service,
            milvus_service=self._milvus_service,
            emotion_result=emotion_result,
            input_truncated=input_truncated,
        )

        return await react.run(
            request=request,
            intent_result=intent_result,
            conversation_id=conversation_id,
            domain=domain,
            intent_steps=intent_steps,
            langfuse_handler=langfuse_handler,
        )

    # =========================================================================
    # 主编排入口
    # =========================================================================

    async def chat_with_agent(self, request: ChatRequest) -> ChatResponse:
        """通用 Agent 多步骤对话（含意图识别 + 路由分发）"""
        from src.modules.monitoring.langfuse_callback import create_langfuse_handler

        domain = getattr(request, 'domain', 'ecommerce')
        conversation_id = request.conversation_id or f"conv_{int(_time.time())}"
        t_overall_start = _time.perf_counter()

        # ── Langfuse: 编排器根 trace（整条管道共享一个 handler）──
        result = create_langfuse_handler(
            session_id=conversation_id,
            tags=[domain, "agent-orchestrator"],
            trace_name=f"{domain}-orchestrator",
            metadata={"domain": domain, "type": "agent-orchestrator"},
        )
        langfuse_handler = None
        langfuse_ctx = None
        if result:
            langfuse_handler, langfuse_ctx = result
            langfuse_ctx.__enter__()

        try:
            # ===== 输入同义词归一化（L1+L2，零 LLM 成本）=====
            t_norm_start = _time.perf_counter()
            normalized_message = await self._normalize_input(request.message, domain)
            t_norm = (_time.perf_counter() - t_norm_start) * 1000
            if normalized_message != request.message:
                logger.info(
                    "输入同义词归一化",
                    original=request.message[:60],
                    normalized=normalized_message[:60],
                )

            # ===== 输入长度管控：Token 预算截断 =====
            # 使用 Qwen3 tokenizer 精确估算，而非字符数（中文 ≈ 1.5 token/字，英文 ≈ 0.3 token/字）
            t_trunc_start = _time.perf_counter()
            max_input_tokens = config.MAX_USER_MESSAGE_TOKENS
            truncation_strategy = config.TRUNCATION_STRATEGY
            _was_truncated = False  # 局部标记，最后注入 response

            token_estimator = get_token_estimator()
            truncated_text, orig_tokens, trunc_tokens = token_estimator.truncate_to_tokens(
                normalized_message,
                max_tokens=max_input_tokens,
                strategy=truncation_strategy,
            )
            t_trunc = (_time.perf_counter() - t_trunc_start) * 1000

            if orig_tokens > trunc_tokens:
                # 发生了截断：重建 request 对象（Pydantic model 是 frozen 的）
                _was_truncated = True
                request = request.model_copy(update={"message": truncated_text})
                logger.info(
                    "用户输入 token 超预算，已智能截断",
                    strategy=truncation_strategy,
                    original_tokens=orig_tokens,
                    truncated_tokens=trunc_tokens,
                    max_tokens=max_input_tokens,
                    duration_trunc_ms=round(t_trunc, 1),
                    preview=truncated_text[:60],
                )
            # 更新 normalized_message 为截断后的版本（意图识别使用）
            normalized_message = request.message

            # ===== 情绪检测（L1 规则 <1ms + L2 本地 ~30ms）=====
            t_emo_start = _time.perf_counter()
            emotion_result: EmotionResult | None = None
            try:
                sentiment_svc = self._ensure_sentiment_service()
                emotion_result = await sentiment_svc.detect(
                    request.message,
                    session_id=conversation_id,
                    skip_cloud=True,  # 默认跳过云端，保持延迟可控
                )
            except Exception:
                logger.debug("情绪检测异常，跳过", exc_info=True)

            t_emo = (_time.perf_counter() - t_emo_start) * 1000

            # 舆情风险 → 立即升级，不走后续管线
            if emotion_result and emotion_result.is_emergency:
                logger.warning(
                    "舆情风险检测，触发立即升级",
                    conversation_id=conversation_id,
                    level=emotion_result.level.name,
                    source=emotion_result.source,
                    keywords=emotion_result.keywords,
                    duration_emo_ms=round(t_emo, 1),
                )
                return ChatResponse(
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

            # ===== 意图识别 =====
            t0 = _time.perf_counter()
            intent_result = await self._intent_recognizer.recognize(
                normalized_message, langfuse_handler=langfuse_handler
            )
            t_intent = (_time.perf_counter() - t0) * 1000

            # 远程 API 意图 → _handle_remote_intent
            if intent_result.intent == "call_remote_api" and intent_result.action:
                response = await self._handle_remote_intent(
                    request, intent_result, domain,
                    langfuse_handler=langfuse_handler,
                    emotion_result=emotion_result,
                    input_truncated=_was_truncated,
                )
                t_overall = (_time.perf_counter() - t_overall_start) * 1000
                logger.debug(
                    "Agent编排耗时统计 [整体]",
                    duration_total_ms=round(t_overall, 1),
                    duration_norm_ms=round(t_norm, 1),
                    duration_intent_ms=round(t_intent, 1),
                    path="remote_api",
                )
                # 注入截断信息
                if _was_truncated:
                    response.input_truncated = True
                    response.input_original_tokens = orig_tokens
                    response.input_truncated_tokens = trunc_tokens
                return response

            # RAG Agent 兜底
            response = await self._chat_with_rag_agent(
                request, domain, langfuse_handler=langfuse_handler,
                input_truncated=_was_truncated,
            )
            t_overall = (_time.perf_counter() - t_overall_start) * 1000
            logger.debug(
                "Agent编排耗时统计 [整体]",
                duration_total_ms=round(t_overall, 1),
                duration_norm_ms=round(t_norm, 1),
                duration_intent_ms=round(t_intent, 1),
                path="rag",
            )
            # 注入截断信息
            if _was_truncated:
                response.input_truncated = True
                response.input_original_tokens = orig_tokens
                response.input_truncated_tokens = trunc_tokens
            return response
        finally:
            if langfuse_ctx:
                langfuse_ctx.__exit__(None, None, None)

# ---------------------------------------------------------------------------
# 私有方法：远程 API 意图处理
# ---------------------------------------------------------------------------

    async def _handle_remote_intent(
        self,
        request: ChatRequest,
        intent_result: IntentResult,
        domain: str,
        langfuse_handler=None,
        emotion_result=None,
        input_truncated: bool = False,
    ) -> ChatResponse:
        """处理 call_remote_api 意图：参数抽取 → 门控 → Tool/ReAct"""
        conversation_id = request.conversation_id or f"conv_{int(_time.time())}"
        t_handler_start = _time.perf_counter()

        intent_steps = [{
            "step_name": "意图识别",
            "step_order": 0,
            "status": "success",
            "output_data": intent_result.model_dump()
        }]

        # ---- 参数抽取 ----
        t0 = _time.perf_counter()
        extracted_params = await self._intent_recognizer.extract_params(
            request.message, intent_result.action, langfuse_handler=langfuse_handler
        )
        t_params = (_time.perf_counter() - t0) * 1000
        if extracted_params:
            existing = intent_result.params or {}
            intent_result.params = {**extracted_params, **existing}
            intent_steps.append({
                "step_name": "参数抽取",
                "step_order": 1,
                "status": "success",
                "output_data": {"extracted_params": extracted_params}
            })

        # ---- 纠纷协调检测（在 ReAct/Tool 之前） ----
        if should_use_dispute_coordinator(
            message=request.message,
            emotion_result=emotion_result,
            intent_action=intent_result.action,
        ):
            logger.info(
                "触发纠纷协调流程",
                action=intent_result.action,
                emotion=emotion_result.level.name if emotion_result else "unknown",
                params=intent_result.params,
            )
            t0 = _time.perf_counter()
            dispute_coordinator = self._ensure_dispute_coordinator()
            response = await dispute_coordinator.resolve(
                request=request,
                emotion_result=emotion_result,
                conversation_id=conversation_id,
                domain=domain,
                intent_steps=intent_steps,
                order_id=intent_result.params.get("order_id") if intent_result.params else None,
                langfuse_handler=langfuse_handler,
            )
            t_dispute = (_time.perf_counter() - t0) * 1000
            t_total = (_time.perf_counter() - t_handler_start) * 1000
            logger.debug(
                "Agent编排耗时统计 [remote_api→dispute]",
                duration_total_ms=round(t_total, 1),
                duration_params_ms=round(t_params, 1),
                duration_dispute_ms=round(t_dispute, 1),
                action=intent_result.action,
            )
            return response

        # ---- 复杂性门控 ----
        if intent_result.complexity == "multi_step":
            logger.info(
                "意图命中但需要ReAct",
                action=intent_result.action,
                complexity=intent_result.complexity,
                reason=intent_result.complexity_reason,
                params=intent_result.params,
            )
            t0 = _time.perf_counter()
            response = await self._chat_with_react_agent(
                request=request,
                intent_result=intent_result,
                conversation_id=conversation_id,
                domain=domain,
                intent_steps=intent_steps,
                langfuse_handler=langfuse_handler,
                emotion_result=emotion_result,
                input_truncated=input_truncated,
            )
            t_react = (_time.perf_counter() - t0) * 1000
            t_total = (_time.perf_counter() - t_handler_start) * 1000
            logger.debug(
                "Agent编排耗时统计 [remote_api→ReAct]",
                duration_total_ms=round(t_total, 1),
                duration_params_ms=round(t_params, 1),
                duration_react_ms=round(t_react, 1),
                action=intent_result.action,
            )
            return response

        # 简单意图：直接 tool dispatch
        t0 = _time.perf_counter()
        tool_response = await self._tool_service.dispatch(
            intent_result.action,
            intent_result.params
        )
        t_tool = (_time.perf_counter() - t0) * 1000

        # ---- 输出内容安全过滤（规则引擎，零 LLM 成本） ----
        output_filter_safe = True
        cf = ContentFilterService.get_instance()
        output_check = cf.filter_output(tool_response, domain)
        if not output_check.is_safe:
            output_filter_safe = False
            logger.warning(
                "Direct Tool 输出安全检查未通过",
                domain=domain,
                risk_categories=output_check.risk_categories,
            )
            if output_check.filtered_text:
                tool_response = output_check.filtered_text
            else:
                tool_response = "抱歉，当前无法处理您的请求，请稍后重试。"

        step_index = len(intent_steps)
        response = ChatResponse(
            message=tool_response,
            conversation_id=conversation_id,
            steps=intent_steps + [{
                "step_name": "Tool调用(直接)",
                "step_order": step_index,
                "status": "success",
                "output_data": {"action": intent_result.action, "params": intent_result.params}
            }],
            documents_used=[],
            safety_passed=output_filter_safe,
            stream_available=True,
            domain=domain,
        )

        t_total = (_time.perf_counter() - t_handler_start) * 1000
        logger.debug(
            "Agent编排耗时统计 [remote_api→direct_tool]",
            duration_total_ms=round(t_total, 1),
            duration_params_ms=round(t_params, 1),
            duration_tool_ms=round(t_tool, 1),
            action=intent_result.action,
        )
        logger.log_business_event(
            "电商Agent Tool直接调用",
            success=True,
            domain=domain,
            action=intent_result.action,
            params=list(intent_result.params.keys()) if intent_result.params else [],
            conversation_id=conversation_id,
            message_length=len(request.message),
            response_length=len(tool_response),
        )
        return response

    async def _chat_with_rag_agent(
        self,
        request: ChatRequest,
        domain: str,
        langfuse_handler=None,
        input_truncated: bool = False,
    ) -> ChatResponse:
        """RAG Agent 兜底流程"""
        from src.modules.chat.agent.executor import GeneralAgentExecutor

        executor = GeneralAgentExecutor(
            domain=domain,
            llm_service=self._llm_service,
            embedding_service=self._embedding_service,
            milvus_service=self._milvus_service,
            redis_cache_service=self._redis_cache_service,
        )

        # 截断信息注入：用户输入已被精简，提示 LLM 关注可能缺失的关键信息
        if input_truncated:
            request = request.model_copy(update={
                "message": (
                    "[系统提示：用户原始输入较长，已被自动精简，部分细节可能丢失。"
                    "如果回复时发现缺少关键信息（如订单号、手机号等），请主动询问用户补充。]\n\n"
                    + request.message
                )
            })

        response = await executor.execute(request, langfuse_handler=langfuse_handler)
        response.domain = domain

        logger.log_business_event(
            f"{executor.agent_name}对话",
            success=True,
            domain=domain,
            conversation_id=response.conversation_id,
            message_length=len(request.message),
            response_length=len(response.message),
            safety_passed=response.safety_passed
        )

        return response
