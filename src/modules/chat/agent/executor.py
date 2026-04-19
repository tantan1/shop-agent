"""
通用Agent执行器
支持多领域配置，通过domain参数切换不同业务场景
"""

import json
import time
import asyncio
from typing import AsyncGenerator, Optional, List, Dict, Any, Union

from src.modules.chat.core.llm_service import LLMService
from src.modules.chat.core.embedding_service import EmbeddingService
from src.modules.chat.core.milvus_service import MilvusService
from src.modules.chat.core.redis_cache_service import RedisCacheService

from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, AIMessage
from src.modules.chat.agent.prompts import PromptTemplateManager
from src.modules.chat.agent.schemas import (
    AgentStepResult,
    SafetyCheckResult,
    QuestionRewriteResult,
    RetrievalResult,
)
from src.modules.chat.schemas import (
    HospitalChatRequest,
    HospitalChatResponse,
)
from src.modules.chat.config import AgentConfig, get_agent_config
from src.shared.logger import APILogger

logger = APILogger("general_agent_executor")


# 类常量定义
_CLEANUP_INTERVAL = 10  # 每N次调用清理一次过期历史


class GeneralAgentExecutor:
    """
    通用Agent执行器
    
    支持通过domain参数切换不同业务领域：
    - medical: 医疗客服
    - ecommerce: 电商客服
    - customer_service: 通用客服
    - general: 通用助手
    """
    
    # 类变量：仅用于初始化锁（共享的异步锁）
    _history_lock: asyncio.Lock = None  # 异步锁
    
    def __init__(
        self,
        domain: str = "medical",
        agent_config: Optional[AgentConfig] = None,
        llm_service: Optional[LLMService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        milvus_service: Optional[MilvusService] = None,
        redis_cache_service: Optional[RedisCacheService] = None,
    ):
        """
        初始化通用Agent执行器
        
        Args:
            domain: 业务领域标识
            agent_config: Agent配置，不提供则自动根据domain加载
            llm_service: LLM服务
            embedding_service: 嵌入服务
            milvus_service: 向量数据库服务
            redis_cache_service: 缓存服务
        """
        self.domain = domain
        self.config = agent_config or get_agent_config(domain)
        
        # 实例变量：每个实例独立的会话历史
        self._history: Dict[str, List[Dict[str, str]]] = {}  # conversation_id -> history
        self._last_access: Dict[str, datetime] = {}  # 追踪最后访问时间
        self._cleanup_counter: int = 0  # 清理计数器
        
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.milvus_service = milvus_service
        self.redis_cache_service = redis_cache_service
        
        self._history_max_age = timedelta(hours=24)
        
        # 初始化锁（类变量，共享的）
        if GeneralAgentExecutor._history_lock is None:
            GeneralAgentExecutor._history_lock = asyncio.Lock()
    
    @property
    def agent_name(self) -> str:
        """获取Agent名称"""
        return self.config.name
    
    @property
    def agent_description(self) -> str:
        """获取Agent描述"""
        return self.config.description
    
    def _get_prompt(self, step_key: str) -> str:
        """获取指定步骤的提示词"""
        step_config = getattr(self.config, step_key, None)
        template_key = step_config.prompt_template_key if step_config else ""
        return PromptTemplateManager.get(self.domain, template_key)
    
    def _cleanup_old_history(self):
        """清理过期的对话历史"""
        now = datetime.now()
        expired = [
            cid for cid, last_access in self._last_access.items()
            if now - last_access > self._history_max_age
        ]
        for cid in expired:
            self._history.pop(cid, None)
            self._last_access.pop(cid, None)
        if expired:
            logger.info(f"[{self.domain}] 已清理 {len(expired)} 条过期对话历史")
    
    async def _get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """获取对话历史（异步安全）"""
        async with self._history_lock:
            if conversation_id not in self._history:
                self._history[conversation_id] = []
            self._last_access[conversation_id] = datetime.now()
            
            self._cleanup_counter += 1
            if self._cleanup_counter >= _CLEANUP_INTERVAL:
                self._cleanup_old_history()
                self._cleanup_counter = 0
            
            return self._history[conversation_id].copy()
    
    async def _add_to_history(self, conversation_id: str, role: str, content: str):
        """添加对话历史（异步安全）"""
        async with self._history_lock:
            history = self._history.get(conversation_id, [])
            history.append({"role": role, "content": content})
            
            if len(history) > self.config.max_history_turns * 2:
                history = history[-self.config.max_history_turns * 2:]
            
            self._history[conversation_id] = history
    
    async def step1_understand(
        self,
        user_question: str,
        conversation_id: str
    ) -> AgentStepResult:
        """
        步骤1：问题理解/改写
        
        Args:
            user_question: 用户问题
            conversation_id: 会话ID
            
        Returns:
            AgentStepResult: 步骤执行结果
        """
        step_config = getattr(self.config, 'step1', None)
        if not step_config or not step_config.enabled:
            return AgentStepResult(
                step_name=step_config.name if step_config else "问题理解",
                step_order=1,
                input_data={"user_question": user_question},
                output_data={"queries": [user_question]},
                status="skipped"
            )
        
        start_time = time.time()
        try:
            history = await self._get_conversation_history(conversation_id)
            template = PromptTemplateManager.get(self.domain, step_config.prompt_template_key)
            
            if template:
                from langchain_core.prompts import ChatPromptTemplate
                prompt = ChatPromptTemplate.from_template(template)
                prompt_obj = prompt.invoke({"user_question": user_question})
                messages = [{"role": "system", "content": prompt_obj.messages[0].content}]
                if len(prompt_obj.messages) > 1:
                    messages.append({"role": "user", "content": prompt_obj.messages[1].content})
            else:
                messages = [
                    {"role": "system", "content": "请理解用户问题并提取关键信息。"},
                    {"role": "user", "content": user_question}
                ]
            
            response = await self.llm_service.chat_qwen(messages)
            queries = [q.strip() for q in response.split('\n') if q.strip()]
            
            result = QuestionRewriteResult(
                original_question=user_question,
                rewritten_queries=queries or [user_question],
                keywords=list(set([w for q in queries for w in q.split()]))
            )
            
            duration = int((time.time() - start_time) * 1000)
            logger.info(f"[{self.domain}] {step_config.name}完成", message_length=len(user_question))
            
            return AgentStepResult(
                step_name=step_config.name,
                step_order=1,
                input_data={"user_question": user_question},
                output_data=result.model_dump(),
                status="success",
                duration_ms=duration
            )
            
        except Exception as e:
            duration = int((time.time() - start_time) * 1000)
            logger.error(f"[{self.domain}] {step_config.name}失败: {str(e)}")
            return AgentStepResult(
                step_name=step_config.name,
                step_order=1,
                input_data={"user_question": user_question},
                status="failed",
                error_message=str(e),
                duration_ms=duration
            )
    
    async def step2_review(
        self,
        user_question: str
    ) -> tuple[AgentStepResult, SafetyCheckResult]:
        """
        步骤2：内容审查/安全检查
        
        Args:
            user_question: 用户问题
            
        Returns:
            tuple: (步骤结果, 审查结果)
        """
        step_config = getattr(self.config, 'step2', None)
        if not step_config or not step_config.enabled:
            # 返回默认安全结果
            default_result = SafetyCheckResult(
                is_safe=True,
                risk_level="low",
                risk_categories=[],
                can_proceed=True
            )
            return AgentStepResult(
                step_name=step_config.name if step_config else "内容审查",
                step_order=2,
                input_data={"user_question": user_question},
                output_data=default_result.model_dump(),
                status="skipped"
            ), default_result
        
        start_time = time.time()
        try:
            template = PromptTemplateManager.get(self.domain, step_config.prompt_template_key)
            
            if template:
                from langchain_core.prompts import ChatPromptTemplate
                prompt = ChatPromptTemplate.from_template(template)
                prompt_obj = prompt.invoke({"user_question": user_question})
                messages = [{"role": "system", "content": prompt_obj.messages[0].content}]
                if len(prompt_obj.messages) > 1:
                    messages.append({"role": "user", "content": prompt_obj.messages[1].content})
            else:
                messages = [
                    {"role": "system", "content": "请检查以下内容是否合规安全。"},
                    {"role": "user", "content": user_question}
                ]
            
            response = await self.llm_service.chat_qwen(messages)
            
            # 解析JSON结果
            try:
                json_str = response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]
                safety_data = json.loads(json_str)
            except json.JSONDecodeError:
                # 保守策略：从配置获取敏感词
                sensitive_keywords = self.config.sensitive_keywords if hasattr(self.config, 'sensitive_keywords') else ["诊断", "处方", "胸痛"]
                detected = [kw for kw in sensitive_keywords if kw in user_question]
                safety_data = {
                    "is_safe": False if detected else True,
                    "risk_level": "high" if detected else "low",
                    "risk_categories": detected or ["解析失败"]
                }
            
            safety_result = SafetyCheckResult(
                is_safe=safety_data.get("is_safe", True),
                risk_level=safety_data.get("risk_level", "low"),
                risk_categories=safety_data.get("risk_categories", []),
                warning_message=safety_data.get("warning_message"),
                can_proceed=safety_data.get("is_safe", True) or safety_data.get("risk_level") != "high"
            )
            
            duration = int((time.time() - start_time) * 1000)
            logger.info(f"[{self.domain}] {step_config.name}完成", is_safe=safety_result.is_safe)
            
            step_result = AgentStepResult(
                step_name=step_config.name,
                step_order=2,
                input_data={"user_question": user_question},
                output_data=safety_result.model_dump(),
                status="success",
                duration_ms=duration
            )
            
            return step_result, safety_result
            
        except Exception as e:
            duration = int((time.time() - start_time) * 1000)
            logger.error(f"[{self.domain}] {step_config.name}失败: {str(e)}")
            
            safety_result = SafetyCheckResult(
                is_safe=False,
                risk_level="high",
                risk_categories=["服务异常"],
                can_proceed=False
            )
            
            return AgentStepResult(
                step_name=step_config.name,
                step_order=2,
                input_data={"user_question": user_question},
                status="failed",
                error_message=str(e),
                duration_ms=duration
            ), safety_result
    
    async def step3_retrieve(
        self,
        queries: List[str],
        top_k: int
    ) -> tuple[AgentStepResult, List[Dict[str, Any]]]:
        """
        步骤3：知识检索
        
        Args:
            queries: 检索查询列表
            top_k: 返回数量
            
        Returns:
            tuple: (步骤结果, 检索文档列表)
        """
        step_config = getattr(self.config, 'step3', None)
        if not step_config or not step_config.enabled:
            return AgentStepResult(
                step_name=step_config.name if step_config else "知识检索",
                step_order=3,
                input_data={"queries": queries},
                output_data={"documents_found": 0},
                status="skipped"
            ), []
        
        start_time = time.time()
        all_documents = []
        
        try:
            for query in queries[:self.config.max_retrieval_queries]:
                if self.embedding_service and self.milvus_service:
                    query_embedding = await self.embedding_service.embed_query(query)
                    docs = self.milvus_service.search_similar(
                        query_embedding=query_embedding,
                        top_k=top_k // 3 + 1
                    )
                    
                    for doc in docs:
                        doc_dict = {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "source_query": query
                        }
                        if doc_dict not in all_documents:
                            all_documents.append(doc_dict)
            
            rag_context = "\n\n".join([
                f"[来源: {doc.get('metadata', {}).get('source', '未知')}]\n{doc['content']}"
                for doc in all_documents[:top_k]
            ]) or "暂无相关检索结果"
            
            duration = int((time.time() - start_time) * 1000)
            logger.info(f"[{self.domain}] {step_config.name}完成", document_count=len(all_documents))
            
            step_result = AgentStepResult(
                step_name=step_config.name,
                step_order=3,
                input_data={"queries": queries, "top_k": top_k},
                output_data={
                    "context": rag_context[:500] + "..." if len(rag_context) > 500 else rag_context,
                    "documents_found": len(all_documents)
                },
                status="success",
                duration_ms=duration
            )
            
            return step_result, all_documents
            
        except Exception as e:
            duration = int((time.time() - start_time) * 1000)
            logger.error(f"[{self.domain}] {step_config.name}失败: {str(e)}")
            
            return AgentStepResult(
                step_name=step_config.name,
                step_order=3,
                input_data={"queries": queries, "top_k": top_k},
                status="failed",
                error_message=str(e),
                duration_ms=duration
            ), []
    
    def _build_safety_reminder(self, safety_result: SafetyCheckResult) -> str:
        """构建安全提醒"""
        if not safety_result.is_safe:
            from src.modules.chat.agent.prompts import WARNING_TEMPLATES
            return WARNING_TEMPLATES.get("default", "").format(
                risk_categories=", ".join(safety_result.risk_categories),
                warning_message=safety_result.warning_message or "请咨询专业人员"
            )
        return "问题已通过审查。"
    
    def _build_safety_check_result(self, safety_result: SafetyCheckResult) -> str:
        """构建安全审查结果描述"""
        result = f"风险等级: {safety_result.risk_level}"
        if safety_result.risk_categories:
            result += f"\n涉及内容: {', '.join(safety_result.risk_categories)}"
        return result
    
    def _quick_evaluate_answer_quality(
        self,
        response: str,
        rag_context: str
    ) -> Dict[str, Any]:
        """快速评估答案质量"""
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
        
        has_low_quality = False
        for pattern in self.config.low_quality_patterns:
            if pattern in response:
                reasons.append(f"包含低质量模式: {pattern}")
                has_low_quality = True
                quality_score -= 2
                break
        
        quality_score = max(0, min(10, quality_score))
        is_solved = quality_score >= 6 and not has_low_quality
        
        return {
            "is_solved": is_solved,
            "quality_score": round(quality_score, 1),
            "eval_reason": "; ".join(reasons)
        }
    
    async def step4_generate(
        self,
        user_question: str,
        rag_context: str,
        safety_result: SafetyCheckResult,
        conversation_id: str
    ) -> tuple[AgentStepResult, str, Dict[str, Any]]:
        """
        步骤4：回答生成
        
        Args:
            user_question: 用户问题
            rag_context: 检索上下文
            safety_result: 安全审查结果
            conversation_id: 会话ID
            
        Returns:
            tuple: (步骤结果, 回答内容, 质量评估)
        """
        step_config = getattr(self.config, 'step4', None)
        if not step_config or not step_config.enabled:
            default_response = "抱歉，暂无法生成回答。"
            return AgentStepResult(
                step_name=step_config.name if step_config else "回答生成",
                step_order=4,
                status="skipped"
            ), default_response, {"is_solved": False, "quality_score": 0}
        
        start_time = time.time()
        
        try:
            current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
            safety_reminder = self._build_safety_reminder(safety_result)
            safety_check_result = self._build_safety_check_result(safety_result)
            
            history = await self._get_conversation_history(conversation_id)
            chat_history = []
            for msg in history[-self.config.max_history_turns:]:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))
            
            # 构建提示
            template = PromptTemplateManager.get(self.domain, step_config.prompt_template_key)
            if template and "{current_time}" in template:
                prompt_content = template.format(
                    current_time=current_time,
                    rag_context=rag_context,
                    user_question=user_question,
                    safety_check_result=safety_check_result,
                    safety_reminder=safety_reminder,
                    chat_history="\n".join([f"{msg.type}: {msg.content}" for msg in chat_history]),
                    product_info=rag_context,
                    knowledge_base=rag_context,
                    context=rag_context,
                    category=""
                )
                messages = [{"role": "system", "content": prompt_content}]
            else:
                # 使用通用格式
                from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
                from src.modules.chat.agent.prompts import ANSWER_GENERATION_PROMPT
                
                prompt_obj = ANSWER_GENERATION_PROMPT.invoke({
                    "current_time": current_time,
                    "rag_context": rag_context,
                    "user_question": user_question,
                    "safety_check_result": safety_check_result,
                    "safety_reminder": safety_reminder,
                    "chat_history": chat_history
                })
                
                def convert_type(msg_type):
                    mapping = {"human": "user", "ai": "assistant", "system": "system"}
                    return mapping.get(msg_type, msg_type)
                
                messages = [{"role": convert_type(msg.type), "content": msg.content} for msg in prompt_obj.messages]
            
            response = await self.llm_service.chat_qwen(messages)
            
            quality_evaluation = self._quick_evaluate_answer_quality(response, rag_context)
            
            await self._add_to_history(conversation_id, "user", user_question)
            await self._add_to_history(conversation_id, "assistant", response)
            
            duration = int((time.time() - start_time) * 1000)
            logger.info(f"[{self.domain}] {step_config.name}完成", response_length=len(response))
            
            return AgentStepResult(
                step_name=step_config.name,
                step_order=4,
                input_data={"user_question": user_question, "context_length": len(rag_context)},
                output_data={"response": response, **quality_evaluation},
                status="success",
                duration_ms=duration
            ), response, quality_evaluation
            
        except Exception as e:
            duration = int((time.time() - start_time) * 1000)
            logger.error(f"[{self.domain}] {step_config.name}失败: {str(e)}")
            
            return AgentStepResult(
                step_name=step_config.name,
                step_order=4,
                status="failed",
                error_message=str(e),
                duration_ms=duration
            ), "抱歉，服务暂时繁忙，请稍后重试。", {"is_solved": False, "quality_score": 0}
    
    async def _generate_fallback_response(
        self,
        user_question: str,
        safety_result: SafetyCheckResult
    ) -> str:
        """生成兜底响应"""
        from src.modules.chat.agent.prompts import WARNING_TEMPLATES, GUIDANCE_TEMPLATES
        
        if not safety_result.is_safe:
            return WARNING_TEMPLATES.get("default", "").format(
                risk_categories=", ".join(safety_result.risk_categories) or "敏感内容",
                warning_message=safety_result.warning_message or "建议咨询专业人员"
            )
        
        return GUIDANCE_TEMPLATES.get("default", "").format(user_question=user_question)
    
    async def _try_get_cached_response(
        self,
        message: str,
        conversation_id: str,
        question_embedding: Optional[List[float]] = None
    ) -> Optional[str]:
        """尝试从缓存获取响应"""
        if not self.config.cache_enabled:
            return None
        
        if not self.redis_cache_service or not self.redis_cache_service.is_available:
            return None
        
        try:
            if question_embedding is None and self.embedding_service:
                question_embedding = await self.embedding_service.embed_query(message)
            
            if question_embedding:
                cached = await self.redis_cache_service.get_cached_response(
                    question=message,
                    question_embedding=question_embedding,
                    threshold=self.config.cache_threshold
                )
                if cached:
                    logger.info(f"[{self.domain}] 命中缓存")
                    return cached
        except Exception as e:
            logger.warning(f"[{self.domain}] 缓存检查失败: {str(e)}")
        
        return None
    
    async def _store_to_cache(
        self,
        message: str,
        response: str,
        conversation_id: str,
        question_embedding: Optional[List[float]] = None
    ) -> None:
        """存储到缓存"""
        if not self.config.cache_enabled:
            return
        
        if not self.redis_cache_service or not self.redis_cache_service.is_available:
            return
        
        try:
            if question_embedding is None and self.embedding_service:
                question_embedding = await self.embedding_service.embed_query(message)
            
            if question_embedding:
                await self.redis_cache_service.store_conversation(
                    conversation_id=conversation_id,
                    question=message,
                    answer=response,
                    embedding=question_embedding
                )
                logger.info(f"[{self.domain}] 对话已缓存")
        except Exception as e:
            logger.warning(f"[{self.domain}] 存储缓存失败: {str(e)}")
    
    async def execute(
        self,
        request: HospitalChatRequest
    ) -> HospitalChatResponse:
        """
        执行完整的Agent流程
        
        Args:
            request: 聊天请求
            
        Returns:
            HospitalChatResponse: Agent执行结果
        """
        conversation_id = request.conversation_id or f"conv_{int(time.time())}"
        steps: List[Dict[str, Any]] = []
        documents_used: List[str] = []
        question_embedding: Optional[List[float]] = None
        
        try:
            # 预计算向量
            if self.embedding_service and self.redis_cache_service and self.redis_cache_service.is_available:
                question_embedding = await self.embedding_service.embed_query(request.message)
            
            # 检查缓存
            cached_response = await self._try_get_cached_response(
                request.message, conversation_id, question_embedding
            )
            if cached_response:
                return HospitalChatResponse(
                    message=cached_response,
                    conversation_id=conversation_id,
                    steps=[{"step_name": "cache_hit", "step_order": 0, "status": "success"}],
                    documents_used=[],
                    safety_passed=True,
                    stream_available=True,
                    cache_hit=True
                )
            
            # 步骤1
            step1_result = await self.step1_understand(request.message, conversation_id)
            steps.append(step1_result.model_dump())
            queries = step1_result.output_data.get("rewritten_queries", [request.message])
            
            # 步骤2
            step2_result, safety_result = await self.step2_review(request.message)
            steps.append(step2_result.model_dump())
            
            if not safety_result.can_proceed:
                warning_response = await self._generate_fallback_response(
                    request.message, safety_result
                )
                return HospitalChatResponse(
                    message=warning_response,
                    conversation_id=conversation_id,
                    steps=steps,
                    documents_used=[],
                    safety_passed=False,
                    stream_available=True
                )
            
            # 步骤3
            step3_result, documents = await self.step3_retrieve(
                queries, self.config.top_k
            )
            steps.append(step3_result.model_dump())
            documents_used = [doc["content"] for doc in documents[:5]]
            
            rag_context = "\n\n".join(documents_used) or "暂无相关检索结果"
            
            # 步骤4
            step4_result, response, quality_evaluation = await self.step4_generate(
                request.message, rag_context, safety_result, conversation_id
            )
            steps.append(step4_result.model_dump())
            
            # 记录业务事件
            logger.log_business_event(
                f"{self.agent_name}对话",
                success=True,
                domain=self.domain,
                conversation_id=conversation_id,
                quality_score=quality_evaluation.get("quality_score"),
                is_solved=quality_evaluation.get("is_solved")
            )
            
            # 缓存高质量答案
            if step4_result.status == "success" and quality_evaluation.get("is_solved"):
                await self._store_to_cache(
                    request.message, response, conversation_id, question_embedding
                )
            
            return HospitalChatResponse(
                message=response,
                conversation_id=conversation_id,
                steps=steps,
                documents_used=documents_used,
                safety_passed=safety_result.is_safe,
                stream_available=True,
                cache_hit=False
            )
            
        except Exception as e:
            logger.error(f"[{self.domain}] Agent执行失败: {str(e)}")
            logger.log_business_event(
                f"{self.agent_name}对话",
                success=False,
                domain=self.domain,
                conversation_id=conversation_id,
                error=str(e)
            )
            
            return HospitalChatResponse(
                message="抱歉，服务暂时繁忙，请稍后重试。",
                conversation_id=conversation_id,
                steps=steps,
                documents_used=documents_used,
                safety_passed=True,
                stream_available=True
            )


# =============================================================================
# 向后兼容的类型别名
# =============================================================================

HospitalAgentExecutor = GeneralAgentExecutor
