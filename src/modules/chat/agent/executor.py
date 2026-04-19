"""
医院客服 Agent 执行器
使用 LangChain 1.2.15 实现多步骤 skill 编排
"""

import json
import time
import asyncio
from typing import AsyncGenerator, Optional, List, Dict, Any

from src.modules.chat.core.llm_service import LLMService
from src.modules.chat.core.embedding_service import EmbeddingService
from src.modules.chat.core.milvus_service import MilvusService
from src.modules.chat.core.redis_cache_service import RedisCacheService

from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, AIMessage
from src.modules.chat.agent.prompts import (
    QUESTION_REWRITING_PROMPT,
    SAFETY_CHECK_PROMPT,
    ANSWER_GENERATION_PROMPT,
    SAFETY_WARNING_TEMPLATE,
    GENERAL_GUIDANCE_TEMPLATE,
)
from src.modules.chat.agent.schemas import (
    AgentStepResult,
    SafetyCheckResult,
    QuestionRewriteResult,
    RetrievalResult,
)
from src.modules.chat.schemas import (
    HospitalChatRequest,
    HospitalChatResponse,
    HospitalAgentConfig,
)
from src.modules.chat.config import chat_config
from src.shared.logger import APILogger

logger = APILogger("hospital_agent_executor")


# 类常量定义
_MAX_HISTORY_MESSAGES_FOR_ANSWER = 10  # 答案生成时使用的历史消息数
_MAX_RETRIEVAL_QUERIES = 3  # 最多使用的检索查询数
_CLEANUP_INTERVAL = 10  # 每N次调用清理一次过期历史


class HospitalAgentExecutor:
    """
    医院客服 Agent 执行器
    
    实现多步骤 skill 编排：
    1. 问题重写 - 将用户问题改写为更适合检索的查询
    2. 安全审查 - 检查问题是否涉及医疗建议、处方、诊断等敏感内容
    3. 知识检索 - 使用现有的 RAG 组件从医疗知识库检索相关内容
    4. 答案生成 - 基于检索结果生成安全、准确的回复
    """
    
    # 类变量：所有实例共享会话历史（使用 asyncio.Lock 保护）
    _history: Dict[str, List[Dict[str, str]]] = {}  # conversation_id -> history
    _last_access: Dict[str, datetime] = {}  # 追踪最后访问时间
    _cleanup_counter: int = 0  # 清理计数器
    _history_lock: asyncio.Lock = None  # 异步锁，用于保护历史记录操作
    
    def __init__(
        self,
        config: Optional[HospitalAgentConfig] = None,
        llm_service: Optional[LLMService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        milvus_service: Optional[MilvusService] = None,
        redis_cache_service: Optional[RedisCacheService] = None,
    ):
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.milvus_service = milvus_service
        self.redis_cache_service = redis_cache_service
        self.config = config or HospitalAgentConfig()
        self._history_max_age = timedelta(hours=24)  # 历史记录过期时间
        # 每个实例维护自己的锁，确保异步安全
        if HospitalAgentExecutor._history_lock is None:
            HospitalAgentExecutor._history_lock = asyncio.Lock()

    def _cleanup_old_history(self):
        """清理过期的对话历史，防止内存泄漏"""
        now = datetime.now()
        expired = [
            cid for cid, last_access in self._last_access.items()
            if now - last_access > self._history_max_age
        ]
        for cid in expired:
            self._history.pop(cid, None)
            self._last_access.pop(cid, None)
        if expired:
            logger.info(f"已清理 {len(expired)} 条过期对话历史")

    async def _get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """获取对话历史（异步安全）"""
        async with self._history_lock:
            if conversation_id not in self._history:
                self._history[conversation_id] = []
            # 更新最后访问时间
            self._last_access[conversation_id] = datetime.now()
            # 定期清理过期历史（每N次调用清理一次）
            self._cleanup_counter += 1
            if self._cleanup_counter >= _CLEANUP_INTERVAL:
                self._cleanup_old_history()
                self._cleanup_counter = 0
            return self._history[conversation_id].copy()  # 返回副本避免外部修改

    async def _add_to_history(self, conversation_id: str, role: str, content: str):
        """添加对话历史（异步安全）"""
        async with self._history_lock:
            history = self._history.get(conversation_id, [])
            history.append({"role": role, "content": content})
            # 保持最大历史轮次
            if len(history) > self.config.max_history_turns * 2:
                history = history[-self.config.max_history_turns * 2:]
            self._history[conversation_id] = history
    
    async def step1_rewrite_question(
        self, 
        user_question: str,
        conversation_id: str
    ) -> AgentStepResult:
        """
        步骤1：问题重写
        
        将用户问题改写为更适合检索的查询
        结合对话历史进行上下文理解
        """
        start_time = time.time()
        
        try:
            # 获取对话历史用于上下文理解
            history = await self._get_conversation_history(conversation_id)
            # 使用 invoke 获取格式化后的消息
            prompt_obj = QUESTION_REWRITING_PROMPT.invoke({"user_question": user_question})
            # 构建消息列表（使用模板自动生成的变量替换）
            messages = [
                {"role": "system", "content": prompt_obj.messages[0].content},
                {"role": "user", "content": prompt_obj.messages[1].content}
            ]
            
            # 调用模型
            response = await self.llm_service.chat_qwen(messages)
            
            # 解析结果
            rewritten_queries = [q.strip() for q in response.split('\n') if q.strip()]
            keywords = list(set([w for q in rewritten_queries for w in q.split()]))
            
            result = QuestionRewriteResult(
                original_question=user_question,
                rewritten_queries=rewritten_queries,
                keywords=keywords
            )
            
            duration = int((time.time() - start_time) * 1000)
            
            # 日志脱敏：只记录消息长度和前20个字符
            logger.info(
                f"问题重写完成",
                message_length=len(user_question),
                message_preview=user_question[:20] + "..." if len(user_question) > 20 else user_question,
                rewritten=rewritten_queries[:2]
            )
            
            return AgentStepResult(
                step_name="question_rewrite",
                step_order=1,
                input_data={"user_question": user_question},
                output_data=result.model_dump(),
                status="success",
                duration_ms=duration
            )
            
        except Exception as e:
            duration = int((time.time() - start_time) * 1000)
            logger.error(f"问题重写失败: {str(e)}")
            return AgentStepResult(
                step_name="question_rewrite",
                step_order=1,
                input_data={"user_question": user_question},
                status="failed",
                error_message=str(e),
                duration_ms=duration
            )
    
    async def step2_safety_check(
        self, 
        user_question: str
    ) -> tuple[AgentStepResult, SafetyCheckResult]:
        """
        步骤2：安全审查
        
        检查问题是否涉及医疗建议、处方、诊断等敏感内容
        """
        start_time = time.time()
        
        try:
            # 使用 invoke 获取格式化后的消息
            prompt_obj = SAFETY_CHECK_PROMPT.invoke({"user_question": user_question})
            # 构建消息列表（使用模板自动生成的变量替换）
            messages = [
                {"role": "system", "content": prompt_obj.messages[0].content},
                {"role": "user", "content": prompt_obj.messages[1].content}
            ]
            
            # 调用模型
            response = await self.llm_service.chat_qwen(messages)
            
            # 解析 JSON 结果
            try:
                # 尝试提取 JSON
                json_str = response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]
                    
                safety_data = json.loads(json_str)
            except json.JSONDecodeError:
                # 保守策略：解析失败时默认中等风险，需人工确认
                # 定义敏感关键词列表
                sensitive_keywords = [
                    "诊断", "处方", "胸痛", "呼吸困难", "昏迷",
                    "休克", "出血", "骨折", "中毒", "自杀",
                    "儿童", "孕妇", "未成年人", "用药", "剂量",
                    "手术", "化疗", "放疗", "预后", "康复"
                ]
                detected_keywords = [
                    kw for kw in sensitive_keywords
                    if kw in user_question
                ]
                safety_data = {
                    "is_safe": False,
                    "risk_level": "medium" if not detected_keywords else "high",
                    "risk_categories": detected_keywords or ["解析失败"],
                    "warning_message": "安全审查服务暂时不可用，建议咨询专业医生"
                }
            
            safety_result = SafetyCheckResult(
                is_safe=safety_data.get("is_safe", True),
                risk_level=safety_data.get("risk_level", "low"),
                risk_categories=safety_data.get("risk_categories", []),
                warning_message=safety_data.get("warning_message"),
                can_proceed=safety_data.get("is_safe", True) or safety_data.get("risk_level") != "high"
            )
            
            duration = int((time.time() - start_time) * 1000)
            
            # 日志脱敏
            logger.info(
                f"安全审查完成",
                message_length=len(user_question),
                message_preview=user_question[:20] + "..." if len(user_question) > 20 else user_question,
                is_safe=safety_result.is_safe,
                risk_level=safety_result.risk_level
            )
            
            step_result = AgentStepResult(
                step_name="safety_check",
                step_order=2,
                input_data={"user_question": user_question},
                output_data=safety_result.model_dump(),
                status="success",
                duration_ms=duration
            )
            
            return step_result, safety_result
            
        except Exception as e:
            duration = int((time.time() - start_time) * 1000)
            logger.error(f"安全审查失败: {str(e)}")
            
            # 保守策略：异常时默认高风险，需人工确认
            safety_result = SafetyCheckResult(
                is_safe=False,
                risk_level="high",
                risk_categories=["安全审查服务异常"],
                warning_message="安全审查服务暂时不可用，建议咨询专业医生",
                can_proceed=False
            )
            
            step_result = AgentStepResult(
                step_name="safety_check",
                step_order=2,
                input_data={"user_question": user_question},
                status="failed",
                error_message=str(e),
                duration_ms=duration
            )
            
            return step_result, safety_result
    
    async def step3_knowledge_retrieval(
        self, 
        rewritten_queries: List[str],
        top_k: int
    ) -> tuple[AgentStepResult, List[Dict[str, Any]]]:
        """
        步骤3：知识检索
        
        使用现有的 RAG 组件从医疗知识库检索相关内容
        """
        start_time = time.time()
        all_documents = []
        retrieval_results = []
        
        try:
            for query in rewritten_queries[:_MAX_RETRIEVAL_QUERIES]:  # 最多使用3个查询
                # 使用嵌入服务和 Milvus 服务检索
                query_embedding = await self.embedding_service.embed_query(query)
                docs = self.milvus_service.search_similar(
                    query_embedding=query_embedding,
                    top_k=top_k // 3 + 1  # 每个查询返回部分结果
                )
                
                for doc in docs:
                    doc_dict = {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source_query": query
                    }
                    # 去重
                    if doc_dict not in all_documents:
                        all_documents.append(doc_dict)
                
                retrieval_results.append(RetrievalResult(
                    query=query,
                    documents=[{"content": d.page_content} for d in docs],
                    scores=[0.0] * len(docs)  # Milvus 不直接返回分数
                ))
            
            # 构建 RAG 上下文
            rag_context = "\n\n".join([
                f"[来源: {doc.get('metadata', {}).get('source', '未知')}]\n{doc['content']}"
                for doc in all_documents[:self.config.top_k]
            ]) or "暂无相关检索结果"
            
            duration = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"知识检索完成",
                query_count=len(rewritten_queries),
                document_count=len(all_documents)
            )
            
            step_result = AgentStepResult(
                step_name="knowledge_retrieval",
                step_order=3,
                input_data={"queries": rewritten_queries, "top_k": top_k},
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
            logger.error(f"知识检索失败: {str(e)}")
            
            step_result = AgentStepResult(
                step_name="knowledge_retrieval",
                step_order=3,
                input_data={"queries": rewritten_queries, "top_k": top_k},
                status="failed",
                error_message=str(e),
                duration_ms=duration
            )
            
            return step_result, []
    
    async def step4_generate_answer(
        self,
        user_question: str,
        rag_context: str,
        safety_result: SafetyCheckResult,
        conversation_id: str
    ) -> tuple[AgentStepResult, str, Dict[str, Any]]:
        """
        步骤4：答案生成（含质量评估）

        基于检索结果生成安全、准确的回复
        同时基于规则快速评估答案质量，返回评估结果

        Returns:
            tuple: (步骤结果, 回答内容, 质量评估信息)
        """
        start_time = time.time()

        try:
            # 获取当前时间
            current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")

            # 构建安全提醒（使用私有方法）
            safety_reminder = self._build_safety_reminder(safety_result)
            safety_check_result = self._build_safety_check_result(safety_result)

            # 获取对话历史
            history = await self._get_conversation_history(conversation_id)
            chat_history = []
            for msg in history[-_MAX_HISTORY_MESSAGES_FOR_ANSWER:]:  # 只使用最近10条
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))

            # 构建提示
            prompt_content = ANSWER_GENERATION_PROMPT.invoke({
                "current_time": current_time,
                "rag_context": rag_context,
                "user_question": user_question,
                "safety_check_result": safety_check_result,
                "safety_reminder": safety_reminder,
                "chat_history": chat_history
            })

            # 构建消息
            def _convert_message_type(msg_type: str) -> str:
                """将 LangChain 消息类型转换为 API 期望的角色"""
                mapping = {"human": "user", "ai": "assistant", "system": "system"}
                return mapping.get(msg_type, msg_type)

            messages = [{"role": _convert_message_type(msg.type), "content": msg.content} for msg in prompt_content.messages]

            # 调用模型
            response = await self.llm_service.chat_qwen(messages)

            # 基于规则快速评估答案质量（避免额外 LLM 调用）
            quality_evaluation = self._quick_evaluate_answer_quality(
                user_question=user_question,
                response=response,
                rag_context=rag_context
            )

            duration = int((time.time() - start_time) * 1000)

            # 更新对话历史
            await self._add_to_history(conversation_id, "user", user_question)
            await self._add_to_history(conversation_id, "assistant", response)

            logger.info(
                f"答案生成完成",
                response_length=len(response),
                duration_ms=duration,
                quality_score=quality_evaluation.get("quality_score"),
                is_solved=quality_evaluation.get("is_solved")
            )

            return AgentStepResult(
                step_name="answer_generation",
                step_order=4,
                input_data={
                    "user_question": user_question,
                    "context_length": len(rag_context),
                    "safety_passed": safety_result.is_safe
                },
                output_data={
                    "response": response,
                    **quality_evaluation  # 包含质量评估信息
                },
                status="success",
                duration_ms=duration
            ), response, quality_evaluation

        except Exception as e:
            duration = int((time.time() - start_time) * 1000)
            logger.error(f"答案生成失败: {str(e)}")

            # 失败时返回默认低质量评估
            default_evaluation = {
                "is_solved": False,
                "quality_score": 0,
                "eval_reason": f"生成失败: {str(e)}"
            }

            return AgentStepResult(
                step_name="answer_generation",
                step_order=4,
                status="failed",
                error_message=str(e),
                duration_ms=duration
            ), "抱歉，服务暂时繁忙，请稍后重试。", default_evaluation
    
    def _build_safety_reminder(self, safety_result: SafetyCheckResult) -> str:
        """构建安全提醒文本"""
        if not safety_result.is_safe:
            return SAFETY_WARNING_TEMPLATE.format(
                risk_categories=", ".join(safety_result.risk_categories),
                warning_message=safety_result.warning_message or "请咨询专业医生"
            )
        return "问题已通过安全审查，可以正常回答。"
    
    def _build_safety_check_result(self, safety_result: SafetyCheckResult) -> str:
        """构建安全审查结果描述"""
        result = f"风险等级: {safety_result.risk_level}"
        if safety_result.risk_categories:
            result += f"\n涉及内容: {', '.join(safety_result.risk_categories)}"
        return result
    
    def _quick_evaluate_answer_quality(
        self,
        user_question: str,
        response: str,
        rag_context: str
    ) -> Dict[str, Any]:
        """
        基于规则快速评估答案质量（无需额外 LLM 调用）

        Returns:
            包含 is_solved, quality_score, eval_reason 的字典
        """
        reasons = []
        quality_score = 5  # 默认中等分数
        is_solved = False

        # 条件1: 检查是否有 RAG 检索结果支撑
        has_rag_context = rag_context and rag_context != "暂无相关检索结果"
        if has_rag_context:
            reasons.append("有RAG检索结果支撑")
            quality_score += 1
        else:
            reasons.append("无RAG检索结果")
            quality_score -= 2

        # 条件2: 检查答案长度是否合理 (50-2000字符)
        response_len = len(response)
        if response_len >= 50:
            reasons.append(f"长度适中({response_len}字)")
            quality_score += 1
        elif response_len >= 20:
            reasons.append(f"长度偏短({response_len}字)")
            quality_score -= 1
        else:
            reasons.append(f"答案过短({response_len}字)")
            quality_score -= 2

        if response_len > 2000:
            reasons.append("长度较长")

        # 条件3: 检查是否包含低质量模式
        low_quality_patterns = [
            ("暂无相关检索结果", "包含'暂无相关检索结果'"),
            ("抱歉，服务暂时繁忙", "包含服务繁忙提示"),
            ("我无法回答", "包含'无法回答'"),
            ("无法提供", "包含'无法提供'"),
            ("未查询到", "表示未查到信息"),
        ]

        has_low_quality = False
        for pattern, desc in low_quality_patterns:
            if pattern in response:
                reasons.append(desc)
                has_low_quality = True
                quality_score -= 2
                break

        # 条件4: 检查内容充实度（非标点字符占比）
        if response_len > 0:
            char_ratio = sum(c.isalnum() for c in response) / response_len
            if char_ratio > 0.5:
                reasons.append("内容充实")
                quality_score += 0.5
            elif char_ratio < 0.3:
                reasons.append("内容可能不够充实")
                quality_score -= 1

        # 条件5: 检查是否引用了 RAG 上下文的关键词（如果有）
        if has_rag_context and response_len > 0:
            # 提取 RAG 上下文中的一些关键词（取前50字）
            rag_keywords = set(rag_context[:500].split())
            response_keywords = set(response.split())
            overlap = rag_keywords & response_keywords
            # 如果有重叠关键词，说明答案引用了 RAG 内容
            if len(overlap) > 5:
                reasons.append("引用了检索内容")
                quality_score += 0.5

        # 限制分数范围
        quality_score = max(0, min(10, quality_score))

        # 判断是否解决问题：分数 >= 6 且不是低质量回复
        is_solved = quality_score >= 6 and not has_low_quality

        return {
            "is_solved": is_solved,
            "quality_score": round(quality_score, 1),
            "eval_reason": "; ".join(reasons) if reasons else "未提供具体原因"
        }

    async def _generate_fallback_response(
        self,
        user_question: str,
        safety_result: SafetyCheckResult
    ) -> str:
        """生成兜底响应"""
        if not safety_result.is_safe:
            return SAFETY_WARNING_TEMPLATE.format(
                risk_categories=", ".join(safety_result.risk_categories) or "医疗敏感内容",
                warning_message=safety_result.warning_message or "建议咨询专业医生"
            )

        return GENERAL_GUIDANCE_TEMPLATE.format(user_question=user_question)

    async def _try_get_cached_response(
        self,
        request: HospitalChatRequest,
        conversation_id: str,
        question_embedding: Optional[List[float]] = None
    ) -> Optional[HospitalChatResponse]:
        """尝试从缓存获取响应"""
        if not self.redis_cache_service or not self.redis_cache_service.is_available:
            return None
        
        try:
            # 使用预计算向量或按需生成
            if question_embedding is None:
                question_embedding = await self.embedding_service.embed_query(request.message)
            
            # 查找相似问题
            cached_response = await self.redis_cache_service.get_cached_response(
                question=request.message,
                question_embedding=question_embedding,
                threshold=chat_config.redis_vector_threshold
            )
            
            if cached_response:
                # 命中缓存，直接返回
                logger.info(f"命中缓存: question={request.message[:30]}...")
                
                return HospitalChatResponse(
                    message=cached_response,
                    conversation_id=conversation_id,
                    steps=[{
                        "step_name": "cache_hit",
                        "step_order": 0,
                        "status": "success",
                        "note": "来自缓存的相似问题回答"
                    }],
                    documents_used=[],
                    safety_passed=True,
                    stream_available=True,
                    cache_hit=True
                )
        except Exception as e:
            logger.warning(f"缓存检查失败，继续正常流程: {str(e)}")
        
        return None
    
    async def _store_to_cache(
        self,
        request: HospitalChatRequest,
        response: str,
        conversation_id: str,
        question_embedding: Optional[List[float]] = None
    ) -> None:
        """将对话存储到 Redis 缓存"""
        if not self.redis_cache_service or not self.redis_cache_service.is_available:
            return
        
        try:
            # 使用预计算向量或按需生成
            if question_embedding is None:
                question_embedding = await self.embedding_service.embed_query(request.message)
            
            # 异步存储（不阻塞返回）
            await self.redis_cache_service.store_conversation(
                conversation_id=conversation_id,
                question=request.message,
                answer=response,
                embedding=question_embedding
            )
            logger.info(f"对话已缓存: conversation_id={conversation_id}")
        except Exception as e:
            logger.warning(f"存储缓存失败: {str(e)}")
    
    async def execute(
        self, 
        request: HospitalChatRequest
    ) -> HospitalChatResponse:
        """
        执行完整的 Agent 流程
        
        Args:
            request: 医院客服聊天请求
            
        Returns:
            HospitalChatResponse: Agent 执行结果
        """
        conversation_id = request.conversation_id or f"conv_{int(time.time())}"
        steps: List[Dict[str, Any]] = []
        documents_used: List[str] = []
        cache_hit = False
        question_embedding: Optional[List[float]] = None
        
        try:
            # 0. 预计算问题向量（用于缓存检查和存储，避免重复计算）
            if self.embedding_service and self.redis_cache_service and self.redis_cache_service.is_available:
                question_embedding = await self.embedding_service.embed_query(request.message)
            
            # 0. 检查 Redis 缓存（问题去重）
            cached_response = await self._try_get_cached_response(request, conversation_id, question_embedding)
            if cached_response:
                return cached_response
            
            # 步骤1：问题重写
            step1_result = await self.step1_rewrite_question(
                request.message, 
                conversation_id
            )
            steps.append(step1_result.model_dump())
            rewritten_queries = step1_result.output_data.get("rewritten_queries", [request.message])
            
            # 步骤2：安全审查
            step2_result, safety_result = await self.step2_safety_check(
                request.message
            )
            steps.append(step2_result.model_dump())
            
            # 如果安全审查失败且风险等级高，直接返回警告
            if not safety_result.can_proceed:
                warning_response = await self._generate_fallback_response(
                    request.message,
                    safety_result
                )
                
                return HospitalChatResponse(
                    message=warning_response,
                    conversation_id=conversation_id,
                    steps=steps,
                    documents_used=[],
                    safety_passed=False,
                    stream_available=True
                )
            
            # 步骤3：知识检索
            step3_result, documents = await self.step3_knowledge_retrieval(
                rewritten_queries,
                self.config.top_k
            )
            steps.append(step3_result.model_dump())
            documents_used = [doc["content"] for doc in documents[:5]]
            
            # 构建 RAG 上下文
            rag_context = "\n\n".join([
                doc["content"] for doc in documents[:5]
            ]) or "暂无相关检索结果"
            
            # 步骤4：答案生成（含质量评估）
            step4_result, response, quality_evaluation = await self.step4_generate_answer(
                request.message,
                rag_context,
                safety_result,
                conversation_id
            )
            steps.append(step4_result.model_dump())

            # 记录业务事件
            logger.log_business_event(
                "医院客服对话",
                success=True,
                conversation_id=conversation_id,
                message_length=len(request.message),
                response_length=len(response),
                steps_count=len(steps),
                cache_hit=cache_hit,
                quality_score=quality_evaluation.get("quality_score"),
                is_solved=quality_evaluation.get("is_solved")
            )

            # 使用 step4 返回的质量评估结果决定是否缓存
            if step4_result.status == "success":
                # is_solved 已经是最终判断结果（分数 >= 6 且非低质量模式）
                if quality_evaluation.get("is_solved"):
                    logger.info(
                        f"答案通过质量评估，可缓存: score={quality_evaluation.get('quality_score')}, reason={quality_evaluation.get('eval_reason')}"
                    )
                    await self._store_to_cache(request, response, conversation_id, question_embedding)
                else:
                    logger.info(
                        f"答案未通过质量评估，不缓存: reason={quality_evaluation.get('eval_reason')}"
                    )

            return HospitalChatResponse(
                message=response,
                conversation_id=conversation_id,
                steps=steps,
                documents_used=documents_used,
                safety_passed=safety_result.is_safe,
                stream_available=True,
                cache_hit=cache_hit
            )
            
        except Exception as e:
            logger.error(f"Agent 执行失败: {str(e)}")
            logger.log_business_event(
                "医院客服对话",
                success=False,
                conversation_id=conversation_id,
                error=str(e)
            )
            
            # 返回兜底响应
            return HospitalChatResponse(
                message="抱歉，服务暂时繁忙，请稍后重试或拨打医院咨询电话：010-69156114",
                conversation_id=conversation_id,
                steps=steps,
                documents_used=documents_used,
                safety_passed=True,
                stream_available=True
            )
    
    async def execute_stream(
        self, 
        request: HospitalChatRequest,
        callback: Optional[Any] = None
    ) -> AsyncGenerator[str, None]:
        """
        执行流式 Agent 流程
        
        Args:
            request: 医院客服聊天请求
            callback: 回调处理器
            
        Yields:
            str: 流式输出的内容块
        """
        conversation_id = request.conversation_id or f"conv_{int(time.time())}"
        
        try:
            # 发送开始事件
            if callback:
                await callback.on_chat_model_start(
                    serialized={},
                    prompts=[],
                    invocation_params={}
                )
            
            # 步骤1：问题重写
            yield json.dumps({
                "event": "step_start",
                "step": "question_rewrite",
                "message": "正在分析您的问题..."
            }, ensure_ascii=False) + "\n"
            
            step1_result = await self.step1_rewrite_question(
                request.message, 
                conversation_id
            )
            
            yield json.dumps({
                "event": "step_complete",
                "step": "question_rewrite",
                "result": step1_result.output_data
            }, ensure_ascii=False) + "\n"
            
            rewritten_queries = step1_result.output_data.get("rewritten_queries", [request.message])
            
            # 步骤2：安全审查
            yield json.dumps({
                "event": "step_start",
                "step": "safety_check",
                "message": "正在进行安全审查..."
            }, ensure_ascii=False) + "\n"
            
            _, safety_result = await self.step2_safety_check(
                request.message
            )
            
            yield json.dumps({
                "event": "step_complete",
                "step": "safety_check",
                "result": safety_result.model_dump()
            }, ensure_ascii=False) + "\n"
            
            # 安全审查失败处理
            if not safety_result.can_proceed:
                warning_response = await self._generate_fallback_response(
                    request.message,
                    safety_result
                )
                yield json.dumps({
                    "event": "content",
                    "content": warning_response,
                    "is_final": True
                }, ensure_ascii=False) + "\n"
                return
            
            # 步骤3：知识检索
            yield json.dumps({
                "event": "step_start",
                "step": "knowledge_retrieval",
                "message": "正在检索相关知识..."
            }, ensure_ascii=False) + "\n"
            
            _, documents = await self.step3_knowledge_retrieval(
                rewritten_queries,
                self.config.top_k
            )
            
            yield json.dumps({
                "event": "step_complete",
                "step": "knowledge_retrieval",
                "result": {
                    "documents_found": len(documents)
                }
            }, ensure_ascii=False) + "\n"
            
            # 构建 RAG 上下文
            rag_context = "\n\n".join([
                doc["content"] for doc in documents[:5]
            ]) or "暂无相关检索结果"
            
            # 步骤4：答案生成（流式）
            yield json.dumps({
                "event": "step_start",
                "step": "answer_generation",
                "message": "正在生成回答..."
            }, ensure_ascii=False) + "\n"
            
            # 获取当前时间
            current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
            
            # 构建安全提醒（使用私有方法）
            safety_reminder = self._build_safety_reminder(safety_result)
            safety_check_result = self._build_safety_check_result(safety_result)
            
            # 获取对话历史
            history = await self._get_conversation_history(conversation_id)
            
            # 构建消息（使用统一的提示词模板）
            chat_history = []
            for msg in history[-_MAX_HISTORY_MESSAGES_FOR_ANSWER:]:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))
            
            # 构建提示
            prompt_content = ANSWER_GENERATION_PROMPT.invoke({
                "current_time": current_time,
                "rag_context": rag_context,
                "user_question": request.message,
                "safety_check_result": safety_check_result,
                "safety_reminder": safety_reminder,
                "chat_history": chat_history
            })
            
            # 转换消息格式
            def _convert_message_type(msg_type: str) -> str:
                mapping = {"human": "user", "ai": "assistant", "system": "system"}
                return mapping.get(msg_type, msg_type)
            
            messages = [{"role": _convert_message_type(msg.type), "content": msg.content} for msg in prompt_content.messages]
            
            # 流式调用通义千问
            full_response = ""
            stream = self.llm_service.qwen_llm.stream(messages)
            
            for chunk in stream:
                if hasattr(chunk, 'content') and chunk.content:
                    delta = chunk.content
                    full_response += delta
                    yield json.dumps({
                        "event": "content",
                        "content": delta,
                        "is_final": False
                    }, ensure_ascii=False) + "\n"
            
            # 更新对话历史
            await self._add_to_history(conversation_id, "user", request.message)
            await self._add_to_history(conversation_id, "assistant", full_response)
            
            # 发送完成事件
            yield json.dumps({
                "event": "done",
                "conversation_id": conversation_id,
                "documents_used": [doc["content"] for doc in documents[:5]],
                "safety_passed": safety_result.is_safe
            }, ensure_ascii=False) + "\n"
            
        except Exception as e:
            logger.error(f"流式 Agent 执行失败: {str(e)}")
            yield json.dumps({
                "event": "error",
                "error": str(e),
                "message": "服务暂时繁忙，请稍后重试。"
            }, ensure_ascii=False) + "\n"
