"""
医院客服 Agent 执行器
使用 LangChain 1.2.15 实现多步骤 skill 编排
"""

import json
import time
from typing import AsyncGenerator, Optional, List, Dict, Any
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, AIMessage

from src.modules.chat.core.embedding_service import EmbeddingService
from src.modules.chat.core.milvus_service import MilvusService
from src.modules.chat.core.llm_service import LLMService
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


class HospitalAgentExecutor:
    """
    医院客服 Agent 执行器
    
    实现多步骤 skill 编排：
    1. 问题重写 - 将用户问题改写为更适合检索的查询
    2. 安全审查 - 检查问题是否涉及医疗建议、处方、诊断等敏感内容
    3. 知识检索 - 使用现有的 RAG 组件从医疗知识库检索相关内容
    4. 答案生成 - 基于检索结果生成安全、准确的回复
    """
    
    def __init__(
        self,
        config: Optional[HospitalAgentConfig] = None,
        llm_service = None,
        embedding_service = None,
        milvus_service = None,
    ):
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.milvus_service = milvus_service
        self.config = config or HospitalAgentConfig()
        self._history: Dict[str, List[Dict[str, str]]] = {}  # conversation_id -> history
        self._last_access: Dict[str, datetime] = {}  # 追踪最后访问时间
        self._history_max_age = timedelta(hours=24)  # 历史记录过期时间

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

    def _get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """获取对话历史"""
        if conversation_id not in self._history:
            self._history[conversation_id] = []
        # 更新最后访问时间
        self._last_access[conversation_id] = datetime.now()
        # 定期清理过期历史
        self._cleanup_old_history()
        return self._history[conversation_id]

    def _add_to_history(self, conversation_id: str, role: str, content: str):
        """添加对话历史"""
        history = self._get_conversation_history(conversation_id)
        history.append({"role": role, "content": content})
        # 保持最大历史轮次
        if len(history) > self.config.max_history_turns * 2:
            self._history[conversation_id] = history[-self.config.max_history_turns * 2:]
    
    async def _call_doubao(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> Any:
        """
        调用 Doubao 模型
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            stream: 是否流式输出
            
        Returns:
            模型响应
        """
        try:
            temperature = temperature or self.config.temperature
            if stream:
                return self.llm_service.chat_doubao(
                    messages=messages,
                    temperature=temperature,
                    stream=True
                )
            else:
                return await self.llm_service.chat_doubao_async(
                    messages=messages,
                    temperature=temperature
                )
                
        except Exception as e:
            logger.error(f"Doubao API 调用失败: {str(e)}")
            raise
    
    async def step1_rewrite_question(
        self, 
        user_question: str,
        conversation_id: str
    ) -> AgentStepResult:
        """
        步骤1：问题重写
        
        将用户问题改写为更适合检索的查询
        """
        start_time = time.time()
        
        try:
            # 使用 invoke 获取格式化后的消息
            prompt_obj = QUESTION_REWRITING_PROMPT.invoke({"user_question": user_question})
            messages = [{"role": "system", "content": prompt_obj.messages[0].content}]
            messages.append({"role": "user", "content": f"用户问题：{user_question}"})
            
            # 调用模型
            response = await self._call_doubao(messages)
            
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
        user_question: str,
        conversation_id: str
    ) -> tuple[AgentStepResult, SafetyCheckResult]:
        """
        步骤2：安全审查
        
        检查问题是否涉及医疗建议、处方、诊断等敏感内容
        """
        start_time = time.time()
        
        try:
            # 使用 invoke 获取格式化后的消息
            prompt_obj = SAFETY_CHECK_PROMPT.invoke({"user_question": user_question})
            messages = [{"role": "system", "content": prompt_obj.messages[0].content}]
            messages.append({"role": "user", "content": f"用户问题：{user_question}"})
            
            # 调用模型
            response = await self._call_doubao(messages)
            
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
            
            # 默认不阻止，但记录错误
            safety_result = SafetyCheckResult(
                is_safe=True,
                risk_level="low",
                risk_categories=[],
                warning_message="安全审查服务暂时不可用",
                can_proceed=True
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
        top_k: int,
        conversation_id: str
    ) -> tuple[AgentStepResult, List[Dict[str, Any]]]:
        """
        步骤3：知识检索
        
        使用现有的 RAG 组件从医疗知识库检索相关内容
        """
        start_time = time.time()
        all_documents = []
        retrieval_results = []
        
        try:
            for query in rewritten_queries[:3]:  # 最多使用3个查询
                # 使用嵌入服务和 Milvus 服务检索
                query_embedding = await self.embedding_service.embed_query(query)
                docs = self.milvus.search_similar(
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
    ) -> tuple[AgentStepResult, str]:
        """
        步骤4：答案生成
        
        基于检索结果生成安全、准确的回复
        """
        start_time = time.time()
        
        try:
            # 获取当前时间
            current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
            
            # 构建安全提醒
            if not safety_result.is_safe:
                safety_reminder = SAFETY_WARNING_TEMPLATE.format(
                    risk_categories=", ".join(safety_result.risk_categories),
                    warning_message=safety_result.warning_message or "请咨询专业医生"
                )
            else:
                safety_reminder = "问题已通过安全审查，可以正常回答。"
            
            # 构建安全审查结果描述
            safety_check_result = f"风险等级: {safety_result.risk_level}"
            if safety_result.risk_categories:
                safety_check_result += f"\n涉及内容: {', '.join(safety_result.risk_categories)}"
            
            # 获取对话历史
            history = self._get_conversation_history(conversation_id)
            chat_history = []
            for msg in history[-10:]:  # 只使用最近10条
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))
            
            # 构建提示
            prompt_content = ANSWER_GENERATION_PROMPT.format(
                current_time=current_time,
                rag_context=rag_context,
                user_question=user_question,
                safety_check_result=safety_check_result,
                safety_reminder=safety_reminder,
                chat_history=chat_history
            )
            
            # 构建消息
            def _convert_message_type(msg_type: str) -> str:
                """将 LangChain 消息类型转换为 API 期望的角色"""
                mapping = {"human": "user", "ai": "assistant", "system": "system"}
                return mapping.get(msg_type, msg_type)
            
            messages = [{"role": _convert_message_type(msg.type), "content": msg.content} for msg in prompt_content.messages]
            
            # 调用模型
            response = await self._call_doubao(messages)
            
            duration = int((time.time() - start_time) * 1000)
            
            # 更新对话历史
            self._add_to_history(conversation_id, "user", user_question)
            self._add_to_history(conversation_id, "assistant", response)
            
            logger.info(
                f"答案生成完成",
                response_length=len(response),
                duration_ms=duration
            )
            
            return AgentStepResult(
                step_name="answer_generation",
                step_order=4,
                input_data={
                    "user_question": user_question,
                    "context_length": len(rag_context),
                    "safety_passed": safety_result.is_safe
                },
                output_data={"response": response},
                status="success",
                duration_ms=duration
            ), response
            
        except Exception as e:
            duration = int((time.time() - start_time) * 1000)
            logger.error(f"答案生成失败: {str(e)}")
            
            return AgentStepResult(
                step_name="answer_generation",
                step_order=4,
                status="failed",
                error_message=str(e),
                duration_ms=duration
            ), "抱歉，服务暂时繁忙，请稍后重试。"
    
    async def _generate_fallback_response(
        self,
        user_question: str,
        safety_result: SafetyCheckResult,
        conversation_id: str
    ) -> str:
        """生成兜底响应"""
        if not safety_result.is_safe:
            return SAFETY_WARNING_TEMPLATE.format(
                risk_categories=", ".join(safety_result.risk_categories) or "医疗敏感内容",
                warning_message=safety_result.warning_message or "建议咨询专业医生"
            )
        
        return GENERAL_GUIDANCE_TEMPLATE.format(user_question=user_question)
    
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
        steps = []
        documents_used = []
        
        try:
            # 步骤1：问题重写
            step1_result = await self.step1_rewrite_question(
                request.message, 
                conversation_id
            )
            steps.append(step1_result.model_dump())
            rewritten_queries = step1_result.output_data.get("rewritten_queries", [request.message])
            
            # 步骤2：安全审查
            step2_result, safety_result = await self.step2_safety_check(
                request.message,
                conversation_id
            )
            steps.append(step2_result.model_dump())
            
            # 如果安全审查失败且风险等级高，直接返回警告
            if not safety_result.can_proceed:
                warning_response = await self._generate_fallback_response(
                    request.message,
                    safety_result,
                    conversation_id
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
                self.config.top_k,
                conversation_id
            )
            steps.append(step3_result.model_dump())
            documents_used = [doc["content"] for doc in documents[:5]]
            
            # 构建 RAG 上下文
            rag_context = "\n\n".join([
                doc["content"] for doc in documents[:5]
            ]) or "暂无相关检索结果"
            
            # 步骤4：答案生成
            step4_result, response = await self.step4_generate_answer(
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
                steps_count=len(steps)
            )
            
            return HospitalChatResponse(
                message=response,
                conversation_id=conversation_id,
                steps=steps,
                documents_used=documents_used,
                safety_passed=safety_result.is_safe,
                stream_available=True
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
            
            step2_result, safety_result = await self.step2_safety_check(
                request.message,
                conversation_id
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
                    safety_result,
                    conversation_id
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
            
            step3_result, documents = await self.step3_knowledge_retrieval(
                rewritten_queries,
                self.config.top_k,
                conversation_id
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
            
            # 构建安全提醒
            if not safety_result.is_safe:
                safety_reminder = SAFETY_WARNING_TEMPLATE.format(
                    risk_categories=", ".join(safety_result.risk_categories),
                    warning_message=safety_result.warning_message or "请咨询专业医生"
                )
            else:
                safety_reminder = "问题已通过安全审查，可以正常回答。"
            
            safety_check_result = f"风险等级: {safety_result.risk_level}"
            if safety_result.risk_categories:
                safety_check_result += f"\n涉及内容: {', '.join(safety_result.risk_categories)}"
            
            # 构建提示
            system_prompt = f"""# Role
你是「医院智能客服助手」，负责为患者提供**准确、安全、合规**的就医咨询服务。

# 核心原则（必须遵守）
1. **优先使用 RAG 检索到的知识**
   - 所有回答必须以 <context> 标签内的内容为主要依据。
   - 若 <context> 中存在与患者问题直接相关的内容，必须直接使用，不得忽略或改写事实。

2. **禁止凭空编造**
   - 严禁基于通用知识"脑补"医院政策、科室设置、医生排班、费用、医保规则等。
   - 若 <context> 中没有相关信息，必须明确告知"暂未查询到相关信息"。

3. **区分事实与通用建议**
   - <context> 内：视为医院官方事实，可明确表述。
   - <context> 外：仅可作为非诊疗性健康科普，并明确说明"仅供参考，请以医生为准"。

4. **安全与合规**
   - 不做诊断、不开处方、不下确定性预后结论。
   - 涉及急危重症时，必须提示立即就医或拨打急救电话。

# 当前时间
{current_time}

# 检索到的知识
<context>
{rag_context}
</context>

# 用户问题
{request.message}

# 安全审查结果
{safety_check_result}

# 重要提醒：
{safety_reminder}"""
            
            # 获取对话历史
            history = self._get_conversation_history(conversation_id)
            
            # 构建消息
            messages = [{"role": "system", "content": system_prompt}]
            for msg in history[-10:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": "请基于以上信息生成回答。"})
            
            # 流式调用
            full_response = ""
            stream = self.llm_service.ark_client.chat.completions.create(
                model=chat_config.chat_model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=4096,
                stream=True,
            )
            
            for chunk in stream:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_response += delta
                        yield json.dumps({
                            "event": "content",
                            "content": delta,
                            "is_final": False
                        }, ensure_ascii=False) + "\n"
            
            # 更新对话历史
            self._add_to_history(conversation_id, "user", request.message)
            self._add_to_history(conversation_id, "assistant", full_response)
            
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
