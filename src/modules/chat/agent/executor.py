"""
通用Agent执行器
支持多领域配置，通过domain参数切换不同业务场景

检索说明：
    - 使用 Milvus 2.6+ 原生混合检索（Dense + Sparse BM25）
    - 无需手动实现 BM25 和 RRF 融合
"""

import json
import time
import asyncio
from typing import AsyncGenerator, Optional, List, Dict, Any, Union

from src.modules.chat.core.llm_service import LLMService
from src.modules.chat.core.embedding_service import EmbeddingService
from src.modules.chat.core.milvus_service import MilvusService
from src.modules.chat.core.redis_cache_service import RedisCacheService
from src.modules.chat.core.reranker_service import RerankerService
from src.modules.monitoring.langchain_callback import get_prometheus_callback
from src.modules.monitoring.langfuse_callback import create_langfuse_handler

from datetime import datetime, timedelta

from src.modules.chat.agent.prompts import PromptTemplateManager
from src.modules.chat.agent.schemas import (
    AgentStepResult,
    SafetyCheckResult,
    QuestionRewriteResult,
    RetrievalResult,
    get_structured_schema,
)
from src.modules.chat.schemas import (
    ChatRequest,
    ChatResponse,
)
from src.modules.chat.config import AgentConfig, get_agent_config
from src.shared.logger import APILogger

logger = APILogger("general_agent_executor")


# 类常量定义
_CLEANUP_INTERVAL = 10  # 每N次调用清理一次过期历史
_MAX_DOC_CHARS = 800  # 每篇文档最大字符数（截断以减少 LLM 输入 token，加快推理）


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
        domain: str = "ecommerce",
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
            
            messages = [
                {"role": "system", "content": template},
                {"role": "user", "content": f"用户问题：{user_question}"}
            ]
            
            response = await self.llm_service.chat_qwen(
                messages, 
                langfuse_handler=getattr(self, '_langfuse_handler', None)
            )
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
            # 构建消息
            template = PromptTemplateManager.get(self.domain, step_config.prompt_template_key)
            messages = [
                {"role": "system", "content": template},
                {"role": "user", "content": f"用户问题：{user_question}"}
            ]
            
            # 根据配置选择输出方式
            safety_data = None
            if step_config.output_format == "json" and step_config.response_schema:
                # 使用结构化输出
                schema_class = get_structured_schema(step_config.response_schema)
                if schema_class:
                    try:
                        structured_result = await self.llm_service.chat_qwen_structured(
                            messages, 
                            schema_class,
                            temperature=0.0,  # 结构化输出用低温
                            langfuse_handler=getattr(self, '_langfuse_handler', None)
                        )
                        # 转换为字典
                        if hasattr(structured_result, 'model_dump'):
                            safety_data = structured_result.model_dump()
                        else:
                            safety_data = structured_result
                        logger.info(f"[{self.domain}] {step_config.name}使用结构化输出")
                    except Exception as e:
                        logger.warning(
                            f"[{self.domain}] {step_config.name}结构化输出失败，降级到JSON解析: {str(e)[:100]}"
                        )
                        # 降级到手动解析
            if safety_data is None:
                response = await self.llm_service.chat_qwen(
                messages, 
                langfuse_handler=getattr(self, '_langfuse_handler', None)
            )
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
        top_k: int,
        user_question: str = "",
        precomputed_embedding: Optional[List[float]] = None
    ) -> tuple[AgentStepResult, List[Dict[str, Any]]]:
        """
        步骤3：知识检索（Milvus 2.6+ 原生混合检索）
        
        使用 Milvus 原生混合检索（Dense向量 + Sparse BM25），
        无需手动 RRF 融合。
        
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
            seen_content = set()  # 用于 O(1) 去重
            rerank_enabled = getattr(self.config, 'rerank_enabled', False)
            # Rerank 开启时从 Milvus 多拿几条供 Reranker 筛选
            milvus_top_k = getattr(self.config, 'rerank_initial_top_k', top_k * 4) if rerank_enabled else top_k
            
            for i, query in enumerate(queries[:self.config.max_retrieval_queries]):
                if self.embedding_service and self.milvus_service:
                    # 如果预计算向量可用且是第一条查询，直接复用，省一次 API 调用
                    if i == 0 and precomputed_embedding is not None:
                        query_embedding = precomputed_embedding
                    else:
                        query_embedding = await self.embedding_service.embed_query(query)
                    
                    # 使用 Milvus 2.6 原生混合检索（Dense + Sparse BM25）
                    docs = self.milvus_service.hybrid_search(
                        query_embedding=query_embedding,
                        query_text=query,
                        top_k=milvus_top_k,
                        rrf_k=getattr(self.config, 'rrf_k', 60)
                    )
                    
                    score_threshold = getattr(self.config, 'retrieval_score_threshold', 0.0)
                    for doc in docs:
                        content = doc.page_content
                        if content not in seen_content:
                            # hybrid_search 使用 RRFRanker，distance 即为 RRF 融合分数（越高越相关）
                            rrf_score = doc.metadata.get('distance', 0)
                            # 按配置的阈值过滤低分文档
                            if rrf_score < score_threshold:
                                continue
                            seen_content.add(content)
                            all_documents.append({
                                "content": content,
                                "metadata": doc.metadata,
                                "source_query": query,
                                "score": round(rrf_score, 4)
                            })
            
            # ============================================================
            # BGE-Reranker 重排序 + 低相关截断
            # ============================================================
            if rerank_enabled and all_documents:
                try:
                    rerank_threshold = getattr(self.config, 'rerank_threshold', 0.3)
                    rerank_top_k = getattr(self.config, 'rerank_top_k', top_k)
                    
                    doc_contents = [doc["content"] for doc in all_documents]
                    reranker = RerankerService.get_instance()
                    # 将同步 CPU 推理放入线程池，避免阻塞 asyncio 事件循环
                    loop = asyncio.get_event_loop()
                    ranked = await loop.run_in_executor(
                        None,
                        lambda: reranker.rerank(
                            query=user_question,
                            documents=doc_contents,
                            top_k=rerank_top_k,
                            threshold=rerank_threshold
                        )
                    )
                    
                    original_count = len(all_documents)
                    # 重建文档列表（保留元数据，更新分数）
                    all_documents = [
                        {
                            **all_documents[idx],
                            "score": round(score, 4)  # Reranker 分数覆盖 RRF 分数
                        }
                        for idx, score, _ in ranked
                    ]
                    
                    discarded = original_count - len(all_documents)
                    if discarded > 0:
                        logger.info(
                            f"[{self.domain}] Rerank 移除了 {discarded} 条低相关文档",
                            before=original_count, after=len(all_documents),
                            threshold=rerank_threshold
                        )
                    if not all_documents:
                        logger.warning(
                            f"[{self.domain}] Rerank 后无文档通过阈值",
                            threshold=rerank_threshold, original_count=original_count
                        )
                except Exception as e:
                    logger.warning(f"[{self.domain}] Rerank 失败，保留原始结果: {str(e)[:150]}")
            
            # LLM 相关性过滤：过滤语义不相关的检索结果
            if getattr(self.config, 'relevance_filter_enabled', True) and all_documents:
                original_count = len(all_documents)
                all_documents = await self._filter_documents_by_relevance(
                    user_question, all_documents
                )
                removed_count = original_count - len(all_documents)
                if removed_count > 0:
                    logger.info(
                        f"[{self.domain}] 相关性过滤移除了 {removed_count} 条不相关文档",
                        remaining=len(all_documents)
                    )
                if not all_documents:
                    logger.warning(
                        f"[{self.domain}] 相关性过滤后无相关文档，将使用空上下文",
                        original_count=original_count
                    )
            
            rag_context = "\n\n".join([
                f"[来源: {doc.get('metadata', {}).get('source', '未知')}]\n{doc['content']}"
                for doc in all_documents[:top_k]
            ]) or "暂无相关检索结果"
            
            duration = int((time.time() - start_time) * 1000)
            logger.info(f"[{self.domain}] {step_config.name}完成",
                        document_count=len(all_documents),
                        hybrid_search="milvus_native")
            
            step_result = AgentStepResult(
                step_name=step_config.name,
                step_order=3,
                input_data={"queries": queries, "top_k": top_k},
                output_data={
                    "context": rag_context[:500] + "..." if len(rag_context) > 500 else rag_context,
                    "documents_found": len(all_documents),
                    "hybrid_search_enabled": True,
                    "hybrid_search_type": "milvus_native_sparse_bm25",
                    "rerank_enabled": rerank_enabled,
                    "rerank_model": "BAAI/bge-reranker-base" if rerank_enabled else None,
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
    
    async def _filter_documents_by_relevance(
        self,
        user_question: str,
        documents: List[Dict[str, Any]],
        max_docs: int = 20
    ) -> List[Dict[str, Any]]:
        """
        使用 LLM 过滤语义不相关的检索文档
        
        Args:
            user_question: 原始用户问题
            documents: 检索到的文档列表
            max_docs: 最多传给 LLM 判断的文档数（节省 token）
            
        Returns:
            过滤后的相关文档列表
        """
        if not documents:
            return []
        
        # 控制传给 LLM 的文档数量
        docs_to_check = documents[:max_docs]
        if len(documents) <= 1:
            return documents  # 只有1条就不过滤了
        
        # 构建判断提示
        doc_list = "\n---\n".join([
            f"[文档{i}] {doc['content'][:300]}"
            for i, doc in enumerate(docs_to_check)
        ])
        
        prompt = f"""你是信息相关性判断助手。请判断以下文档是否与用户问题相关。

用户问题：{user_question}

判断标准：
- 相关：文档内容能直接帮助回答问题
- 不相关：文档内容是无关领域（如公司请假制度 vs 电商咨询）、或与问题完全无关

检索到的文档：
{doc_list}

请严格按JSON格式输出，只输出不相关文档的编号列表：
```json
{{"irrelevant_ids": [1, 3]}}
```
如果没有不相关文档，输出：
```json
{{"irrelevant_ids": []}}
```"""
        
        try:
            response = await self.llm_service.chat_qwen(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                langfuse_handler=getattr(self, '_langfuse_handler', None)
            )
            
            # 解析 JSON
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            
            result = json.loads(json_str.strip())
            irrelevant_ids = set(result.get("irrelevant_ids", []))
            
            if not irrelevant_ids:
                return documents
            
            # 过滤掉不相关的文档
            filtered = [
                doc for i, doc in enumerate(documents)
                if i not in irrelevant_ids
            ]
            
            logger.info(
                f"[{self.domain}] LLM 相关性过滤完成",
                before=len(documents),
                after=len(filtered),
                removed_ids=list(irrelevant_ids)[:5]
            )
            return filtered
            
        except Exception as e:
            logger.warning(f"[{self.domain}] 相关性过滤失败，保留所有文档: {str(e)[:100]}")
            return documents  # 失败时保守策略：不过滤
    
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
            
            # 获取提示词模板（提前获取以判断是否需要 chat_history）
            template = PromptTemplateManager.get(self.domain, step_config.prompt_template_key)
            needs_chat_history = "{chat_history}" in template
            
            history = await self._get_conversation_history(conversation_id)
            chat_history_str = ""
            if needs_chat_history and history:
                # 限制历史条数 + 每条截断到 300 字，减少不必要 token
                recent = history[-self.config.max_history_turns:]
                parts = []
                for msg in recent:
                    role = "用户" if msg["role"] == "user" else "助手"
                    content = msg["content"][:300]
                    if len(msg["content"]) > 300:
                        content += "..."
                    parts.append(f"{role}: {content}")
                chat_history_str = "\n".join(parts)
            
            # 构建提示
            prompt_content = template.format(
                    current_time=current_time,
                    rag_context=rag_context,
                    user_question=user_question,
                    safety_check_result=safety_check_result,
                    safety_reminder=safety_reminder,
                    chat_history=chat_history_str,
                    product_info=rag_context,
                    knowledge_base=rag_context,
                    context=rag_context,
                    category=""
                )
            messages = [{"role": "system", "content": prompt_content}]
            
            # 回答生成用较低 temperature，减少随机采样开销，输出更稳定
            response = await self.llm_service.chat_qwen(
                messages, 
                temperature=0.3,
                langfuse_handler=getattr(self, '_langfuse_handler', None)
            )
            
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
        request: ChatRequest,
        langfuse_handler=None,
    ) -> ChatResponse:
        """
        执行完整的Agent流程
        
        Args:
            request: 聊天请求
            langfuse_handler: 外部 Langfuse CallbackHandler（由上层编排器传入）
            
        Returns:
            ChatResponse: Agent执行结果
        """
        # 获取 Prometheus 回调
        callback = get_prometheus_callback()
        executor_start_time = time.time()
        
        # Langfuse v4.x: 外部传入 handler 时复用，否则内部创建
        if langfuse_handler is None:
            conversation_id = request.conversation_id or f"conv_{int(time.time())}"
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
        
        steps: List[Dict[str, Any]] = []
        documents_used: List[str] = []
        question_embedding: Optional[List[float]] = None
        
        try:
            # 预计算向量（供 step3 复用，避免重复调 embedding API）
            if self.embedding_service:
                question_embedding = await self.embedding_service.embed_query(request.message)
            
            # 检查缓存
            # cached_response = await self._try_get_cached_response(
            #     request.message, conversation_id, question_embedding
            # )
            # if cached_response:
            #     return ChatResponse(
            #         message=cached_response,
            #         conversation_id=conversation_id,
            #         steps=[{"step_name": "cache_hit", "step_order": 0, "status": "success"}],
            #         documents_used=[],
            #         safety_passed=True,
            #         stream_available=True,
            #         cache_hit=True
            #     )
            
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
                return ChatResponse(
                    message=warning_response,
                    conversation_id=conversation_id,
                    steps=steps,
                    documents_used=[],
                    safety_passed=False,
                    stream_available=True
                )
            
            # 步骤3 - 传入预计算向量，避免重复调 embedding API
            step3_result, documents = await self.step3_retrieve(
                queries, self.config.top_k, request.message,
                precomputed_embedding=question_embedding
            )
            steps.append(step3_result.model_dump())
            # 截断每篇文档到 _MAX_DOC_CHARS，减少 LLM 输入 token 数
            documents_used = [
                doc["content"][:_MAX_DOC_CHARS] for doc in documents[:5]
            ]
            
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
            
            # 记录执行器级别 Prometheus 指标
            executor_duration = time.time() - executor_start_time
            callback.on_chain_end(
                outputs={"status": "success", "steps": len(steps)},
                run_id=f"executor_{conversation_id}"
            )
            logger.info(f"[{self.domain}] 执行器完成", 
                        duration_ms=int(executor_duration * 1000),
                        steps=len(steps))
            
            # 缓存高质量答案
            if step4_result.status == "success" and quality_evaluation.get("is_solved"):
                await self._store_to_cache(
                    request.message, response, conversation_id, question_embedding
                )
            
            return ChatResponse(
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
            
            # 记录执行器失败指标
            callback.on_chain_error(
                error=e,
                run_id=f"executor_{conversation_id}"
            )
            
            return ChatResponse(
                message="抱歉，服务暂时繁忙，请稍后重试。",
                conversation_id=conversation_id,
                steps=steps,
                documents_used=documents_used,
                safety_passed=True,
                stream_available=True
            )
        finally:
            # Langfuse v4.x: 退出 propagate_attributes 上下文
            if self._langfuse_ctx:
                self._langfuse_ctx.__exit__(None, None, None)


# =============================================================================
# 向后兼容的类型别名
# =============================================================================

HospitalAgentExecutor = GeneralAgentExecutor
