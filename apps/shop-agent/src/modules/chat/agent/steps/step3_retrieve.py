"""步骤3：知识检索（Milvus 2.6+ 原生混合检索 + BGE-Reranker）。"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

from src.modules.chat.agent.schemas import AgentStepResult
from src.modules.chat.agent.steps.base import AgentContext, BaseStep
from src.modules.chat.core.reranker_service import RerankerService
from src.shared.logger import APILogger

logger = APILogger("step3_retrieve")

_MAX_DOC_CHARS = 800


@dataclass
class StepSuccessContext:
    """步骤成功执行上下文，封装 _build_success_result 所需参数。"""
    ctx: AgentContext
    step_config: Any
    queries: list
    top_k: int
    all_documents: List[Dict[str, Any]]
    duration: int


@dataclass
class RetrieveQueryContext:
    """单次检索查询上下文，封装 _retrieve_single_query 所需参数。"""
    ctx: AgentContext
    query: str
    query_index: int
    milvus_top_k: int
    seen_content: Set[str]
    all_documents: List[Dict[str, Any]]


class RetrieveStep(BaseStep):
    """Milvus 混合检索 + Reranker 重排序 + LLM 相关性过滤。"""

    step_name = "知识检索"
    step_order = 3

    async def execute(self, ctx: AgentContext) -> Tuple[AgentStepResult, List[Dict[str, Any]]]:
        step_config = getattr(ctx.config, "step3", None)
        if not step_config or not step_config.enabled:
            return (
                AgentStepResult(
                    step_name=step_config.name if step_config else "知识检索",
                    step_order=self.step_order,
                    input_data={"queries": [ctx.request.message]},
                    output_data={"documents_found": 0},
                    status="skipped",
                ),
                [],
            )

        start_time = time.time()
        all_documents: List[Dict[str, Any]] = []

        try:
            queries = [ctx.request.message]
            seen_content = set()
            rerank_enabled = getattr(ctx.config, "rerank_enabled", False)
            top_k = getattr(ctx.config, "top_k", 5)
            milvus_top_k = (
                getattr(ctx.config, "rerank_initial_top_k", top_k * 4) if rerank_enabled else top_k
            )

            for i, query in enumerate(queries[: getattr(ctx.config, "max_retrieval_queries", 3)]):
                await self._retrieve_single_query(
                    RetrieveQueryContext(
                        ctx=ctx,
                        query=query,
                        query_index=i,
                        milvus_top_k=milvus_top_k,
                        seen_content=seen_content,
                        all_documents=all_documents,
                    )
                )

            if rerank_enabled and all_documents:
                all_documents = await self._apply_rerank(ctx, all_documents, top_k)

            if getattr(ctx.config, "relevance_filter_enabled", True) and all_documents:
                all_documents = await self._filter_documents_by_relevance(ctx, all_documents)

            duration = int((time.time() - start_time) * 1000)

            logger.info(
                f"[{ctx.domain}] {step_config.name}完成",
                document_count=len(all_documents),
                hybrid_search="milvus_native",
            )

            return self._build_success_result(
                StepSuccessContext(
                    ctx=ctx,
                    step_config=step_config,
                    queries=queries,
                    top_k=top_k,
                    all_documents=all_documents,
                    duration=duration,
                )
            )

        except Exception as e:
            duration = int((time.time() - start_time) * 1000)
            logger.error(f"[{ctx.domain}] {step_config.name}失败: {str(e)}")
            return self._build_failure_result(step_config, top_k, duration, e, ctx.request.message)

    async def _retrieve_single_query(
        self,
        rq_ctx: RetrieveQueryContext,
    ):
        """执行单个查询的混合检索并收集结果。"""
        if not (rq_ctx.ctx.embedding_service and rq_ctx.ctx.milvus_service):
            return

        query_embedding = (
            rq_ctx.ctx.question_embedding
            if rq_ctx.query_index == 0 and rq_ctx.ctx.question_embedding is not None
            else await rq_ctx.ctx.embedding_service.embed_query(rq_ctx.query)
        )

        docs = []
        hybrid_failed = False
        try:
            docs = rq_ctx.ctx.milvus_service.hybrid_search(
                query_embedding=query_embedding,
                query_text=rq_ctx.query,
                top_k=rq_ctx.milvus_top_k,
                rrf_k=getattr(rq_ctx.ctx.config, "rrf_k", 60),
            )
        except Exception as hybrid_err:
            hybrid_failed = True
            logger.warning(
                f"[{rq_ctx.ctx.domain}] 混合检索异常，回退到纯向量检索",
                error=str(hybrid_err)[:120],
            )

        if (not docs) or hybrid_failed:
            if hybrid_failed or rq_ctx.query_index == 0:
                try:
                    docs = rq_ctx.ctx.milvus_service.search_similar(query_embedding, top_k=rq_ctx.milvus_top_k)
                    if not hybrid_failed:
                        logger.info(
                            f"[{rq_ctx.ctx.domain}] 混合检索返回0结果，已回退到纯Dense检索",
                            dense_count=len(docs),
                        )
                except Exception as dense_err:
                    logger.warning(f"[{rq_ctx.ctx.domain}] Dense回退也失败: {str(dense_err)[:120]}")
                    docs = []

        score_threshold = getattr(rq_ctx.ctx.config, "retrieval_score_threshold", 0.0)
        for doc in docs:
            content = doc.page_content
            if content not in rq_ctx.seen_content:
                rrf_score = doc.metadata.get("distance", 0)
                if rrf_score < score_threshold:
                    continue
                rq_ctx.seen_content.add(content)
                rq_ctx.all_documents.append(
                    {
                        "content": content,
                        "metadata": doc.metadata,
                        "source_query": rq_ctx.query,
                        "score": round(rrf_score, 4),
                    }
                )

    async def _apply_rerank(
        self,
        ctx: AgentContext,
        all_documents: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """使用 BGE-Reranker 重排序。"""
        try:
            rerank_threshold = getattr(ctx.config, "rerank_threshold", 0.3)
            rerank_top_k = getattr(ctx.config, "rerank_top_k", top_k)

            doc_contents = [doc["content"] for doc in all_documents]
            reranker = RerankerService.get_instance()
            loop = asyncio.get_event_loop()
            ranked = await loop.run_in_executor(
                None,
                lambda: reranker.rerank(
                    query=ctx.request.message,
                    documents=doc_contents,
                    top_k=rerank_top_k,
                    threshold=rerank_threshold,
                ),
            )

            original_count = len(all_documents)
            all_documents = [
                {**all_documents[idx], "score": round(score, 4)}
                for idx, score, _ in ranked
            ]

            discarded = original_count - len(all_documents)
            if discarded > 0:
                logger.info(
                    f"[{ctx.domain}] Rerank 移除了 {discarded} 条低相关文档",
                    before=original_count,
                    after=len(all_documents),
                    threshold=rerank_threshold,
                )
            return all_documents

        except Exception as e:
            logger.warning(f"[{ctx.domain}] Rerank 失败，保留原始结果: {str(e)[:150]}")
            return all_documents

    def _build_rag_context(self, all_documents: List[Dict[str, Any]], top_k: int) -> str:
        """从检索结果构建 RAG 上下文文本。"""
        return (
            "\n\n".join(
                [
                    f"[来源: {doc.get('metadata', {}).get('source', '未知')}]\n{doc['content']}"
                    for doc in all_documents[:top_k]
                ]
            )
            or "暂无相关检索结果"
        )

    def _build_success_result(
        self,
        success_ctx: StepSuccessContext,
    ) -> Tuple[AgentStepResult, List[Dict[str, Any]]]:
        """构建成功执行结果。"""
        rag_context = self._build_rag_context(success_ctx.all_documents, success_ctx.top_k)
        return (
            AgentStepResult(
                step_name=success_ctx.step_config.name,
                step_order=self.step_order,
                input_data={"queries": success_ctx.queries, "top_k": success_ctx.top_k},
                output_data={
                    "context": rag_context[:500] + "..." if len(rag_context) > 500 else rag_context,
                    "documents_found": len(success_ctx.all_documents),
                    "hybrid_search_enabled": True,
                    "hybrid_search_type": "milvus_native_sparse_bm25",
                    "rerank_enabled": getattr(success_ctx.ctx.config, "rerank_enabled", False),
                    "rerank_model": (
                        "BAAI/bge-reranker-base"
                        if getattr(success_ctx.ctx.config, "rerank_enabled", False)
                        else None
                    ),
                },
                status="success",
                duration_ms=success_ctx.duration,
            ),
            success_ctx.all_documents,
        )

    def _build_failure_result(
        self,
        step_config,
        top_k: int,
        duration: int,
        error: Exception,
        request_message: str,
    ) -> Tuple[AgentStepResult, List[Dict[str, Any]]]:
        """构建失败执行结果。"""
        return (
            AgentStepResult(
                step_name=step_config.name,
                step_order=self.step_order,
                input_data={"queries": [request_message], "top_k": top_k},
                status="failed",
                error_message=str(error),
                duration_ms=duration,
            ),
            [],
        )

    async def _filter_documents_by_relevance(
        self,
        ctx: AgentContext,
        documents: List[Dict[str, Any]],
        max_docs: int = 20,
    ) -> List[Dict[str, Any]]:
        """使用 LLM 过滤语义不相关的检索文档。"""
        if not documents:
            return []

        docs_to_check = documents[:max_docs]
        if len(documents) <= 1:
            return documents

        doc_list = "\n---\n".join(
            [f"[文档{i}] {doc['content'][:300]}" for i, doc in enumerate(docs_to_check)]
        )

        prompt = f"""你是信息相关性判断助手。请判断以下文档是否与用户问题相关。

用户问题：{ctx.request.message}

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
            response = await ctx.llm_service.chat_qwen(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                langfuse_handler=ctx.langfuse_handler,
            )

            result = _parse_json_from_llm(response)
            if result is None:
                logger.warning(f"[{ctx.domain}] 相关性过滤 JSON 解析失败，保留所有文档")
                return documents
            irrelevant_ids = set(result.get("irrelevant_ids", []))

            if not irrelevant_ids:
                return documents

            filtered = [doc for i, doc in enumerate(documents) if i not in irrelevant_ids]
            logger.info(
                f"[{ctx.domain}] LLM 相关性过滤完成",
                before=len(documents),
                after=len(filtered),
                removed_ids=list(irrelevant_ids)[:5],
            )
            return filtered

        except Exception as e:
            logger.warning(f"[{ctx.domain}] 相关性过滤失败，保留所有文档: {str(e)[:100]}")
            return documents


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
