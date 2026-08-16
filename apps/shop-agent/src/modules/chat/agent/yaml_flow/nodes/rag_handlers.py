"""RAG 固定四步原子节点处理器（阶段 4.2）。

把现状 ``Pipeline``（step1-step4 黑盒）拆成 4 个 YAML 图节点：
  - rag_rewrite  ：步骤1 问题理解/改写（UnderstandStep）
  - rag_review   ：步骤2 安全审查（ReviewStep）
  - rag_retrieve ：步骤3 知识检索（RetrieveStep）
  - rag_generate ：步骤4 回答生成+质量评估（GenerateStep）

设计要点（对应 29 篇「控制流外移 / 数据流外移」）：
- 四步之间不靠 ``AgentContext`` 闭包隐式传递，而走显式 GraphState 的 ``rag`` channel
  （承载 rewritten_queries / safety / documents / rag_context / response / quality / question_embedding）。
- 节点函数不感知上下游名字：输入从 ``state['rag']`` 读，产出写回 ``state['rag']`` 对应 key。
- 每个节点自行构建 ``AgentContext``（依赖由 compiler 注入的 llm/embedding/milvus/redis），
  question_embedding 只在首个执行的 RAG 节点计算并缓存到 ``rag['question_embedding']``，
  供后续节点复用，避免重复调 embedding API（与现状 Pipeline 预计算一致）。
- 安全审查（rag_review）不通过时，由图的 **条件边** 短路到 fallback 节点（控制流外移，
  而非在节点内 return），fail-closed 兜底由步骤2 自身实现（异常即视作高风险拦截）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.shared.logger import APILogger

logger = APILogger("yaml_flow_rag_nodes")


def _build_agent_context(state: Dict[str, Any], *, llm_service=None, embedding_service=None,
                         milvus_service=None, redis_cache_service=None, domain: str = "ecommerce"):
    """构建 steps 共用的 AgentContext（阶段 4.2）。

    复用现有 ``steps.base.AgentContext``，从 GraphState 抽取必要依赖。
    question_embedding 优先从 state['rag'] 取缓存，否则现场计算。
    """
    from src.modules.chat.agent.steps.base import AgentContext
    from src.modules.chat.config import get_agent_config
    from src.modules.chat.schemas import ChatRequest

    conversation_id = state.get("thread_id", "") or "rag-default"
    message = ""
    msgs = state.get("messages") or []
    if msgs:
        last = msgs[-1]
        message = last.get("content", "") if isinstance(last, dict) else str(last)

    request = ChatRequest(
        message=message,
        conversation_id=conversation_id,
        domain=domain,
    )
    config = get_agent_config(domain)
    # 阶段 4.2：业务在 YAML 显式声明了 RAG 原子节点，语义上就是要执行对应步骤。
    # 强制启用 step1-step4，避免「config 未启用 → 节点静默跳过」破坏编排预期
    # （对应阶段 5 安全固化：编排层显式声明即视为强制，不依赖 config 开关回落放行）。
    for _step in ("step1", "step2", "step3", "step4"):
        _sc = getattr(config, _step, None)
        if _sc is not None:
            try:
                _sc.enabled = True
            except Exception:
                pass
    rag = state.get("rag") or {}
    question_embedding = rag.get("question_embedding")
    return AgentContext(
        request=request,
        domain=domain,
        config=config,
        llm_service=llm_service,
        embedding_service=embedding_service,
        milvus_service=milvus_service,
        redis_cache_service=redis_cache_service,
        conversation_id=conversation_id,
        question_embedding=question_embedding,
    )


async def _ensure_embedding(ctx, embedding_service) -> Optional[list]:
    """若 ctx.question_embedding 为空则现场计算（并回写 ctx，供下游复用）。"""
    if ctx.question_embedding is not None:
        return ctx.question_embedding
    if embedding_service is None or not ctx.request.message:
        return None
    emb = await embedding_service.embed_query(ctx.request.message)
    ctx.question_embedding = emb
    return emb


class RagRewriteHandler:
    """步骤1：问题理解/改写节点（rag_rewrite）。

    读 state['messages'][-1] → 调 UnderstandStep → 把 rewritten_queries 写回 rag['rewritten_queries']。
    """

    needs_llm = True

    def __init__(self, *, llm_service=None, embedding_service=None, milvus_service=None,
                 redis_cache_service=None, domain: str = "ecommerce"):
        self._llm = llm_service
        self._embedding = embedding_service
        self._milvus = milvus_service
        self._redis = redis_cache_service
        self._domain = domain

    async def run(self, state, inputs):
        from src.modules.chat.agent.steps.step1_understand import UnderstandStep

        ctx = _build_agent_context(
            state, llm_service=self._llm, embedding_service=self._embedding,
            milvus_service=self._milvus, redis_cache_service=self._redis, domain=self._domain,
        )
        step = UnderstandStep()
        result = await step.execute(ctx)
        queries = (result.output_data or {}).get("rewritten_queries", []) or [ctx.request.message]

        # 元描述回退（与现状 Pipeline 一致）：丢弃非检索词的输出
        _META_PREFIXES = ("问题类型：", "核心需求：", "关键实体：", "问题分类：",
                          "问题类型:", "核心需求:", "关键实体:", "问题分类:")
        if any(q.startswith(_META_PREFIXES) for q in queries):
            logger.warning(f"[{self._domain}] rag_rewrite 输出元描述，已回退为原问题")
            queries = [ctx.request.message]
        elif ctx.request.message not in queries:
            queries.append(ctx.request.message)

        rag = dict(state.get("rag") or {})
        rag["rewritten_queries"] = queries
        if ctx.question_embedding is not None:
            rag["question_embedding"] = ctx.question_embedding
        return {"rag": rag}


class RagReviewHandler:
    """步骤2：安全审查节点（rag_review）。

    调 ReviewStep → 把 SafetyCheckResult 写回 rag['safety'] 与 rag['can_proceed']。
    审查不通过时 **不在此节点短路**，而是由图的 ``on_xxx``/``always`` 条件边把流程导向
    fallback 节点（对应 YAML 中 `rag_fallback_on_review_fail` 控制是否提供该短路边）。
    """

    needs_llm = True

    def __init__(self, *, llm_service=None, embedding_service=None, milvus_service=None,
                 redis_cache_service=None, domain: str = "ecommerce"):
        self._llm = llm_service
        self._embedding = embedding_service
        self._milvus = milvus_service
        self._redis = redis_cache_service
        self._domain = domain

    async def run(self, state, inputs):
        from src.modules.chat.agent.steps.step2_review import ReviewStep

        ctx = _build_agent_context(
            state, llm_service=self._llm, embedding_service=self._embedding,
            milvus_service=self._milvus, redis_cache_service=self._redis, domain=self._domain,
        )
        step = ReviewStep()
        step_result, safety = await step.execute(ctx)

        rag = dict(state.get("rag") or {})
        rag["safety"] = safety.model_dump()
        rag["can_proceed"] = bool(safety.can_proceed)
        if ctx.question_embedding is not None:
            rag["question_embedding"] = ctx.question_embedding
        return {"rag": rag}


class RagRetrieveHandler:
    """步骤3：知识检索节点（rag_retrieve）。

    调 RetrieveStep（复用其 Milvus 混合检索 + Reranker + 相关性过滤）→
    把文档列表写回 rag['documents']，把拼接好的 rag_context 写回 rag['rag_context']。
    """

    needs_llm = True

    def __init__(self, *, llm_service=None, embedding_service=None, milvus_service=None,
                 redis_cache_service=None, domain: str = "ecommerce", top_k: Optional[int] = None):
        self._llm = llm_service
        self._embedding = embedding_service
        self._milvus = milvus_service
        self._redis = redis_cache_service
        self._domain = domain
        self._top_k = top_k

    async def run(self, state, inputs):
        # fail-closed：检索必须依赖向量底座
        if self._embedding is None or self._milvus is None:
            raise ValueError(
                "rag_retrieve 节点需要 embedding_service 与 milvus_service，"
                "请在 FlowCompiler 注入（缺失会导致无检索能力的盲生成）"
            )

        ctx = _build_agent_context(
            state, llm_service=self._llm, embedding_service=self._embedding,
            milvus_service=self._milvus, redis_cache_service=self._redis, domain=self._domain,
        )
        # 复用上游 rag_rewrite 的缓存 embedding（避免重复计算）
        await _ensure_embedding(ctx, self._embedding)

        # top_k 覆盖：节点 config.rag_top_k 优先于 AgentConfig.top_k
        if self._top_k is not None:
            ctx.config.top_k = self._top_k

        from src.modules.chat.agent.steps.step3_retrieve import RetrieveStep

        step = RetrieveStep()
        step_result, documents = await step.execute(ctx)

        rag_context = "\n\n".join(
            [f"[来源: {d.get('metadata', {}).get('source', '未知')}]\n{d['content']}" for d in documents[:5]]
        ) or "暂无相关检索结果"

        rag = dict(state.get("rag") or {})
        rag["documents"] = documents
        rag["rag_context"] = rag_context
        rag["documents_found"] = len(documents)
        if ctx.question_embedding is not None:
            rag["question_embedding"] = ctx.question_embedding
        return {"rag": rag}


class RagGenerateHandler:
    """步骤4：回答生成 + 质量评估节点（rag_generate）。

    调 GenerateStep（构建 prompt → token 预算守卫 → LLM 生成 → 输出过滤 + 质量评估）→
    把最终回答写回 rag['response']，质量评估写回 rag['quality']。
    依赖上游 rag_review 的 safety 与 rag_retrieve 的 rag_context。
    """

    needs_llm = True

    def __init__(self, *, llm_service=None, embedding_service=None, milvus_service=None,
                 redis_cache_service=None, domain: str = "ecommerce", output_filter: bool = True):
        self._llm = llm_service
        self._embedding = embedding_service
        self._milvus = milvus_service
        self._redis = redis_cache_service
        self._domain = domain
        self._output_filter = output_filter

    async def run(self, state, inputs):
        from src.modules.chat.agent.schemas import SafetyCheckResult
        from src.modules.chat.agent.steps.step4_generate import GenerateStep

        ctx = _build_agent_context(
            state, llm_service=self._llm, embedding_service=self._embedding,
            milvus_service=self._milvus, redis_cache_service=self._redis, domain=self._domain,
        )
        rag = dict(state.get("rag") or {})

        # 恢复上游 safety（rag_review 写入），缺失则默认放行（与步骤2 默认 safe 一致）
        safety_data = rag.get("safety") or {}
        safety = SafetyCheckResult(
            is_safe=safety_data.get("is_safe", True),
            risk_level=safety_data.get("risk_level", "low"),
            risk_categories=safety_data.get("risk_categories", []),
            warning_message=safety_data.get("warning_message"),
            can_proceed=safety_data.get("can_proceed", True),
        )
        rag_context = rag.get("rag_context", "暂无相关检索结果")

        # 用节点级开关覆盖 config 的内容过滤开关（阶段 4.2 专用字段）
        ctx.config.content_filter_enabled = self._output_filter

        step = GenerateStep()
        step_result, response, quality = await step.execute(
            ctx, safety_result=safety, rag_context=rag_context, graph_context=""
        )
        rag["response"] = response
        rag["quality"] = quality
        return {"rag": rag, "tool_result": response}


__all__ = [
    "RagRewriteHandler",
    "RagReviewHandler",
    "RagRetrieveHandler",
    "RagGenerateHandler",
]
