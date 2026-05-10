"""
ReAct Agent —— 基于 LangChain create_agent 的 ReAct 循环

Agent 自主决策：
1. 是否需要先查 RAG（政策/规则类知识）
2. 调用对应的业务 tool（查订单/查物流/退货/余额/优惠券）
3. 将工具结果与 RAG 结果融合成最终回复

工具选择策略（P0 + P1 + P2 三层过滤）：
- P0 意图前置过滤：IntentResult.action → 缩减候选池到 2-5 个
- P2 Embedding 语义重排（FAISS HNSW）：user query × tool description 余弦相似度 + 意图加权 → Top-3/5
  比 LLM 选择更精准（语义级匹配），比纯规则更鲁棒（捕捉隐含意图）
- P1 本地模型最终确认：用本地 Qwen2.5-1.5B 从 Top-3/5 中选出最相关的（兜底消歧）
  有本地模型时优先用本地，否则回退到云端 LLMToolSelectorMiddleware

Skill SOP 注入（替代 deepagents 渐进式披露）：
- 所有 skill 定义在 skills/*/SKILL.md 文件中（YAML frontmatter + Markdown 正文）
- 新增 Skill 只需创建新目录 + SKILL.md，无需改代码
- 启动时：SkillLoader 读取 frontmatter + 正文（body）存入 SkillRegistry
- 运行时：P0/P2/P1 确定工具后，反向查找命中 skill，将正文内联注入 system prompt
- 无额外 LLM 调用开销，无 prompt injection 风险（SOP 在 system role）
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, Dict, List, Set

import faiss
import numpy as np
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain.agents.middleware import LLMToolSelectorMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from src.modules.chat.schemas import ChatRequest, ChatResponse, IntentResult
from src.modules.chat.agent.skill_loader import (
    SkillRegistry,
    get_skill_registry,
)
from src.shared.logger import APILogger
from src.modules.chat.core.local_model_service import LocalModelService
from src.modules.monitoring.langfuse_callback import create_langfuse_handler
from langfuse import observe

if TYPE_CHECKING:
    from src.modules.chat.core.llm_service import LLMService
    from src.modules.chat.core.embedding_service import EmbeddingService
    from src.modules.chat.core.milvus_service import MilvusService
    from src.modules.chat.core.tool_registry import ToolService

logger = APILogger("react_agent")

# ── 人在回路：中断存储（用于退款确认后恢复执行）──
# thread_id → (agent_graph, config, conversation_id, intent_steps, domain, order_id, reason)
_INTERRUPT_STORE: Dict[str, tuple] = {}


def _store_interrupt(
    thread_id: str,
    graph,
    config: dict,
    conversation_id: str,
    intent_steps: list,
    domain: str,
    order_id: str,
    reason: str,
):
    """保存被中断的 graph 上下文，供后续 resume 使用。"""
    _INTERRUPT_STORE[thread_id] = (graph, config, conversation_id, intent_steps, domain, order_id, reason)


def _pop_interrupt(thread_id: str):
    """取出并删除中断上下文。"""
    return _INTERRUPT_STORE.pop(thread_id, None)

# ════════════════════════════════════════════════════════════════════════
# 方案 A：Skill 注册表 —— 从 skills/*/SKILL.md 自动加载
# ════════════════════════════════════════════════════════════════════════

# 全局懒加载单例（首次访问时扫描 skills/ 目录）
_SKILL_REGISTRY: SkillRegistry | None = None


def _skill_registry() -> SkillRegistry:
    """获取全局 Skill 注册表（懒加载）。"""
    global _SKILL_REGISTRY
    if _SKILL_REGISTRY is None:
        _SKILL_REGISTRY = get_skill_registry()
    return _SKILL_REGISTRY


def _get_intent_tool_map() -> Dict[str, Set[str]]:
    return _skill_registry().intent_tool_map

# ════════════════════════════════════════════════════════════════════════
# P1: 本地模型工具选择器 —— 语义二次筛选
# ════════════════════════════════════════════════════════════════════════

# 工具选择器专用的 system prompt ——
# 与主 Agent 的 prompt 分离，专注于工具路由消歧
_TOOL_SELECTOR_PROMPT = """\
你是一个电商客服工具路由器。根据用户消息，从可用工具列表中选择最相关的工具。

每个工具的 description 已包含其功能说明，请根据语义进行匹配。
如果多个工具功能相似，选择最直接、最精准的。
当用户同时涉及多个操作时（如订单+物流），可以同时选中。

仅选择回答查询所直接需要的工具。"""

# middleware 实例（延迟创建，仅当本地模型不可用时作为 fallback 使用）
_TOOL_SELECTOR_MIDDLEWARE: LLMToolSelectorMiddleware | None = None


def _get_tool_selector_middleware(llm_service) -> LLMToolSelectorMiddleware:
    """获取工具选择器 middleware（云端 fallback）。"""
    global _TOOL_SELECTOR_MIDDLEWARE
    if _TOOL_SELECTOR_MIDDLEWARE is not None:
        return _TOOL_SELECTOR_MIDDLEWARE

    try:
        model = llm_service.tool_selector_llm if llm_service else None
    except Exception:
        model = None

    _TOOL_SELECTOR_MIDDLEWARE = LLMToolSelectorMiddleware(
        model=model,
        max_tools=3,
        always_include=["knowledge_search"],
        system_prompt=_TOOL_SELECTOR_PROMPT,
    )
    return _TOOL_SELECTOR_MIDDLEWARE


# P1 工具的上下限：最少选 1 个，最多选 3 个
_P1_MIN_TOOLS = 1
_P1_MAX_TOOLS = 3


async def _local_p1_tool_select(
    user_query: str,
    tool_names: set[str],
    tool_descriptions: dict[str, str],
    *,
    p2_ranked: list[str] | None = None,
) -> set[str]:
    """P1 本地模型工具选择：从候选工具中选出最相关的。

    优先使用本地模型（Qwen2.5-1.5B），本地不可用时返回原始候选集，
    由上层 fallback 到 LLMToolSelectorMiddleware。

    Args:
        p2_ranked: P2 Embedding 排序后的工具列表（最相关在前），
                   用于交叉校验——如果 P1 把 P2 得分最高的工具踢掉且分差大，
                   强制保留 P2 top-1 作为安全兜底。
    """
    if len(tool_names) <= 2:
        return tool_names  # 已经足够少，无需再选

    local_svc = LocalModelService.get_instance()
    names_list = list(tool_names)
    selected = await local_svc.chat_classify(
        user_query=user_query,
        tool_names=names_list,
        tool_descriptions=tool_descriptions,
        system_prompt=_TOOL_SELECTOR_PROMPT,
    )
    # 只保留仍在候选池内的（防止模型幻觉出不存在的工具名）
    result = {n for n in selected if n in tool_names}
    if not result:
        return tool_names  # 解析失败时保留全部候选

    # ── 交叉校验: P2 top-1 兜底 ──
    # 如果 P1 把 P2 Embedding 得分最高的工具从结果中踢掉了，
    # 说明小模型可能判断失误 —— 强制保留 P2 top-1
    if p2_ranked and len(p2_ranked) >= 2:
        p2_top1 = p2_ranked[0]
        if p2_top1 in tool_names and p2_top1 not in result:
            logger.warning(
                "P1 丢弃了 P2 top-1 工具，疑似小模型误判，追加回结果集",
                p2_top1=p2_top1,
                p1_selected=sorted(result),
                p2_ranked=p2_ranked[:3],
            )
            result.add(p2_top1)

    # ── 数量约束: 至少 _P1_MIN_TOOLS 个，最多 _P1_MAX_TOOLS 个 ──
    if len(result) > _P1_MAX_TOOLS:
        # 优先保留 P2 排序靠前的
        ranked = [n for n in (p2_ranked or []) if n in result]
        result = set(ranked[:_P1_MAX_TOOLS])

    return result

# ════════════════════════════════════════════════════════════════════════
# P2: Embedding 语义匹配器 —— 用向量相似度做工具重排
# ════════════════════════════════════════════════════════════════════════

class EmbeddingToolMatcher:
    """基于 FAISS HNSW 图索引的工具语义匹配器。

    工作流程：
    1. 首次使用时，为所有工具描述预计算 embedding，构建 FAISS IndexHNSWFlat（O(log N) 搜索）
    2. 每次请求：计算用户 query 的向量，HNSW 近似搜索（L2 距离 → 余弦相似度）
    3. 叠加意图加权：P0 已匹配的工具 ×1.5 权重
    4. 返回 Top-K，供 P1 middleware 做最终确认

    HNSW vs IndexFlatIP:
    - IndexFlatIP: O(N) 暴力搜索，适合 N<30
    - IndexHNSWFlat: O(log N) 图搜索，适合 N>50，skill 越多优势越明显
    """

    def __init__(
        self,
        tool_descriptions: Dict[str, str],
        embedding_service: EmbeddingService,
        *,
        intent_boost: float = 1.5,
    ):
        self._descriptions = tool_descriptions
        self._emb_service = embedding_service
        self._intent_boost = intent_boost
        self._intent_tool_map: Dict[str, Set[str]] = _get_intent_tool_map()  # 启动时快照，不支持运行时热加载
        self._index: faiss.Index | None = None
        self._tool_names: List[str] = []
        self._ready = False
        self._init_failed = False
        self._init_lock = asyncio.Lock()

    async def _ensure_index(self):
        """预计算所有工具描述的 embedding 向量并构建 FAISS 索引（只执行一次）。"""
        if self._ready:
            return
        async with self._init_lock:
            if self._ready:
                return

            if not self._descriptions:
                logger.warning("工具描述为空，跳过 FAISS 索引构建")
                self._ready = True
                return

            texts: List[str] = []
            for name, desc in self._descriptions.items():
                texts.append(f"工具名称：{name}；功能描述：{desc}")
                self._tool_names.append(name)

            try:
                vectors = await self._emb_service.embed_texts(texts)
            except Exception:
                logger.exception("工具 embedding 向量化失败，将在下次 rank() 时重试")
                self._init_failed = True
                return

            vecs_np = np.array(vectors, dtype=np.float32)

            # L2 归一化后，对于归一化向量: IndexHNSWFlat + L2 距离 ↔ 余弦相似度
            faiss.normalize_L2(vecs_np)

            # HNSW 图索引：O(log N) 近似搜索，比 IndexFlatIP(O(N)) 快数十倍（skill>30 时明显）
            M = 16  # 每层连接数，trade-off: 16=小内存高速度，32=高精度
            self._index = faiss.IndexHNSWFlat(vecs_np.shape[1], M)
            self._index.hnsw.efConstruction = 64  # 构建时搜索宽度，越高越精准但越慢
            self._index.hnsw.efSearch = 32         # 查询时搜索宽度

            self._index.add(vecs_np)

            self._ready = True
            logger.info(
                "工具 embedding 索引构建完成",
                tool_count=len(self._tool_names),
                dim=vecs_np.shape[1],
            )

    @observe(name="tool_matcher.embedding_rerank")
    async def rank(
        self,
        user_query: str,
        candidate_names: Set[str],
        intent_action: str | None,
        top_k: int = 3,
    ) -> List[str]:
        """对候选工具做语义重排，返回 Top-K 名称列表。

        Args:
            user_query: 用户原始消息
            candidate_names: P0 过滤后的工具名称集合
            intent_action: 意图识别结果（用于加权）
            top_k: 返回多少个

        Returns:
            Ordered tool name list（最相关 → 最不相关）
        """
        await self._ensure_index()

        if self._index is None or self._index.ntotal == 0:
            if self._init_failed:
                logger.warning("FAISS 索引初始化失败，回退到 P0 规则过滤")
            else:
                logger.warning("FAISS 索引为空，回退到 candidate_names")
            fallback = list(candidate_names)
            return fallback[:top_k]

        # 计算用户 query 向量并归一化
        query_vec = np.array(
            await self._emb_service.embed_query(user_query),
            dtype=np.float32,
        ).reshape(1, -1)
        faiss.normalize_L2(query_vec)

        # HNSW 近似搜索 → 候选集大时扩大搜索宽度
        search_k = min(self._index.ntotal, max(top_k, top_k * 3))
        if len(candidate_names) > 10:
            # 候选集过大（100 skills 下意图映射更容易命中 10+）：扩大搜索范围防止遗漏
            search_k = min(self._index.ntotal, max(16, top_k * 5))
        scores, indices = self._index.search(query_vec, search_k)

        # HNSW 返回 L2 距离 → 归一化向量 d ∈ [0, 2] → 转为余弦相似度 ∈ [0, 1]
        intent_tools = self._intent_tool_map.get(intent_action, set()) if intent_action else set()
        scored: List[tuple[str, float]] = []
        for idx, dist in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self._tool_names):
                continue
            name = self._tool_names[idx]
            if name not in candidate_names:
                continue

            # 归一化向量: L2_distance ∈ [0, 2] → cosine_sim = 1 - d²/2
            # 近似: similarity = max(0, 1 - dist/2) 避免浮点误差产生负值
            score = max(0.0, 1.0 - float(dist) / 2.0)

            # 意图加权：P0 已匹配的意图工具 ×1.5
            if name in intent_tools:
                score *= self._intent_boost

            scored.append((name, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_names = [name for name, _ in scored[:top_k]]

        logger.info(
            "Embedding 工具重排完成",
            candidates=len(candidate_names),
            top_k=len(top_names),
            scores=[(name, round(score, 4)) for name, score in scored[:top_k]],
        )
        return top_names


# ════════════════════════════════════════════════════════════════════════
# System prompt
# ════════════════════════════════════════════════════════════════════════

_REACT_SYSTEM_PROMPT = """你是电商公司的智能客服助手，能够自主调用工具来帮助用户。

## 工作流程
1. 理解用户的意图和问题
2. 如果需要查询订单、物流、余额、优惠券或处理退货，**必须调用对应的工具**
3. 如果需要了解公司政策、退货规则等，可以先调用 knowledge_search 查知识库
4. 将工具返回的信息整理成自然、友好的回复

## 重要规则
- 用户需要操作时，**必须调用工具**，不要凭空编造数据
- 如果工具返回错误或需要更多信息，如实告知用户
- **不要用相同参数重复调用同一个工具**：如果参数不同（如查不同订单号）可以多次调用，但相同参数不要重复
- knowledge_search 若首次结果不相关，可以换关键词重新搜索
- **request-return 是终端操作**：调用一次后提交即完成，直接告知用户结果，不要再次调用
- 保持回复简洁、自然、友好
"""


class ReActAgent:
    """真正的 ReAct Agent —— 具备 tool-calling 迭代循环 + 意图前置工具过滤"""

    def __init__(
        self,
        *,
        llm_service: LLMService,
        tool_service: ToolService,
        embedding_service: EmbeddingService | None = None,
        milvus_service: MilvusService | None = None,
        max_iterations: int = 5,
    ):
        self._llm_service = llm_service
        self._tool_service = tool_service
        self._embedding_service = embedding_service
        self._milvus_service = milvus_service
        self._max_iterations = max_iterations

        # 加载 skill 注册表（从 skills/*/SKILL.md）
        self._skill_registry = _skill_registry()

        # 构建全量工具池（所有业务工具 + RAG）
        self._all_tools = self._build_tools()

        # P2: embedding 工具匹配器 —— 预先计算工具描述向量
        self._tool_matcher: EmbeddingToolMatcher | None = None
        if self._embedding_service:
            self._tool_matcher = EmbeddingToolMatcher(
                self._skill_registry.tool_descriptions, self._embedding_service
            )

        # 人在回路：退款审批等待标志（run() 内跨 tool → post-invoke 传递）
        self._pending_approval: tuple[str, str] | None = None

    # ── 工具构建 ──────────────────────────────────────────────────────

    def _build_tools(self):
        """构建 LangChain tool 列表（全量，从 skill 注册表生成）"""
        tools = []

        for action in self._skill_registry.tool_descriptions:
            if action == "request-return":
                # 退款工具需要人在回路确认
                tools.append(self._make_refund_tool_with_confirmation())
            else:
                tools.append(self._make_business_tool(action))

        # RAG 知识库检索（如果有 Milvus）
        if self._milvus_service and self._embedding_service:
            tools.append(self._make_knowledge_search_tool())

        return tools

    def _make_business_tool(self, action: str):
        """为一个业务 action 创建 LangChain tool（描述来自 SKILL.md）"""

        desc = self._skill_registry.tool_descriptions.get(action, f"执行{action}操作")
        _action = action  # 显式绑定，防止闭包延迟绑定导致 future 重构引入 bug
        _dispatch = self._tool_service.dispatch  # 显式绑定 dispatch 引用，防止多实例/并发下 self 指向错误

        @tool(_action, description=desc)
        async def business_tool(**kwargs: Any) -> str:
            """动态绑定的业务工具函数"""
            return await _dispatch(_action, kwargs or None)

        return business_tool

    def _make_refund_tool_with_confirmation(self):
        """创建带人在回路（Human-in-the-Loop）确认的退款工具。

        流程：
        1. 调用 mock API 打印退款确认信息
        2. 返回提醒文案 + 设置 _pending_approval 标志位（避免 raise 导致 Langfuse 报 error）
        3. run() 在 ainvoke 正常完成后检测标志位，走人在回路分支
        4. 人工通过 /agent/refund/confirm API 批准后，direct dispatch 退款
        5. 拒绝则返回取消消息

        为什么不用 langgraph_interrupt()：
        interrupt() 只能在 graph node 函数内调用，tool 在 agent 的
        tool-calling node 内部执行，不在 node 的直接上下文中，调用会 hang。
        """
        from src.modules.chat.core.tool_registry import ToolService

        desc = self._skill_registry.tool_descriptions.get("request-return", "申请退货退款")
        _dispatch = self._tool_service.dispatch
        _agent = self  # 闭包外引用，用于设置标志位

        @tool("request-return", description=desc)
        async def request_return_with_confirm(
            order_id: str = "",
            reason: str = "未说明",
            **kwargs: Any,
        ) -> str:
            """退款工具 —— 带人在回路确认"""

            # ── Mock API: 打印退款确认信息（模拟调用外部审批系统）──
            await ToolService.mock_refund_confirmation(
                order_id=order_id,
                reason=reason,
                refund_amount=0.0,
            )

            # ── 人在回路：设置标志位并返回提醒，让 invoke 正常结束 ──
            # 防御：如果 LLM 不遵守 system prompt 重复调用，记录 warning 且不覆盖已有标志
            if _agent._pending_approval is not None:
                logger.warning(
                    "request-return 被重复调用，忽略（已存在待审批申请）",
                    existing_order_id=_agent._pending_approval[0],
                    duplicate_order_id=order_id,
                )
                return f"退款申请已在处理中（订单号: {_agent._pending_approval[0]}），无需重复提交。"
            _agent._pending_approval = (order_id, reason)
            return f"✓ 退款申请已成功提交（订单号: {order_id}，原因: {reason}）。该申请已进入人工审批队列，审批通过后退款将在 1-3 个工作日内原路返回。请等待管理员审批结果通知。"

        return request_return_with_confirm

    def _make_knowledge_search_tool(self):
        """创建 RAG 知识库检索工具"""

        @tool(description="搜索知识库，查询公司政策、退货规则、产品信息等。参数: query(搜索内容)")
        async def knowledge_search(query: str) -> str:
            """从 Milvus + embedding 检索知识库"""
            try:
                emb = self._embedding_service.get_embeddings()
                query_vec = await emb.aembed_query(query)
                docs = self._milvus_service.search_similar(query_vec, top_k=3)

                if not docs:
                    return "知识库中未找到相关信息。"

                results = []
                for i, doc in enumerate(docs, 1):
                    content = doc.page_content[:600]
                    results.append(f"[文档{i}] {content}")
                return "\n\n".join(results)
            except Exception as e:
                logger.error(f"RAG检索失败: {e}")
                return "知识库检索暂时不可用。"

        return knowledge_search

    # ════════════════════════════════════════════════════════════════════
    # P0 + P2: 意图过滤 + Embedding 语义重排
    # ════════════════════════════════════════════════════════════════════

    async def _select_tools_for_intent(
        self, action: str | None, user_query: str
    ) -> List:
        """
        三层工具精选流水线：

        P0：意图规则过滤 —— INTENT_TOOL_MAP 缩小候选池
        P2：Embedding 语义重排 —— 余弦相似度 + 意图加权 → Top-3/5
        P1：本地模型最终确认 —— Qwen2.5-1.5B 从候选选出最相关的

        Args:
            action: 意图 action
            user_query: 用户原始消息（用于 embedding 匹配）

        Returns:
            filtered_tools: 精简后的 LangChain tool 对象列表（通常 2-4 个）
        """
        # ── P0: 意图规则过滤 ──
        intent_map = self._skill_registry.intent_tool_map
        tool_names: Set[str] = intent_map.get(
            action or "unknown",
            intent_map["unknown"],
        )

        logger.info(
            "P0 意图过滤完成",
            action=action,
            candidates=len(tool_names),
            tool_names=sorted(tool_names),
        )

        # ── P2: Embedding 语义重排 ──
        p2_ranked: list[str] = []  # P2 排序结果（最相关→最不相关），供 P1 交叉校验用
        if self._tool_matcher and len(tool_names) > 1:
            try:
                # "unknown" 意图：无规则依据，放宽 Top-K 让 P1 有更多选项可选
                p2_top_k = 5 if (action is None or action == "unknown") else 4
                ranked_names = await self._tool_matcher.rank(
                    user_query=user_query,
                    candidate_names=tool_names,
                    intent_action=action,
                    top_k=p2_top_k,
                )
                if ranked_names:
                    p2_ranked = list(ranked_names)
                    tool_names = set(p2_ranked)
                else:
                    logger.warning("P2 重排返回空结果，保持 P0 候选集")
            except Exception as e:
                logger.warning(f"Embedding 重排失败，回退到 P0 结果: {e}")

        # ── P1: 本地模型最终确认（优先本地，不可用则 fallback 到 middleware）──
        if self._tool_matcher and len(tool_names) > 2:
            tool_descs = self._skill_registry.tool_descriptions
            try:
                selected = await _local_p1_tool_select(
                    user_query=user_query,
                    tool_names=tool_names,
                    tool_descriptions=tool_descs,
                    p2_ranked=p2_ranked,
                )
                if selected and selected != tool_names:
                    logger.info(
                        "P1 本地模型工具选择完成",
                        before=sorted(tool_names),
                        after=sorted(selected),
                    )
                    tool_names = selected
                # 如果 local_p1 返回全量（本地模型不可用），
                # 则保留在 _build_graph 中用 LLMToolSelectorMiddleware 兜底
            except Exception as e:
                logger.warning(f"P1 本地模型工具选择失败，保留 P2 结果: {e}")

        # 从全量池中选取对应的 tool 对象
        filtered: List = []
        for t in self._all_tools:
            name = getattr(t, "name", "")
            if name in tool_names:
                filtered.append(t)
            # knowledge_search 永远附带
            elif name == "knowledge_search":
                filtered.append(t)

        logger.info(
            "P0+P1+P2 工具过滤最终结果",
            action=action,
            total_tools=len(self._all_tools),
            filtered=len(filtered),
            tool_names=sorted([getattr(t, "name", "") for t in filtered]),
        )
        return filtered

    # ── Agent 构建 ────────────────────────────────────────────────────

    def _build_graph(self, tools: List | None = None, checkpointer=None):
        """
        构建 LangChain create_agent。

        三层工具过滤：
        - P0 / P2 / P1(local)：已在 _select_tools_for_intent 中完成
        - P1(fallback)：本地模型不可用时，LLMToolSelectorMiddleware 兜底

        Skill SOP 注入：
        - 根据选中工具反向查找命中 skill，将正文内联到 system prompt
        - 替代 deepagents SkillsMiddleware 渐进式披露，无额外 LLM 调用

        Args:
            tools: 工具列表。None 时使用全量工具池。
            checkpointer: LangGraph checkpointer（用于 Time Travel）。None 时不启用。
        """
        llm = self._llm_service.qwen_llm
        if llm is None:
            raise RuntimeError("LLM 未初始化，无法构建 Agent")
        selected_tools = tools if tools is not None else self._all_tools

        # P1 本地模型可用时已精准过滤，无需 middleware
        # 工具数 > 3 说明本地模型没起作用，用 middleware 兜底
        middleware = []
        tool_count = len([t for t in selected_tools if not isinstance(t, dict)])
        if tool_count > 3:
            middleware = [_get_tool_selector_middleware(self._llm_service)]

        # 动态拼装 system prompt（base 核心指令 + 命中 skill 的 SOP）
        tool_names = {getattr(t, "name", "") for t in selected_tools}
        prompt = self._build_system_prompt(tool_names)

        graph = create_agent(
            model=llm,
            tools=selected_tools,
            system_prompt=prompt,
            middleware=middleware,
            checkpointer=checkpointer,
        )
        # 设置 graph 级默认 recursion_limit，防止无穷循环
        return graph.with_config({"recursion_limit": self._max_iterations * 2 + 2})

    def _build_system_prompt(self, tool_names: set[str]) -> str:
        """动态拼装 system prompt：base 核心指令 + 命中 skill 的 SOP 正文。

        按 tool_name → skill.allowed_tools 反向查找，注入对应的操作指引。
        不注入 knowledge_search 的 SOP（它不是业务 skill）。
        """
        prompt = _REACT_SYSTEM_PROMPT  # 纯静态核心指令

        # 排除 knowledge_search（它不是业务 skill）
        biz_tool_names = tool_names - {"knowledge_search"}

        # 反向查找：匹配 allowed_tools 包含当前工具名的 skill
        matched_skills = [
            s for s in self._skill_registry.skills
            if set(s.allowed_tools) & biz_tool_names
        ]
        if matched_skills:
            bodies = [s.body for s in matched_skills if s.body]
            if bodies:
                prompt += "\n\n## 当前场景操作指南\n" + "\n\n".join(bodies)

        return prompt

    # ── 执行入口 ──────────────────────────────────────────────────────

    async def run(
        self,
        request: ChatRequest,
        intent_result: IntentResult,
        conversation_id: str,
        domain: str,
        intent_steps: list,
        langfuse_handler=None,
    ) -> ChatResponse:
        """
        执行 ReAct 循环。

        Args:
            request: 原始聊天请求
            intent_result: 意图识别结果
            conversation_id: 会话 ID
            domain: 业务域
            intent_steps: 前置步骤列表（意图识别 + 参数抽取）
            langfuse_handler: 外部 Langfuse CallbackHandler（由上层编排器传入）。
                             传入时复用此 handler，不再创建新 trace；
                             为 None 时内部自动创建（向后兼容独立调用）。

        Returns:
            ChatResponse
        """
        # ── P0 + P2: 按意图过滤 + Embedding 语义重排 ──
        selected_tools = await self._select_tools_for_intent(
            intent_result.action, user_query=request.message
        )

        # 使用共享的 MemorySaver 启用 checkpointer（人在回路的 interrupt() 必需）
        memory_saver = MemorySaver()
        agent_graph = self._build_graph(tools=selected_tools, checkpointer=memory_saver)

        # 构造增强消息（意图上下文作为提示）
        params_str = ""
        if intent_result.params:
            params_str = f"已抽取参数: {json.dumps(intent_result.params, ensure_ascii=False)}. "

        intent_context = (
            f"用户意图: {intent_result.action}, "
            f"复杂程度: {intent_result.complexity}. "
            f"{params_str}"
        )
        enhanced_message = f"[系统提示] {intent_context}\n用户问题: {request.message}"

        logger.info(
            "ReAct Agent 开始执行",
            action=intent_result.action,
            params=intent_result.params,
            message_length=len(request.message),
            tools_count=len(selected_tools),
        )

        react_start = time.monotonic()

        # Langfuse v4.x: 外部传入 handler 时直接复用，否则内部创建
        langfuse_ctx = None
        if langfuse_handler is None:
            result = create_langfuse_handler(
                session_id=conversation_id,
                tags=[domain, "react-agent"],
                trace_name=f"{domain}-react-{intent_result.action}",
                metadata={
                    "domain": domain,
                    "action": intent_result.action,
                    "complexity": intent_result.complexity,
                },
            )
            if result:
                langfuse_handler, langfuse_ctx = result
                langfuse_ctx.__enter__()

        try:
            # LangGraph invoke: 传入 messages 列表 + callbacks + thread_id
            config = {
                "recursion_limit": self._max_iterations * 2 + 2,
                "configurable": {"thread_id": conversation_id},
            }
            if langfuse_handler:
                config["callbacks"] = [langfuse_handler]

            result = await agent_graph.ainvoke(
                {"messages": [HumanMessage(content=enhanced_message)]},
                config=config,
            )

            # ── 人在回路：检测 request-return 工具是否被调用 ──
            # 工具 return 正常字符串（而非 raise），Langfuse 不再报 error
            # 由 _pending_approval 标志位接管人在回路流程
            if self._pending_approval is not None:
                order_id, reason = self._pending_approval
                self._pending_approval = None

                logger.info(
                    "ReAct Agent 进入人在回路等待",
                    conversation_id=conversation_id,
                    action=intent_result.action,
                    order_id=order_id,
                )
                # 保存上下文（含订单信息）以便后续 resume 执行
                _store_interrupt(
                    thread_id=conversation_id,
                    graph=agent_graph,
                    config=config,
                    conversation_id=conversation_id,
                    intent_steps=intent_steps,
                    domain=domain,
                    order_id=order_id,
                    reason=reason,
                )

                # 退出 langfuse 上下文
                if langfuse_ctx:
                    langfuse_ctx.__exit__(None, None, None)
                    langfuse_ctx = None

                return ChatResponse(
                    message="退款申请需要人工确认，请通过审批系统进行操作。",
                    conversation_id=conversation_id,
                    steps=intent_steps + [{
                        "step_name": "人在回路-退款确认",
                        "step_order": len(intent_steps),
                        "status": "waiting",
                        "output_data": {
                            "action": "request-return",
                            "order_id": order_id,
                            "reason": reason,
                            "message": "等待人工审批确认退款",
                        },
                    }],
                    documents_used=[],
                    safety_passed=True,
                    stream_available=True,
                    domain=domain,
                    status="waiting_for_confirmation",
                    interrupt_data={
                        "type": "refund_confirmation",
                        "action": "request-return",
                        "conversation_id": conversation_id,
                        "order_id": order_id,
                        "reason": reason,
                    },
                )
        except Exception as e:
            self._pending_approval = None  # 异常路径下清除标志位，避免状态残留
            logger.error(f"ReAct Agent 执行失败: {e}")
            return ChatResponse(
                message="抱歉，处理您的请求时遇到了问题，请稍后重试。",
                conversation_id=conversation_id,
                steps=intent_steps + [{
                    "step_name": "ReAct执行",
                    "step_order": len(intent_steps),
                    "status": "failed",
                    "error_message": str(e)[:200],
                }],
                documents_used=[],
                safety_passed=True,
                stream_available=True,
                domain=domain,
            )
        finally:
            # Langfuse v4.x: 退出 propagate_attributes 上下文
            if langfuse_ctx:
                langfuse_ctx.__exit__(None, None, None)

        elapsed_ms = int((time.monotonic() - react_start) * 1000)

        # 从 messages 中提取最终输出和中间步骤
        messages: list = result.get("messages", [])
        final_output, intermediate_steps = self._parse_messages(messages)
        if not final_output:
            final_output = "抱歉，我暂时无法处理这个请求。"

        react_steps = self._format_intermediate_steps(intermediate_steps, elapsed_ms)

        logger.log_business_event(
            "电商Agent ReAct调用",
            success=True,
            domain=domain,
            action=intent_result.action,
            complexity=intent_result.complexity,
            conversation_id=conversation_id,
            message_length=len(request.message),
            response_length=len(final_output),
            react_iterations=len(intermediate_steps),
            duration_ms=elapsed_ms,
        )

        return ChatResponse(
            message=final_output,
            conversation_id=conversation_id,
            steps=intent_steps + react_steps,
            documents_used=[],
            safety_passed=True,
            stream_available=True,
            domain=domain,
        )

    @staticmethod
    def _parse_messages(messages: list) -> tuple[str, list]:
        """
        从 LangGraph messages 列表中提取最终回复和中间步骤。

        Returns:
            (final_output, [(tool_name, tool_input, observation), ...])
        """
        final_output = ""
        intermediate_steps = []

        # 遍历消息，配对 tool_calls 与 tool 结果
        pending_calls = {}  # tool_call_id → (name, args)

        for msg in messages:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    # 记录待执行的工具调用
                    for tc in msg.tool_calls:
                        pending_calls[tc["id"]] = (
                            tc.get("name", "unknown"),
                            tc.get("args", {}),
                        )
                elif msg.content and not msg.tool_calls:
                    # 无 tool_calls 的 AIMessage = 最终回复
                    final_output = str(msg.content)
                elif msg.content:
                    # 同时有 content 和 tool_calls：部分文本回复（保存备用）
                    final_output = str(msg.content)
            elif isinstance(msg, ToolMessage):
                tc_id = getattr(msg, "tool_call_id", "")
                if tc_id in pending_calls:
                    tool_name, tool_input = pending_calls.pop(tc_id)
                    intermediate_steps.append((tool_name, tool_input, str(msg.content)))
                else:
                    tool_name = getattr(msg, "name", "unknown")
                    intermediate_steps.append((tool_name, {}, str(msg.content)))

        return final_output, intermediate_steps

    @staticmethod
    def _format_intermediate_steps(
        intermediate_steps: list,
        total_elapsed_ms: int,
    ) -> list:
        """将 (tool_name, tool_input, observation) 元组列表转成统一 step 格式"""
        steps = []
        for idx, (tool_name, tool_input, observation) in enumerate(intermediate_steps):
            obs_str = str(observation)[:300]
            name = str(tool_name)

            steps.append({
                "step_name": f"ReAct-{name}",
                "step_order": idx + 1,
                "status": "success",
                "output_data": {
                    "tool": name,
                    "tool_input": tool_input,
                    "observation": obs_str,
                },
            })

        steps.append({
            "step_name": "ReAct-总结",
            "step_order": len(steps) + 1,
            "status": "success",
            "output_data": {
                "total_iterations": len(intermediate_steps),
                "duration_ms": total_elapsed_ms,
            },
        })

        return steps

    # ── 人在回路：恢复执行 ──────────────────────────────────────────

    @staticmethod
    async def resume_execution(
        thread_id: str,
        confirm: bool,
        tool_service: "ToolService | None" = None,
    ) -> "ChatResponse | None":
        """恢复被人在回路中断的退款执行。

        Args:
            thread_id: 中断时的 conversation_id
            confirm: True=批准退款, False=拒绝退款

        Returns:
            ChatResponse（含最终执行结果），或 None（未找到对应中断）
        """
        stored = _pop_interrupt(thread_id)
        if stored is None:
            logger.warning(f"未找到待恢复的中断: thread_id={thread_id}")
            return None

        _, __, conversation_id, intent_steps, domain, order_id, reason = stored

        if not confirm:
            logger.log_business_event("退款审批-拒绝", conversation_id=conversation_id, order_id=order_id)
            return ChatResponse(
                message=f"退款申请已被取消（订单号: {order_id}）。如有需要，请重新发起申请。",
                conversation_id=conversation_id,
                steps=intent_steps + [{
                    "step_name": "人在回路-退款拒绝",
                    "step_order": len(intent_steps),
                    "status": "success",
                    "output_data": {"order_id": order_id, "confirm": False},
                }],
                documents_used=[],
                safety_passed=True,
                stream_available=True,
                domain=domain,
                status="completed",
            )

        # ── 审批通过，直接 dispatch 退款 ──
        logger.info("退款审批通过，执行退款", conversation_id=conversation_id, order_id=order_id)
        react_start = time.monotonic()
        try:
            from src.modules.chat.core.tool_registry import ToolService
            svc = tool_service or ToolService()
            dispatch_result = await svc.dispatch("request-return", {"order_id": order_id, "reason": reason})
        except Exception as e:
            logger.error(f"退款执行失败: {e}", thread_id=thread_id, order_id=order_id)
            return ChatResponse(
                message="审批已通过，但退款操作执行失败，请稍后重试。",
                conversation_id=conversation_id,
                status="completed",
                domain=domain,
            )

        elapsed_ms = int((time.monotonic() - react_start) * 1000)

        logger.info("人在回路恢复执行完成", thread_id=thread_id, confirm=True, elapsed_ms=elapsed_ms)

        return ChatResponse(
            message=dispatch_result,
            conversation_id=conversation_id,
            steps=intent_steps + [{
                "step_name": "人在回路-退款确认执行",
                "step_order": len(intent_steps),
                "status": "success",
                "output_data": {
                    "order_id": order_id,
                    "reason": reason,
                    "confirm": True,
                    "result": dispatch_result[:300],
                    "duration_ms": elapsed_ms,
                },
            }],
            documents_used=[],
            safety_passed=True,
            stream_available=True,
            domain=domain,
            status="completed",
        )


# ════════════════════════════════════════════════════════════════════════
# LangGraph Studio 导出函数 —— Time Travel Web 调试入口
# ════════════════════════════════════════════════════════════════════════

def get_graph():
    """LangGraph Studio 入口：返回带 MemorySaver 的编译后 graph。

    LangGraph Studio 调用此函数获取 graph + checkpointer，
    从而在 Web UI 中实现 Time Travel（状态快照、回溯、重放）。

    使用方式：
        langgraph dev   # 启动本地 Web 服务器
        或双击 LangGraph Studio Desktop 打开 langgraph.json
    """
    from langchain_core.tools import tool

    # 1. 初始化 LLM（复用项目配置）
    from src.modules.chat.core.llm_service import LLMService
    llm_svc = LLMService()
    llm_svc.initialize()
    llm = llm_svc.qwen_llm
    if llm is None:
        raise RuntimeError("LLM 未初始化，请检查 TONGYI_API_KEY 配置")

    # 2. 从 skill 注册表生成 stub 工具（只保留名称+描述，不做实际 dispatch）
    registry = _skill_registry()
    stub_tools = []

    def _make_stub(tool_name: str, tool_desc: str):
        @tool(tool_name, description=tool_desc)
        def stub(**kwargs: Any) -> str:
            return f"[Studio] 工具 {tool_name} 被调用（LangGraph Studio 调试模式，未接入真实后端）"
        return stub

    for name, desc in registry.tool_descriptions.items():
        stub_tools.append(_make_stub(name, desc))

    # 知识库检索 stub
    @tool(description="搜索知识库，查询公司政策、退货规则、产品信息等。参数: query(搜索内容)")
    def knowledge_search_stub(query: str = "") -> str:
        return f"[Studio] 知识库检索: {query}（调试模式）"

    stub_tools.append(knowledge_search_stub)

    # 3. 构建 graph + MemorySaver（Time Travel 必需）
    graph = create_agent(
        model=llm,
        tools=stub_tools,
        system_prompt=_REACT_SYSTEM_PROMPT,
        middleware=[],
        checkpointer=MemorySaver(),
    )
    return graph.with_config({"recursion_limit": 24})
