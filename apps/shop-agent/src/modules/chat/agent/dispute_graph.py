"""纠纷协调 LangGraph 实现。

将现有 `composite.py` 中的 AgentNode 适配为 LangGraph 节点，
构建可序列化、可持久化的纠纷解决流程。
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from src.modules.chat.agent.composite import AgentContext
from src.modules.chat.agent.dispute_agents import (
    BuyerAnalysisAgent,
    FactCollectorAgent,
    MediatorAgent,
    SellerAnalysisAgent,
)
from src.modules.chat.agent.dispute_common import (
    AgentPerspective,
    EmotionLevel,
)
from src.shared.logger import APILogger

logger = APILogger("dispute_graph")


# ═══════════════════════════════════════════════════════════════════════
# State 定义
# ═══════════════════════════════════════════════════════════════════════


class DisputeState(TypedDict):
    """LangGraph 共享状态。"""
    request_message: str
    facts: Dict[str, str]
    metadata: Annotated[Dict[str, Any], lambda a, b: {**(a or {}), **(b or {})}]
    emotion_level: Optional[EmotionLevel]
    order_id: Optional[str]
    conversation_id: str
    domain: str
    # 流程执行结果
    buyer_result: Optional[AgentPerspective]
    seller_result: Optional[AgentPerspective]
    mediator_result: Optional[AgentPerspective]
    # 错误信息
    error: Optional[str]


# ═══════════════════════════════════════════════════════════════════════
# 节点函数
# ═══════════════════════════════════════════════════════════════════════


class _FactCollectorNode:
    """LangGraph 节点：事实收集。"""

    def __init__(self, tool_service: Any):
        self._tool_service = tool_service
        self._agent = FactCollectorAgent(tool_service)

    async def __call__(self, state: DisputeState) -> Dict[str, Any]:
        ctx = AgentContext(
            request_message=state["request_message"],
            facts=state.get("facts", {}),
            metadata=state.get("metadata", {}),
            emotion_level=state.get("emotion_level"),
            order_id=state.get("order_id"),
            conversation_id=state.get("conversation_id", ""),
            domain=state.get("domain", "ecommerce"),
        )
        await self._agent.execute(ctx)
        return {"facts": ctx.facts, "metadata": ctx.metadata}


class _BuyerAnalysisNode:
    """LangGraph 节点：买家分析。"""

    def __init__(self, llm_service: Any):
        self._llm = llm_service
        self._agent = BuyerAnalysisAgent(llm_service)

    async def __call__(self, state: DisputeState) -> Dict[str, Any]:
        ctx = AgentContext(
            request_message=state["request_message"],
            facts=state.get("facts", {}),
            metadata=state.get("metadata", {}),
            emotion_level=state.get("emotion_level"),
            order_id=state.get("order_id"),
            conversation_id=state.get("conversation_id", ""),
            domain=state.get("domain", "ecommerce"),
        )
        result = await self._agent.execute(ctx)
        return {"buyer_result": result, "metadata": ctx.metadata}


class _SellerAnalysisNode:
    """LangGraph 节点：卖家分析。"""

    def __init__(self, llm_service: Any):
        self._llm = llm_service
        self._agent = SellerAnalysisAgent(llm_service)

    async def __call__(self, state: DisputeState) -> Dict[str, Any]:
        ctx = AgentContext(
            request_message=state["request_message"],
            facts=state.get("facts", {}),
            metadata=state.get("metadata", {}),
            emotion_level=state.get("emotion_level"),
            order_id=state.get("order_id"),
            conversation_id=state.get("conversation_id", ""),
            domain=state.get("domain", "ecommerce"),
        )
        result = await self._agent.execute(ctx)
        return {"seller_result": result, "metadata": ctx.metadata}


class _MediatorNode:
    """LangGraph 节点：调停裁决。"""

    def __init__(self, llm_service: Any):
        self._llm = llm_service
        self._agent = MediatorAgent(llm_service)

    async def __call__(self, state: DisputeState) -> Dict[str, Any]:
        metadata = dict(state.get("metadata", {}))
        if state.get("buyer_result"):
            metadata["buyer_analysis_result"] = state["buyer_result"]
        if state.get("seller_result"):
            metadata["seller_analysis_result"] = state["seller_result"]

        ctx = AgentContext(
            request_message=state["request_message"],
            facts=state.get("facts", {}),
            metadata=metadata,
            emotion_level=state.get("emotion_level"),
            order_id=state.get("order_id"),
            conversation_id=state.get("conversation_id", ""),
            domain=state.get("domain", "ecommerce"),
        )
        result = await self._agent.execute(ctx)
        return {"mediator_result": result, "metadata": metadata}


# ═══════════════════════════════════════════════════════════════════════
# 路由函数
# ═══════════════════════════════════════════════════════════════════════


def _route_after_fact_collector(
    state: DisputeState,
) -> List[str]:
    """事实收集后，并行路由到 buyer + seller。"""
    return ["buyer_analysis", "seller_analysis"]


# ═══════════════════════════════════════════════════════════════════════
# 图构建
# ═══════════════════════════════════════════════════════════════════════


def build_dispute_graph(
    tool_service: Any,
    llm_service: Any,
) -> StateGraph:
    """构建纠纷协调 LangGraph。

    结构：
        fact_collector → [buyer_analysis, seller_analysis] → mediator → END

    Args:
        tool_service: 工具服务实例
        llm_service: LLM 服务实例

    Returns:
        编译后的 LangGraph StateGraph
    """
    workflow = StateGraph(DisputeState)

    # 添加节点
    workflow.add_node("fact_collector", _FactCollectorNode(tool_service))
    workflow.add_node("parallel_start", lambda state: state)
    workflow.add_node("buyer_analysis", _BuyerAnalysisNode(llm_service))
    workflow.add_node("seller_analysis", _SellerAnalysisNode(llm_service))
    workflow.add_node("mediator", _MediatorNode(llm_service))

    # 添加边
    workflow.add_edge("fact_collector", "parallel_start")

    async def _parallel_router(state: DisputeState) -> List[Send]:
        return [
            Send("buyer_analysis", state),
            Send("seller_analysis", state),
        ]

    workflow.add_conditional_edges(
        "parallel_start",
        _parallel_router,
        ["buyer_analysis", "seller_analysis"],
    )
    workflow.add_edge("buyer_analysis", "mediator")
    workflow.add_edge("seller_analysis", "mediator")
    workflow.add_edge("mediator", END)

    # 设置入口
    workflow.set_entry_point("fact_collector")

    return workflow


class DisputeLangGraph:
    """纠纷协调 LangGraph 封装。"""

    def __init__(
        self,
        tool_service: Any,
        llm_service: Any,
    ):
        self._tool_service = tool_service
        self._llm_service = llm_service
        self._graph = build_dispute_graph(tool_service, llm_service).compile()

    async def resolve(  # noqa: PLR0913
        self,
        request_message: str,
        emotion_level: Optional[EmotionLevel] = None,
        order_id: Optional[str] = None,
        conversation_id: str = "",
        domain: str = "ecommerce",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """执行纠纷协调流程。

        Returns:
            dict with keys: facts, buyer_result, seller_result, mediator_result
        """
        initial_state: DisputeState = {
            "request_message": request_message,
            "facts": {},
            "metadata": metadata or {},
            "emotion_level": emotion_level,
            "order_id": order_id,
            "conversation_id": conversation_id,
            "domain": domain,
            "buyer_result": None,
            "seller_result": None,
            "mediator_result": None,
            "error": None,
        }

        try:
            result = await self._graph.ainvoke(initial_state)
            return {
                "facts": result.get("facts", {}),
                "buyer_result": result.get("buyer_result"),
                "seller_result": result.get("seller_result"),
                "mediator_result": result.get("mediator_result"),
                "error": result.get("error"),
            }
        except Exception as e:
            logger.error(f"LangGraph 纠纷协调失败: {str(e)[:100]}")
            return {
                "facts": {},
                "buyer_result": None,
                "seller_result": None,
                "mediator_result": None,
                "error": str(e),
            }
