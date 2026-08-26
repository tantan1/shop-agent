"""YAML 编排运行时状态与图封装（阶段 2.7 / 2.9）。

定义 LangGraph 用的 GraphState（与 YAML state.fields 对齐），以及编译后图的
ainvoke 入口封装。checkpointer 默认 RedisCheckpointSaver（Redis 不可用时
自动降级为内存 MemorySaver）。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph
from langgraph.types import Command

from src.modules.chat.agent.yaml_flow.checkpointer import RedisCheckpointSaver

# GraphState：节点间共享状态（与 YAML state.fields 对齐，阶段 1.5a）。
# - messages：对话历史（list[dict]，累加 reducer）
# - intent：意图识别结果（dict，含 intent/action/complexity）
# - params：抽取/硬强制出的业务参数（dict）
# - tool_result：节点产出（any，read_from/write_to 落点，LastValue 覆盖）
# - hitl_pending：人在回路挂起上下文（any）
# - thread_id：跨请求恢复标识（str）
#
# 每个字段是独立 channel，避免 dict 整体当 __root__ 时的并发写入冲突。
class GraphState(TypedDict, total=False):
    messages: Annotated[List[Dict[str, Any]], operator.add]
    intent: Dict[str, Any]
    params: Dict[str, Any]
    tool_result: Any
    hitl_pending: Any
    thread_id: str
    emergency: bool
    # 阶段 4.2：RAG 四步拆节点之间的共享中间态。
    # 承载 {rewritten_queries, safety, documents, rag_context, response, quality}。
    # 四步节点各自 read_from/write_to 该 channel 的不同 key，避免与 tool_result 混用。
    rag: Dict[str, Any]


def new_graph_state(thread_id: str, *, messages: Optional[list] = None, intent: Optional[dict] = None, params: Optional[dict] = None) -> Dict[str, Any]:
    """构造初始 state。"""
    return {
        "messages": messages or [],
        "intent": intent or {},
        "params": params or {},
        "tool_result": None,
        "hitl_pending": None,
        "thread_id": thread_id,
        "emergency": False,
        "rag": {},
    }


class CompiledFlow:
    """编译后的流程图封装（阶段 2.9）。

    提供 ainvoke 入口，内部用 checkpointer 支撑 human_approval 的 interrupt/resume。
    """

    def __init__(
        self,
        workflow: StateGraph,
        *,
        checkpointer=None,
        execution_store=None,
        recursion_limit: int = 25,
    ):
        self._checkpointer = checkpointer or RedisCheckpointSaver()
        self._execution_store = execution_store
        self._recursion_limit = recursion_limit
        self._graph = workflow.compile(
            checkpointer=self._checkpointer,
            # interrupt 在 human_approval 节点触发（阶段 3 完善跨请求恢复）
        )

    @property
    def graph(self):
        return self._graph

    async def ainvoke(self, state: Dict[str, Any], thread_id: str | None = None) -> Dict[str, Any]:
        """驱动图执行，返回最终 state。

        thread_id 用于 checkpointer 定位（human_approval 挂起/恢复）。

        注意：当图中存在 human_approval 节点时，首次 ainvoke 会在
        LangGraph ``interrupt()`` 处挂起（state 已写入 checkpointer），
        并返回含挂起信息的 state。调用方应检测 ``hitl_pending`` 字段并
        转成「等待审批」响应（阶段 3.1 / 3.2）。
        """
        tid = thread_id or state.get("thread_id") or "default"
        try:
            result = await self._graph.ainvoke(
                state,
                config={"configurable": {"thread_id": tid}, "recursion_limit": self._recursion_limit},
            )
        except Exception as exc:
            if self._execution_store is not None:
                await self._execution_store.save_execution(
                    thread_id=tid,
                    state=state,
                    current_node=state.get("current_node", ""),
                    status="failed",
                    error_message=str(exc),
                )
            raise

        if self._execution_store is not None:
            is_suspended = result.get("hitl_pending") is not None
            status = "suspended" if is_suspended else "completed"
            await self._execution_store.save_execution(
                thread_id=tid,
                state=result,
                current_node=result.get("current_node", ""),
                status=status,
            )
        return result

    async def aresume(self, thread_id: str, confirm: bool) -> Dict[str, Any]:
        """从 human_approval 中断点恢复图执行（阶段 3.2）。

        用同一个 checkpointer + ``Command(resume=confirm)`` 驱动图从
        ``interrupt()`` 挂起点继续；``confirm`` 作为中断点的返回值
        （即 HumanApprovalHandler.run 内 ``interrupt()`` 的返回值）。

        Args:
            thread_id: 与 ainvoke 时一致的 thread_id（跨请求恢复标识）
            confirm: True=审批通过，False=审批拒绝

        Returns:
            恢复执行后的最终 state（dict）
        """
        if self._execution_store is not None:
            await self._execution_store.append_event(
                thread_id=thread_id,
                event_type="resumed",
                payload={"confirm": confirm},
                node_name="human_approval",
            )
        result = await self._graph.ainvoke(
            Command(resume=confirm),
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": self._recursion_limit},
        )
        if self._execution_store is not None:
            await self._execution_store.save_execution(
                thread_id=thread_id,
                state=result,
                current_node=result.get("current_node", ""),
                status="completed",
            )
        return result

    def get_graph(self):
        """LangGraph Studio 调试入口（阶段 2.10）。"""
        return self._graph
