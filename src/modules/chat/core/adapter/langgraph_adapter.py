"""
LangGraph 适配器

将 LangGraph 的图节点、路由、构建器适配到我们的抽象协议
"""
from __future__ import annotations

from typing import Dict, Any, Literal, Optional
from typing_extensions import TypedDict

from src.modules.chat.core.abstract.graph_node import (
    GraphState,
    GraphNode,
    GraphRouter,
    GraphBuilder,
    GraphExecutor,
)


class LangGraphState(TypedDict):
    """LangGraph 状态实现"""
    pass


class LangGraphNodeAdapter(GraphNode):
    """LangGraph 节点适配器"""

    def __init__(self, name: str, func):
        self._name = name
        self._func = func

    async def execute(self, state: GraphState) -> Dict[str, Any]:
        return self._func(state.to_dict())

    @property
    def name(self) -> str:
        return self._name


class LangGraphRouterAdapter(GraphRouter):
    """LangGraph 路由适配器"""

    def __init__(self, name: str, func):
        self._name = name
        self._func = func

    async def route(self, state: GraphState) -> str:
        return self._func(state.to_dict())

    @property
    def name(self) -> str:
        return self._name


class LangGraphBuilderAdapter(GraphBuilder):
    """LangGraph 构建器适配器"""

    def __init__(self, state_schema):
        from langgraph.graph import StateGraph
        self._builder = StateGraph(state_schema)
        self._state_schema = state_schema

    def add_node(self, name: str, node: GraphNode) -> None:
        async def wrapper(state):
            result = await node.execute(LangGraphStateAdapter(state))
            return result
        self._builder.add_node(name, wrapper)

    def add_edge(self, from_node: str, to_node: str) -> None:
        self._builder.add_edge(from_node, to_node)

    def add_conditional_edge(self, source: str, router: GraphRouter) -> None:
        async def wrapper(state):
            return await router.route(LangGraphStateAdapter(state))
        self._builder.add_conditional_edges(source, wrapper)

    def set_entry_point(self, node_name: str) -> None:
        self._builder.set_entry_point(node_name)

    def compile(self) -> GraphExecutor:
        app = self._builder.compile()
        return LangGraphExecutorAdapter(app)


class LangGraphExecutorAdapter(GraphExecutor):
    """LangGraph 执行器适配器"""

    def __init__(self, app):
        self._app = app

    async def run(self, initial_state: Dict[str, Any], **kwargs) -> GraphState:
        result = await self._app.ainvoke(initial_state, **kwargs)
        return LangGraphStateAdapter(result)

    async def stream(self, initial_state: Dict[str, Any], **kwargs):
        async for event in self._app.astream(initial_state, **kwargs):
            yield {k: LangGraphStateAdapter(v) if isinstance(v, dict) else v for k, v in event.items()}


class LangGraphStateAdapter(GraphState):
    """LangGraph 状态适配器"""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LangGraphStateAdapter":
        return cls(data)