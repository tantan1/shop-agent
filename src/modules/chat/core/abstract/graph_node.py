"""
Graph Node 抽象协议

定义与框架无关的工作流节点接口，业务代码只依赖此协议，
具体实现由适配器层提供（如 LangGraphAdapter、其他工作流引擎）
"""
from __future__ import annotations

from typing import Protocol, Dict, Any, Optional


class GraphState(Protocol):
    """图状态抽象协议"""

    def get(self, key: str, default: Any = None) -> Any:
        ...

    def set(self, key: str, value: Any) -> None:
        ...

    def to_dict(self) -> Dict[str, Any]:
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphState":
        ...


class GraphNode(Protocol):
    """图节点抽象协议"""

    async def execute(self, state: GraphState) -> Dict[str, Any]:
        """
        执行节点逻辑

        Args:
            state: 当前图状态

        Returns:
            需要更新到状态的键值对
        """
        ...

    @property
    def name(self) -> str:
        """节点名称"""
        ...


class GraphRouter(Protocol):
    """图路由抽象协议（条件边）"""

    async def route(self, state: GraphState) -> str:
        """
        根据状态决定下一个节点

        Args:
            state: 当前图状态

        Returns:
            目标节点名称
        """
        ...

    @property
    def name(self) -> str:
        """路由名称"""
        ...


class GraphBuilder(Protocol):
    """图构建器抽象协议"""

    def add_node(self, name: str, node: GraphNode) -> None:
        """添加节点"""
        ...

    def add_edge(self, from_node: str, to_node: str) -> None:
        """添加确定性边"""
        ...

    def add_conditional_edge(self, source: str, router: GraphRouter) -> None:
        """添加条件边"""
        ...

    def set_entry_point(self, node_name: str) -> None:
        """设置入口节点"""
        ...

    def compile(self) -> "GraphExecutor":
        """编译图，返回执行器"""
        ...


class GraphExecutor(Protocol):
    """图执行器抽象协议"""

    async def run(self, initial_state: Dict[str, Any], **kwargs) -> GraphState:
        """
        执行图

        Args:
            initial_state: 初始状态

        Returns:
            最终状态
        """
        ...

    async def stream(self, initial_state: Dict[str, Any], **kwargs):
        """流式执行图"""
        ...