"""
Agent 运行器抽象协议

定义与框架无关的 Agent 编排接口，业务代码只依赖此协议，
具体实现由适配器层提供（如 LangChainAdapter、LangGraphAdapter）
"""
from __future__ import annotations

from typing import Protocol, Any


class AgentRunner(Protocol):
    """Agent 运行器抽象协议"""

    async def run(
        self,
        request: Any,
        **kwargs
    ) -> Any:
        """
        运行 Agent

        Args:
            request: 请求对象

        Returns:
            响应对象
        """
        ...