"""
工具执行器抽象协议

定义与框架无关的工具调用接口，业务代码只依赖此协议，
具体实现由适配器层提供
"""
from __future__ import annotations

from typing import Protocol, Dict, Any, List


class ToolResult:
    """工具执行结果"""
    def __init__(self, success: bool, content: str, error: str = ""):
        self.success = success
        self.content = content
        self.error = error


class ToolExecutor(Protocol):
    """工具执行器抽象协议"""

    async def execute(
        self,
        tool_name: str,
        tool_params: Dict[str, Any],
        **kwargs
    ) -> ToolResult:
        """
        执行工具

        Args:
            tool_name: 工具名称
            tool_params: 工具参数

        Returns:
            工具执行结果
        """
        ...

    def get_tool_names(self) -> List[str]:
        """
        获取所有可用工具名称

        Returns:
            工具名称列表
        """
        ...

    def get_tool_description(self, tool_name: str) -> str:
        """
        获取工具描述

        Args:
            tool_name: 工具名称

        Returns:
            工具描述
        """
        ...