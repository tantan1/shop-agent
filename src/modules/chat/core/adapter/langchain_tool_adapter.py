"""
LangChain Tool 适配器

将 LangChain 的工具调用接口适配到我们的抽象协议
"""
from __future__ import annotations

from typing import Dict, Any, List

from src.modules.chat.core.abstract.tool_executor import ToolExecutor, ToolResult


class LangChainToolAdapter(ToolExecutor):
    """LangChain Tool 适配器"""

    def __init__(self, tool_service):
        self._tool_service = tool_service

    async def execute(
        self,
        tool_name: str,
        tool_params: Dict[str, Any],
        **kwargs
    ) -> ToolResult:
        try:
            result = await self._tool_service.dispatch(tool_name, tool_params)
            return ToolResult(success=True, content=result)
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))

    def get_tool_names(self) -> List[str]:
        self._tool_service._ensure_registry()
        return list(self._tool_service._registry.keys())

    def get_tool_description(self, tool_name: str) -> str:
        self._tool_service._ensure_registry()
        tool_fn = self._tool_service._registry.get(tool_name)
        if tool_fn and tool_fn.__doc__:
            return tool_fn.__doc__
        return f"工具 {tool_name}"