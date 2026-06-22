"""
LangChain Agent 适配器

将 LangChain 的 Agent 编排接口适配到我们的抽象协议
"""
from __future__ import annotations

from typing import Any

from src.modules.chat.core.abstract.agent_runner import AgentRunner


class LangChainAgentAdapter(AgentRunner):
    """LangChain Agent 适配器"""

    def __init__(self, agent_instance):
        self._agent = agent_instance

    async def run(
        self,
        request: Any,
        **kwargs
    ) -> Any:
        return await self._agent.run(request=request, **kwargs)