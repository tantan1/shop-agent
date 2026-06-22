"""
适配器工厂

根据配置创建对应的适配器实例，实现框架解耦
"""
from __future__ import annotations

from typing import Optional

from src.modules.chat.core.abstract.llm_client import LLMClient
from src.modules.chat.core.abstract.tool_executor import ToolExecutor
from src.modules.chat.core.abstract.agent_runner import AgentRunner
from src.modules.chat.config import chat_config


class AdapterFactory:
    """适配器工厂"""

    _llm_adapter: Optional[LLMClient] = None
    _tool_adapter: Optional[ToolExecutor] = None

    @classmethod
    def get_llm_adapter(cls) -> LLMClient:
        """获取 LLM 适配器"""
        if cls._llm_adapter is None:
            adapter_type = getattr(chat_config, "LLM_ADAPTER_TYPE", "langchain")
            
            if adapter_type == "mock":
                from src.modules.chat.core.adapter.mock_llm_adapter import MockLLMAdapter
                cls._llm_adapter = MockLLMAdapter()
            else:
                from src.modules.chat.core.adapter.langchain_llm_adapter import LangChainLLMAdapter
                from src.modules.chat.core.llm_service import LLMService
                cls._llm_adapter = LangChainLLMAdapter(LLMService.get_instance())
        
        return cls._llm_adapter

    @classmethod
    def get_tool_adapter(cls) -> ToolExecutor:
        """获取工具执行器适配器"""
        if cls._tool_adapter is None:
            from src.modules.chat.core.adapter.langchain_tool_adapter import LangChainToolAdapter
            from src.modules.chat.core.tool_registry import ToolService
            cls._tool_adapter = LangChainToolAdapter(ToolService())
        return cls._tool_adapter

    @classmethod
    def create_agent_adapter(cls, agent_instance) -> AgentRunner:
        """创建 Agent 适配器"""
        from src.modules.chat.core.adapter.langchain_agent_adapter import LangChainAgentAdapter
        return LangChainAgentAdapter(agent_instance)

    @classmethod
    def reset(cls):
        """重置适配器实例（用于测试或配置变更）"""
        cls._llm_adapter = None
        cls._tool_adapter = None