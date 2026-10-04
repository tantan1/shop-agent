"""
Mock ChatModel 适配器

实现 LangChain BaseChatModel 接口，内部委托给 MockLLMAdapter。
用于在 LLM_ADAPTER_TYPE=mock 时替代 ChatOpenAI，返回结构化工具调用。
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, Type, Union

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import RunnableBinding
from pydantic import BaseModel, Field

from src.modules.chat.core.adapter.mock_llm_adapter import MockLLMAdapter
from src.modules.chat.config import chat_config


class MockRunnableBinding(RunnableBinding):
    """自定义 RunnableBinding，支持 bind_tools 方法。"""
    
    def bind_tools(
        self,
        tools: Sequence[Dict[str, Any] | Type | Callable | BaseTool],
        **kwargs: Any,
    ) -> "MockRunnableBinding":
        """绑定工具：返回新的 MockRunnableBinding。"""
        formatted_tools = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.args_schema.schema() if tool.args_schema else {"type": "object", "properties": {}}
                    }
                })
            elif isinstance(tool, dict) and "function" in tool:
                formatted_tools.append(tool)
            elif hasattr(tool, "__name__"):  # Callable
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.__name__,
                        "description": getattr(tool, "__doc__", ""),
                        "parameters": {"type": "object", "properties": {}}
                    }
                })
        # 合并现有工具和新工具
        existing_tools = self.kwargs.get("tools", [])
        all_tools = existing_tools + formatted_tools
        # 创建新 binding，并在 bound model 上设置 tools
        new_binding = MockRunnableBinding(bound=self.bound, kwargs={"tools": all_tools}, config_factories=self.config_factories)
        # 将 tools 存储到 bound model 上（如果是 MockChatModel）
        if hasattr(self.bound, '_set_bound_tools'):
            new_binding.bound._set_bound_tools(all_tools)
        return new_binding

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        """异步调用，将绑定的 tools 存储到 model 上。"""
        tools = self.kwargs.get("tools", [])
        if hasattr(self.bound, '_set_bound_tools'):
            self.bound._set_bound_tools(tools)
        return await self.bound.ainvoke(input, config, **kwargs)

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        """同步调用，将绑定的 tools 存储到 model 上。"""
        tools = self.kwargs.get("tools", [])
        if hasattr(self.bound, '_set_bound_tools'):
            self.bound._set_bound_tools(tools)
        return self.bound.invoke(input, config, **kwargs)


class MockChatModel(BaseChatModel):
    """Mock ChatModel 适配器，实现 LangChain BaseChatModel 接口。"""

    mock_adapter: MockLLMAdapter = Field(default_factory=MockLLMAdapter)
    model_name: str = "mock-llm"
    bound_tools: List[Dict] = Field(default_factory=list, exclude=True)

    def _set_bound_tools(self, tools: List[Dict]) -> None:
        """设置绑定的工具（供 RunnableBinding 调用）。"""
        self.bound_tools = tools

    def _convert_messages_to_dict(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """将 LangChain 消息转换为字典格式。"""
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, ToolMessage):
                role = "tool"
            else:
                role = "user"
            result.append({"role": role, "content": msg.content})
        return result

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """同步生成（委托给异步版本）。"""
        import asyncio

        async def _async_generate():
            return await self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_async_generate())
        else:
            return loop.run_until_complete(_async_generate())

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步生成，委托给 MockLLMAdapter。"""
        message_dicts = self._convert_messages_to_dict(messages)
        temperature = kwargs.get("temperature", 0.7)

        # 使用实例变量存储的绑定 tools
        bound_tools = getattr(self, 'bound_tools', [])

        response_text = await self.mock_adapter.chat(message_dicts, temperature=temperature, tools=bound_tools)

        # 如果响应已经是工具调用格式，直接返回 AIMessage
        # 否则包装为普通回复
        try:
            parsed = json.loads(response_text)
            if "tool_calls" in parsed:
                # 构造带 tool_calls 的 AIMessage
                ai_msg = AIMessage(content="")
                ai_msg.tool_calls = parsed["tool_calls"]
                generation = ChatGeneration(message=ai_msg)
            else:
                generation = ChatGeneration(message=AIMessage(content=response_text))
        except (json.JSONDecodeError, TypeError):
            generation = ChatGeneration(message=AIMessage(content=response_text))

        return ChatResult(generations=[generation])

    async def astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGeneration]:
        """流式生成（简单实现：一次性返回）。"""
        result = await self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        for gen in result.generations:
            yield gen

    def with_structured_output(
        self,
        schema: Union[Dict[str, Any], Type[BaseModel]],
        **kwargs: Any,
    ) -> "MockChatModel":
        """返回自身，structured output 由 MockLLMAdapter.chat_structured 处理。"""
        return self

    def bind_tools(
        self,
        tools: Sequence[Dict[str, Any] | Type | Callable | BaseTool],
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        """绑定工具：返回自定义 MockRunnableBinding。"""
        formatted_tools = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.args_schema.schema() if tool.args_schema else {"type": "object", "properties": {}}
                    }
                })
            elif isinstance(tool, dict) and "function" in tool:
                formatted_tools.append(tool)
            elif hasattr(tool, "__name__"):  # Callable
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.__name__,
                        "description": getattr(tool, "__doc__", ""),
                        "parameters": {"type": "object", "properties": {}}
                    }
                })
        return MockRunnableBinding(bound=self, kwargs={"tools": formatted_tools}, config_factories=[])

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}

    def bind_tools(self, tools, **kwargs: Any) -> "MockChatModel":
        """支持 bind_tools，返回自身（Mock 模式下工具调用由 MockLLMAdapter 内部处理）。"""
        return self


class MockRunnableBinding(RunnableBinding):
    """Mock 模式下的 RunnableBinding，支持链式 bind_tools 调用。"""
    
    def bind_tools(self, tools, **kwargs: Any) -> "MockRunnableBinding":
        return self
    
    @property
    def bound(self) -> "MockRunnableBinding":
        return self