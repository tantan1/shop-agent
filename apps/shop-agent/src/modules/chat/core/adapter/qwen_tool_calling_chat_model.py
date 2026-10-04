"""LangChain wrapper for Qwen models that output tool calls as JSON in content."""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig


class QwenToolCallingChatModel(ChatOpenAI):
    """包装 Qwen 模型，将 content 中的 JSON 转为标准 tool_calls。

    支持的输出格式：
    1. 直接 JSON: {"name": "tool_name", "arguments": {"key": "value"}}
    2. 代码块 JSON: ```json {"name": "...", "arguments": {...}} ```
    3. 普通代码块: ``` {"name": "...", "arguments": {...}} ```
    """

    def _extract_tool_calls(self, content: str) -> List[ToolCall]:
        """从文本内容中提取工具调用。"""
        tool_calls: List[ToolCall] = []
        
        # 模式按优先级排序
        patterns = [
            # 标准代码块 JSON
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            # 直接 JSON 对象（贪婪匹配最外层）
            r'(\{"name":\s*"[^"]+",\s*"arguments":\s*\{.*?\}\})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    if isinstance(data, dict) and "name" in data:
                        tool_calls.append(ToolCall(
                            name=data["name"],
                            args=data.get("arguments", {}),
                            id=f"call_{len(tool_calls)}",
                            type="tool_call",
                        ))
                except (json.JSONDecodeError, KeyError):
                    continue
                    
        return tool_calls

    def _process_message(self, message: AIMessage) -> AIMessage:
        """处理消息，提取 tool_calls。"""
        if not message.content:
            return message
            
        tool_calls = self._extract_tool_calls(message.content)
        if tool_calls:
            # 创建新消息保留原有属性
            new_msg = AIMessage(
                content="",
                tool_calls=tool_calls,
                response_metadata=message.response_metadata,
                id=message.id,
            )
            # 复制其他属性
            for attr in ["usage_metadata", "additional_kwargs"]:
                if hasattr(message, attr):
                    setattr(new_msg, attr, getattr(message, attr))
            return new_msg
        return message

    async def _agenerate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = await super()._agenerate(messages, stop, run_manager, **kwargs)
        
        # 处理每个 generation
        new_generations = []
        for gen in result.generations:
            if isinstance(gen.message, AIMessage):
                processed_msg = self._process_message(gen.message)
                new_generations.append(gen.__class__(message=processed_msg))
            else:
                new_generations.append(gen)
                
        return ChatResult(generations=new_generations)

    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = super()._generate(messages, stop, run_manager, **kwargs)
        
        new_generations = []
        for gen in result.generations:
            if isinstance(gen.message, AIMessage):
                processed_msg = self._process_message(gen.message)
                new_generations.append(gen.__class__(message=processed_msg))
            else:
                new_generations.append(gen)
                
        return ChatResult(generations=new_generations)