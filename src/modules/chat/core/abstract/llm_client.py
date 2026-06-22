"""
LLM 客户端抽象协议

定义与框架无关的 LLM 调用接口，业务代码只依赖此协议，
具体实现由适配器层提供（如 LangChainAdapter、NativeAPIAdapter）
"""
from __future__ import annotations

from typing import Protocol, List, Dict, Optional, Type, Any
from pydantic import BaseModel


class LLMClient(Protocol):
    """LLM 客户端抽象协议"""

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        聊天接口

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数

        Returns:
            模型回复内容
        """
        ...

    async def chat_structured(
        self,
        messages: List[Dict[str, str]],
        output_schema: Type[BaseModel],
        temperature: float = 0.0,
        **kwargs
    ) -> BaseModel:
        """
        结构化输出接口

        Args:
            messages: 消息列表
            output_schema: Pydantic Schema
            temperature: 温度参数

        Returns:
            结构化对象（Pydantic Model 实例）
        """
        ...

    async def embed(self, text: str) -> List[float]:
        """
        文本嵌入接口

        Args:
            text: 输入文本

        Returns:
            嵌入向量
        """
        ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量文本嵌入接口

        Args:
            texts: 输入文本列表

        Returns:
            嵌入向量列表
        """
        ...


class TokenUsage:
    """Token 使用情况"""
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens