"""
Mock LLM 适配器

用于测试和验证抽象协议，不依赖任何框架
"""
from __future__ import annotations

from typing import List, Dict, Type, Any
from pydantic import BaseModel

from src.modules.chat.core.abstract.llm_client import LLMClient


class MockLLMAdapter(LLMClient):
    """Mock LLM 适配器"""

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        last_user_message = messages[-1]["content"] if messages else ""
        return f"[Mock LLM] 收到消息：{last_user_message[:50]}..."

    async def chat_structured(
        self,
        messages: List[Dict[str, str]],
        output_schema: Type[BaseModel],
        temperature: float = 0.0,
        **kwargs
    ) -> BaseModel:
        return output_schema()

    async def embed(self, text: str) -> List[float]:
        return [0.1] * 768

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * 768 for _ in texts]