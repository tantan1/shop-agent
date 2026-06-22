"""
LangChain LLM 适配器

将 LangChain 的 LLM 接口适配到我们的抽象协议
"""
from __future__ import annotations

from typing import List, Dict, Type, Any
from pydantic import BaseModel

from src.modules.chat.core.abstract.llm_client import LLMClient


class LangChainLLMAdapter(LLMClient):
    """LangChain LLM 适配器"""

    def __init__(self, llm_service):
        self._llm_service = llm_service

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        return await self._llm_service.chat_qwen(
            messages,
            temperature=temperature,
            **kwargs
        )

    async def chat_structured(
        self,
        messages: List[Dict[str, str]],
        output_schema: Type[BaseModel],
        temperature: float = 0.0,
        **kwargs
    ) -> BaseModel:
        return await self._llm_service.chat_qwen_structured(
            messages,
            output_schema=output_schema,
            temperature=temperature,
            **kwargs
        )

    async def embed(self, text: str) -> List[float]:
        from src.modules.chat.core.embedding_service import EmbeddingService
        embedding_service = EmbeddingService.get_instance()
        return await embedding_service.embed_query(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        from src.modules.chat.core.embedding_service import EmbeddingService
        embedding_service = EmbeddingService.get_instance()
        return await embedding_service.aembed_documents(texts)