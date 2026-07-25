"""
Mock LLM 适配器

用于测试和验证抽象协议，不依赖任何框架。
LLM_ADAPTER_TYPE=mock 时通过本适配器/LLMService 仿真延迟与错误率（压测 0 Token 消耗）。
"""

from __future__ import annotations

import asyncio
import random
from typing import Dict, List, Type

from pydantic import BaseModel

from src.modules.chat.config import chat_config
from src.modules.chat.core.abstract.llm_client import LLMClient


class MockLLMAdapter(LLMClient):
    """Mock LLM 适配器（支持延迟/错误率仿真）"""

    def _simulate(self) -> None:
        """按配置仿真延迟与错误。"""
        latency_min = getattr(chat_config, "mock_llm_latency_min", 100) / 1000.0
        latency_max = getattr(chat_config, "mock_llm_latency_max", 500) / 1000.0
        error_rate = getattr(chat_config, "mock_llm_error_rate", 0.01)

        # 非事件循环上下文（如 ThreadPool）时退化为 time.sleep
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            import time

            time.sleep(random.uniform(latency_min, latency_max))
        else:
            # 仅简单同步路径用；异步场景由 LLMService._mock_simulate 处理
            pass

        if random.random() < error_rate:
            raise TimeoutError("Mock LLM timeout (simulated)")

    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, **kwargs) -> str:
        await asyncio.sleep(
            random.uniform(
                getattr(chat_config, "mock_llm_latency_min", 100) / 1000.0,
                getattr(chat_config, "mock_llm_latency_max", 500) / 1000.0,
            )
        )
        if random.random() < getattr(chat_config, "mock_llm_error_rate", 0.01):
            raise TimeoutError("Mock LLM timeout (simulated)")
        last_user_message = messages[-1]["content"] if messages else ""
        return f"[Mock LLM] 收到消息：{last_user_message[:50]}..."

    async def chat_structured(
        self,
        messages: List[Dict[str, str]],
        output_schema: Type[BaseModel],
        temperature: float = 0.0,
        **kwargs,
    ) -> BaseModel:
        return output_schema()

    async def embed(self, text: str) -> List[float]:
        return [0.1] * 768

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * 768 for _ in texts]
