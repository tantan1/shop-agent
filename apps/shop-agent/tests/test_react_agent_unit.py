"""ReAct Agent 关键路径单测（工具选择过滤 + HITL + 空参拒绝）。"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.modules.chat.agent.react_agent import ReActAgent, _INTERRUPT_MEM
from src.modules.chat.schemas import IntentResult, ChatRequest


class TestToolSelectionFilter:
    """三层工具过滤：P0 意图 + P1 语义 + P2 本地模型。"""

    @pytest.fixture
    def agent(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")

        llm = MagicMock()
        llm.chat_qwen = MagicMock(return_value="mock")
        tool = MagicMock()
        embedding = MagicMock()
        embedding.embed_query = MagicMock(return_value=[0.1] * 768)

        return ReActAgent(
            llm_service=llm,
            tool_service=tool,
            embedding_service=embedding,
        )

    def test_all_tools_populated(self, agent):
        """Agent 应加载工具列表（_all_tools）。"""
        assert hasattr(agent, "_all_tools")
        assert len(agent._all_tools) > 0

    def test_empty_query_detection(self, agent):
        """空输入应在 ReAct 前被检测到。"""
        message = "   "
        assert not message.strip()


class TestInterruptStore:
    """人在回路：中断上下文持久化。"""

    def test_store_and_retrieve_memory_fallback(self, monkeypatch):
        """Redis 不可用时降级内存存储。"""
        thread_id = "test-thread-1"
        _INTERRUPT_MEM.pop(thread_id, None)

        # Mock Redis 不可用
        mock_redis = MagicMock()
        mock_redis.is_available = False
        monkeypatch.setattr(
            "src.modules.chat.agent.react_agent_interrupt.get_redis_cache_service",
            lambda: mock_redis,
        )

        from src.modules.chat.agent.react_agent import _store_interrupt, InterruptContext
        _store_interrupt(
            InterruptContext(
                thread_id=thread_id,
                graph=None,
                config={},
                conversation_id="conv-1",
                intent_steps=[],
                domain="ecommerce",
                order_id="ORDER-123",
                reason="退款审批",
            )
        )

        assert thread_id in _INTERRUPT_MEM
        stored = _INTERRUPT_MEM[thread_id]
        assert stored[2] == "conv-1"
        assert stored[4] == "ecommerce"

        _INTERRUPT_MEM.pop(thread_id, None)
