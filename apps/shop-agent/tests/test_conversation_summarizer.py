"""对话历史摘要生成器单测。"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock

from src.modules.chat.agent.conversation_summarizer import ConversationSummarizer


class TestConversationSummarizer:
    """摘要逻辑：短历史直接返回，长历史 LLM 摘要 + 近期保留。"""

    @pytest.fixture
    def summarizer(self):
        llm = MagicMock()
        llm.chat_qwen = AsyncMock(return_value="用户之前咨询了订单 123456 的物流状态。")
        return ConversationSummarizer(llm_service=llm)

    @pytest.mark.asyncio
    async def test_short_history_no_summary(self, summarizer):
        """短历史未超预算，直接返回格式化文本。"""
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "您好，有什么可以帮您？"},
        ]
        result = await summarizer.summarize_if_needed(messages, max_tokens=5000)
        assert "用户: 你好" in result
        assert "助手: 您好" in result

    @pytest.mark.asyncio
    async def test_long_history_triggers_summary(self, summarizer):
        """长历史超预算，触发 LLM 摘要。"""
        messages = [
            {"role": "user", "content": f"历史消息 {i} " * 100}
            for i in range(20)
        ]
        result = await summarizer.summarize_if_needed(messages, max_tokens=100)
        # 应调用 LLM 生成摘要
        summarizer._llm.chat_qwen.assert_called_once()
        assert "【历史摘要】" in result
        assert "【最近对话】" in result

    @pytest.mark.asyncio
    async def test_empty_messages(self, summarizer):
        """空消息列表返回空字符串。"""
        result = await summarizer.summarize_if_needed([])
        assert result == ""

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self, summarizer):
        """LLM 调用失败时降级为截断。"""
        summarizer._llm.chat_qwen = AsyncMock(side_effect=Exception("LLM error"))
        messages = [
            {"role": "user", "content": f"历史消息 {i} " * 100}
            for i in range(20)
        ]
        result = await summarizer.summarize_if_needed(messages, max_tokens=100)
        # 降级：直接截断
        assert "..." in result
