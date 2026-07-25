"""AgentOrchestrator 关键路径单测（情绪升级 + 路由分发）。"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.modules.chat.agent.orchestrator import AgentOrchestrator
from src.modules.chat.schemas import ChatRequest, IntentResult
from src.modules.chat.core.sentiment_service import EmotionResult, EmotionLevel, SentimentService


class TestEmotionEscalation:
    """舆情风险检测：触发立即升级。"""

    @pytest.fixture
    def orchestrator(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")

        llm = MagicMock()
        llm.chat_qwen = AsyncMock(return_value="mock response")
        llm.chat_qwen_structured = AsyncMock(return_value=MagicMock())

        embedding = MagicMock()
        embedding.embed_query = AsyncMock(return_value=[0.1] * 768)

        milvus = MagicMock()
        milvus.hybrid_search = AsyncMock(return_value=[])

        intent = MagicMock()
        intent.recognize = AsyncMock(return_value=IntentResult(action="query-order", confidence=0.9))

        tool = MagicMock()
        tool.dispatch = AsyncMock(return_value="tool result")

        orch = AgentOrchestrator(
            llm_service=llm,
            embedding_service=embedding,
            milvus_service=milvus,
            intent_recognizer=intent,
            tool_service=tool,
        )
        return orch

    @pytest.mark.asyncio
    async def test_emotion_escalation_short_circuits(self, orchestrator):
        """紧急情绪应直接返回升级响应，不进入后续管线。"""
        emergency_emotion = EmotionResult(
            level=EmotionLevel.EMERGENCY,
            confidence=0.95,
            escalate=True,
            is_emergency=True,
            keywords=["愤怒", "投诉"],
            source="rule",
        )
        mock_sentiment = MagicMock()
        mock_sentiment.detect = AsyncMock(return_value=emergency_emotion)

        with patch.object(orchestrator, '_ensure_sentiment_service', return_value=mock_sentiment):
            request = ChatRequest(message="我要退货！", domain="ecommerce")
            response = await orchestrator.chat_with_agent(request)
            assert response.status == "escalated"
            assert "高级专员" in response.message
            assert len(response.steps) == 1
            assert response.steps[0]["status"] == "escalated"

    @pytest.mark.asyncio
    async def test_normal_emotion_continues(self, orchestrator):
        """普通情绪应继续后续管线（不升级）。"""
        normal_emotion = EmotionResult(
            level=EmotionLevel.NEUTRAL,
            confidence=0.5,
            escalate=False,
            is_emergency=False,
            keywords=[],
            source="rule",
        )
        mock_sentiment = MagicMock()
        mock_sentiment.detect = AsyncMock(return_value=normal_emotion)

        with patch.object(orchestrator, '_ensure_sentiment_service', return_value=mock_sentiment):
            request = ChatRequest(message="你好", domain="ecommerce")
            response = await orchestrator.chat_with_agent(request)
            assert response.status != "escalated"
