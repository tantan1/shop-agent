"""AgentOrchestrator 关键路径单测（情绪升级 + 路由分发 + skill_id 硬路由）。"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.modules.chat.agent.orchestrator import AgentOrchestrator
from src.modules.chat.schemas import ChatRequest, ChatResponse, IntentResult
from src.modules.chat.core.intent.candidate import ExecutionPlan
from src.modules.chat.core.sentiment_service import EmotionResult, EmotionLevel, SentimentService


def _neutral_emotion() -> EmotionResult:
    return EmotionResult(
        level=EmotionLevel.NEUTRAL,
        confidence=0.5,
        escalate=False,
        is_emergency=False,
        keywords=[],
        source="rule",
    )


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
        intent.recognize = AsyncMock(
            return_value=IntentResult(
                plan=ExecutionPlan(mode="direct_tool", skill="query-order"), action="query-order"
            )
        )

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

        with patch.object(orchestrator, '_sentiment_service', mock_sentiment):
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

        with patch.object(orchestrator, '_sentiment_service', mock_sentiment):
            request = ChatRequest(message="你好", domain="ecommerce")
            response = await orchestrator.chat_with_agent(request)
            assert response.status != "escalated"


class TestSkillIdHardRouting:
    """A2A / MCP 传入 skill_id 时应跳过意图识别，把 action 直接钉死。

    意图识别是概率性的；对端既已在 Agent Card 里确认过能力，就不该再猜一次。
    """

    @pytest.fixture
    def orchestrator(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")

        llm = MagicMock()
        llm.chat_qwen = AsyncMock(return_value="mock response")

        embedding = MagicMock()
        embedding.embed_query = AsyncMock(return_value=[0.1] * 768)

        milvus = MagicMock()
        milvus.hybrid_search = AsyncMock(return_value=[])

        # 若硬路由生效，recognize 不应被调用；被调用则说明回退到了意图识别
        intent = MagicMock()
        intent.recognize = AsyncMock(
            return_value=IntentResult(
                plan=ExecutionPlan(mode="direct_tool", skill="check-shipping"),
                action="check-shipping",
            )
        )

        orch = AgentOrchestrator(
            llm_service=llm,
            embedding_service=embedding,
            milvus_service=milvus,
            intent_recognizer=intent,
            tool_service=MagicMock(),
        )
        return orch

    @staticmethod
    def _capture_route(captured: dict):
        """拦截 _route_intent：既能拿到 prepare 产出的 ctx，又能让流程正常收尾。"""

        async def _route(self, ctx):
            captured["intent_result"] = ctx.intent_result
            captured["steps"] = ctx.intent_steps
            return ChatResponse(message="ok", conversation_id="conv_test")

        return _route

    async def _run(self, orchestrator: AgentOrchestrator, request: ChatRequest) -> dict:
        mock_sentiment = MagicMock()
        mock_sentiment.detect = AsyncMock(return_value=_neutral_emotion())
        captured: dict = {}

        with patch.object(
            orchestrator, "_sentiment_service", mock_sentiment
        ), patch.object(
            AgentOrchestrator, "_route_intent", new=self._capture_route(captured)
        ):
            await orchestrator.chat_with_agent(request)

        return captured

    @pytest.mark.asyncio
    async def test_skill_id_bypasses_intent_recognition(self, orchestrator):
        """核心断言：skill_id 存在时 recognize 不被调用，action 被钉死。"""
        request = ChatRequest(
            message="帮我看看这个",
            domain="ecommerce",
            skill_id="query-order",
            context={"order_id": "WB202405270001"},
        )
        captured = await self._run(orchestrator, request)

        orchestrator._intent_recognizer.recognize.assert_not_called()
        assert captured["intent_result"].action == "query-order"
        assert captured["intent_result"].mode == "direct_tool"
        assert captured["intent_result"].params == {"order_id": "WB202405270001"}
        # direct_tool → 走确定性执行路径，不进 ReAct 自主规划
        assert captured["intent_result"].plan.skill == "query-order"
        assert captured["steps"][0]["step_name"] == "意图识别（skill_id 硬路由）"

    @pytest.mark.asyncio
    async def test_without_skill_id_falls_back_to_recognition(self, orchestrator):
        """未传 skill_id 时必须走常规意图识别 —— 零行为回归。"""
        request = ChatRequest(message="我的快递到哪了", domain="ecommerce")
        captured = await self._run(orchestrator, request)

        orchestrator._intent_recognizer.recognize.assert_called_once()
        assert captured["intent_result"].action == "check-shipping"
        assert captured["steps"][0]["step_name"] == "意图识别"
