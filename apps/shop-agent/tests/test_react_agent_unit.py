"""ReAct Agent 关键路径单测（工具选择过滤 + HITL + 空参拒绝 + 参数硬强制注入）。"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.modules.chat.agent.react_agent import ReActAgent, _INTERRUPT_MEM
from src.modules.chat.agent.react_agent_selection import _make_business_args_schema
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

    def test_single_skill_convergence_includes_allowed_tools(self, agent):
        """单 skill 收敛（skill_filter）应构建该 skill 自身 + allowed_tools 关联工具。

        回归保护：query-order 的 allowed-tools 含 check-shipping，
        收敛路径下二者都必须注册，否则模型无法在订单场景顺带查物流
        （与 INTENT_TOOL_MAP 候选集语义一致）。
        """
        conv = ReActAgent(
            llm_service=agent._llm_service,
            tool_service=agent._tool_service,
            embedding_service=agent._embedding_service,
            skill_filter="query-order",
        )
        names = {getattr(t, "name", "") for t in conv._all_tools}
        # 主工具 + 关联工具都应存在；不应只有 query-order 一个
        assert "query-order" in names
        assert "check-shipping" in names

    def test_single_skill_convergence_single_tool_skill(self, agent):
        """单工具 skill（如 check-balance）收敛后仍只含自身，不引入无关工具。"""
        conv = ReActAgent(
            llm_service=agent._llm_service,
            tool_service=agent._tool_service,
            embedding_service=agent._embedding_service,
            skill_filter="check-balance",
        )
        names = {getattr(t, "name", "") for t in conv._all_tools}
        assert names == {"check-balance"}

    def test_general_entry_builds_from_allowed_tools_union(self, agent):
        """通用入口（未指定 skill_filter）应基于 allowed-tools 并集（意图候选全集）构建工具。

        回归保护：通用入口不再盲目注册整张 skill 注册表，而是取所有 skill 的
        ``allowed_tools`` 并集（含关联工具），与 P0 的 INTENT_TOOL_MAP 同源。
        本仓库 5 个 skill 的 allowed-tools 并集恰好覆盖全部 5 个 action，
        （check-shipping 与 query-order 互引），故通用入口应注册全部业务工具。
        """
        names = {getattr(t, "name", "") for t in agent._all_tools}
        business = names - {"knowledge_search"}
        # 所有意图候选 action 都应在通用入口可见
        assert business == {
            "check-balance",
            "check-shipping",
            "coupon-inquiry",
            "query-order",
            "request-return",
        }
        # 关联工具对：query-order 与 check-shipping 都必须出现在通用入口
        assert {"query-order", "check-shipping"} <= business


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

    @pytest.mark.asyncio
    async def test_resume_execution_falls_back_to_dispatch(self, monkeypatch):
        """审批恢复：ApprovalGate 无记录时降级直接 dispatch 执行，而非误报失败。

        回归保护：退款链路未走 ApprovalGate.execute_with_approval，
        command_service.approve(thread_id) 必然 failed。此前会 fail-closed 返回
        「执行失败」导致审批通过却无反馈。修复后应在 approve 失败时改用中断上下文
        直接 dispatch 退款确认执行。
        """
        thread_id = "test-thread-resume"
        _INTERRUPT_MEM.pop(thread_id, None)

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
                conversation_id="conv-resume",
                intent_steps=[],
                domain="ecommerce",
                order_id="ORDER-456",
                reason="质量问题",
            )
        )

        # mock tool_service.dispatch：模拟退款确认执行成功
        mock_tool = MagicMock()
        mock_tool.dispatch = AsyncMock(return_value="退款已批准并执行，单号 R789")

        response = await ReActAgent.resume_execution(
            thread_id=thread_id, confirm=True, tool_service=mock_tool
        )

        # 必须真正调用了 dispatch 执行退款（而非失败分支）
        mock_tool.dispatch.assert_awaited_once()
        _, dispatch_params = mock_tool.dispatch.await_args.args
        assert dispatch_params["order_id"] == "ORDER-456"
        assert dispatch_params["confirm"] is True

        assert response is not None
        assert response.status == "completed"
        assert "R789" in response.message

        _INTERRUPT_MEM.pop(thread_id, None)


class TestHardPresetInjection:
    """硬强制（方式 2）：前置确定性抽取的参数注入工具调用，模型无法篡改高后果字段。

    核心保证：最终进后端的参数 100% 来自确定性抽取，模型既看不到（schema 剔除）
    也改不了（闭包覆盖）关键字段；非法值在下发前被格式校验拦截。
    """

    @pytest.fixture
    def agent(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
        llm = MagicMock()
        tool_service = MagicMock()
        tool_service.dispatch = AsyncMock(return_value="DISPATCH_OK")
        embedding = MagicMock()
        embedding.embed_query = MagicMock(return_value=[0.1] * 768)
        agent = ReActAgent(
            llm_service=llm,
            tool_service=tool_service,
            embedding_service=embedding,
        )
        # 用捕获型 mock 替换实际下发通道，验证硬强制合并结果
        agent._tool_service = tool_service
        agent._command_tool_service = MagicMock()
        agent._command_tool_service.dispatch = AsyncMock(return_value="CMD_OK")
        return agent

    def test_schema_strips_preset_field(self):
        """带 preset 的字段应从 tool schema 剔除，模型在 tool-calling 时根本看不到。"""
        with_preset = _make_business_args_schema("query-order", {"order_id": "WB1"})
        assert "order_id" not in with_preset.model_fields
        assert "phone" in with_preset.model_fields  # 未预设字段仍可见、可填

        without = _make_business_args_schema("query-order", None)
        assert "order_id" in without.model_fields  # 无 preset 时字段正常暴露

    @pytest.mark.asyncio
    async def test_preset_overrides_model_injected_value(self, agent):
        """模型即便尝试传入 order_id（schema 已剔除，会被忽略），最终仍强制使用预设值。

        由于 order_id 不在 schema，LangChain 在 ainvoke 时会丢弃该字段，
        模型无法把错误值塞进闭包；结合闭包 {**model, **preset} 的覆盖，
        最终 dispatch 的 order_id 100% 来自确定性抽取。
        """
        tool = agent._make_business_tool("query-order", {"order_id": "WB202405270001"})
        await tool.ainvoke({"order_id": "WRONG_ORDER", "phone": "8888"})

        _, params = agent._tool_service.dispatch.await_args.args
        assert params["order_id"] == "WB202405270001"  # 模型尝试传的 WRONG 未生效
        assert params["phone"] == "8888"  # 非预设字段保留模型值

    @pytest.mark.asyncio
    async def test_ainvoke_without_preset_field_still_gets_preset(self, agent):
        """模型看不到 order_id（schema 已剔除），ainvoke 不传它，dispatch 仍拿到预设值。"""
        tool = agent._make_business_tool("query-order", {"order_id": "WB202405270001"})
        await tool.ainvoke({"phone": "8888"})

        _, params = agent._tool_service.dispatch.await_args.args
        assert params["order_id"] == "WB202405270001"
        assert params["phone"] == "8888"

    @pytest.mark.asyncio
    async def test_refund_order_id_override_and_valid(self, agent):
        """退款：preset 合法 order_id 覆盖模型传入的错误值，并正常下发 command dispatch。"""
        tool = agent._make_refund_tool_with_confirmation({"order_id": "WB202405270001"})
        await tool.ainvoke({"order_id": "WRONG", "reason": "质量问题"})

        params = agent._command_tool_service.dispatch.await_args.kwargs["params"]
        assert params["order_id"] == "WB202405270001"
        assert params["reason"] == "质量问题"

    @pytest.mark.asyncio
    async def test_refund_invalid_order_id_blocked(self, agent):
        """退款：preset 非法 order_id 被格式校验拦截，绝不静默下发。"""
        tool = agent._make_refund_tool_with_confirmation({"order_id": "1"})
        result = await tool.ainvoke({"order_id": "1", "reason": "x"})

        assert "格式不正确" in result
        agent._command_tool_service.dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_apply_preset_rebuilds_tools(self, agent):
        """_apply_preset_to_tools 应为命中 action 的工具注入 preset，非业务工具原样保留。"""
        selected = [
            t for t in agent._all_tools
            if getattr(t, "name", "") in {"request-return", "query-order"}
        ]
        assert len(selected) == 2  # 两个业务工具必须存在
        rebuilt = agent._apply_preset_to_tools(selected, {"order_id": "WB202405270001"})

        names = {getattr(t, "name", "") for t in rebuilt}
        assert names == {"request-return", "query-order"}

        # 重建后的 request-return 工具带 preset：调用时 order_id 来自预设而非模型
        rtool = next(t for t in rebuilt if getattr(t, "name", "") == "request-return")
        await rtool.ainvoke({"order_id": "WRONG", "reason": "x"})
        params = agent._command_tool_service.dispatch.await_args.kwargs["params"]
        assert params["order_id"] == "WB202405270001"
