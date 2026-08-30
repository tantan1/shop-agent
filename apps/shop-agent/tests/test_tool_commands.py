"""命令模式工具服务单测（ToolCommand + ApprovalGate）。"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.modules.chat.agent.tool_commands import (
    ApprovalGate,
    CheckBalanceCommand,
    CheckShippingCommand,
    CouponInquiryCommand,
    QueryOrderCommand,
    RefundCommand,
    ToolContext,
    ToolResult,
)
from src.modules.chat.agent.command_tool_service import CommandToolService


class TestToolCommands:
    """具体命令的 execute/undo。"""

    @pytest.mark.asyncio
    async def test_query_order_command(self):
        ctx = ToolContext(action="query-order", params={"order_id": "TEST123"})
        cmd = QueryOrderCommand()
        result = await cmd.execute(ctx)
        assert result.status == "success"
        assert "TEST123" in result.message

    @pytest.mark.asyncio
    async def test_refund_command_requires_approval(self):
        cmd = RefundCommand()
        assert cmd.requires_approval() is True

    @pytest.mark.asyncio
    async def test_query_order_undo(self):
        ctx = ToolContext(action="query-order", params={})
        cmd = QueryOrderCommand()
        result = await cmd.undo(ctx)
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_refund_command_execute(self):
        cmd = RefundCommand(order_service_url="")
        ctx = ToolContext(action="request-return", params={"order_id": "123", "reason": "test"})
        result = await cmd.execute(ctx)
        assert result.status == "pending_approval"
        assert result.undo_data is not None


class TestApprovalGate:
    """审批门：execute_with_approval / approve / reject。"""

    @pytest.fixture
    def gate(self):
        return ApprovalGate()

    @pytest.mark.asyncio
    async def test_non_approval_command_passes_through(self, gate):
        cmd = QueryOrderCommand()
        ctx = ToolContext(action="query-order", params={})
        result = await gate.execute_with_approval(cmd, ctx)
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_approval_command_returns_waiting(self, gate):
        cmd = RefundCommand(order_service_url="")
        ctx = ToolContext(action="request-return", params={"order_id": "123"})
        result = await gate.execute_with_approval(cmd, ctx)
        assert result.status == "waiting_for_confirmation"
        assert result.approval_id is not None

    @pytest.mark.asyncio
    async def test_approve_executes_command(self, gate):
        cmd = RefundCommand(order_service_url="")
        ctx = ToolContext(action="request-return", params={"order_id": "123"})
        result = await gate.execute_with_approval(cmd, ctx)
        approval_id = result.approval_id

        # 审批通过
        approved = await gate.approve(approval_id)
        assert approved.status == "pending_approval"

    @pytest.mark.asyncio
    async def test_reject_undoes_command(self, gate):
        cmd = RefundCommand(order_service_url="")
        ctx = ToolContext(action="request-return", params={"order_id": "123"})
        result = await gate.execute_with_approval(cmd, ctx)
        approval_id = result.approval_id

        # 审批拒绝
        rejected = await gate.reject(approval_id)
        assert rejected.status == "success"


class TestCommandToolService:
    """CommandToolService 集成测试。

    说明：CommandToolService 是「审批聚合入口」，只暴露 approve(approval_id) /
    reject(approval_id)，**不提供 dispatch()**（派发由 ToolService 负责，审批通过
    ApprovalGate 触发）。旧测试按 dispatch()/has_pending_approval 编写，与实现不符，
    现改为验证真实 API 语义。
    """

    @pytest.fixture
    def service(self):
        return CommandToolService()

    @pytest.mark.asyncio
    async def test_reject_unknown_approval_returns_failed(self, service):
        """审批不存在的记录 → failed（不抛异常）。"""
        result = await service.reject("approval:does-not-exist")
        assert result.status == "failed"
        assert "不存在" in (result.error or "")

    @pytest.mark.asyncio
    async def test_approve_unknown_approval_returns_failed(self, service):
        """审批不存在的记录 → failed（不抛异常）。"""
        result = await service.approve("approval:does-not-exist")
        assert result.status == "failed"
        assert "不存在" in (result.error or "")

    @pytest.mark.asyncio
    async def test_approve_delegates_to_gate(self, service):
        """approve 委托给内部 ApprovalGate —— 用 mock 验证委派关系。"""
        expected = ToolResult(status="pending_approval", message="ok")
        service._gate.approve = AsyncMock(return_value=expected)

        result = await service.approve("approval:abc")

        assert result is expected
        service._gate.approve.assert_awaited_once_with("approval:abc")

    @pytest.mark.asyncio
    async def test_reject_delegates_to_gate(self, service):
        """reject 委托给内部 ApprovalGate。"""
        expected = ToolResult(status="success", message="undone")
        service._gate.reject = AsyncMock(return_value=expected)

        result = await service.reject("approval:abc")

        assert result is expected
        service._gate.reject.assert_awaited_once_with("approval:abc")
