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
    """CommandToolService 集成测试。"""

    @pytest.fixture
    def service(self):
        return CommandToolService()

    @pytest.mark.asyncio
    async def test_dispatch_non_approval_tool(self, service):
        result = await service.dispatch("query-order", {"order_id": "123"})
        assert "123" in result

    @pytest.mark.asyncio
    async def test_dispatch_approval_tool_returns_waiting(self, service):
        result = await service.dispatch("request-return", {"order_id": "123", "reason": "test"})
        assert "等待人工审批" in result or "审批" in result

    @pytest.mark.asyncio
    async def test_has_pending_approval(self, service):
        await service.dispatch("request-return", {"order_id": "123"})
        assert service.has_pending_approval is True
        service.pop_pending_approval()
        assert service.has_pending_approval is False
