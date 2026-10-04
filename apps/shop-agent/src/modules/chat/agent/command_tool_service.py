"""命令式工具审批服务（适配层）。

薄封装 ``tool_commands.ApprovalGate``，向调用方（ReActAgent / resume_execution）
暴露 ``approve(approval_id)`` / ``reject(approval_id)`` 接口，返回 ``ToolResult``。

真实审批存储与执行由 ``ApprovalGate`` 负责（Redis 优先，内存降级），
本层不重复实现审批逻辑，仅做对象聚合与类型对齐。
"""

from __future__ import annotations

from typing import Optional

from src.modules.chat.agent.tool_commands import ApprovalGate, ToolResult, _APPROVAL_MEM
from src.shared.logger import APILogger

logger = APILogger("command_tool_service")


class CommandToolService:
    """审批服务：聚合 ApprovalGate，提供 approve/reject 入口（命令模式）。"""

    def __init__(self, *, tool_service=None, approval_store=None):
        # tool_service 预留（后续如需在审批执行时回调业务工具可接入）
        self._tool_service = tool_service
        self._gate = ApprovalGate(approval_store=approval_store)

    async def dispatch(self, action: str, params: dict, conversation_id: str = "", domain: str = "") -> str:
        """执行工具调用（委托给 tool_service）。"""
        if self._tool_service is None:
            raise RuntimeError("CommandToolService 未配置 tool_service")
        return await self._tool_service.dispatch(action, params)

    async def approve(self, approval_id: str) -> ToolResult:
        """审批通过：确认执行（委托 ApprovalGate）。"""
        return await self._gate.approve(approval_id)

    async def reject(self, approval_id: str) -> ToolResult:
        """审批拒绝：撤销执行（委托 ApprovalGate）。"""
        return await self._gate.reject(approval_id)

    @property
    def has_pending_approval(self) -> bool:
        return bool(_APPROVAL_MEM)

    def pop_pending_approval(self) -> tuple[str, str, dict]:
        if not _APPROVAL_MEM:
            raise ValueError("no pending approval")
        approval_id, payload = next(iter(_APPROVAL_MEM.items()))
        del _APPROVAL_MEM[approval_id]
        command_name = payload.get("command_name", "")
        ctx = payload.get("context", {})
        params = ctx.get("params", {})
        return command_name, approval_id, params
