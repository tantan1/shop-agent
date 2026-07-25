"""命令模式工具服务：在现有 ToolService 之上封装 Command 层。

设计原则：
- 不破坏现有 ToolService 接口（向后兼容）
- 通过 ApprovalGate 统一处理 HITL 审批流
- 支持 undo 操作（退款撤销）
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.modules.chat.agent.tool_commands import (
    ApprovalGate,
    CheckBalanceCommand,
    CheckShippingCommand,
    CouponInquiryCommand,
    QueryOrderCommand,
    RefundCommand,
    ToolCommand,
    ToolContext,
    ToolResult,
)
from src.modules.chat.core.tool_registry import ToolService
from src.shared.logger import APILogger

logger = APILogger("command_tool_service")


class CommandToolService:
    """命令模式工具服务（装饰器模式包装现有 ToolService）。"""

    def __init__(self, tool_service: Optional[ToolService] = None, approval_gate: Optional[ApprovalGate] = None):
        self._tool_service = tool_service or ToolService()
        self._approval_gate = approval_gate or ApprovalGate()
        self._commands: Dict[str, ToolCommand] = {}
        self._setup_commands()

    def _setup_commands(self):
        """初始化命令映射。"""
        self._commands = {
            "query-order": QueryOrderCommand(),
            "check-shipping": CheckShippingCommand(),
            "request-return": RefundCommand(),
            "check-balance": CheckBalanceCommand(),
            "coupon-inquiry": CouponInquiryCommand(),
        }

    def get_command(self, action: str) -> Optional[ToolCommand]:
        """获取对应 action 的命令对象。"""
        return self._commands.get(action)

    async def dispatch(self, action: str, params: Optional[Dict[str, Any]] = None, **context) -> str:
        """分发工具调用（通过命令模式）。

        Returns:
            工具执行结果字符串
        """
        params = params or {}
        command = self.get_command(action)
        if command is None:
            logger.warning(f"未找到命令: {action}，回退到原始 ToolService")
            return await self._tool_service.dispatch(action, params)

        ctx = ToolContext(
            action=action,
            params=params,
            conversation_id=context.get("conversation_id", ""),
            domain=context.get("domain", ""),
            user_id=context.get("user_id", ""),
            metadata=context.get("metadata", {}),
        )

        result = await self._approval_gate.execute_with_approval(command, ctx)

        if result.status == "waiting_for_confirmation":
            # 存储 pending approval 信息，供上层查询
            self._pending_approval = (action, result.approval_id, ctx.params)
            return result.message
        if result.error:
            return f"工具执行失败: {result.error}"
        return result.message or str(result.data)

    @property
    def has_pending_approval(self) -> bool:
        """是否有待审批的工具调用。"""
        return hasattr(self, '_pending_approval') and self._pending_approval is not None

    def pop_pending_approval(self) -> Optional[tuple]:
        """取出并清除待审批信息。"""
        if hasattr(self, '_pending_approval'):
            approval = self._pending_approval
            self._pending_approval = None
            return approval
        return None

    async def approve(self, approval_id: str) -> ToolResult:
        """审批通过。"""
        return await self._approval_gate.approve(approval_id)

    async def reject(self, approval_id: str) -> ToolResult:
        """审批拒绝。"""
        return await self._approval_gate.reject(approval_id)

    @property
    def tool_service(self) -> ToolService:
        """底层原始 ToolService（兼容旧代码）。"""
        return self._tool_service
