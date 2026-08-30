"""命令模式：工具调用 + 撤销 + HITL 审批。

将每个业务工具封装为可执行/可撤销的命令对象，通过 ApprovalGate 统一处理人在回路流程。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.shared.logger import APILogger

logger = APILogger("tool_command")


# ── 数据结构 ──────────────────────────────────────────────────────────


@dataclass
class ToolContext:
    """工具执行上下文。"""
    action: str
    params: Dict[str, Any]
    conversation_id: str = ""
    domain: str = ""
    user_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """工具执行结果。"""
    status: str  # success / failed / pending_approval / waiting_for_confirmation
    data: Any = None
    message: str = ""
    error: Optional[str] = None
    approval_id: Optional[str] = None
    undo_data: Optional[Dict[str, Any]] = None  # 撤销所需的数据快照


# ── 抽象基类 ─────────────────────────────────────────────────────────


class ToolCommand(ABC):
    """工具命令抽象基类（命令模式）。"""

    command_name: str = ""

    @abstractmethod
    async def execute(self, ctx: ToolContext) -> ToolResult:
        """执行工具调用（可能触发副作用，如退款）。"""
        ...

    @abstractmethod
    async def undo(self, ctx: ToolContext) -> ToolResult:
        """撤销工具调用（如取消退款）。"""
        ...

    def requires_approval(self) -> bool:
        """是否需要人工审批（默认不需要）。"""
        return False

    def description(self) -> str:
        """工具描述（用于日志/审计）。"""
        return f"{self.command_name} tool"


# ── 审批门 ───────────────────────────────────────────────────────────


class ApprovalGate:
    """人在回路审批门。

    工作流程：
    1. 命令执行前检查是否需要审批
    2. 需要审批 → 执行命令（产生 pending 状态）→ 存储审批记录 → 返回 waiting
    3. 人工审批通过 → 确认执行（通常是幂等写库）
    4. 人工审批拒绝 → 撤销执行（undo）
    """

    def __init__(self, approval_store: Optional[Any] = None):
        self._store = approval_store

    async def execute_with_approval(self, command: ToolCommand, ctx: ToolContext) -> ToolResult:
        """执行命令，如需审批则返回 waiting 状态。"""
        if not command.requires_approval():
            return await command.execute(ctx)

        result = await command.execute(ctx)
        if result.status != "pending_approval":
            return result

        if self._store is not None:
            approval_id = await self._store.create_approval(command, ctx, result)
        else:
            approval_id = await self._store_approval(command, ctx, result)
        logger.info(
            "Tool 等待审批",
            command=command.command_name,
            approval_id=approval_id,
            conversation_id=ctx.conversation_id,
        )
        return ToolResult(
            status="waiting_for_confirmation",
            message=result.message,
            approval_id=approval_id,
            undo_data=result.undo_data,
        )

    async def create_pending_approval(
        self, command: ToolCommand, ctx: ToolContext, message: str = ""
    ) -> str:
        """仅创建待审批记录（不执行命令副作用，用于「先拦截后执行」模型）。

        MCP 远程写操作采用此模型：dispatch 时只生成审批单 + approval_id，
        人工 approve 时（ApprovalGate.approve → command.execute）才真实调用远程服务，
        避免未授权副作用被提前触发。
        """
        if self._store is not None:
            approval_id = await self._store.create_approval(
                command, ctx, ToolResult(status="pending_approval", message=message)
            )
        else:
            approval_id = await self._store_approval(
                command, ctx, ToolResult(status="pending_approval", message=message)
            )
        logger.info(
            "MCP Tool 待审批（未执行）",
            command=command.command_name,
            approval_id=approval_id,
            conversation_id=ctx.conversation_id,
        )
        return approval_id

    async def approve(self, approval_id: str) -> ToolResult:
        """审批通过：确认执行。

        存储后端与 execute_with_approval 保持对称：显式传入 store 则用 store，
        否则走内置降级路径（Redis → 内存），与 _store_approval 对应。
        """
        if self._store is not None:
            approval = await self._store.get_approval(approval_id)
        else:
            approval = await self._get_approval(approval_id)
        if not approval:
            return ToolResult(status="failed", error=f"审批记录不存在: {approval_id}")

        command: ToolCommand = approval.get("command")
        ctx: ToolContext = approval["context"]
        if command is None:
            return ToolResult(status="failed", error=f"审批命令对象缺失: {approval_id}")
        result = await command.execute(ctx)
        logger.info(
            "Tool 审批通过",
            command=command.command_name,
            approval_id=approval_id,
        )
        return result

    async def reject(self, approval_id: str) -> ToolResult:
        """审批拒绝：撤销执行。

        与 approve 一致：store 未显式配置时走内置降级路径（Redis → 内存）。
        """
        if self._store is not None:
            approval = await self._store.get_approval(approval_id)
        else:
            approval = await self._get_approval(approval_id)
        if not approval:
            return ToolResult(status="failed", error=f"审批记录不存在: {approval_id}")

        command: ToolCommand = approval.get("command")
        ctx: ToolContext = approval["context"]
        if command is None:
            return ToolResult(status="failed", error=f"审批命令对象缺失: {approval_id}")
        result = await command.undo(ctx)
        logger.info(
            "Tool 审批拒绝",
            command=command.command_name,
            approval_id=approval_id,
        )
        return result

    async def _store_approval(self, command: ToolCommand, ctx: ToolContext, result: ToolResult) -> str:
        """存储审批记录（优先 Redis，降级内存）。"""
        import uuid
        approval_id = f"approval:{uuid.uuid4().hex[:8]}"
        payload = {
            "command_name": command.command_name,
            "context": {
                "action": ctx.action,
                "params": ctx.params,
                "conversation_id": ctx.conversation_id,
                "domain": ctx.domain,
                "user_id": ctx.user_id,
                "metadata": ctx.metadata,
            },
            "result": {
                "status": result.status,
                "message": result.message,
                "undo_data": result.undo_data,
            },
            "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        }

        # 尝试 Redis
        try:
            from src.modules.chat.core.redis_cache_service import get_redis_cache_service
            svc = get_redis_cache_service()
            if svc and svc.is_available:
                svc.set_json(approval_id, payload, ex=3600)
                return approval_id
        except Exception as e:
            logger.warning(f"审批记录写入 Redis 失败，降级内存: {str(e)[:100]}")

        # 内存降级：存储 command 对象引用（仅内存模式支持）
        payload["_command_ref"] = command
        _APPROVAL_MEM[approval_id] = payload
        return approval_id

    async def _get_approval(self, approval_id: str) -> Optional[dict]:
        """获取审批记录。"""
        try:
            from src.modules.chat.core.redis_cache_service import get_redis_cache_service
            svc = get_redis_cache_service()
            if svc and svc.is_available:
                data = svc.get_json(approval_id)
                if data is not None:
                    # Redis 中只有 command_name，需要重建 command 对象
                    if "command" not in data:
                        cmd_name = data.get("command_name")
                        data["command"] = _COMMAND_REGISTRY.get(cmd_name) if cmd_name else None
                    # 重建 ToolContext（Redis 中存储的是 dict）
                    ctx_data = data.get("context", {})
                    if isinstance(ctx_data, dict):
                        data["context"] = ToolContext(
                            action=ctx_data.get("action", ""),
                            params=ctx_data.get("params", {}),
                            conversation_id=ctx_data.get("conversation_id", ""),
                            domain=ctx_data.get("domain", ""),
                            user_id=ctx_data.get("user_id", ""),
                            metadata=ctx_data.get("metadata", {}),
                        )
                    return data
        except Exception:
            pass
        # 内存降级：与 Redis 分支一致地重建 command / context 对象
        mem = _APPROVAL_MEM.get(approval_id)
        if mem is not None:
            if "command" not in mem and "_command_ref" in mem:
                mem["command"] = mem["_command_ref"]
            ctx_data = mem.get("context")
            if isinstance(ctx_data, dict):
                mem["context"] = ToolContext(
                    action=ctx_data.get("action", ""),
                    params=ctx_data.get("params", {}),
                    conversation_id=ctx_data.get("conversation_id", ""),
                    domain=ctx_data.get("domain", ""),
                    user_id=ctx_data.get("user_id", ""),
                    metadata=ctx_data.get("metadata", {}),
                )
        return mem


# ── 具体命令实现 ─────────────────────────────────────────────────────


class RefundCommand(ToolCommand):
    """退款命令（需审批）。"""

    command_name = "request-return"

    def __init__(self, order_service_url: str = "", timeout: int = 5):
        self._order_service_url = order_service_url
        self._timeout = timeout

    def requires_approval(self) -> bool:
        return True

    async def execute(self, ctx: ToolContext) -> ToolResult:
        """记录退款确认请求（真实写库，待人工审批）。"""
        order_id = ctx.params.get("order_id", "未指定")
        reason = ctx.params.get("reason", "未说明")
        refund_amount = ctx.params.get("refund_amount", 0.0)

        if not self._order_service_url:
            logger.warning("订单服务未配置，跳过退款确认记录")
            return ToolResult(
                status="pending_approval",
                message=f"退款申请已提交（订单号: {order_id}），等待人工审批。",
                undo_data={"order_id": order_id, "reason": reason},
            )

        url = f"{self._order_service_url.rstrip('/')}/api/refunds/confirm"
        try:
            import httpx
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    url,
                    json={
                        "order_id": order_id,
                        "reason": reason,
                        "refund_amount": refund_amount,
                    },
                )
                resp.raise_for_status()
        except Exception as e:
            logger.error(f"记录退款确认失败: {e}")
            return ToolResult(
                status="failed",
                error=str(e),
                undo_data={"order_id": order_id, "reason": reason},
            )

        return ToolResult(
            status="pending_approval",
            message=f"退款申请已提交（订单号: {order_id}，原因: {reason}），等待人工审批。",
            undo_data={"order_id": order_id, "reason": reason},
        )

    async def undo(self, ctx: ToolContext) -> ToolResult:
        """撤销退款申请。"""
        order_id = ctx.params.get("order_id", "未指定")
        logger.info(f"撤销退款申请: order_id={order_id}")
        return ToolResult(
            status="success",
            message=f"退款申请（订单号: {order_id}）已取消。",
        )


class QueryOrderCommand(ToolCommand):
    """查询订单命令（无需审批）。"""

    command_name = "query-order"

    async def execute(self, ctx: ToolContext) -> ToolResult:
        from src.modules.chat.core.tool_registry import ToolService
        result_str = await ToolService._tool_query_order(ctx.params)
        return ToolResult(status="success", data=result_str, message=result_str)

    async def undo(self, ctx: ToolContext) -> ToolResult:
        return ToolResult(status="success", message="查询操作无需撤销。")


class CheckShippingCommand(ToolCommand):
    """查询物流命令（无需审批）。"""

    command_name = "check-shipping"

    async def execute(self, ctx: ToolContext) -> ToolResult:
        from src.modules.chat.core.tool_registry import ToolService
        result_str = await ToolService._tool_check_shipping(ctx.params)
        return ToolResult(status="success", data=result_str, message=result_str)

    async def undo(self, ctx: ToolContext) -> ToolResult:
        return ToolResult(status="success", message="查询操作无需撤销。")


class CheckBalanceCommand(ToolCommand):
    """查询余额命令（无需审批）。"""

    command_name = "check-balance"

    async def execute(self, ctx: ToolContext) -> ToolResult:
        from src.modules.chat.core.tool_registry import ToolService
        result_str = await ToolService._tool_check_balance(ctx.params)
        return ToolResult(status="success", data=result_str, message=result_str)

    async def undo(self, ctx: ToolContext) -> ToolResult:
        return ToolResult(status="success", message="查询操作无需撤销。")


class CouponInquiryCommand(ToolCommand):
    """查询优惠券命令（无需审批）。"""

    command_name = "coupon-inquiry"

    async def execute(self, ctx: ToolContext) -> ToolResult:
        from src.modules.chat.core.tool_registry import ToolService
        result_str = await ToolService._tool_coupon_inquiry(ctx.params)
        return ToolResult(status="success", data=result_str, message=result_str)

    async def undo(self, ctx: ToolContext) -> ToolResult:
        return ToolResult(status="success", message="查询操作无需撤销。")


# 命令注册表：用于从名称重建命令对象（需在所有命令类定义之后）
_COMMAND_REGISTRY: Dict[str, ToolCommand] = {
    "query-order": QueryOrderCommand(),
    "check-shipping": CheckShippingCommand(),
    "request-return": RefundCommand(),
    "check-balance": CheckBalanceCommand(),
    "coupon-inquiry": CouponInquiryCommand(),
}

# 内存审批存储（Redis 不可用时的降级）
_APPROVAL_MEM: Dict[str, dict] = {}
