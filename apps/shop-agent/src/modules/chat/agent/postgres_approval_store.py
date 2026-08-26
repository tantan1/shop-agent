"""PostgreSQL 人在回路审批存储。"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.chat.agent.models_approval import HumanApproval
from src.modules.chat.agent.tool_commands import (
    ToolCommand,
    ToolContext,
    ToolResult,
    _COMMAND_REGISTRY,
)
from src.shared.logger import APILogger
from src.shared.redact import redact_dict

logger = APILogger("postgres_approval_store")


class PostgresApprovalStore:
    """PostgreSQL 审批存储，替代 Redis JSON 存储。"""

    def __init__(self, db: AsyncSession, *, tool_service: Any = None):
        self._db = db
        self._tool_service = tool_service

    async def create_approval(
        self,
        command: ToolCommand,
        ctx: ToolContext,
        result: ToolResult,
    ) -> str:
        """创建审批记录，返回 approval_id。"""
        approval_id = f"approval:{uuid.uuid4().hex[:8]}"
        masked_params = redact_dict(ctx.params) if ctx.params else {}

        approval = HumanApproval(
            approval_id=approval_id,
            command_name=command.command_name,
            action=ctx.action,
            params=dict(ctx.params) if ctx.params else {},
            params_masked=masked_params,
            conversation_id=ctx.conversation_id or "",
            domain=ctx.domain or "ecommerce",
            user_id=ctx.user_id or "",
            custom_metadata=dict(ctx.metadata) if ctx.metadata else {},
            status="pending_approval",
            message=result.message or "",
            undo_data=dict(result.undo_data) if result.undo_data else {},
        )
        self._db.add(approval)

        await self._db.execute(
            text("""
                INSERT INTO agent_events (thread_id, event_type, node_name, payload, operator_id)
                VALUES (:thread_id, 'approval_created', :node_name, :payload, 'system')
            """),
            {
                "thread_id": ctx.conversation_id or approval_id,
                "node_name": "human_approval",
                "payload": json.dumps({
                    "approval_id": approval_id,
                    "command_name": command.command_name,
                    "action": ctx.action,
                    "params_masked": masked_params,
                    "status": "pending_approval",
                }, ensure_ascii=False),
            },
        )
        await self._db.flush()

        logger.info(
            "审批记录创建",
            approval_id=approval_id,
            command=command.command_name,
            conversation_id=ctx.conversation_id,
        )
        return approval_id

    async def get_approval(self, approval_id: str) -> Optional[Dict[str, Any]]:
        """获取审批记录，重建 command/context 对象。"""
        result = await self._db.execute(
            text("SELECT * FROM human_approvals WHERE approval_id = :aid"),
            {"aid": approval_id},
        )
        row = result.mappings().first()
        if not row:
            return None

        data = dict(row)
        data["command"] = _COMMAND_REGISTRY.get(data.get("command_name", ""))
        data["context"] = ToolContext(
            action=data.get("action", ""),
            params=data.get("params", {}),
            conversation_id=data.get("conversation_id", ""),
            domain=data.get("domain", "ecommerce"),
            user_id=data.get("user_id", ""),
            metadata=data.get("custom_metadata", {}),
        )
        return data

    async def approve(
        self,
        approval_id: str,
        operator_id: str = "system",
        note: str = "",
    ) -> Dict[str, Any]:
        """审批通过。"""
        result = await self._db.execute(
            text("""
                UPDATE human_approvals
                SET status = 'approved',
                    resolved_at = now(),
                    resolved_by = :operator_id,
                    resolution_note = :note,
                    updated_at = now()
                WHERE approval_id = :aid AND status = 'pending_approval'
                RETURNING *
            """),
            {"aid": approval_id, "operator_id": operator_id, "note": note},
        )
        row = result.mappings().first()
        if not row:
            return {"status": "failed", "error": f"审批记录不存在或已处理: {approval_id}"}

        await self._db.execute(
            text("""
                INSERT INTO agent_events (thread_id, event_type, node_name, payload, operator_id)
                VALUES (:thread_id, 'approval_approved', 'human_approval', :payload, :operator_id)
            """),
            {
                "thread_id": row["conversation_id"],
                "payload": json.dumps({"approval_id": approval_id, "note": note}, ensure_ascii=False),
                "operator_id": operator_id,
            },
        )
        await self._db.flush()
        return dict(row)

    async def reject(
        self,
        approval_id: str,
        operator_id: str = "system",
        note: str = "",
    ) -> Dict[str, Any]:
        """审批拒绝。"""
        result = await self._db.execute(
            text("""
                UPDATE human_approvals
                SET status = 'rejected',
                    resolved_at = now(),
                    resolved_by = :operator_id,
                    resolution_note = :note,
                    updated_at = now()
                WHERE approval_id = :aid AND status = 'pending_approval'
                RETURNING *
            """),
            {"aid": approval_id, "operator_id": operator_id, "note": note},
        )
        row = result.mappings().first()
        if not row:
            return {"status": "failed", "error": f"审批记录不存在或已处理: {approval_id}"}

        await self._db.execute(
            text("""
                INSERT INTO agent_events (thread_id, event_type, node_name, payload, operator_id)
                VALUES (:thread_id, 'approval_rejected', 'human_approval', :payload, :operator_id)
            """),
            {
                "thread_id": row["conversation_id"],
                "payload": json.dumps({"approval_id": approval_id, "note": note}, ensure_ascii=False),
                "operator_id": operator_id,
            },
        )
        await self._db.flush()
        return dict(row)
