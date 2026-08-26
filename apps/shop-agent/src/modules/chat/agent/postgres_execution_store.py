"""PostgreSQL 执行状态存储。"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.chat.agent.models_execution import AgentEvent, AgentExecution
from src.shared.logger import APILogger
from src.shared.redact import redact_obj

logger = APILogger("postgres_execution_store")


class PostgresExecutionStore:
    """PostgreSQL 执行状态存储，负责 agent_executions 和 agent_events 的读写。"""

    def __init__(self, db: AsyncSession):
        self._db = db

    async def save_execution(
        self,
        thread_id: str,
        state: Dict[str, Any],
        current_node: str = "",
        *,
        status: str = "running",
        error_message: str | None = None,
    ) -> None:
        """保存/更新执行状态（upsert），同时追加事件。"""
        state_masked = redact_obj(state)
        payload = {
            "current_node": current_node,
            "status": status,
            "error_message": error_message,
        }

        await self._db.execute(
            text("""
                INSERT INTO agent_executions (thread_id, status, current_node, state_snapshot, context, version)
                VALUES (:thread_id, :status, :current_node, :state_snapshot, :context, 1)
                ON CONFLICT (thread_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    current_node = EXCLUDED.current_node,
                    state_snapshot = EXCLUDED.state_snapshot,
                    context = EXCLUDED.context,
                    version = agent_executions.version + 1,
                    updated_at = now()
            """),
            {
                "thread_id": thread_id,
                "status": status,
                "current_node": current_node,
                "state_snapshot": json.dumps(state_masked, ensure_ascii=False),
                "context": json.dumps({}, ensure_ascii=False),
            },
        )
        await self._db.flush()

        event_type = "state_updated"
        if status in ("completed", "failed"):
            event_type = "execution_ended"
        await self._append_event(
            thread_id=thread_id,
            event_type=event_type,
            payload=payload,
            node_name=current_node or None,
        )

    async def get_execution(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """获取执行状态。"""
        result = await self._db.execute(
            text("SELECT * FROM agent_executions WHERE thread_id = :tid"),
            {"tid": thread_id},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    async def append_event(
        self,
        thread_id: str,
        event_type: str,
        payload: Dict[str, Any],
        *,
        node_name: str | None = None,
        operator_id: str = "system",
    ) -> None:
        """追加事件到 agent_events。"""
        await self._append_event(
            thread_id=thread_id,
            event_type=event_type,
            payload=payload,
            node_name=node_name,
            operator_id=operator_id,
        )

    async def get_events(
        self,
        thread_id: str,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """获取事件流（用于重建状态）。"""
        sql = "SELECT * FROM agent_events WHERE thread_id = :tid"
        params: Dict[str, Any] = {"tid": thread_id}
        if event_type:
            sql += " AND event_type = :et"
            params["et"] = event_type
        sql += " ORDER BY event_id ASC LIMIT :lim"
        params["lim"] = limit

        result = await self._db.execute(text(sql), params)
        return [dict(row) for row in result.mappings().all()]

    async def _append_event(
        self,
        thread_id: str,
        event_type: str,
        payload: Dict[str, Any],
        node_name: str | None = None,
        operator_id: str = "system",
    ) -> None:
        event = AgentEvent(
            thread_id=thread_id,
            event_type=event_type,
            node_name=node_name,
            payload=redact_obj(payload),
            operator_id=operator_id,
        )
        self._db.add(event)
        await self._db.flush()
        logger.debug(
            "追加事件",
            thread_id=thread_id,
            event_type=event_type,
            node_name=node_name,
        )
