"""Agent 人在回路审批模型。"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Index,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase

from src.shared.database import Base


class HumanApproval(Base):
    """人在回路审批记录表。"""

    __tablename__ = "human_approvals"

    approval_id = Column(String(100), primary_key=True, comment="approval:{uuid}")
    command_name = Column(String(100), nullable=False, comment="命令名称")
    action = Column(String(100), nullable=False, comment="对应 ToolContext.action")
    params = Column(JSONB, nullable=False, comment="工具参数（脱敏前，原始数据）")
    params_masked = Column(JSONB, nullable=True, comment="脱敏后参数（用于展示）")
    conversation_id = Column(String(200), nullable=False, index=True, comment="会话 ID")
    domain = Column(String(50), nullable=False, default="ecommerce", comment="领域")
    user_id = Column(String(100), nullable=True, comment="用户 ID")
    custom_metadata = Column(JSONB, nullable=False, default=dict, comment="扩展字段")
    status = Column(
        String(50), nullable=False, default="pending_approval", comment="审批状态"
    )
    message = Column(Text, nullable=True, comment="给用户看的提示语")
    undo_data = Column(JSONB, nullable=True, comment="撤销所需数据快照")
    graph_state_snapshot = Column(JSONB, nullable=True, comment="interrupt 时的图状态快照")
    created_at = Column(DateTime, default=datetime.now, nullable=False, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False, comment="更新时间")
    resolved_at = Column(DateTime, nullable=True, comment="审批解决时间")
    resolved_by = Column(String(100), nullable=True, comment="审批人")
    resolution_note = Column(Text, nullable=True, comment="审批备注")


Index("idx_human_approvals_conversation", HumanApproval.conversation_id)
Index("idx_human_approvals_status", HumanApproval.status)
Index("idx_human_approvals_created_at", HumanApproval.created_at)
