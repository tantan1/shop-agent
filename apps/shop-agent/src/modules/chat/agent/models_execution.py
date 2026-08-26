"""Agent 执行状态模型。"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Index,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase

from src.shared.database import Base


class AgentExecution(Base):
    """Agent 执行状态表。"""

    __tablename__ = "agent_executions"

    thread_id = Column(String(200), primary_key=True, comment="LangGraph thread_id")
    status = Column(
        String(50), nullable=False, default="running", comment="running/suspended/completed/failed"
    )
    current_node = Column(String(100), nullable=True, comment="当前执行到的节点名")
    state_snapshot = Column(JSONB, nullable=False, comment="完整图状态（GraphState JSON）")
    context = Column(JSONB, nullable=False, default=dict, comment="执行上下文")
    version = Column(BigInteger, nullable=False, default=0, comment="乐观锁版本号")
    error_message = Column(Text, nullable=True, comment="失败时的错误信息")
    created_at = Column(DateTime, default=datetime.now, nullable=False, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False, comment="更新时间")
    completed_at = Column(DateTime, nullable=True, comment="执行完成/失败时间")


Index("idx_agent_executions_status", AgentExecution.status)
Index("idx_agent_executions_created_at", AgentExecution.created_at)


class AgentEvent(Base):
    """Agent 事件溯源表（按 created_at 时间分区，父表只存元数据）。"""

    __tablename__ = "agent_events"
    __table_args__ = {"postgresql_partition_by": "RANGE (created_at)"}

    event_id = Column(BigInteger, primary_key=True, comment="事件 ID（自增）")
    thread_id = Column(String(200), nullable=False, index=True, comment="关联 execution thread_id")
    event_type = Column(String(50), nullable=False, comment="事件类型")
    node_name = Column(String(100), nullable=True, comment="节点名（node 事件专用）")
    payload = Column(JSONB, nullable=False, comment="事件详情")
    operator_id = Column(String(100), nullable=True, comment="操作人")
    source = Column(String(50), nullable=False, default="shop-agent", comment="事件来源")
    created_at = Column(DateTime, default=datetime.now, nullable=False, comment="事件时间")


Index("idx_agent_events_type", AgentEvent.event_type)
Index("idx_agent_events_created_at", AgentEvent.created_at)
