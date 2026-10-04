from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import JSONB

from src.shared.database import Base


class User(Base):
    """轻量用户模型"""

    __tablename__ = "users"

    id = Column(String(64), primary_key=True, comment="user_id")
    user_type = Column(
        String(32), nullable=False, default="anonymous",
        comment="用户类型: registered / anonymous / test",
    )
    created_at = Column(DateTime, nullable=False, default=datetime.now, comment="首次出现时间")
    last_seen_at = Column(
        DateTime, nullable=False, default=datetime.now,
        onupdate=datetime.now, comment="最近活动时间",
    )
    user_metadata = Column(JSONB, nullable=False, default=dict, comment="扩展字段")


class UserProfile(Base):
    """用户画像模型（记忆架构 Phase 2 使用）"""

    __tablename__ = "user_profiles"

    user_id = Column(String(64), primary_key=True, comment="关联 users.id")
    preferences = Column(JSONB, nullable=False, default=dict, comment="用户偏好")
    vip_level = Column(String(32), nullable=False, default="normal", comment="VIP 等级")
    first_seen_at = Column(DateTime, nullable=False, default=datetime.now, comment="首次对话时间")
    last_seen_at = Column(DateTime, nullable=False, default=datetime.now, comment="最近对话时间")
    total_orders = Column(Integer, nullable=False, default=0, comment="历史订单数")
    total_complaints = Column(Integer, nullable=False, default=0, comment="历史投诉数")
    pending_issues = Column(JSONB, nullable=False, default=list, comment="未解决事项")
    created_at = Column(DateTime, nullable=False, default=datetime.now, comment="创建时间")
    updated_at = Column(
        DateTime, nullable=False, default=datetime.now,
        onupdate=datetime.now, comment="更新时间",
    )
