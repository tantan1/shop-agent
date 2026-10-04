import re
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.user.models import User, UserProfile


class UserService:
    """轻量用户服务"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def resolve_user_id(
        self,
        x_user_id: Optional[str] = None,
        cookie_user_id: Optional[str] = None,
    ) -> str:
        """解析 user_id：header > cookie > 生成匿名 ID"""
        user_id = x_user_id or cookie_user_id
        if not user_id:
            user_id = f"anon_{uuid.uuid4().hex[:8]}"

        # 最小验证：格式检查
        if not self._is_valid_user_id(user_id):
            raise ValueError("Invalid user_id format")

        # 记录用户活动（upsert）
        await self._touch_user(user_id)

        return user_id

    def _is_valid_user_id(self, user_id: str) -> bool:
        """user_id 格式校验：字母数字 + 下划线，长度 4-64"""
        return bool(re.match(r"^[a-zA-Z0-9_]{4,64}$", user_id))

    async def _touch_user(self, user_id: str) -> None:
        """更新用户最后活动时间，不存在则创建"""
        now = datetime.now()
        stmt = select(User).where(User.id == user_id)
        result = await self.db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            user = User(
                id=user_id,
                user_type="registered" if not user_id.startswith("anon_") else "anonymous",
                created_at=now,
                last_seen_at=now,
                user_metadata={},
            )
            self.db.add(user)
        else:
            user.last_seen_at = now

        await self.db.flush()

    async def get_or_create_profile(self, user_id: str) -> UserProfile:
        """获取或创建用户画像"""
        stmt = select(UserProfile).where(UserProfile.user_id == user_id)
        result = await self.db.execute(stmt)
        profile = result.scalar_one_or_none()

        if not profile:
            profile = UserProfile(user_id=user_id)
            self.db.add(profile)
            await self.db.flush()

        return profile

    async def update_profile(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """更新用户偏好"""
        profile = await self.get_or_create_profile(user_id)
        profile.preferences = {**profile.preferences, **preferences}
        profile.last_seen_at = datetime.now()
        await self.db.flush()
