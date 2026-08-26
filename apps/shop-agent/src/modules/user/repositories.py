from datetime import datetime
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.user.models import User, UserProfile


class UserRepository:
    """用户数据访问"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, user_id: str) -> Optional[User]:
        stmt = select(User).where(User.id == user_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, user_id: str, user_type: str = "anonymous", metadata: dict = None) -> User:
        user = User(
            id=user_id,
            user_type=user_type,
            metadata=metadata or {},
        )
        self.db.add(user)
        await self.db.flush()
        return user

    async def touch(self, user_id: str) -> None:
        """更新最后活动时间"""
        stmt = update(User).where(User.id == user_id).values(last_seen_at=datetime.now())
        await self.db.execute(stmt)
        await self.db.flush()

    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        stmt = select(UserProfile).where(UserProfile.user_id == user_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def create_profile(self, user_id: str) -> UserProfile:
        profile = UserProfile(user_id=user_id)
        self.db.add(profile)
        await self.db.flush()
        return profile

    async def update_profile_preferences(self, user_id: str, preferences: dict) -> UserProfile:
        profile = await self.get_or_create_profile(user_id)
        profile.preferences = {**profile.preferences, **preferences}
        await self.db.flush()
        return profile
