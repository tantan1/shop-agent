from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.user.schemas import UserProfileResponse
from src.modules.user.services import UserService
from src.shared.database import get_db

router = APIRouter(prefix="/users", tags=["用户管理"])


async def get_user_service(db: AsyncSession = Depends(get_db)) -> UserService:
    return UserService(db)


@router.get("/me", response_model=UserProfileResponse, summary="获取当前用户信息")
async def get_current_user(
    user_service: UserService = Depends(get_user_service),
    user_id: str = Depends(lambda: ...),  # 由 auth/middleware 注入
):
    """获取当前用户的基本信息和画像"""
    profile = await user_service.get_or_create_profile(user_id)
    return UserProfileResponse(
        user_id=profile.user_id,
        preferences=profile.preferences,
        vip_level=profile.vip_level,
        total_orders=profile.total_orders,
        total_complaints=profile.total_complaints,
    )


@router.delete("/me/memory", summary="删除用户所有记忆（GDPR）")
async def delete_user_memory(
    user_service: UserService = Depends(get_user_service),
    user_id: str = Depends(lambda: ...),  # 由 auth/middleware 注入
):
    """删除用户所有记忆数据（L1/L2/L3）"""
    # TODO: 实现级联删除逻辑
    return {"message": "记忆删除功能待实现"}
