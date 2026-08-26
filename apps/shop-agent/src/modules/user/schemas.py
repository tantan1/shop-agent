from typing import Any, Dict

from pydantic import BaseModel, Field


class UserProfileResponse(BaseModel):
    """用户画像响应"""

    user_id: str
    preferences: Dict[str, Any] = Field(default_factory=dict)
    vip_level: str = "normal"
    total_orders: int = 0
    total_complaints: int = 0
    pending_issues: list = Field(default_factory=list)
