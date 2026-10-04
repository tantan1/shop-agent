"""认证领域模型。

为避免与权限核心重复，直接复用 `src.core.permissions` 的权威定义
（Role / ClientInfo），并补充接口层所需的请求模型。
"""
from src.core.permissions import ClientInfo, Role  # noqa: F401

__all__ = ["ClientInfo", "Role", "ApiKeyAuth"]

from pydantic import BaseModel, Field


class ApiKeyAuth(BaseModel):
    """API Key 认证请求体（用于换取会话/临时凭证的端点）。"""

    api_key: str = Field(..., min_length=1, description="API 密钥")
