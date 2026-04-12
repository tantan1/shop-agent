from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel


class BaseResponse(BaseModel):
    """基础响应模型"""

    success: bool
    code: int
    message: str


class SuccessResponse(BaseResponse):
    """成功响应模型"""

    success: bool = True
    code: int = 200
    message: str = "操作成功"
    data: Optional[Any] = None


class ErrorResponse(BaseResponse):
    """错误响应模型"""

    success: bool = False
    code: int = 400
    message: str = "操作失败"
    error_detail: Optional[str] = None


def success_response(
    data: Any = None, message: str = "操作成功", code: int = 200
) -> dict:
    """创建成功响应"""
    return SuccessResponse(data=data, message=message, code=code).model_dump()


def error_response(
    message: str = "操作失败", error_detail: Optional[str] = None, code: int = 400
) -> dict:
    """创建错误响应"""
    return ErrorResponse(
        message=message, error_detail=error_detail, code=code
    ).model_dump()
