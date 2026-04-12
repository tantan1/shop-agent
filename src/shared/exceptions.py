from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from src.shared.responses import error_response
import structlog

logger = structlog.get_logger()


class BusinessException(Exception):
    """业务异常基类"""
    def __init__(self, message: str, error_detail: str = None, code: int = 400):
        self.message = message
        self.error_detail = error_detail
        self.code = code
        super().__init__(self.message)


class AuthenticationException(BusinessException):
    """认证异常"""
    def __init__(self, message: str = "认证失败", error_detail: str = None):
        super().__init__(message, error_detail, 401)


class AuthorizationException(BusinessException):
    """授权异常"""
    def __init__(self, message: str = "权限不足", error_detail: str = None):
        super().__init__(message, error_detail, 403)


class NotFoundException(BusinessException):
    """资源不存在异常"""
    def __init__(self, message: str = "资源不存在", error_detail: str = None):
        super().__init__(message, error_detail, 404)


class ValidationException(BusinessException):
    """数据验证异常"""
    def __init__(self, message: str = "数据验证失败", error_detail: str = None):
        super().__init__(message, error_detail, 422)


class DatabaseException(BusinessException):
    """数据库异常"""
    def __init__(self, message: str = "数据库操作失败", error_detail: str = None):
        super().__init__(message, error_detail, 500)


async def business_exception_handler(request: Request, exc: BusinessException):
    """业务异常处理器"""
    logger.error(
        "业务异常",
        exception=exc.__class__.__name__,
        message=exc.message,
        error_detail=exc.error_detail,
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=exc.code,
        content=error_response(
            message=exc.message,
            error_detail=exc.error_detail,
            code=exc.code
        )
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    logger.error(
        "HTTP异常",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
            message="请求处理失败",
            error_detail=str(exc.detail),
            code=exc.status_code
        )
    )


async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器"""
    logger.error(
        "系统异常",
        exception=exc.__class__.__name__,
        detail=str(exc),
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response(
            message="系统内部错误",
            error_detail="服务器内部错误，请稍后重试",
            code=500
        )
    )
