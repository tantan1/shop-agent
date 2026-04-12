import time
from typing import Callable
import structlog
from fastapi import Request, Response
from src.core.config import config


def configure_logging():
    """配置结构化日志"""
    # 配置structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if config.LOG_FORMAT == "json" 
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None):
    """获取日志记录器"""
    return structlog.get_logger(name)


async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """请求日志中间件"""
    start_time = time.time()
    
    # 获取请求信息
    logger = get_logger("api")
    request_id = id(request)  # 简单的请求ID
    
    # 记录请求开始
    logger.info(
        "请求开始",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    
    try:
        # 处理请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 记录请求完成
        logger.info(
            "请求完成",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=f"{process_time:.4f}s",
        )
        
        # 添加处理时间到响应头
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 记录请求异常
        logger.error(
            "请求异常",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            exception=e.__class__.__name__,
            error_detail=str(e),
            process_time=f"{process_time:.4f}s",
        )
        
        # 重新抛出异常，让异常处理器处理
        raise


class APILogger:
    """API业务日志记录器"""
    
    def __init__(self, name: str = "business"):
        self.logger = get_logger(name)
    
    def info(self, msg: str, **kwargs):
        """记录信息日志"""
        self.logger.info(msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """记录错误日志"""
        self.logger.error(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """记录警告日志"""
        self.logger.warning(msg, **kwargs)
    
    def log_api_call(self, api_key: str, endpoint: str, success: bool, **kwargs):
        """记录API调用"""
        self.logger.info(
            "API调用",
            api_key=api_key[:8] + "****" if api_key else None,  # 脱敏处理
            endpoint=endpoint,
            success=success,
            **kwargs
        )
    
    def log_database_operation(self, operation: str, table: str, success: bool, **kwargs):
        """记录数据库操作"""
        self.logger.info(
            "数据库操作",
            operation=operation,
            table=table,
            success=success,
            **kwargs
        )
    
    def log_business_event(self, event: str, **kwargs):
        """记录业务事件"""
        self.logger.info(
            "业务事件",
            business_event=event,
            **kwargs
        )
