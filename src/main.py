import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi import __version__ as fastapi_version

from src.core.config import config
from src.shared.logger import configure_logging, logging_middleware
from src.shared.exceptions import (
    BusinessException,
    business_exception_handler,
    http_exception_handler,
    general_exception_handler
)
from src.shared.responses import success_response
from src.modules.auth.routers import router as auth_router
from src.modules.items.routers import router as reports_router
from src.modules.chat.routers import router as chat_router


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    configure_logging()  # 配置日志
    yield
    # 关闭时执行（如果需要）


app = FastAPI(
    title="大数据服务API",
    description="为业务方提供数据查询服务的API接口",
    version="1.0.0",
    debug=config.DEBUG_MODE,
    docs_url="/docs" if config.DEBUG_MODE else None,  # 开发环境启用文档
    redoc_url="/redoc" if config.DEBUG_MODE else None,
    lifespan=lifespan
)

# 添加日志中间件
app.middleware("http")(logging_middleware)

# 注册异常处理器
app.add_exception_handler(BusinessException, business_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# 注册路由
app.include_router(auth_router, prefix=config.API_V1_PREFIX)
app.include_router(reports_router, prefix=config.API_V1_PREFIX)
app.include_router(chat_router, prefix=config.API_V1_PREFIX)


@app.get("/health", include_in_schema=False)
async def health_check():
    """健康检查接口"""
    return success_response(
        data={
            "server_status": "running",
            "fastapi_version": fastapi_version,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "debug_mode": config.DEBUG_MODE
        },
        message="服务运行正常"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)