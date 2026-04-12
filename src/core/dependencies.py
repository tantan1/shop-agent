from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.shared.database import get_db
from src.shared.logger import APILogger


def get_api_logger() -> APILogger:
    """获取API日志记录器依赖"""
    return APILogger()


def get_database_session() -> AsyncSession:
    """获取数据库会话依赖"""
    return Depends(get_db)