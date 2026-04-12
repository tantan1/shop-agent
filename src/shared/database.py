from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from src.core.config import config


class Base(DeclarativeBase):
    """数据库模型基类"""
    pass


# 创建异步数据库引擎
engine = create_async_engine(
    config.database_url,
    echo=config.DEBUG_MODE,  # 开发环境下打印SQL语句
    pool_pre_ping=True,      # 连接池预检查
    pool_recycle=3600,       # 连接回收时间（秒）
)


def get_async_session():
    """创建异步会话"""
    return AsyncSession(engine, expire_on_commit=False)


async def get_db():
    """获取数据库会话的依赖注入函数"""
    session = get_async_session()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()