from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.core.config import config


class Base(DeclarativeBase):
    """数据库模型基类"""

    pass


def _check_driver(url: str) -> None:
    """校验连接串所需的异步驱动是否已安装，避免静默失败"""
    if url.startswith("postgresql+asyncpg"):
        try:
            import asyncpg  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "database_url 使用 PostgreSQL(asyncpg)，但镜像未安装 asyncpg。"
                "请在 requirements 中加入 asyncpg 后重建镜像。"
            )


# 创建异步数据库引擎
_check_driver(config.database_url)
engine = create_async_engine(
    config.database_url,
    echo=config.DEBUG_MODE,  # 开发环境下打印SQL语句
    pool_pre_ping=True,  # 连接池预检查
    pool_recycle=3600,  # 连接回收时间（秒）
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
