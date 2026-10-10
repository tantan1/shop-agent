"""
GrowthBook Data Source 写入模块（significance 数据入口）

职责（design.md §3.4 / §4.8）：
  - 通过 psycopg2 同步连接池（复用 pgvector_service 的 SimpleConnectionPool 模式）
    向现有 pgvector Postgres 的 shop_agent 库写入曝光/指标事件。
  - 表：gb_exposures(feature_key, variation_key, user_id, domain, timestamp)
        gb_metrics(feature_key, variant_key, metric_name, metric_value, user_id, timestamp)
  - 复用设计：GB 显著性分析直接在该库上跑 SQL（Data Source 注册见 GB UI），
    数据不出内网。

所有写操作 best-effort：失败仅记日志、计数丢失，绝不抛异常影响主链路
（scope §4.9 红线 + design.md §4 层3/层4）。

注意：因 v3.2.0 Python SDK 已移除旧版 cacheConnection/RedisConnection，
本模块只负责「写」；缓存持久化由 growthbook_client 的快照逻辑负责。
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

from psycopg2.pool import SimpleConnectionPool

from src.core.config import config
from src.shared.logger import APILogger

logger = APILogger("growthbook_datasource")


class GrowthBookDataSource:
    """GB Data Source 写入单例（同步 psycopg2 连接池）。

    与 pgvector_service 同源（pgvector:5432/shop_agent），但使用**同步 scheme**
    （postgresql://）连接串；独立管理一个轻量写池（minconn=1, maxconn=5），
    与 pgvector 读/写路径互不干扰。
    """

    _instance: Optional["GrowthBookDataSource"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._pool: Optional[SimpleConnectionPool] = None
        self._dsn: str = ""
        self._initialized: bool = False
        self._lost_exposures: int = 0
        self._lost_metrics: int = 0

    @classmethod
    def get_instance(cls) -> "GrowthBookDataSource":
        """进程内单例（线程安全）。"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = GrowthBookDataSource()
            return cls._instance

    # ════════════════════════════════════════════════════════════════════════
    # 连接串构建
    # ════════════════════════════════════════════════════════════════════════

    def _build_dsn(self) -> str:
        """构建同步 postgresql:// 连接串。

        优先级：
          1. GROWTHBOOK_DATASOURCE_URL（显式，应已为同步 scheme）
          2. 环境变量 DATABASE_URL（asyncpg scheme → 改写为 postgresql://）
          3. 回退：拼装 config 的 pgvector 字段
        """
        url = getattr(config, "GROWTHBOOK_DATASOURCE_URL", "") or ""
        if url:
            return url

        env_url = os.getenv("DATABASE_URL") or ""
        if env_url:
            if env_url.startswith("postgresql+asyncpg://"):
                env_url = "postgresql://" + env_url[len("postgresql+asyncpg://"):]
            elif env_url.startswith("postgresql+psycopg://"):
                env_url = "postgresql://" + env_url[len("postgresql+psycopg://"):]
            elif env_url.startswith("postgres://"):
                env_url = "postgresql://" + env_url[len("postgres://"):]
            return env_url

        # 最终回退：按 config 的 pgvector 字段拼接（与 pgvector_service 同源）
        return (
            f"postgresql://{config.PGVECTOR_USER}:{config.PGVECTOR_PASSWORD}"
            f"@{config.PGVECTOR_HOST}:{config.PGVECTOR_PORT}/{config.PGVECTOR_DB}"
        )

    # ════════════════════════════════════════════════════════════════════════
    # 初始化 + schema
    # ════════════════════════════════════════════════════════════════════════

    def initialize(self) -> None:
        """初始化连接池（best-effort，失败不抛，仅记日志）。"""
        if self._initialized:
            return
        try:
            self._dsn = self._build_dsn()
            self._pool = SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                dsn=self._dsn,
            )
            self._initialized = True
            logger.info(
                f"GrowthBookDataSource 初始化完成: dsn=***@{self._dsn.split('@')[-1]}"
            )
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"GrowthBookDataSource 初始化失败（曝光/指标写入将静默跳过）: {e}"
            )
            self._pool = None

    def ensure_schema(self) -> None:
        """建表（幂等）。连接池未就绪时先尝试 lazy 初始化。"""
        if self._pool is None:
            self.initialize()
        if self._pool is None:
            return
        conn = None
        try:
            conn = self._pool.getconn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS gb_exposures (
                        id SERIAL PRIMARY KEY,
                        feature_key TEXT NOT NULL,
                        variation_key TEXT NOT NULL,
                        user_id TEXT,
                        domain TEXT,
                        timestamp TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS gb_metrics (
                        id SERIAL PRIMARY KEY,
                        feature_key TEXT NOT NULL,
                        variant_key TEXT,
                        metric_name TEXT NOT NULL,
                        metric_value DOUBLE PRECISION,
                        user_id TEXT,
                        timestamp TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
            conn.commit()
            logger.info("GB Data Source schema 就绪（gb_exposures / gb_metrics）")
        except Exception as e:  # noqa: BLE001
            logger.error(f"GB Data Source schema 初始化失败: {e}")
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    pass
        finally:
            if conn is not None:
                self._pool.putconn(conn)

    # ════════════════════════════════════════════════════════════════════════
    # 写入 API
    # ════════════════════════════════════════════════════════════════════════

    def record_exposure(
        self,
        feature_key: str,
        variation_key: str,
        user_id: str,
        domain: str = "ecommerce",
        timestamp: Optional[Any] = None,
    ) -> None:
        """写入一条曝光事件（best-effort，吞异常）。

        Args:
            feature_key: GB feature key（即 experiment.id）
            variation_key: 命中的 variation key / id
            user_id: 用户标识（如 conversation_id）
            domain: 业务领域
            timestamp: 可选显式时间戳（datetime/timestamp 字面量）
        """
        if self._pool is None:
            self.initialize()
        if self._pool is None:
            self._lost_exposures += 1
            logger.warning("GB DataSource 未就绪，曝光事件丢弃（best-effort）")
            return
        conn = None
        try:
            conn = self._pool.getconn()
            with conn.cursor() as cur:
                if timestamp is not None:
                    cur.execute(
                        "INSERT INTO gb_exposures "
                        "(feature_key, variation_key, user_id, domain, timestamp) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (feature_key, str(variation_key), user_id, domain, timestamp),
                    )
                else:
                    cur.execute(
                        "INSERT INTO gb_exposures "
                        "(feature_key, variation_key, user_id, domain) "
                        "VALUES (%s, %s, %s, %s)",
                        (feature_key, str(variation_key), user_id, domain),
                    )
            conn.commit()
        except Exception as e:  # noqa: BLE001
            self._lost_exposures += 1
            logger.error(f"GB 曝光写入失败（忽略）: {e}")
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    pass
        finally:
            if conn is not None:
                self._pool.putconn(conn)

    def record_metric(
        self,
        feature_key: str,
        variant_key: str,
        metric_name: str,
        value: float,
        user_id: Optional[str] = None,
        timestamp: Optional[Any] = None,
    ) -> None:
        """写入一条自定义指标事件（best-effort，吞异常）。"""
        if self._pool is None:
            self.initialize()
        if self._pool is None:
            self._lost_metrics += 1
            logger.warning("GB DataSource 未就绪，指标事件丢弃（best-effort）")
            return
        conn = None
        try:
            conn = self._pool.getconn()
            with conn.cursor() as cur:
                if timestamp is not None:
                    cur.execute(
                        "INSERT INTO gb_metrics "
                        "(feature_key, variant_key, metric_name, metric_value, user_id, timestamp) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        (
                            feature_key,
                            variant_key,
                            metric_name,
                            value,
                            user_id,
                            timestamp,
                        ),
                    )
                else:
                    cur.execute(
                        "INSERT INTO gb_metrics "
                        "(feature_key, variant_key, metric_name, metric_value, user_id) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (feature_key, variant_key, metric_name, value, user_id),
                    )
            conn.commit()
        except Exception as e:  # noqa: BLE001
            self._lost_metrics += 1
            logger.error(f"GB 指标写入失败（忽略）: {e}")
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    pass
        finally:
            if conn is not None:
                self._pool.putconn(conn)

    # ════════════════════════════════════════════════════════════════════════
    # 健康检查 + 关闭
    # ════════════════════════════════════════════════════════════════════════

    def health(self) -> Dict[str, Any]:
        """返回 Data Source 健康状态（供 /health / readyz 聚合展示）。"""
        reachable = False
        if self._pool is not None:
            conn = None
            try:
                conn = self._pool.getconn()
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
                reachable = True
            except Exception:  # noqa: BLE001
                reachable = False
            finally:
                if conn is not None:
                    self._pool.putconn(conn)
        return {
            "initialized": self._initialized,
            "reachable": reachable,
            "lost_exposures": self._lost_exposures,
            "lost_metrics": self._lost_metrics,
        }

    def close(self) -> None:
        """关闭连接池（best-effort）。"""
        if self._pool is not None:
            try:
                self._pool.closeall()
            except Exception:  # noqa: BLE001
                pass
            self._pool = None
        self._initialized = False
