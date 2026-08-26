"""
记忆批量任务：L2 清理 + L3 归档 + 遗忘
由 K8s CronJob 每天凌晨 2 点调用
"""
import asyncio
import time

from src.modules.chat.core.forgetting_policy import ForgettingPolicy
from src.modules.chat.core.long_term_memory_archiver import LongTermMemoryArchiver
from src.modules.chat.core.memory_milvus_service import MemoryBlockService
from src.modules.chat.core.memory_observability import MemoryObservability
from src.modules.chat.core.redis_cache_service import RedisCacheService
from src.modules.chat.core.short_term_memory_cleaner import ShortTermMemoryCleaner
from src.modules.chat.agent.orchestrator_history import run_l3_daily_backfill
from src.shared.logger import APILogger

logger = APILogger("batch_memory_job")
obs = MemoryObservability()


async def run_batch_job() -> None:
    """每日批量任务：L2 清理 + L3 归档 + 遗忘 + L3 兜底补提"""
    start_time = time.perf_counter()
    logger.info("记忆批量任务开始")

    try:
        milvus = MemoryBlockService.get_instance()
        milvus.initialize()

        # 1. 清理过期 L2 记忆
        l2_cleaner = ShortTermMemoryCleaner(milvus)
        deleted = await l2_cleaner.clean_expired()
        obs.record_forgetting_job("delete", time.perf_counter() - start_time, count=deleted)
        logger.info(f"L2 清理完成，删除 {deleted} 条过期记录")

        # 2. 归档旧 L3 记忆
        l3_archiver = LongTermMemoryArchiver(milvus)
        archived = await l3_archiver.archive_old_records(days=90)
        obs.record_forgetting_job("archive", time.perf_counter() - start_time, count=archived)
        logger.info(f"L3 归档完成，归档 {archived} 条记录")

        # 3. 执行遗忘策略
        forgetting = ForgettingPolicy()
        to_delete = await forgetting.find_expired_blocks(milvus)
        if to_delete:
            ids_str = ", ".join([f"'{bid}'" for bid in to_delete])
            milvus.collection.delete(f"block_id in [{ids_str}]")
            milvus.collection.flush()
            obs.record_forgetting_job("delete", time.perf_counter() - start_time, count=len(to_delete))
            logger.info(f"遗忘完成，删除 {len(to_delete)} 条记录")
        else:
            logger.info("遗忘完成，无记录删除")

        # 4. L3 每日兜底：补提有进展但未提取的会话
        redis = RedisCacheService.get_instance()
        if getattr(redis, "is_available", False):
            backfilled = await run_l3_daily_backfill(redis)
            logger.info(f"L3 每日兜底完成，补提 {backfilled} 个会话")
        else:
            logger.info("Redis 不可用，跳过 L3 每日兜底")

        duration = time.perf_counter() - start_time
        logger.info(f"记忆批量任务完成，耗时 {duration:.2f}s")
    except Exception as e:
        logger.error(f"记忆批量任务失败: {e}")


if __name__ == "__main__":
    asyncio.run(run_batch_job())
