"""
L3 长期记忆归档器：将旧记忆迁移到冷存储
"""
from datetime import datetime, timedelta
from typing import Optional

from src.modules.chat.core.memory_milvus_service import MemoryBlockService
from src.shared.logger import APILogger

logger = APILogger("long_term_memory_archiver")


class LongTermMemoryArchiver:
    """L3 长期记忆归档器"""

    def __init__(self, milvus_service: Optional[MemoryBlockService] = None):
        self._milvus = milvus_service or MemoryBlockService.get_instance()

    async def archive_old_records(self, days: int = 90, batch_size: int = 1000) -> int:
        """归档超过指定天数的记忆记录

        当前实现：将过期记忆的 importance 降为 1（可遗忘），并延长 expires_at
        后续可替换为对象存储迁移
        """
        now = datetime.now()
        archive_threshold = now - timedelta(days=days)
        archived_count = 0

        try:
            results = self._milvus.collection.query(
                expr=(
                    f"last_accessed_at != '' and last_accessed_at < '{archive_threshold.isoformat()}' "
                    f"and importance > 1"
                ),
                output_fields=["block_id", "importance"],
                limit=batch_size,
            )

            if not results:
                return 0

            for r in results:
                try:
                    self._milvus.collection.update(
                        expr=f"block_id == '{r['block_id']}'",
                        field_values={
                            "importance": 1,
                            "expires_at": (now + timedelta(days=30)).isoformat(),
                        },
                    )
                    archived_count += 1
                except Exception as e:
                    logger.warning(f"归档记忆块 {r['block_id']} 失败: {e}")

            logger.info(f"L3 归档完成，归档 {archived_count} 条记录")
            return archived_count
        except Exception as e:
            logger.error(f"L3 归档失败: {e}")
            return 0
