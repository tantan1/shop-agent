"""
L2 短期记忆清理器：删除已过期的摘要
"""
from datetime import datetime
from typing import Optional

from src.modules.chat.core.memory_milvus_service import MemoryBlockService
from src.shared.logger import APILogger

logger = APILogger("short_term_memory_cleaner")


class ShortTermMemoryCleaner:
    """L2 短期记忆清理器"""

    def __init__(self, milvus_service: Optional[MemoryBlockService] = None):
        self._milvus = milvus_service or MemoryBlockService.get_instance()

    async def clean_expired(self, batch_size: int = 1000) -> int:
        """清理 expires_at < 当前时间 的 L2 记忆块，返回删除数量"""
        now = datetime.now()
        try:
            results = self._milvus.collection.query(
                expr=f"expires_at != '' and expires_at < '{now.isoformat()}'",
                output_fields=["block_id"],
                limit=batch_size,
            )
            if not results:
                return 0
            ids = [r["block_id"] for r in results]
            self._milvus.collection.delete(f"block_id in {ids}")
            self._milvus.collection.flush()
            logger.info(f"L2 清理完成，删除 {len(ids)} 条过期记录")
            return len(ids)
        except Exception as e:
            logger.error(f"L2 清理失败: {e}")
            return 0
