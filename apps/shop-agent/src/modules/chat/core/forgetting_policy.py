"""
遗忘策略：决定记忆何时归档或删除
"""
from datetime import datetime, timedelta
from typing import List

from src.modules.chat.core.memory_milvus_service import MemoryBlockService
from src.shared.logger import APILogger

logger = APILogger("forgetting_policy")


class ForgettingPolicy:
    """遗忘策略：基于时间衰减 + 重要性阈值 + 访问频率"""

    def __init__(
        self,
        archive_days: int = 90,
        delete_days: int = 365,
        half_life_days: int = 30,
    ):
        self.archive_days = archive_days
        self.delete_days = delete_days
        self.half_life_days = half_life_days

    def should_archive(self, block: dict) -> bool:
        """判断是否应归档（移出热存储）"""
        now = datetime.now()
        last_accessed = self._parse_time(block.get("last_accessed_at", ""))
        expires_at = self._parse_time(block.get("expires_at", ""))
        importance = block.get("importance", 3)

        days_since_access = (now - last_accessed).days if last_accessed else 999

        # 策略 1：超过 archive_days 未访问且重要性 <= 2
        if days_since_access > self.archive_days and importance <= 2:
            return True

        # 策略 2：超过 2*archive_days 未访问且从未被召回过
        if days_since_access > self.archive_days * 2 and (block.get("access_count", 0) == 0):
            return True

        # 策略 3：已过期
        if expires_at and now > expires_at:
            return True

        return False

    def should_delete(self, block: dict) -> bool:
        """判断是否应永久删除"""
        now = datetime.now()
        last_accessed = self._parse_time(block.get("last_accessed_at", ""))
        days_since_access = (now - last_accessed).days if last_accessed else 999
        importance = block.get("importance", 3)

        # 超过 delete_days 未访问且重要性 = 1
        if days_since_access > self.delete_days and importance == 1:
            return True

        return False

    async def find_expired_blocks(self, milvus_service: MemoryBlockService, batch_size: int = 1000) -> List[str]:
        """查找应删除的过期记忆块 ID 列表"""
        now = datetime.now()
        expired_ids = []

        # 查询已过期的块
        try:
            results = milvus_service.collection.query(
                expr=f"expires_at != '' and expires_at < '{now.isoformat()}'",
                output_fields=["block_id"],
                limit=batch_size,
            )
            expired_ids.extend([r["block_id"] for r in results])
        except Exception as e:
            logger.error(f"查询过期记忆块失败: {e}")

        # 查询应归档的块（超过 archive_days 未访问且 importance <= 2）
        try:
            archive_threshold = (now - timedelta(days=self.archive_days)).isoformat()
            results = milvus_service.collection.query(
                expr=(
                    f"last_accessed_at != '' and last_accessed_at < '{archive_threshold}' "
                    f"and importance <= 2 and access_count == 0"
                ),
                output_fields=["block_id"],
                limit=batch_size,
            )
            expired_ids.extend([r["block_id"] for r in results])
        except Exception as e:
            logger.error(f"查询可归档记忆块失败: {e}")

        return list(set(expired_ids))

    def _parse_time(self, time_str: str) -> datetime:
        """解析 ISO 时间字符串"""
        if not time_str:
            return datetime.now() - timedelta(days=999)
        try:
            return datetime.fromisoformat(time_str)
        except (ValueError, TypeError):
            return datetime.now() - timedelta(days=999)
