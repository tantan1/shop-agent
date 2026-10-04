"""
记忆服务主入口：L2 短期记忆 + L3 长期记忆
"""
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.modules.chat.core.memory_milvus_service import MemoryBlockService, MemoryBlockType, SearchFilter
from src.modules.chat.core.memory_observability import MemoryObservability
from src.shared.logger import APILogger

logger = APILogger("memory_service")
obs = MemoryObservability()


class ShortTermMemory:
    """短期记忆服务：最近 7 天对话摘要（L2）"""

    def __init__(self):
        self._milvus = MemoryBlockService.get_instance()

    async def store_summary(  # noqa: PLR0913
        self,
        user_id: str,
        conversation_id: str,
        summary: str,
        key_entities: List[str],
        turn_number: int = 0,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """存储对话摘要到 Milvus"""
        if embedding is None:
            raise ValueError("embedding 不能为空，请先调用 embedding_service.embed_query()")

        now = datetime.now()
        block = {
            "user_id": user_id,
            "block_type": MemoryBlockType.INTERACTION,
            "label": f"对话摘要 {conversation_id[:8]}",
            "value": summary,
            "embedding": embedding,
            "importance": 3,
            "source": "conversation",
            "expires_at": (now + timedelta(days=7)).isoformat(),
            "metadata": {
                "conversation_id": conversation_id,
                "entities": key_entities,
                "turn_number": turn_number,
            },
        }
        block_id = self._milvus.insert_block(block)
        obs.record_block_created(block["block_type"])
        logger.info(
            "L2 摘要已保存",
            user_id=user_id,
            conversation_id=conversation_id[:8],
            block_id=block_id[:8],
        )
        return block_id

    async def recall_recent(
        self,
        user_id: str,
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """召回最近 7 天的相关摘要"""
        now = datetime.now().isoformat()
        search_filter = SearchFilter(
            user_id=user_id,
            block_type=MemoryBlockType.INTERACTION,
            importance_threshold=1,
            filter_expr=f"expires_at >= '{now}'",
        )
        results = self._milvus.search(
            query_embedding=query_embedding,
            search_filter=search_filter,
            top_k=top_k,
        )
        obs.record_recall("l2", 0, count=len(results))
        logger.debug("L2 召回完成", user_id=user_id, count=len(results))
        return results


class LongTermMemory:
    """长期记忆服务：用户画像 + 历史记忆（L3）"""

    def __init__(self, pg_session, milvus_service: MemoryBlockService, embedding_service):
        self.pg = pg_session
        self.milvus = milvus_service
        self.embedding = embedding_service

    async def _with_session(self, op):
        """pg_session 由调用方注入时直接复用；为 None 时自开会话。

        executor / memory_extraction_trigger 当前均传 pg_session=None，
        故走自开路径，使 L3 用户画像真正持久化（此前返回空 mock）。
        """
        if self.pg is not None:
            return await op(self.pg)
        from src.shared.database import get_async_session

        async with get_async_session() as db:
            return await op(db)

    async def _ensure_profile_table(self, db) -> None:
        """幂等建表：user_profiles 属 Phase 2 画像，当前库可能未 migrate。

        表结构以 user 模块的 UserProfile 模型为唯一权威定义，此处仅在其
        不存在时按模型自动建表（避免手写字面量 DDL 与模型定义分叉）。
        """
        from src.modules.user.models import UserProfile

        await db.run_sync(
            lambda sess: UserProfile.__table__.create(sess.get_bind(), checkfirst=True)
        )

    async def get_or_create_profile(self, user_id: str) -> Dict[str, Any]:
        """获取或创建用户画像（委托 user 模块的 UserRepository 访问 PostgreSQL）"""
        from src.modules.user.repositories import UserRepository

        async def _do(db):
            await self._ensure_profile_table(db)
            repo = UserRepository(db)
            profile = await repo.get_profile(user_id)
            if profile is None:
                profile = await repo.create_profile(user_id)
                await db.commit()
            return {
                "user_id": profile.user_id,
                "preferences": profile.preferences or {},
                "vip_level": profile.vip_level,
                "total_orders": profile.total_orders,
                "total_complaints": profile.total_complaints,
                "pending_issues": profile.pending_issues or [],
            }

        return await self._with_session(_do)

    async def _update_profile(
        self, user_id: str, extracted_memories: List[Dict[str, Any]]
    ) -> None:
        """根据提取的记忆更新并持久化用户画像（委托 UserRepository）"""
        from src.modules.user.repositories import UserRepository

        async def _do(db):
            await self._ensure_profile_table(db)
            repo = UserRepository(db)
            profile = await repo.get_profile(user_id)
            if profile is None:
                profile = await repo.create_profile(user_id)
            preferences = dict(profile.preferences or {})
            for memory in extracted_memories:
                if memory["type"] == "preference":
                    # 合并用户偏好
                    key = memory["label"]
                    preferences[key] = memory["value"]
            await repo.update_profile_preferences(user_id, preferences)
            await db.commit()

        await self._with_session(_do)

    async def update_from_conversation(
        self, user_id: str, extracted_memories: List[Dict[str, Any]]
    ) -> List[str]:
        """将 MemoryExtractor 提取的记忆存入 L3"""
        # 1. 更新用户画像
        await self._update_profile(user_id, extracted_memories)

        # 2. 向量化记忆存入 Milvus
        block_ids = []
        for memory in extracted_memories:
            block_id = self.milvus.insert_block({
                "user_id": user_id,
                "block_type": memory["type"],
                "label": memory["label"],
                "value": memory["value"],
                "embedding": await self.embedding.embed_query(memory["value"]),
                "importance": memory.get("importance", 3),
                "source": "conversation",
                "metadata": memory.get("metadata", {}),
            })
            block_ids.append(block_id)
            obs.record_block_created(memory["type"])
        return block_ids

    async def recall_relevant(
        self,
        user_id: str,
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """召回用户相关的长期记忆"""
        now = datetime.now().isoformat()
        search_filter = SearchFilter(
            user_id=user_id,
            importance_threshold=2,
            filter_expr=f"expires_at >= '{now}' or expires_at == ''",
        )
        results = self.milvus.search(
            query_embedding=query_embedding,
            search_filter=search_filter,
            top_k=top_k,
        )
        obs.record_recall("l3", 0, count=len(results))
        return results
