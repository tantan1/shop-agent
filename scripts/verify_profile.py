import asyncio
from src.modules.chat.core.memory_service import LongTermMemory
from src.shared.database import get_async_session

UID = "verify_user_001"


async def main():
    async with get_async_session() as db:
        lt = LongTermMemory(pg_session=db, milvus_service=None, embedding_service=None)
        before = await lt.get_or_create_profile(UID)
        print("PROFILE_BEFORE:", before)
        await lt._update_profile(
            UID, [{"type": "preference", "label": "物流偏好", "value": "顺丰"}]
        )
        after = await lt.get_or_create_profile(UID)
        print("PROFILE_AFTER:", after)
        assert after["preferences"].get("物流偏好") == "顺丰", "preferences not persisted!"
        print("OK: L3 profile persisted to PostgreSQL (not mock)")


asyncio.run(main())
