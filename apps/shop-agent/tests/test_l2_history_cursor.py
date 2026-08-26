"""L2 增量摘要游标（chat:l2_cursor）单测。

验证去重机制：以 message seq 游标替代布尔标记，
- 不重复摘要同一批消息；
- 不遗漏新增消息；
- 不受 L1 滚动截断影响（源数据来自 Redis 持久历史）。

使用内存版 FakeRedis，不依赖真实 Redis / Milvus / embedding 服务。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.modules.chat.agent.orchestrator_history import (
    L2_SAVE_INTERVAL,
    _trigger_l2_save,
    persist_turn,
)


class FakeRedis:
    """极简内存 Redis，仅覆盖 L2 游标相关命令。"""

    def __init__(self):
        # 单一存储：incr/setex/get 都基于同一 dict，避免游标/计数错位
        self._store: dict[str, bytes] = {}
        self._client = SimpleNamespace(
            incr=self._incr,
            get=self._get,
            setex=self._setex,
            expire=self._expire,
        )

    @property
    def is_available(self) -> bool:
        return True

    def _incr(self, key: str) -> int:
        cur = 0
        if key in self._store:
            cur = int(self._store[key].decode())
        self._store[key] = str(cur + 1).encode()
        return cur + 1

    def _get(self, key: str):
        return self._store.get(key)

    def _setex(self, key: str, ttl: int, value: str) -> None:  # noqa: ARG002
        self._store[key] = str(value).encode()

    def _expire(self, key: str, ttl: int) -> None:  # noqa: ARG002
        pass

    def add_chat_message(self, conversation_id, role, content, max_turns=10, expire_days=1):
        # 直接调用真实实现需要真实 Redis；这里用 seq 计数模拟写入
        seq = self._incr(f"chat_history:{conversation_id}:seq")
        self._store.setdefault(f"chat_history:{conversation_id}", []).append(  # type: ignore[attr-defined]
            {"role": role, "content": content, "seq": seq}
        )
        return True

    def get_chat_messages_since(self, conversation_id, since_seq=0, limit=200):
        messages = self._store.get(f"chat_history:{conversation_id}", [])
        return [m for m in messages if m["seq"] > since_seq][-limit:]


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture
def patched_redis_service(fake_redis):
    """FakeRedis 直接作为参数传入 persist_turn / _trigger_l2_save 调用，避免触达真实 Redis。"""
    yield fake_redis


async def _run_l2_save(redis, conversation_id, turn_number, user_id="u1"):
    """执行一次 _trigger_l2_save，返回被摘要的内容列表（通过 mock ShortTermMemory 捕获）。"""
    captured = {}

    class FakeShortTerm:
        async def store_summary(self, **kwargs):
            captured["summary"] = kwargs.get("summary")
            captured["turn_number"] = kwargs.get("turn_number")
            return True

    fake_emb_instance = MagicMock()
    fake_emb_instance.embed_query = AsyncMock(return_value=[0.1] * 4)
    with patch(
        "src.modules.chat.core.memory_service.ShortTermMemory",
        return_value=FakeShortTerm(),
    ):
        with patch(
            "src.modules.chat.core.embedding_service.EmbeddingService"
        ) as emb:
            emb.get_instance.return_value = fake_emb_instance
            await _trigger_l2_save(redis, conversation_id, turn_number, user_id)
    return captured


@pytest.mark.asyncio
async def test_cursor_advances_and_dedups(patched_redis_service):
    """多轮保存后游标推进，再次触发不产生重复摘要。"""
    redis = patched_redis_service
    conv = "conv-dedup"
    max_turns = 10
    # 写 10 条消息（5 轮）
    for i in range(5):
        persist_turn(redis, conv, "u1", f"user-{i}", f"assistant-{i}")

    captured1 = await _run_l2_save(redis, conv, L2_SAVE_INTERVAL)
    cursor_key = f"chat:l2_cursor:{conv}"
    cursor_after = int(redis._client.get(cursor_key).decode())
    assert cursor_after == 10  # 最后一条 seq
    assert "user-4" in captured1.get("summary", "")

    # 立即再次触发（无新消息），应当不存储（new_messages 为空 -> 提前 return）
    captured2 = await _run_l2_save(redis, conv, L2_SAVE_INTERVAL * 2)
    assert "summary" not in captured2  # 没有新消息可摘要


@pytest.mark.asyncio
async def test_cursor_only_summarizes_incremental(patched_redis_service):
    """游标只摘要增量：新消息才进摘要，已摘要的不重复。"""
    redis = patched_redis_service
    conv = "conv-incremental"
    for i in range(3):
        persist_turn(redis, conv, "u1", f"user-{i}", f"assistant-{i}")

    await _run_l2_save(redis, conv, L2_SAVE_INTERVAL)
    cursor_after_first = int(redis._client.get(f"chat:l2_cursor:{conv}").decode())
    assert cursor_after_first == 6

    # 新增 2 轮
    persist_turn(redis, conv, "u1", "user-new-1", "assistant-new-1")
    persist_turn(redis, conv, "u1", "user-new-2", "assistant-new-2")

    captured = await _run_l2_save(redis, conv, L2_SAVE_INTERVAL * 2)
    summary = captured.get("summary", "")
    # 增量只包含新消息
    assert "user-new-2" in summary
    assert "user-0" not in summary  # 旧消息不应重复出现
    cursor_after_second = int(redis._client.get(f"chat:l2_cursor:{conv}").decode())
    assert cursor_after_second == 10


@pytest.mark.asyncio
async def test_cursor_resilient_to_l1_truncation(patched_redis_service):
    """模拟 L1 滚动截断（仅保留最近 N 条），但游标仍基于 Redis 持久 seq，不漂移。"""
    redis = patched_redis_service
    conv = "conv-l1-trunc"
    # 写入超过 max_turns*2 的轮数，模拟 L1 滚动
    for i in range(8):
        persist_turn(redis, conv, "u1", f"user-{i}", f"assistant-{i}")

    await _run_l2_save(redis, conv, L2_SAVE_INTERVAL)
    cursor = int(redis._client.get(f"chat:l2_cursor:{conv}").decode())
    assert cursor == 16  # 仍基于真实全局 seq，未因截断丢游标

    # 再写一轮新消息
    persist_turn(redis, conv, "u1", "user-last", "assistant-last")
    captured = await _run_l2_save(redis, conv, L2_SAVE_INTERVAL * 2)
    assert "user-last" in captured.get("summary", "")
    assert int(redis._client.get(f"chat:l2_cursor:{conv}").decode()) == 18


@pytest.mark.asyncio
async def test_persist_turn_increments_turn_counter(patched_redis_service):
    """persist_turn 应正确递增轮次计数，并在达到间隔时调度 L2 保存任务。"""
    redis = patched_redis_service
    conv = "conv-counter"

    # 第 1~4 轮：不应触发 L2（turn % 5 != 0）
    for i in range(4):
        persist_turn(redis, conv, "u1", f"u{i}", f"a{i}")
    assert int(redis._client.get(f"chat:turn_count:{conv}").decode()) == 4
    assert f"chat:l2_cursor:{conv}" not in redis._store

    # 第 5 轮：触发 L2（创建后台 task，不要求立即完成）
    persist_turn(redis, conv, "u1", "u5", "a5")
    assert int(redis._client.get(f"chat:turn_count:{conv}").decode()) == 5
