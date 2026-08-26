"""L3 长期记忆触发逻辑单测。

验证 03-L3 篇设计的"L2/L3 解耦 + 长对话每 5 轮兜底 + 优先级"：
- 不应无差别每 5 轮提取（10 轮以内普通对话不触发）
- 仅在 turn_number > LONG_CONVERSATION_THRESHOLD 且 %5==0 时长对话兜底
- 优先级：对话结束 > 高价值意图 > 长对话兜底 > 关键词
- persist_turn 在第 5/10 轮同时异步派发 L2 与 L3 任务（解耦）

全程 mock，不依赖真实 Redis / LLM / Milvus / PG。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.modules.chat.core.memory_extraction_trigger import (
    LONG_CONVERSATION_THRESHOLD,
    ExtractionContext,
    MemoryExtractionTrigger,
)
from src.modules.chat.agent.orchestrator_history import persist_turn


class FakeRedis:
    """极简内存 Redis，仅覆盖 L3 测试所需命令。"""

    def __init__(self):
        self._store: dict[str, bytes] = {}
        self._client = SimpleNamespace(
            incr=self._incr,
            get=self._get,
            setex=self._setex,
            expire=self._expire,
            scan=self._scan,
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
        seq = self._incr(f"chat_history:{conversation_id}:seq")
        self._store.setdefault(f"chat_history:{conversation_id}", []).append(
            {"role": role, "content": content, "seq": seq}
        )
        return True

    def get_chat_messages_since(self, conversation_id, since_seq=0, limit=200):
        messages = self._store.get(f"chat_history:{conversation_id}", [])
        return [m for m in messages if m["seq"] > since_seq][-limit:]

    def _scan(self, cursor=0, match="*", count=100):
        import fnmatch

        keys = [k for k in self._store if fnmatch.fnmatch(k, match)]
        # 简单分页：一次性返回（测试数据量小）
        return 0, keys


def _ctx(turn_number=0, is_ended=False, last_intent=None, user_message="", history=None):
    return ExtractionContext(
        user_id="u1",
        conversation_id="c1",
        chat_history=history or [{"role": "user", "content": "hi"}],
        user_message=user_message,
        is_ended=is_ended,
        turn_number=turn_number,
        last_intent=last_intent,
    )


def test_should_extract_not_every_5_turns_for_normal_dialog():
    """普通对话（<=10 轮）不应无差别每 5 轮提取。"""
    trig = MemoryExtractionTrigger(llm := MagicMock())
    # 第 5 轮、第 10 轮（恰好是 L2 节奏）但非长对话 → 不触发
    assert trig.should_extract(False, 5, None, [], "你好") is False
    assert trig.should_extract(False, 10, None, [], "你好") is False


def test_should_extract_long_conversation_every_5():
    """长对话（>阈值）且是 5 的倍数时才兜底触发。"""
    trig = MemoryExtractionTrigger(MagicMock())
    assert trig.should_extract(False, LONG_CONVERSATION_THRESHOLD + 5, None, [], "你好") is True
    # 长对话但非 5 的倍数 → 不触发
    assert trig.should_extract(False, LONG_CONVERSATION_THRESHOLD + 3, None, [], "你好") is False


def test_should_extract_priority_ended():
    """优先级1：对话结束必须提取（即便短对话）。"""
    trig = MemoryExtractionTrigger(MagicMock())
    assert trig.should_extract(True, 2, None, [], "随便聊聊") is True


def test_should_extract_priority_high_value_intent():
    """优先级2：高价值意图（退货/投诉/纠纷）立即触发。"""
    trig = MemoryExtractionTrigger(MagicMock())
    for intent in ("request-return", "complaint", "dispute"):
        assert trig.should_extract(False, 3, intent, [], "我要退货") is True


def test_should_extract_priority_keyword():
    """优先级4：关键词命中（订单/偏好/尺码等）触发。"""
    trig = MemoryExtractionTrigger(MagicMock())
    assert trig.should_extract(False, 3, None, [], "我的订单什么时候到") is True
    assert trig.should_extract(False, 3, None, [], "我偏好蓝色") is True


def test_should_extract_no_trigger_plain():
    """普通闲聊、短对话、无关键词 → 不触发（避免 L3 高频高成本）。"""
    trig = MemoryExtractionTrigger(MagicMock())
    assert trig.should_extract(False, 3, None, [], "好的谢谢") is False


@pytest.mark.asyncio
async def test_trigger_l3_extract_runs_when_should_extract_true():
    """_trigger_l3_extract 在 should_extract 命中时调用提取器并落库。"""
    captured = {}

    class FakeLongTerm:
        async def update_from_conversation(self, user_id, memories):
            captured["memories"] = memories
            return ["id1"]

    class FakeExtractor:
        async def extract(self, chat_history, user_message):
            return [{"type": "order", "label": "订单12345", "value": "x", "importance": 4, "metadata": {}}]

    with patch(
        "src.modules.chat.core.memory_extractor.MemoryExtractor",
        return_value=FakeExtractor(),
    ):
        with patch(
            "src.modules.chat.core.embedding_service.EmbeddingService"
        ) as emb:
            emb.get_instance.return_value.embed_query = AsyncMock(return_value=[0.1] * 4)
            with patch(
                "src.modules.chat.core.memory_milvus_service.MemoryBlockService"
            ) as mbs:
                mbs.get_instance.return_value.insert_block.return_value = "blk"
                with patch(
                    "src.modules.chat.core.memory_service.LongTermMemory",
                    return_value=FakeLongTerm(),
                ):
                    from src.modules.chat.core.llm_service import LLMService

                    with patch.object(LLMService, "get_instance", return_value=MagicMock()):
                        trigger = MemoryExtractionTrigger(MagicMock())
                        ctx = _ctx(
                            turn_number=15,
                            user_message="我的订单 12345 什么时候到",
                        )
                        ctx.chat_history = [
                            {"role": "user", "content": "我的订单 12345 什么时候到"},
                            {"role": "assistant", "content": "正在查询"},
                        ]
                        result = await trigger.try_extract(ctx)
    assert result is not None
    assert captured.get("memories")


@pytest.mark.asyncio
async def test_persist_turn_dispatches_l2_and_l3_tasks():
    """persist_turn 在第 5 轮同时异步派发 L2 与 L3 任务（解耦、互不阻塞）。"""
    fake_redis = FakeRedis()
    dispatched = {"l2": False, "l3": False}
    tasks = []

    async def fake_l2(*a, **k):
        dispatched["l2"] = True

    async def fake_l3(*a, **k):
        dispatched["l3"] = True

    def _capture_create_task(coro):
        t = asyncio.ensure_future(coro)
        tasks.append(t)
        return t

    with patch(
        "src.modules.chat.agent.orchestrator_history._trigger_l2_save",
        side_effect=fake_l2,
    ):
        with patch(
            "src.modules.chat.agent.orchestrator_history._trigger_l3_extract",
            side_effect=fake_l3,
        ):
            with patch(
                "src.modules.chat.agent.orchestrator_history.asyncio.create_task",
                side_effect=_capture_create_task,
            ):
                for i in range(5):
                    persist_turn(fake_redis, "c1", "u1", f"u{i}", f"a{i}")
                # 驱动已派发的任务在事件循环中运行
                if tasks:
                    await asyncio.gather(*tasks)
    assert dispatched["l2"] is True
    assert dispatched["l3"] is True


@pytest.mark.asyncio
async def test_l3_extract_writes_success_flag():
    """L3 提取成功后写入 chat:l3_last_extract 标记（供每日兜底去重）。"""
    from src.modules.chat.agent.orchestrator_history import _trigger_l3_extract

    fake_redis = FakeRedis()
    fake_redis.add_chat_message("c1", "user", "我的订单 12345 什么时候到")
    fake_redis.add_chat_message("c1", "assistant", "正在查询")

    with patch("src.modules.chat.core.memory_extractor.MemoryExtractor") as me:
        me.return_value.extract = AsyncMock(
            return_value=[
                {"type": "order", "label": "订单12345", "value": "x", "importance": 4, "metadata": {}}
            ]
        )
        with patch(
            "src.modules.chat.core.embedding_service.EmbeddingService"
        ) as emb:
            emb.get_instance.return_value.embed_query = AsyncMock(return_value=[0.1] * 4)
            with patch(
                "src.modules.chat.core.memory_milvus_service.MemoryBlockService"
            ) as mbs:
                mbs.get_instance.return_value.insert_block.return_value = "blk"
                with patch(
                    "src.modules.chat.core.memory_service.LongTermMemory"
                ) as ltm:
                    ltm.return_value.update_from_conversation = AsyncMock(return_value=["id1"])
                    from src.modules.chat.core.llm_service import LLMService

                    with patch.object(LLMService, "get_instance", return_value=MagicMock()):
                        ok = await _trigger_l3_extract(
                            fake_redis, "c1", "u1", "我的订单 12345 什么时候到", 6
                        )
    assert ok is True
    flag = fake_redis._client.get("chat:l3_last_extract:c1")
    assert flag is not None
    import json as _json

    flag_data = _json.loads(flag.decode())
    assert flag_data["turn"] == 6
    assert flag_data["uid"] == "u1"


@pytest.mark.asyncio
async def test_l3_daily_backfill_force_extracts_unflagged():
    """每日兜底：未提取标记（或标记轮次落后）的会话被强制补提；已齐平则跳过。"""
    from src.modules.chat.agent.orchestrator_history import run_l3_daily_backfill

    fake_redis = FakeRedis()
    # 会话 A：有 6 轮、无提取标记 → 应补提
    for i in range(6):
        fake_redis.add_chat_message("A", "user", f"u{i}")
        fake_redis.add_chat_message("A", "assistant", f"a{i}")
    fake_redis._client.setex("chat:turn_count:A", 999, "6")  # 轮次计数
    # 会话 B：有 6 轮，但已提取标记轮次=6（齐平）→ 跳过
    for i in range(6):
        fake_redis.add_chat_message("B", "user", f"u{i}")
        fake_redis.add_chat_message("B", "assistant", f"a{i}")
    fake_redis._client.setex("chat:turn_count:B", 999, "6")
    import json as _json

    fake_redis._client.setex(
        "chat:l3_last_extract:B", 999, _json.dumps({"turn": 6, "uid": "u1"})
    )
    # 会话 C：仅 2 轮（过短）→ 跳过
    fake_redis.add_chat_message("C", "user", "hi")
    fake_redis.add_chat_message("C", "assistant", "hello")
    fake_redis._client.setex("chat:turn_count:C", 999, "2")

    extract_calls = []

    # 包装 _trigger_l3_extract 记录被补提的会话 id（force=True）
    import src.modules.chat.agent.orchestrator_history as oh

    real_extract = oh._trigger_l3_extract

    async def _spy(redis, conv_id, uid, msg, turn, force=False):
        extract_calls.append(conv_id)
        return await real_extract(redis, conv_id, uid, msg, turn, force=force)

    with patch(
        "src.modules.chat.core.memory_extractor.MemoryExtractor"
    ) as me:
        me.return_value.extract = AsyncMock(
            return_value=[{"type": "preference", "label": "x", "value": "y", "importance": 3}]
        )
        with patch(
            "src.modules.chat.core.embedding_service.EmbeddingService"
        ) as emb:
            emb.get_instance.return_value.embed_query = AsyncMock(return_value=[0.1] * 4)
            with patch(
                "src.modules.chat.core.memory_milvus_service.MemoryBlockService"
            ) as mbs:
                mbs.get_instance.return_value.insert_block.return_value = "blk"
                with patch(
                    "src.modules.chat.core.memory_service.LongTermMemory"
                ) as ltm:
                    ltm.return_value.update_from_conversation = AsyncMock(return_value=["id1"])
                    from src.modules.chat.core.llm_service import LLMService

                    with patch.object(LLMService, "get_instance", return_value=MagicMock()):
                        with patch.object(oh, "_trigger_l3_extract", side_effect=_spy):
                            extracted = await run_l3_daily_backfill(fake_redis)

    # 仅 A 被补提（B 已齐平、C 过短）
    assert extracted == 1
    assert "A" in extract_calls
    assert "B" not in extract_calls
    assert "C" not in extract_calls
