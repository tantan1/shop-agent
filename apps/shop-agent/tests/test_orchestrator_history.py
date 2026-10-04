"""验证 L2 落库接入真实 ConversationSummarizer + 轮次触发/游标推进等真实分支。

设计原则（兑现文章 23「测试必须失败」）：
  - 不把核心语义完全打桩成固定返回值，而是断言"真实调用发生了什么"
  - 覆盖 persist_turn 的轮次取模触发、游标推进、空消息早退等分支，
    使注入变异（布尔/比较符翻转、算术边界）能被测试感知。
"""

from __future__ import annotations

import asyncio
import types

import pytest

from src.modules.chat.agent import orchestrator_history as oh
from src.modules.chat.agent.conversation_summarizer import ConversationSummarizer
from src.modules.chat.core.memory_service import ShortTermMemory


# ── fakes ────────────────────────────────────────────────────────────────────
class _FakeClient:
    """最小 redis client：incr 自增、get/setex 记录。"""

    def __init__(self, cursor=None):
        self._counts: dict = {}
        self._kv: dict = {}
        self._expire_calls: list = []
        self._setex_calls: list = []
        if cursor is not None:
            self._kv["chat:l2_cursor:conv1"] = str(cursor).encode()

    def incr(self, key):
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def expire(self, key, ttl):
        self._expire_calls.append((key, ttl))

    def get(self, key):
        # 源码对 get 结果调用 .decode()，故此处返回 bytes（与项目所用 redis 客户端一致）
        return self._kv.get(key)

    def setex(self, key, ttl, value):
        self._setex_calls.append((key, ttl, value))
        self._kv[key] = str(value).encode()

    def scan(self, cursor=0, match=None, count=100):
        # 返回已注册的 chat:turn_count:* 键（供 L3 兜底扫描遍历）
        keys = [k for k in self._counts if k.startswith("chat:turn_count:")]
        return 0, keys


class _FakeRedis:
    def __init__(self, messages, cursor=None):
        self._client = _FakeClient(cursor)
        self.is_available = True
        self._messages = messages
        self.added: list = []
        self.since_calls: list = []

    def add_chat_message(self, conv_id, role, content, max_turns=None):
        self.added.append({"role": role, "content": content})

    def get_chat_messages_since(self, conv_id, since_seq=0, limit=200):
        self.since_calls.append(since_seq)
        return [m for m in self._messages if int(m.get("seq", 0)) > since_seq]


class _SpySummarizer:
    """spy：记录入参并返回与拼接不同的压缩摘要（保留真实调用可观测性）。"""

    def __init__(self):
        self.calls: list = []

    async def summarize_if_needed(self, messages):
        self.calls.append(list(messages))
        return "压缩摘要"  # 与拼接串必然不同


class _FakeEmbedding:
    async def embed_query(self, text):
        return [0.1, 0.2, 0.3]


# ── tests ────────────────────────────────────────────────────────────────────
@pytest.fixture
def collected():
    return []


async def _fake_store(collected, **kwargs):
    """store_summary 是 async，替身必须返回可 await 对象。"""
    collected.append(kwargs)
    return "block-id"


def _patch_store(monkeypatch, collected):
    monkeypatch.setattr(ShortTermMemory, "__init__", lambda self: None)
    monkeypatch.setattr(
        ShortTermMemory,
        "store_summary",
        lambda self, **kwargs: _fake_store(collected, **kwargs),
    )


@pytest.mark.asyncio
async def test_l2_save_calls_real_summarizer_with_incremental_messages(monkeypatch, collected):
    """L2 落库必须调用真实摘要器，且入参是游标之后的增量消息。"""
    messages = [
        {"role": "user", "content": f"u{i}", "seq": i} for i in range(1, 6)
    ]
    fake_redis = _FakeRedis(messages, cursor=2)
    spy = _SpySummarizer()
    monkeypatch.setattr(ConversationSummarizer, "summarize_if_needed", spy.summarize_if_needed)
    _patch_store(monkeypatch, collected)
    monkeypatch.setattr(
        "src.modules.chat.core.embedding_service.EmbeddingService",
        types.SimpleNamespace(get_instance=lambda: _FakeEmbedding()),
    )

    await oh._trigger_l2_save(fake_redis, "conv1", 5, "user1")

    # 1) 真实调用发生，且只传游标(seq>2)之后的增量
    assert spy.calls, "summarize_if_needed 未被调用"
    fed = spy.calls[0]
    assert [m["content"] for m in fed] == ["u3", "u4", "u5"]
    # 2) 落库用真摘要，不是拼接
    assert collected, "store_summary 未被调用"
    assert collected[0]["summary"] == "压缩摘要"
    concat = "; ".join(m["content"] for m in messages)
    assert collected[0]["summary"] != concat
    # 3) embedding 透传
    assert collected[0]["embedding"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_l2_save_advances_cursor_to_max_seq(monkeypatch, collected):
    """游标必须推进到本次处理的最大 seq（算术边界变异可捕获）。"""
    messages = [{"role": "user", "content": "a", "seq": 7},
                {"role": "user", "content": "b", "seq": 9}]
    fake_redis = _FakeRedis(messages, cursor=0)
    spy = _SpySummarizer()
    monkeypatch.setattr(ConversationSummarizer, "summarize_if_needed", spy.summarize_if_needed)
    _patch_store(monkeypatch, collected)
    monkeypatch.setattr(
        "src.modules.chat.core.embedding_service.EmbeddingService",
        types.SimpleNamespace(get_instance=lambda: _FakeEmbedding()),
    )

    await oh._trigger_l2_save(fake_redis, "conv1", 5, "user1")

    # 游标 key 写入的必须是 max(seq)=9
    cursor_writes = [c for c in fake_redis._client._setex_calls if "l2_cursor" in c[0]]
    assert cursor_writes, "游标未推进"
    assert cursor_writes[-1][2] == "9"


@pytest.mark.asyncio
async def test_l2_save_returns_early_when_no_new_messages(monkeypatch, collected):
    """无增量消息时必须早退，不落库（布尔翻转可捕获）。"""
    fake_redis = _FakeRedis([{"role": "user", "content": "old", "seq": 1}], cursor=99)
    spy = _SpySummarizer()
    monkeypatch.setattr(ConversationSummarizer, "summarize_if_needed", spy.summarize_if_needed)
    _patch_store(monkeypatch, collected)

    await oh._trigger_l2_save(fake_redis, "conv1", 5, "user1")

    assert collected == [], "无增量消息时不应落库"
    assert spy.calls == [], "无增量消息时不应调用摘要器"


@pytest.mark.asyncio
async def test_persist_turn_triggers_l2_only_every_n_turns(monkeypatch):
    """轮次取模触发：第 5 轮触发 L2+L3，非 5 轮不触发。"""
    triggered: list = []

    async def fake_l2(redis, conv, turn, uid):
        triggered.append(("l2", turn))

    async def fake_l3(redis, conv, uid, msg, turn, force=False):
        triggered.append(("l3", turn))

    monkeypatch.setattr(oh, "_trigger_l2_save", fake_l2)
    monkeypatch.setattr(oh, "_trigger_l3_extract", fake_l3)

    # 第 1、2 轮：不触发
    fake = _FakeRedis([])
    oh.persist_turn(fake, "conv1", "u1", "hi", "hello")
    oh.persist_turn(fake, "conv1", "u1", "hi2", "hello2")
    await asyncio.sleep(0)
    assert triggered == [], f"前 2 轮不应触发，实际 {triggered}"

    # 第 3、4、5 轮：第 5 轮触发
    for i in range(3, 6):
        oh.persist_turn(fake, "conv1", "u1", f"m{i}", "r{i}")
    await asyncio.sleep(0.01)
    assert ("l2", 5) in triggered, f"第 5 轮应触发 L2，实际 {triggered}"
    assert ("l3", 5) in triggered, f"第 5 轮应触发 L3，实际 {triggered}"


def test_persist_turn_skips_when_redis_unavailable():
    """redis 不可用时直接返回，不写任何消息。"""
    fake = _FakeRedis([])
    fake.is_available = False
    oh.persist_turn(fake, "conv1", "u1", "hi", "hello")
    assert fake.added == [], "redis 不可用不应写入"


# ── L3 兜底扫描：覆盖 turn_number < 3 / 标记 >= 两个比较边界 ──────────────────
@pytest.mark.asyncio
async def test_l3_backfill_skips_short_conversations(monkeypatch):
    """轮次 < 3 的会话应跳过兜底（比较符翻转可捕获）。"""
    called: list = []

    async def fake_l3(redis, conv, uid, msg, turn, force=False):
        called.append(conv)
        return True

    monkeypatch.setattr(oh, "_trigger_l3_extract", fake_l3)

    fake = _FakeRedis([])
    # 源码通过 get("chat:turn_count:<conv>") 读轮次，scan 只负责枚举键
    # 轮次 = 2（< 3）→ 应跳过
    fake._client._counts["chat:turn_count:c_short"] = 2
    fake._client._kv["chat:turn_count:c_short"] = b"2"
    # 轮次 = 5（>= 3）→ 应补提
    fake._client._counts["chat:turn_count:c_long"] = 5
    fake._client._kv["chat:turn_count:c_long"] = b"5"

    extracted = await oh.run_l3_daily_backfill(fake)

    assert "c_short" not in called, "轮次<3 不应兜底"
    assert "c_long" in called, "轮次>=3 应兜底"
    assert extracted == 1, f"应只补提 1 个，实际 {extracted}"


@pytest.mark.asyncio
async def test_l3_backfill_skips_already_extracted(monkeypatch):
    """已提取标记 turn >= 当前轮次时跳过（>= 翻转为 < 可捕获）。"""
    called: list = []

    async def fake_l3(redis, conv, uid, msg, turn, force=False):
        called.append(conv)
        return True

    monkeypatch.setattr(oh, "_trigger_l3_extract", fake_l3)

    import json

    fake = _FakeRedis([])
    # 边界：标记 turn=5 == 当前 turn=5 → already，跳过
    fake._client._counts["chat:turn_count:c_eq"] = 5
    fake._client._kv["chat:turn_count:c_eq"] = b"5"
    fake._client._kv["chat:l3_last_extract:c_eq"] = json.dumps(
        {"turn": 5, "uid": "u1"}
    ).encode()
    # 标记 turn=3 < 当前 turn=8 → 有新进展，应补提
    fake._client._counts["chat:turn_count:c_new"] = 8
    fake._client._kv["chat:turn_count:c_new"] = b"8"
    fake._client._kv["chat:l3_last_extract:c_new"] = json.dumps(
        {"turn": 3, "uid": "u1"}
    ).encode()

    extracted = await oh.run_l3_daily_backfill(fake)

    assert "c_eq" not in called, "已提取(turn>=当前)应跳过"
    assert "c_new" in called, "有新进展(turn<当前)应补提"
    assert extracted == 1, f"应只补提 1 个，实际 {extracted}"


@pytest.mark.asyncio
async def test_l3_backfill_passes_last_user_message(monkeypatch):
    """兜底补提应把'最后一条用户消息'作为上下文（比较翻转 != -> == 可捕获）。"""
    captured_msg: list = []

    async def fake_l3(redis, conv, uid, msg, turn, force=False):
        captured_msg.append(msg)
        return True

    monkeypatch.setattr(oh, "_trigger_l3_extract", fake_l3)

    fake = _FakeRedis([])
    # 轮次 = 5（>= 3 触发），构造多条历史：最后是助手消息，倒数第二是用户消息
    fake._client._counts["chat:turn_count:c_lastuser"] = 5
    fake._client._kv["chat:turn_count:c_lastuser"] = b"5"
    fake._messages = [
        {"role": "user", "content": "用户A", "seq": 1},
        {"role": "assistant", "content": "助手B", "seq": 2},
        {"role": "user", "content": "用户C", "seq": 3},
        {"role": "assistant", "content": "助手D", "seq": 4},
    ]

    await oh.run_l3_daily_backfill(fake)

    assert captured_msg, "兜底应调用 L3 提取"
    # 必须取最后一条用户消息（用户C），而非最后一条助手消息（助手D）
    assert captured_msg[0] == "用户C", f"应传最后用户消息，实际传了 {captured_msg[0]!r}"
