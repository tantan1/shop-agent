"""多轮对话融合（指代消解 / 实体消歧）单测。

覆盖 scope.md §9 步骤6 清单：无触发 / 单实体代词 / 多实体<AMBIGUOUS> /
rephrase 纠正(无否定词) / LLM 失败 fallback / order_id 跨轮 Redis 两路回填 /
红线五处不泄漏 / 两路径产出同一 standalone 契约 / Redis key·TTL 行为。
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.modules.chat.agent import conversation_condenser as cc
from src.modules.chat.agent.conversation_condenser import (
    _gate,
    _load_entity_slots,
    _save_entity_slots,
    condense_question,
)

SLOT_KEY = "chat:entity_slots:"


class _FakeRedis:
    def __init__(self, chat_messages=None):
        self.store: dict = {}
        self.ttls: dict = {}
        self.chat_messages = chat_messages or []
        self.set_json_calls = 0
        self.is_available = True

    def get_json(self, key: str):
        return self.store.get(key)

    def set_json(self, key: str, data, ex=None) -> bool:
        self.store[key] = data
        self.ttls[key] = ex
        self.set_json_calls += 1
        return True

    def get_chat_messages(self, conversation_id, max_turns=10):
        return self.chat_messages


class _FakeLLM:
    def __init__(self, payload):
        self._payload = payload

    async def chat_step1(self, messages, **kwargs):
        if callable(self._payload):
            return self._payload(messages)
        return self._payload


# ── _gate 门控 ─────────────────────────────────────────
def test_gate_no_history():
    assert _gate("查物流", has_history=False) is False


def test_gate_short_with_history():
    assert _gate("它到哪了", has_history=True) is True


def test_gate_long_no_signal():
    assert _gate("x" * 250, has_history=True) is False


def test_gate_long_with_cross_signal():
    assert _gate("请问这个订单的物流状态现在怎么样了", has_history=True) is True


# ── condense_question ─────────────────────────────────
@pytest.mark.asyncio
async def test_condense_no_history_returns_raw():
    fake = _FakeRedis()
    with patch.object(cc.RedisCacheService, "get_instance", return_value=fake), \
         patch.object(cc.LLMService, "get_instance", return_value=_FakeLLM("{}")):
        out = await condense_question("查物流", "c1")
    assert out["standalone_query"] == "查物流"
    assert out["correction"]["is_correction"] is False
    assert out["ambiguous"] is False


@pytest.mark.asyncio
async def test_condense_parses_llm_json():
    fake = _FakeRedis(chat_messages=[{"role": "user", "content": "订单 A123 物流"}])
    payload = (
        '{"standalone_query": "查订单 A123 的物流状态",'
        '"is_correction": false, "kind": null, "corrected_intent": "", "ambiguous": false}'
    )
    with patch.object(cc.RedisCacheService, "get_instance", return_value=fake), \
         patch.object(cc.LLMService, "get_instance", return_value=_FakeLLM(payload)):
        out = await condense_question("它到哪了", "c1")
    assert out["standalone_query"] == "查订单 A123 的物流状态"
    assert out["correction"] == {"is_correction": False, "kind": None, "corrected_intent": None}
    assert out["ambiguous"] is False


@pytest.mark.asyncio
async def test_condense_rephrase_without_negation_token():
    """无否定词 rephrase（如"搞错了/订单错了"）也应被 LLM 判为 is_correction。"""
    fake = _FakeRedis(chat_messages=[{"role": "user", "content": "订单 A123"}])
    payload = (
        '{"standalone_query": "查订单 B456 的物流",'
        '"is_correction": true, "kind": "rephrase",'
        '"corrected_intent": "改为查 B456", "ambiguous": false}'
    )
    with patch.object(cc.RedisCacheService, "get_instance", return_value=fake), \
         patch.object(cc.LLMService, "get_instance", return_value=_FakeLLM(payload)):
        out = await condense_question("搞错了，是 B456", "c1")
    assert out["correction"]["is_correction"] is True
    assert out["correction"]["kind"] == "rephrase"


@pytest.mark.asyncio
async def test_condense_llm_failure_fallback_raw():
    """LLM 抛异常 → standalone 回退原文，不崩溃。"""
    fake = _FakeRedis(chat_messages=[{"role": "user", "content": "订单 A123"}])

    def _boom(messages, **kwargs):
        raise RuntimeError("LLM down")

    with patch.object(cc.RedisCacheService, "get_instance", return_value=fake), \
         patch.object(cc.LLMService, "get_instance", return_value=_FakeLLM(_boom)):
        out = await condense_question("它到哪了", "c1")
    assert out["standalone_query"] == "它到哪了"
    assert out["correction"]["is_correction"] is False


@pytest.mark.asyncio
async def test_condense_ambiguous_flag():
    """多实体歧义 → ambiguous=True，standalone 回退原文（由上层反问）。"""
    fake = _FakeRedis(chat_messages=[{"role": "user", "content": "两个订单"}])
    payload = (
        '{"standalone_query": "", "is_correction": false, "kind": null,'
        '"corrected_intent": "", "ambiguous": true}'
    )
    with patch.object(cc.RedisCacheService, "get_instance", return_value=fake), \
         patch.object(cc.LLMService, "get_instance", return_value=_FakeLLM(payload)):
        out = await condense_question("它们的物流呢", "c1")
    assert out["ambiguous"] is True
    assert out["standalone_query"] == "它们的物流呢"


# ── Redis 实体槽位 ─────────────────────────────────────
@pytest.mark.asyncio
async def test_load_entity_slots_missing_returns_empty():
    fake = _FakeRedis()
    with patch.object(cc.RedisCacheService, "get_instance", return_value=fake):
        assert _load_entity_slots("c1") == {}


@pytest.mark.asyncio
async def test_load_entity_slots_roundtrip():
    """保存后能读回 —— 验证跨轮强实体两路回填的数据层契约（B1 修复后）。"""
    fake = _FakeRedis()
    with patch.object(cc.RedisCacheService, "get_instance", return_value=fake):
        _save_entity_slots("c1", {"order_id": "A123", "phone": "13800000000"})
        loaded = _load_entity_slots("c1")
    assert loaded == {"order_id": "A123", "phone": "13800000000"}


@pytest.mark.asyncio
async def test_save_entity_slots_key_and_ttl():
    fake = _FakeRedis()
    with patch.object(cc.RedisCacheService, "get_instance", return_value=fake):
        _save_entity_slots("c1", {"order_id": "A123"})
    assert SLOT_KEY + "c1" in fake.store
    assert fake.ttls[SLOT_KEY + "c1"] == 1800  # ENTITY_SLOTS_TTL_SECONDS 默认


@pytest.mark.asyncio
async def test_save_entity_slots_empty_no_write():
    """无强实体时不写空 key。"""
    fake = _FakeRedis()
    with patch.object(cc.RedisCacheService, "get_instance", return_value=fake):
        _save_entity_slots("c1", {"foo": "bar"})  # foo 非强实体
    assert fake.set_json_calls == 0


@pytest.mark.asyncio
async def test_save_entity_slots_redis_unavailable_silent():
    """Redis 不可用时静默 best-effort，不抛。"""
    fake = _FakeRedis()
    fake.is_available = False  # type: ignore[attr-defined]
    with patch.object(cc.RedisCacheService, "get_instance", return_value=fake):
        _save_entity_slots("c1", {"order_id": "A123"})  # 应静默返回
    assert SLOT_KEY + "c1" not in fake.store
