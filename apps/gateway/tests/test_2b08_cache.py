"""批次 08 验收（语义缓存）转为正式测试。

迁移自 apps/gateway/_verify_2b08.py，按当前 gateway 实现核对：
- hooks.cache_store(stage, payload, answer) 写入；同 prompt 经 cache_lookup 命中返回答案（短路）
- 不同 prompt 不命中（返回 None）
- cache_enabled=False 时 lookup 直接返回 None（缓存可关闭，不引入 SPOF）
不发起真实网络请求。
"""

import asyncio

from conftest import reload_gateway


def _hooks():
    mod = reload_gateway("config", "hooks.governance", "cache.store").get("hooks.governance")
    return mod.hooks


def _payload(text):
    return {"model": "qwen3.7-plus-2026-05-26", "messages": [{"role": "user", "content": text}]}


def test_cache_store_then_lookup_hit():
    """写入后同 prompt 命中，返回缓存答案（语义缓存短路）。"""
    hooks = _hooks()
    p = _payload("今天北京天气如何")
    asyncio.run(hooks.cache_store("egress", p, "晴，20度"))
    hit = asyncio.run(hooks.cache_lookup("ingress", p))
    assert hit == "晴，20度", "同 prompt 应命中缓存"


def test_cache_miss_on_different_prompt():
    """不同 prompt 不命中，返回 None。"""
    hooks = _hooks()
    p1 = _payload("今天北京天气如何")
    asyncio.run(hooks.cache_store("egress", p1, "晴"))
    p2 = _payload("上海今天天气如何")
    miss = asyncio.run(hooks.cache_lookup("ingress", p2))
    assert miss is None, "不同 prompt 不应命中"


def test_cache_disabled_returns_none(monkeypatch):
    """cache_enabled=False 时 lookup 直接返回 None（缓存可关闭）。"""
    monkeypatch.setenv("CACHE_ENABLED", "false")
    hooks = _hooks()  # reload 后读取 CACHE_ENABLED=false
    p = _payload("任意问题")
    asyncio.run(hooks.cache_store("egress", p, "答案"))
    # 即使写入，关闭状态下 lookup 也不应短路返回缓存
    assert asyncio.run(hooks.cache_lookup("ingress", p)) is None
