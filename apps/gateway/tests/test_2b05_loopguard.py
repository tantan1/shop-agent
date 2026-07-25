"""批次 05 验收（失控循环防护）转为正式测试。

迁移自 apps/gateway/_verify_2b05.py，按当前 gateway 实现核对：
- loopguard.check(tenant, model, prompt) 基于指纹（tenant+model+归一化 prompt）滑动窗计数
- 同指纹窗口内超 loop_guard_max → deny（429）；reset() 清空状态
- proxy 端到端：同一请求重复超限 → 429（防 Agent 自我循环烧钱）
不发起真实网络请求（stub AsyncClient）。
"""

import json

from conftest import reload_gateway, stub_proxy


class _OkClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def request(self, method, url, **kwargs):
        return _Resp(200, json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode())


class _Resp:
    def __init__(self, status_code, content, headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    async def aiter_bytes(self):
        yield self.content


# --------------------------------------------------------------------------
# 单元：循环检测
# --------------------------------------------------------------------------

def test_loopguard_allows_below_threshold(monkeypatch):
    """低于阈值 Repeated 相同请求应放行。"""
    mods = reload_gateway("config", "loopguard")
    settings = mods["config"].settings
    monkeypatch.setattr(settings, "loop_guard_max", 3)
    monkeypatch.setattr(settings, "loop_guard_window_sec", 60)
    lg = mods["loopguard"]
    lg.reset()
    for _ in range(3):
        v = lg.check("t", "gpt-4o", b"same prompt")
        assert v.allowed is True


def test_loopguard_denies_above_threshold(monkeypatch):
    """超过阈值后同指纹请求被 deny。"""
    mods = reload_gateway("config", "loopguard")
    settings = mods["config"].settings
    monkeypatch.setattr(settings, "loop_guard_max", 2)
    monkeypatch.setattr(settings, "loop_guard_window_sec", 60)
    lg = mods["loopguard"]
    lg.reset()
    assert lg.check("t", "gpt-4o", b"same prompt").allowed is True
    assert lg.check("t", "gpt-4o", b"same prompt").allowed is True
    v = lg.check("t", "gpt-4o", b"same prompt")
    assert v.allowed is False, "超阈值应 deny"
    assert "loop" in v.reason.lower()


def test_loopguard_reset_clears_state(monkeypatch):
    """reset() 后重新计数，重新放行。"""
    mods = reload_gateway("config", "loopguard")
    settings = mods["config"].settings
    monkeypatch.setattr(settings, "loop_guard_max", 1)
    monkeypatch.setattr(settings, "loop_guard_window_sec", 60)
    lg = mods["loopguard"]
    lg.reset()
    lg.check("t", "gpt-4o", b"same")
    assert lg.check("t", "gpt-4o", b"same").allowed is False
    lg.reset()
    assert lg.check("t", "gpt-4o", b"same").allowed is True


# --------------------------------------------------------------------------
# 端到端：proxy 返回 429
# --------------------------------------------------------------------------

def test_proxy_429_on_loop_guard(monkeypatch):
    """端到端：相同请求重复超限 → proxy 返回 429。"""
    monkeypatch.setenv("LOOP_GUARD_MAX", "1")
    monkeypatch.setenv("LOOP_GUARD_WINDOW_SEC", "60")
    client, _ = stub_proxy(
        monkeypatch, _OkClient,
        reload_modules=("controllers.proxy", "router", "config", "loopguard", "main"),
    )
    headers = {"X-Tenant-Id": "t-loop"}
    payload = {"model": "qwen3.7-plus-2026-05-26", "messages": [{"role": "user", "content": "repeat me"}]}
    r1 = client.post("/v1/chat/completions", json=payload, headers=headers)
    r2 = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert r1.status_code in (200, 429)
    assert r2.status_code == 429, f"失控循环应 429，得到 {r2.status_code}: {r2.text}"
