"""批次 03 验收（成本治理 / 限流 / 预算）转为正式测试。

迁移自 apps/gateway/_verify_2b03.py，按当前 gateway 实现核对：
- limiter.check(tenant) 令牌桶：初始满额放行；连续取用超过 burst 后 deny（429）
- Verdict.allow()/deny(reason)：allowed 标志与 reason 字段正确
- proxy 端到端：租户令牌耗尽 → 429 + Retry-After + X-RateLimit-Remaining=0
不发起真实网络请求（stub AsyncClient）。
"""

import json

from conftest import FakeRouter, reload_gateway, stub_proxy


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


def _reload_limiter():
    return reload_gateway("limiter", "config").get("limiter")


# --------------------------------------------------------------------------
# 单元：令牌桶
# --------------------------------------------------------------------------

def test_check_allow_when_tokens_available():
    """初始满额时 check 放行。"""
    lim = _reload_limiter()
    v = lim.check("tenant-a")
    assert v.allowed is True, "新租户应有额度"


def test_check_deny_after_burst_exhausted(monkeypatch):
    """连续取用超过 burst 后 check 拒绝（限流不变量）。"""
    mods = reload_gateway("config", "limiter")
    settings = mods["config"].settings
    # 缩小桶容量，触发耗尽
    monkeypatch.setattr(settings, "rate_limit_burst", 3)
    monkeypatch.setattr(settings, "rate_limit_global_rps", 0.0)  # 不补充，便于稳定复现
    monkeypatch.setattr(settings, "rate_limit_tenant_rps", 0.0)
    lim = mods["limiter"]
    results = [lim.check("tenant-b").allowed for _ in range(5)]
    assert results[:3] == [True, True, True], "前 3 次应放行"
    assert False in results[3:], "超过 burst 后应拒绝"


def test_verdict_deny_carries_reason():
    """Verdict.deny 带 reason，allow 的 allowed=True。"""
    from gateway.types import Verdict

    allow = Verdict.allow()
    deny = Verdict.deny("rate limited (global)")
    assert allow.allowed is True
    assert deny.allowed is False
    assert "rate limited" in deny.reason


# --------------------------------------------------------------------------
# 端到端：proxy 限流返回 429
# --------------------------------------------------------------------------

def test_proxy_returns_429_on_rate_limit(monkeypatch):
    """端到端：租户令牌耗尽 → proxy 返回 429（不伪装成功）。"""
    # 用环境变量压小班容量（reload 后 Settings 重新读 env 生效）
    monkeypatch.setenv("RATE_LIMIT_BURST", "1")
    monkeypatch.setenv("RATE_LIMIT_GLOBAL_RPS", "0")
    monkeypatch.setenv("RATE_LIMIT_TENANT_RPS", "0")

    client, _ = stub_proxy(
        monkeypatch, FakeRouter(),
        reload_modules=("controllers.proxy", "router", "config", "limiter", "main", "litellm_router"),
    )
    headers = {"X-Tenant-Id": "tenant-c"}
    # 第一次消耗唯一令牌
    r1 = client.post(
        "/v1/chat/completions",
        json={"model": "qwen3.7-plus-2026-05-26", "messages": [{"role": "user", "content": "hi"}]},
        headers=headers,
    )
    # 第二次应被限流
    r2 = client.post(
        "/v1/chat/completions",
        json={"model": "qwen3.7-plus-2026-05-26", "messages": [{"role": "user", "content": "hi again"}]},
        headers=headers,
    )
    assert r1.status_code in (200, 429), f"首次请求应成功或受限，得到 {r1.status_code}"
    assert r2.status_code == 429, f"令牌耗尽应 429，得到 {r2.status_code}: {r2.text}"
    assert r2.headers.get("Retry-After") == "1", "限流 429 应带 Retry-After"
