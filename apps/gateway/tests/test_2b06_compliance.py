"""批次 06 验收（合规护栏 / 违禁词 blocklist）转为正式测试。

迁移自 apps/gateway/_verify_2b06.py，按当前 gateway 实现核对：
- hooks.guardrails_check(stage, text) 精确命中 GUARDRAILS_BLOCKLIST → deny
- fuzzy 边界词（词尾 (fuzzy)）→ 放行 + human_review（不阻断，E3 三通道落点）
- 无命中 → allow
- proxy 端到端：请求含违禁词 → 400（fail-closed 不静默放行）
不发起真实网络请求（stub AsyncClient）。
"""

import asyncio
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


def _reload_governance():
    mod = reload_gateway("config", "hooks.governance").get("hooks.governance")
    return mod.hooks  # GovernanceHooks 实例（模块级单例）


# --------------------------------------------------------------------------
# 单元：护栏扫描
# --------------------------------------------------------------------------

def test_guardrails_deny_on_exact_blocklist(monkeypatch):
    """精确命中违禁词 → deny。"""
    monkeypatch.setenv("GUARDRAILS_BLOCKLIST", "炸弹,违禁")
    gov = _reload_governance()
    v = asyncio.run(gov.guardrails_check("ingress", "这是炸弹内容"))
    assert v.allowed is False, "精确命中应 deny"
    assert "blocklist" in v.reason.lower() or "炸弹" in v.reason


def test_guardrails_allow_on_clean_text(monkeypatch):
    """无命中 → allow。"""
    monkeypatch.setenv("GUARDRAILS_BLOCKLIST", "炸弹")
    gov = _reload_governance()
    v = asyncio.run(gov.guardrails_check("ingress", "今天天气不错"))
    assert v.allowed is True


def test_guardrails_fuzzy_is_review_not_deny(monkeypatch):
    """fuzzy 边界词 → 放行但带 human_review（不阻断）。"""
    monkeypatch.setenv("GUARDRAILS_BLOCKLIST", "敏感词(fuzzy)")
    gov = _reload_governance()
    v = asyncio.run(gov.guardrails_check("ingress", "提到敏感词"))
    assert v.allowed is True, "fuzzy 边界应放行"
    assert v.review is not None, "fuzzy 边界应落 human_review"


# --------------------------------------------------------------------------
# 端到端：proxy 拦截违禁词
# --------------------------------------------------------------------------

def test_proxy_blocks_blocklisted_word(monkeypatch):
    """端到端：请求含违禁词 → 400（fail-closed 不静默放行）。"""
    monkeypatch.setenv("GUARDRAILS_BLOCKLIST", "炸弹")
    client, _ = stub_proxy(
        monkeypatch, _OkClient,
        reload_modules=("controllers.proxy", "router", "config", "hooks.governance", "main"),
    )
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen3.7-plus-2026-05-26", "messages": [{"role": "user", "content": "这里有炸弹"}]},
    )
    assert r.status_code == 400, f"违禁词应拦截 400，得到 {r.status_code}: {r.text}"
    body = r.json()
    assert body.get("error") == "blocked by guardrails"
