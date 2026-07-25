"""批次 2b-12：LiteLLM Router 适配层单元测试（gateway/litellm_router.py）。

聚焦嵌入 LiteLLM Router 后的网关侧契约，不依赖真实上游：
  - model_list 生成：YAML 驱动（权威）与 legacy 字段回退两条路径；
  - mock_cloud_override=1：云端 deployment 整体改路由 mock 上游（压测零真实云调用）；
  - router_settings：routing_strategy / num_retries / timeout 透传到底层 Router；
  - failover：acompletion 遇上游异常统一抛 GovernanceError，由 proxy 按 fail_mode 分流；
  - X-Fallback 证据头：used_fallback 经 acompletion 回传给 proxy 写响应头。

FakeRouter 复用 conftest 协议：(dict, model, used_fallback) / async generator of (chunk, model)。
"""

from __future__ import annotations

import importlib
import os
import tempfile

import pytest

from conftest import FakeRouter, reload_gateway, stub_proxy


# --------------------------------------------------------------------------
# 单元：model_list 生成（legacy 回退）
# --------------------------------------------------------------------------

def test_legacy_model_list_uses_explicit_provider():
    """legacy build_model_list 必须为裸模型名显式标 custom_llm_provider，
    避免 LiteLLM 对 qwen3-unified 等未知名做 provider 推断失败（生产级默认回退路径）。"""
    mods = reload_gateway("config")
    settings = mods["config"].settings
    model_list = settings.build_model_list()

    by_name = {}
    for dep in model_list:
        by_name.setdefault(dep["model_name"], []).append(dep["litellm_params"])

    # 本地统一模型以 openai 兼容 vLLM 提供
    assert any(
        p.get("custom_llm_provider") == "openai" and p.get("model") == "qwen3-unified"
        for p in by_name.get("qwen3-unified", [])
    ), "qwen3-unified 应以 openai provider 走 vLLM"
    # gpt-* 含 azure 主 + openai 备选两条 deployment（故障转移）
    gpt_deps = by_name.get("gpt-*", [])
    providers = {p.get("custom_llm_provider") for p in gpt_deps}
    assert {"azure", "openai"} <= providers, "gpt-* 应含 azure+openai 双 deployment 作故障转移"


def test_legacy_model_list_has_glob_names():
    """model_list 的 model_name 使用 glob（gpt-*/claude*/qwen*/local/*/mock*），
    decide() 的 fnmatch 据此判定注册，避免逐模型硬编码。"""
    mods = reload_gateway("config")
    settings = mods["config"].settings
    names = {dep["model_name"] for dep in settings.build_model_list()}
    for glob in ("gpt-*", "claude*", "qwen*", "local/*", "mock*"):
        assert glob in names, f"model_list 应含 glob 名 {glob}"


# --------------------------------------------------------------------------
# 单元：mock_cloud_override 劫持云端 deployment
# --------------------------------------------------------------------------

def test_mock_cloud_override_rewrites_cloud_deployments():
    """mock_cloud_override=1：云端 deployment 改路由 mock 上游，本地 vLLM 与 mock 自身不被劫持。"""
    mods = reload_gateway("config", "litellm_router")
    settings = mods["config"].settings
    lr = mods["litellm_router"]

    settings.mock_cloud_override = True
    model_list = settings.build_model_list()
    out = lr._apply_mock_override(model_list)

    cloud = [d for d in out if d["model_name"] in ("gpt-*", "claude*", "qwen*", "default")]
    local = [d for d in out if d["model_name"] in ("qwen3-unified", "local/*", "mock", "mock*")]
    assert cloud, "应有云端 deployment 被劫持"
    for d in cloud:
        p = d["litellm_params"]
        assert p["api_base"] == settings.mock_base_url, "云端应被改写到 mock_base_url"
        assert p["custom_llm_provider"] == "openai"
        assert p["model"] == "mock"
        assert "api_key" not in p, "mock 上游不应带云端 api_key"
    for d in local:
        p = d["litellm_params"]
        # 本地 vLLM 不被劫持：仍指向 vllm_base_url
        # mock / mock* 本就指向 mock_base_url，且不被 override 改写（model 仍为 mock）
        if d["model_name"] == "qwen3-unified" or d["model_name"] == "local/*":
            assert p["api_base"] == settings.vllm_base_url, "本地 vLLM 不应被劫持到 mock"
        assert p["model"] == "mock" or p["model"] == "qwen3-unified", \
            "本地模型不应被 override 改写模型名"


def test_no_mock_override_keeps_cloud_intact():
    """mock_cloud_override=0：云端 deployment 保持原配置（azure/openai/bedrock/百炼）。"""
    mods = reload_gateway("config", "litellm_router")
    settings = mods["config"].settings
    lr = mods["litellm_router"]

    settings.mock_cloud_override = False
    model_list = settings.build_model_list()
    out = lr._apply_mock_override(model_list)
    gpt = [d for d in out if d["model_name"] == "gpt-*"]
    assert any(d["litellm_params"]["api_base"] == settings.azure_openai_base_url for d in gpt), \
        "未开 override 时 gpt-* 仍含 azure 主后端"


# --------------------------------------------------------------------------
# 单元：YAML 驱动 model_list 为权威来源
# --------------------------------------------------------------------------

def test_yaml_config_path_drives_model_list(monkeypatch):
    """配 litellm_config_path 时，build() 读 YAML 的 model_list + router_settings，
    而非 legacy 字段（生产级权威来源）。"""
    mods = reload_gateway("config", "litellm_router")
    settings = mods["config"].settings
    lr = mods["litellm_router"]

    yaml_text = """
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: gpt-4o
      custom_llm_provider: openai
      api_base: https://example-openai/v1
      api_key: sk-env
router_settings:
  routing_strategy: least-busy
  num_retries: 5
  timeout: 30
"""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    tmp.write(yaml_text)
    tmp.close()
    monkeypatch.setenv("LITELLM_CONFIG_PATH", tmp.name)
    # 重建 settings 读取新 env，并 reload litellm_router 绑定新 settings
    settings = mods["config"].Settings()
    import gateway.config as config_mod
    monkeypatch.setattr(config_mod, "settings", settings)
    importlib.reload(lr)

    lr.llm_router.build()
    names = {d["model_name"] for d in lr.llm_router._model_list}
    assert "gpt-4o" in names, "YAML 中的 model 应被注册"
    assert lr.llm_router._routing_strategy == "least-busy"
    assert lr.llm_router._num_retries == 5
    assert lr.llm_router._timeout == 30.0


# --------------------------------------------------------------------------
# 单元：acompletion 异常包装 / failover 证据
# --------------------------------------------------------------------------

def test_acompletion_wraps_exception_as_governance_error():
    """Router 调用抛异常时，acompletion 统一包成 GovernanceError（stage=egress），
    供 proxy 按 fail_mode 分流，保留 fail-closed 不变量。"""
    from gateway.types import GovernanceError

    mods = reload_gateway("litellm_router")
    lr = mods["litellm_router"]
    # 注入一个必抛异常的 fake router
    class _Boom:
        async def acompletion(self, *a, **k):
            from litellm.exceptions import RateLimitError
            raise RateLimitError(message="boom", model="gpt-4o", llm_provider="openai")

    lr.llm_router._router = _Boom()
    import asyncio
    with pytest.raises(GovernanceError) as exc:
        asyncio.run(
            lr.llm_router.acompletion("gpt-4o", [{"role": "user", "content": "hi"}])
        )
    assert exc.value.stage == "egress"


def test_acompletion_returns_used_fallback_flag():
    """acompletion 透传 used_fallback，供 proxy 写 X-Fallback 证据头。"""
    mods = reload_gateway("litellm_router")
    lr = mods["litellm_router"]
    lr.llm_router._router = FakeRouter(used_fallback=True, content="hi")

    import asyncio
    resp, actual, used = asyncio.run(
        lr.llm_router.acompletion("gpt-4o", [{"role": "user", "content": "hi"}])
    )
    assert used is True, "FakeRouter 标记 used_fallback 应被透传"
    assert actual == "gpt-4o"
    assert resp["choices"][0]["message"]["content"] == "hi"


# --------------------------------------------------------------------------
# 端到端：X-Fallback 证据头（经由真实 proxy → llm_router.acompletion）
# --------------------------------------------------------------------------

def test_proxy_x_fallback_header_from_router(monkeypatch):
    """proxy 经 llm_router.acompletion 调用，把 used_fallback 写进 X-Fallback 响应头。"""
    client, _ = stub_proxy(
        monkeypatch, FakeRouter(used_fallback=True, content="ok"),
        reload_modules=("controllers.proxy", "router", "config", "main", "litellm_router"),
    )
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen3.7-plus-2026-05-26", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, r.text
    assert r.headers.get("X-Fallback") == "true", "used_fallback 应回写 X-Fallback: true"
    assert r.headers.get("X-Upstream-Model") == "qwen3.7-plus-2026-05-26"


def test_proxy_x_upstream_model_header(monkeypatch):
    """proxy 写 X-Upstream-Model 证据头，标明实际发往 Router 的模型名。"""
    client, _ = stub_proxy(
        monkeypatch, FakeRouter(used_fallback=False, content="ok"),
        reload_modules=("controllers.proxy", "router", "config", "main", "litellm_router"),
    )
    r = client.post(
        "/v1/chat/completions",
        json={"model": "local/foo", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, r.text
    # 本地任务键应改写为 qwen3-unified（router.route 行为）
    assert r.headers.get("X-Upstream-Model") == "qwen3-unified"
    assert r.headers.get("X-Fallback") == "false"
