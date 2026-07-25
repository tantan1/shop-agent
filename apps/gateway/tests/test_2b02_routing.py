"""批次 02 验收（路由 + 故障转移 fallback）转为正式测试。

迁移自 apps/gateway/_verify_2b02.py，按当前 gateway 实现（LiteLLM Router 接管调度）核对：
- router.route(model) 返回 RouteDecision（backend / upstream_model / fallback_chain 占位）
- 本地任务键（local/*、tool_select、param、qwen3-unified）→ backend="vllm"，upstream_model 改写 qwen3-unified
- 云端键（gpt-*/claude*/qwen*/mock*）→ backend 分类，upstream_model 留空（原样透传 Router）
- fallback_chain 统一占位 ["__litellm_router__"]，声明故障转移由 Router 内部接管（02 §4）
- mock_cloud_override 改在 Router model_list 层生效（覆盖 litellm_router._apply_mock_override）
- 端到端：经 FakeRouter 验证 fail-closed（上游失败→503，不伪装成功）、X-Fallback 证据头回填
不发起真实网络请求（FakeRouter 注入 llm_router）。
"""

from conftest import FakeRouter, reload_gateway, stub_proxy


# --------------------------------------------------------------------------
# 单元：路由决策（网关侧，仅决策 + 改写）
# --------------------------------------------------------------------------

def test_route_local_tasks_to_vllm():
    """local/* 与 tool_select/param 及 qwen3-unified → vLLM，统一上游模型 qwen3-unified。"""
    mods = reload_gateway("router", "config")
    for name in ("local/foo", "local/qwen3", "tool_select", "param", "qwen3-unified"):
        d = mods["router"].route(name)
        assert d.backend == "vllm", f"{name} 应路由 vLLM（不得走 ollama）"
        assert d.fallback_chain == ["__litellm_router__"], f"{name} 应声明 fallback 归 Router"
        assert d.upstream_model == "qwen3-unified", f"{name} 上游模型名应为 qwen3-unified"


def test_route_cloud_keys_classified():
    """gpt-*/claude*/qwen*/mock* → backend 分类，upstream_model 留空（原样透传 Router）。"""
    mods = reload_gateway("router", "config")
    cases = {
        "gpt-4o": "cloud",
        "claude-3": "cloud",
        "qwen3.7-plus-2026-05-26": "cloud",
        "mock-llm": "mock",
        "mock/1": "mock",
    }
    for name, want in cases.items():
        d = mods["router"].route(name)
        assert d.backend == want, f"{name} backend 应为 {want}，得到 {d.backend}"
        assert d.upstream_model == "", f"{name} 不应改写上游模型名"
        assert d.fallback_chain == ["__litellm_router__"]


def test_route_default_fallback_placeholder():
    """未知模型 → default backend，fallback 占位声明。"""
    mods = reload_gateway("router", "config")
    d = mods["router"].route("some-unknown-model")
    assert d.backend == "default"
    assert d.fallback_chain == ["__litellm_router__"]


def test_mock_override_applies_to_model_list(monkeypatch):
    """压测开关 MOCK_CLOUD_OVERRIDE=1：云端 deployment 在 Router model_list 层切到 mock。

    校验 litellm_router._apply_mock_override 行为（route() 本身不受影响，本地仍 vLLM）。
    """
    monkeypatch.setenv("MOCK_CLOUD_OVERRIDE", "1")
    monkeypatch.setenv("MOCK_BASE_URL", "http://mock-cloud:0/v1")
    mods = reload_gateway("config", "litellm_router")
    config = mods["config"].settings
    lr = mods["litellm_router"]

    base_list = config.build_model_list()
    overridden = lr._apply_mock_override(base_list)
    # 云端 deployment 的 api_base 应改为 mock_base_url，本地 vLLM/mock 不变
    cloud_bases = {
        d["litellm_params"].get("api_base")
        for d in overridden
        if d["model_name"] in ("gpt-*", "claude*", "qwen*", "default")
    }
    assert config.mock_base_url in cloud_bases, "云端 deployment 应切到 mock_base_url"
    vllm_bases = {
        d["litellm_params"].get("api_base")
        for d in overridden
        if d["model_name"] in ("qwen3-unified", "tool_select", "param")
    }
    assert vllm_bases == {config.vllm_base_url}, "本地 vLLM 不应被压测开关劫持"


# --------------------------------------------------------------------------
# 端到端：用 FakeRouter 注入 llm_router（无真实网络）
# --------------------------------------------------------------------------

def test_no_fallback_when_primary_ok(monkeypatch):
    """主后端可用时不触发 fallback，X-Fallback=false。"""
    fake = FakeRouter(seq=[200], used_fallback=False)
    client, _ = stub_proxy(monkeypatch, fake)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen3.7-plus-2026-05-26", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, f"应 200，得到 {r.status_code}: {r.text}"
    assert r.headers.get("X-Fallback") == "false", "主后端可用不应 fallback"
    assert r.headers.get("X-Upstream-Model") == "qwen3.7-plus-2026-05-26"


def test_vllm_upstream_model_rewritten(monkeypatch):
    """vLLM 后端：发往 Router 的模型名改写为 qwen3-unified（证据头反映实际上游名）。"""
    fake = FakeRouter(seq=[200], used_fallback=False)
    client, _ = stub_proxy(monkeypatch, fake)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "param", "messages": [{"role": "user", "content": "rewrite check"}]},
    )
    assert r.status_code == 200, f"应 200，得到 {r.status_code}: {r.text}"
    assert r.headers.get("X-Upstream-Model") == "qwen3-unified", "证据头应反映实际上游模型名"
    assert fake.last_model == "qwen3-unified", "发往 Router 的 model 应为 qwen3-unified"


def test_fail_closed_when_all_upstreams_down(monkeypatch):
    """上游全部失败 → fail-closed，不伪装成功（返回 503）。"""
    fake = FakeRouter(seq=[500, 500])
    client, _ = stub_proxy(monkeypatch, fake)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503, f"全上游故障应网关 503，得到 {r.status_code}: {r.text}"
    assert r.json().get("error") == "all backends unavailable", "应返回网关级失败信号"


def test_fail_closed_single_backend_5xx(monkeypatch):
    """单后端返回 5xx → 网关级 503，不误报 X-Fallback。"""
    fake = FakeRouter(seq=[500])
    client, _ = stub_proxy(monkeypatch, fake)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "local/foo", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503, f"唯一上游故障应网关 503，得到 {r.status_code}"
    assert r.headers.get("X-Fallback") == "false", "单后端无 fallback，不应标记 X-Fallback=true"


def test_fallback_evidence_header(monkeypatch):
    """Router 实际发生 fallback 时，X-Fallback=true 回填。"""
    fake = FakeRouter(seq=[200], used_fallback=True)
    client, _ = stub_proxy(monkeypatch, fake)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.headers.get("X-Fallback") == "true", "Router 内部 fallback 应回填 X-Fallback=true"
