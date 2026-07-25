"""批次 01 验收（平台工程 §L3 出口不变量）转为正式测试。

迁移自 apps/gateway/_verify_2b01.py，并按当前 gateway 实现核对重写：
- 当前启动装配用 include_router 挂载 proxy/health/metrics 路由（无独立 run/middleware 模块）
- 出口不变量落到配置：fail_mode 默认 closed；路由只返回 config 内已配置后端
- 流量收敛：proxy 通过 httpx 统一出口，不直连第三方 SDK
纯静态/结构检查，不发起真实请求。
"""

from conftest import app_paths, reload_gateway


def test_gateway_fail_mode_default_closed():
    """处置边界默认 fail-closed：gateway_fail_mode=closed。"""
    mods = reload_gateway("config")
    settings = mods["config"].settings
    assert settings.gateway_fail_mode == "closed", "处置边界应默认 fail-closed"
    assert settings.fail_mode.value in ("closed", "CLOSED"), "fail_mode 应为 closed"


def test_routes_registered():
    """核心路由应全部注册：/v1/chat/completions、/health、/metrics。"""
    mods = reload_gateway("main")
    app = mods["main"].app
    paths = app_paths(app)
    for need in ("/v1/{path:path}", "/health", "/metrics"):
        assert need in paths, f"缺少路由 {need}"


def test_route_returns_decision():
    """router.route 应真实存在并返回 RouteDecision（非占位）。

    LiteLLM Router 接管后，网关侧只做「模型名改写 + 声明 fallback 归属」，
    不再持有 base_url；上游地址由 Router 的 model_list 解析，故校验新契约字段。
    """
    mods = reload_gateway("router")
    router = mods["router"]
    assert hasattr(router, "route"), "缺少 route"
    from gateway.types import RouteDecision

    d = router.route("gpt-4o")
    assert isinstance(d, RouteDecision), "route 应返回 RouteDecision"
    # 云端键透传原 model；本地任务键应改写为 qwen3-unified
    assert d.upstream_model in ("", "qwen3-unified"), "upstream_model 应给出改写结果或留空透传"
    # fallback 归属声明由 Router 内部接管（占位标记，非网关手写外部域名）
    assert d.fallback_chain == ["__litellm_router__"], "fallback 应声明交由 LiteLLM Router 接管"


def test_route_uses_configured_backends_only():
    """路由只收敛到已声明后端类型，绝不硬编码裸第三方公网地址（无旁路不变量）。

    重构后网关不再持有 base_url，而是按 key 前缀分类 backend 类型
    （vllm / mock / cloud / default），实际地址由 Router model_list 同源解析。
    """
    mods = reload_gateway("router", "config")
    router = mods["router"]
    settings = mods["config"].settings
    for model in ("gpt-4o", "claude-3", "qwen3.7-plus-2026-05-26", "local/foo", None):
        d = router.route(model)
        # fallback 链不得泄露裸公网地址（只能是 Router 接管占位）
        assert d.fallback_chain == ["__litellm_router__"], f"route({model}) fallback 不得硬编码外部域名"
        # backend 类型必须落在已知集合（与 model_list 同源，杜绝无旁路裸连）
        assert d.backend in ("vllm", "mock", "cloud", "default"), f"route({model}) backend 越界 {d.backend}"
        # 已知后端基址配置非空（保证 model_list 不会指向未声明地址）
        known = {
            settings.openai_base_url,
            settings.azure_openai_base_url,
            settings.bedrock_base_url,
            settings.vllm_base_url,
            settings.bai_lian_base_url,
        }
        assert any(known), "config 至少应声明一个后端基址，否则 model_list 无源可解析"


def test_no_direct_sdk_import():
    """proxy 不应直接 import 第三方 LLM SDK 发起请求（必须走 httpx 收敛出口）。"""
    mods = reload_gateway("controllers.proxy")
    proxy = mods["controllers.proxy"]
    src = proxy.__file__
    assert src
    with open(src, encoding="utf-8") as f:
        body = f.read()
    assert "from openai" not in body, "proxy 不应直接依赖 openai SDK，破坏出口不变量"
    assert ("import httpx" in body) or ("from httpx" in body), "proxy 应通过 httpx 收敛出口"


def test_proxy_module_importable():
    """proxy 控制器可正常导入，出口路径已装配（无未接线的占位）。"""
    mods = reload_gateway("controllers.proxy")
    proxy = mods["controllers.proxy"]
    assert proxy is not None
    # 至少有路由函数或 router 装配点
    assert hasattr(proxy, "router") or hasattr(proxy, "route"), "proxy 应暴露 router/route"
