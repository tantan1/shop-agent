"""批次 09 验收（业务级可观测 / 指标维度）转为正式测试。

迁移自 apps/gateway/_verify_2b09.py，按当前 gateway 实现核对：
- 指标支持业务维度标签（tenant/model），可按维度分桶累计（可追溯、可告警）
- /metrics 端点暴露 Prometheus exposition 文本（业务指标 pull 可见）
不发起模型请求（仅 GET /metrics）。
"""

from fastapi.testclient import TestClient

from conftest import reload_gateway, stub_proxy


def test_metrics_support_tenant_model_labels():
    """指标可按 (tenant, model) 业务维度分桶累计。"""
    metrics = reload_gateway("metrics").get("metrics")
    c = metrics.register_counter("gateway_test_biz", "biz", ("tenant", "model"))
    c.inc(labels={"tenant": "t1", "model": "gpt-4o"})
    c.inc(labels={"tenant": "t1", "model": "gpt-4o"})
    c.inc(labels={"tenant": "t2", "model": "qwen3.7-plus-2026-05-26"})
    assert c.get(labels={"tenant": "t1", "model": "gpt-4o"}) == 2
    assert c.get(labels={"tenant": "t2", "model": "qwen3.7-plus-2026-05-26"}) == 1


def test_metrics_endpoint_exposes_text():
    """/metrics 端点暴露 Prometheus exposition 文本。"""
    mods = reload_gateway("main")
    client = TestClient(mods["main"].app)
    r = client.get("/metrics")
    assert r.status_code == 200, f"/metrics 应 200，得到 {r.status_code}"
    assert "TYPE" in r.text or "# HELP" in r.text, "应暴露 Prometheus exposition 文本"
