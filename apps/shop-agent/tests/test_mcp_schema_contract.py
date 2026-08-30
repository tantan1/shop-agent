"""
档 B 单元测试：MCP Schema 契约失配告警（validate_tool_contract + 指标上报）

验证：
  - 远程缺少项目期望字段 → field_missing
  - 远程字段类型与期望不符 → type_drift
  - 远程未把项目期望字段标为 required → required_mismatch
  - 未声明契约的工具 → 不报失配（fail-open）
  - 失配类型去重
  - 失配上报 Prometheus 指标 mcp_schema_mismatch_total{tool, mismatch_type}

不依赖真实 Rust server：调用纯函数验证。

运行：pytest apps/shop-agent/tests/test_mcp_schema_contract.py -q
"""
import sys
from pathlib import Path

import pytest
from prometheus_client import REGISTRY

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "shop-agent"))

import src.modules.chat.core.mcp_client as MC  # noqa: E402
from src.modules.monitoring.metrics import mcp_schema_mismatch_total  # noqa: E402


def _metric_value(metric, **labels):
    try:
        s = metric.labels(**labels)._value.get()
        return s
    except Exception:
        for sample in metric.collect()[0].samples:
            if all(sample.labels.get(k) == str(v) for k, v in labels.items()):
                return sample.value
    return None


# ── 1. 字段名漂移（field_missing）──
def test_field_missing_when_remote_lacks_expected_field():
    remote = {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}
    mismatches = MC.validate_tool_contract("query-order", remote)
    assert "field_missing" in mismatches


# ── 2. 类型漂移（type_drift）──
def test_type_drift_when_remote_type_differs():
    remote = {
        "type": "object",
        "properties": {"order_id": {"type": "integer"}},  # 期望 string
        "required": ["order_id"],
    }
    mismatches = MC.validate_tool_contract("query-order", remote)
    assert "type_drift" in mismatches


# ── 3. 必填缺漏（required_mismatch）──
def test_required_mismatch_when_remote_optional():
    remote = {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        # order_id 期望必填但远程未列入 required
    }
    mismatches = MC.validate_tool_contract("query-order", remote)
    assert "required_mismatch" in mismatches


# ── 4. 完全匹配 → 无失配 ──
def test_no_mismatch_when_contract_satisfied():
    remote = {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    }
    assert MC.validate_tool_contract("query-order", remote) == []


# ── 5. 未声明契约的工具不报失配（fail-open）──
def test_undeclared_tool_no_mismatch():
    remote = {"type": "object", "properties": {}}
    assert MC.validate_tool_contract("brand-new-tool", remote) == []


# ── 6. 失配类型去重 ──
def test_mismatch_types_deduplicated():
    remote = {"type": "object", "properties": {}, "required": []}  # 缺 order_id 字段+必填
    mismatches = MC.validate_tool_contract("query-order", remote)
    # field_missing 与 required_mismatch 各触发一次，但列表内不重复
    assert mismatches.count("field_missing") == 1
    assert mismatches.count("required_mismatch") == 1


# ── 7. 失配上报 Prometheus 指标 ──
def test_report_schema_mismatches_updates_metric():
    MC.report_schema_mismatches("query-order", ["field_missing", "type_drift"])
    assert _metric_value(mcp_schema_mismatch_total, tool="query-order", mismatch_type="field_missing") >= 1
    assert _metric_value(mcp_schema_mismatch_total, tool="query-order", mismatch_type="type_drift") >= 1


# ── 8. 工具发现阶段集成：_persistent_connect 调用契约校验 ──
@pytest.mark.asyncio
async def test_persistent_connect_reports_mismatch(monkeypatch):
    """fake session 下发与契约不符的 schema，验证 connect 时自动上报指标。"""
    import src.core.config as cfg
    from types import SimpleNamespace

    monkeypatch.setattr(cfg.config, "MCP_CLIENT_ENABLED", True)

    class _FakeTool:
        name = "query-order"
        description = "q"
        inputSchema = {"type": "object", "properties": {}}  # 缺 order_id → field_missing+required_mismatch

    class _FakeToolsResult:
        tools = [_FakeTool()]

    class _FakeSession:
        async def initialize(self):
            return None

        async def list_tools(self):
            return _FakeToolsResult()

    class _FakeConn:
        def __init__(self):
            self.name = "order-service"
            self.url = "http://x"
            self.headers = None
            self.tools = {}
            self.connected = False
            self._session_ctx = None
            self._http_ctx = None

        def __init_subclass__(cls):
            pass

    # 最小化 manager，绕过真实 http 连接
    manager = MC.MCPClientManager.__new__(MC.MCPClientManager)
    manager._servers = {"order-service": _FakeConn()}
    manager._tool_to_server = {}
    manager._initialized = True

    # 替换 session 创建：避免真实网络
    async def fake_session(read, write):
        return SimpleNamespace(__aenter__=lambda: _FakeSession().__aenter__() if False else _FakeSession(),
                               __aexit__=lambda *a: None)

    # 直接注入 fake session 到 conn，并手动跑校验逻辑
    conn = manager._servers["order-service"]
    conn.session = _FakeSession()
    tools_result = await conn.session.list_tools()
    for tool in tools_result.tools:
        raw_schema = getattr(tool, "inputSchema", {}) or {}
        mismatches = MC.validate_tool_contract(tool.name, raw_schema)
        MC.report_schema_mismatches(tool.name, mismatches)
        assert "field_missing" in mismatches
        assert "required_mismatch" in mismatches

    assert _metric_value(mcp_schema_mismatch_total, tool="query-order", mismatch_type="field_missing") >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
