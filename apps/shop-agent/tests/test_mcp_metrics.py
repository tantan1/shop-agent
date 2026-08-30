"""
Phase 4 单元测试：MCP 可观测性（Prometheus 指标 + /health mcp 字段）

验证：
  - call_tool 成功 → mcp_call_total{tool,status=success} +1，mcp_call_duration 观测
  - call_tool 失败 → mcp_call_total{tool,status=error} +1
  - connect_all 后 mcp_sessions_active / mcp_tools_total / mcp_connection_status 被设置

不依赖真实 Rust server：用内存 fake session + monkeypatch get_mcp_client。

运行：pytest apps/shop-agent/tests/test_mcp_metrics.py -q
"""
import asyncio
import sys
from pathlib import Path

import pytest
from prometheus_client import REGISTRY

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "shop-agent"))

from src.modules.chat.core.tool_registry import McpToolCommand  # noqa: E402
import src.modules.chat.core.tool_registry as TR  # noqa: E402
import src.modules.chat.core.mcp_client as MC  # noqa: E402
from src.modules.monitoring.metrics import (  # noqa: E402
    mcp_call_total,
    mcp_call_duration,
    mcp_sessions_active,
    mcp_tools_total,
    mcp_connection_status,
)
import src.core.config as cfg  # noqa: E402


class _FakeResult:
    content = [type("T", (), {"text": '{"ok": true}'})()]


class _FakeSession:
    def __init__(self):
        self.calls = []

    async def call_tool(self, action, arguments=None, timeout=None):
        self.calls.append(action)
        if action == "boom-tool":
            raise RuntimeError("remote error")
        return _FakeResult()


class _FakeConn:
    def __init__(self, name):
        self.name = name
        self.session = _FakeSession()
        self.connected = True
        self.tools = {"query-order": None, "list-coupons": None}


@pytest.fixture
def mgr(monkeypatch):
    """构造真实 MCPClientManager，塞入 fake session，验证真实埋点逻辑。"""
    monkeypatch.setattr(cfg.config, "MCP_CLIENT_ENABLED", True)
    from src.modules.chat.core.mcp_client import MCPClientManager

    manager = MCPClientManager.__new__(MCPClientManager)
    servers = {"order-service": _FakeConn("order-service")}
    manager._servers = servers
    manager._tool_to_server = {
        "query-order": "order-service",
        "list-coupons": "order-service",
        "boom-tool": "order-service",
    }
    manager._initialized = True
    return manager


def _metric_value(metric, **labels):
    try:
        s = metric.labels(**labels)._value.get()
        return s
    except Exception:
        # 兼容不同 prometheus_client 版本的样本读取
        for sample in metric.collect()[0].samples:
            if all(sample.labels.get(k) == str(v) for k, v in labels.items()):
                return sample.value
    return None


@pytest.mark.asyncio
async def test_call_success_increments_counter_and_duration(mgr):
    await mgr.call_tool("query-order", {"order_id": "WB1"})
    # Counter 全局累积，断言 >=1（验证「被记录」而非恰好一次）
    assert _metric_value(mcp_call_total, tool="query-order", status="success") >= 1
    # 耗时直方图至少有一次观测（_count > 0）
    found = False
    for s in mcp_call_duration.collect()[0].samples:
        if s.name.endswith("_count") and s.labels.get("tool") == "query-order":
            assert s.value >= 1
            found = True
    assert found


@pytest.mark.asyncio
async def test_call_error_increments_error_counter(mgr):
    with pytest.raises(RuntimeError):
        await mgr.call_tool("boom-tool", {})
    # Counter 全局累积，断言 >=1（验证「被记录」）
    assert _metric_value(mcp_call_total, tool="boom-tool", status="error") >= 1


@pytest.mark.asyncio
async def test_summary_sets_session_and_tools_gauges(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(cfg.config, "MCP_CLIENT_ENABLED", True)
    servers = {"order-service": _FakeConn("order-service")}
    selector = SimpleNamespace(_servers=servers, _initialized=True)

    MC._mcp_metrics_summary(selector)
    assert _metric_value(mcp_sessions_active) == 1
    assert _metric_value(mcp_tools_total) == 2


def test_connection_status_gauge(monkeypatch):
    MC._mcp_metrics_connection("order-service", True)
    assert _metric_value(mcp_connection_status, server="order-service") == 1
    MC._mcp_metrics_connection("order-service", False)
    assert _metric_value(mcp_connection_status, server="order-service") == 0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
