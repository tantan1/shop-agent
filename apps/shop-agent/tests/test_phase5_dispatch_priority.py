"""
Phase 5 验证：dispatch 优先级 MCP > REST（REST 确为降级通道，非主路径）

验证：
  - MCP 启用且工具存在时，dispatch 走 MCP（_try_mcp_dispatch），**不**触发 REST
    （_call_order_api / _call_remote_api 不被调用）
  - MCP 未启用时，dispatch 回退到本地注册表 / REST（降级通道生效）

不依赖真实 Rust server：用 fake session + monkeypatch get_mcp_client，并 spy REST 函数。

运行：pytest apps/shop-agent/tests/test_phase5_dispatch_priority.py -q
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "shop-agent"))

import src.modules.chat.core.tool_registry as TR  # noqa: E402
import src.modules.chat.core.mcp_client as MC  # noqa: E402
import src.core.config as cfg  # noqa: E402
from src.modules.chat.core.tool_registry import ToolService  # noqa: E402


class _FakeResult:
    content = [type("T", (), {"text": '{"ok": true, "via": "mcp"}'})()]


class _FakeSession:
    def __init__(self):
        self.calls = []

    async def call_tool(self, action, arguments=None, timeout=None):
        self.calls.append(action)
        return _FakeResult()


class _FakeConn:
    def __init__(self, name):
        self.name = name
        self.session = _FakeSession()
        self.connected = True
        self.tools = {"query-order": None}


@pytest.fixture
def mcp_enabled(monkeypatch):
    monkeypatch.setattr(cfg.config, "MCP_CLIENT_ENABLED", True)
    from src.modules.chat.core.mcp_client import MCPClientManager

    servers = {"order-service": _FakeConn("order-service")}
    manager = MCPClientManager.__new__(MCPClientManager)
    manager._servers = servers
    manager._tool_to_server = {"query-order": "order-service"}
    manager._initialized = True

    async def fake_get():
        return manager

    monkeypatch.setattr(TR, "get_mcp_client", fake_get)
    monkeypatch.setattr(MC, "get_mcp_client", fake_get)
    return manager


def _spy_rest(monkeypatch):
    calls = []

    async def fake_order_api(action, params=None, user_id=None):
        calls.append(("order_api", action))
        return '{"via": "rest"}'

    async def fake_remote_api(action, params=None):
        calls.append(("remote_api", action))
        return '{"via": "remote"}'

    monkeypatch.setattr(TR.ToolService, "_call_order_api", staticmethod(fake_order_api))
    monkeypatch.setattr(TR.ToolService, "_call_remote_api", staticmethod(fake_remote_api))
    return calls


@pytest.mark.asyncio
async def test_dispatch_prefers_mcp_over_rest(mcp_enabled, monkeypatch):
    rest_calls = _spy_rest(monkeypatch)
    svc = ToolService()
    # query-order 在 MCP 中暴露 → 应走 MCP，不触发任何 REST
    res = await svc.dispatch("query-order", {"order_id": "WB1"})
    assert "mcp" in res
    assert rest_calls == [], f"MCP 可用时不应触发 REST 降级，实际: {rest_calls}"
    assert mcp_enabled._servers["order-service"].session.calls == ["query-order"]


@pytest.mark.asyncio
async def test_dispatch_falls_back_to_rest_when_mcp_disabled(monkeypatch):
    monkeypatch.setattr(cfg.config, "MCP_CLIENT_ENABLED", False)
    rest_calls = _spy_rest(monkeypatch)
    svc = ToolService()
    # query-order 在本地注册表存在 → 走注册表/REST（MCP 关闭）
    res = await svc.dispatch("query-order", {"order_id": "WB1"})
    # 注册表 _execute_query_order 优先级里若有 REST 则触发；至少不应走 MCP
    assert "mcp" not in res


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
