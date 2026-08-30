"""
Phase 3 单元测试：MCP 敏感工具 HITL（先拦截后执行）

验证：
  - dispatch 敏感工具（check-balance/request-return/refund-confirm）返回 waiting_for_confirmation JSON
  - 此时**未**真实调用远程 MCP（无副作用）
  - approve 后才真实调用远程 MCP
  - reject 不调用远程 MCP

不依赖真实 Rust server：用内存 fake session + monkeypatch get_mcp_client。

运行：pytest apps/shop-agent/tests/test_mcp_hitl.py -q
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "shop-agent"))

from src.modules.chat.core.tool_registry import (  # noqa: E402
    ToolService,
    McpToolCommand,
    MCP_SENSITIVE_TOOLS,
)


class _FakeResult:
    content = [type("T", (), {"text": '{"ok": true}'})()]


class _FakeSession:
    def __init__(self):
        self.calls = []
        self.order_id_injected = None

    async def call_tool(self, action, arguments=None, timeout=None):
        self.calls.append((action, dict(arguments or {})))
        if arguments:
            self.order_id_injected = arguments.get("order_id")
        return _FakeResult()


class _FakeConn:
    def __init__(self, name):
        self.name = name
        self.session = _FakeSession()
        self.connected = True
        self.tools = {}


def _patch_mcp(monkeypatch, tools):
    """用一个内存 MCP manager 替换 get_mcp_client，并启用 MCP_CLIENT_ENABLED。"""
    from types import SimpleNamespace

    import src.core.config as cfg

    monkeypatch.setattr(cfg.config, "MCP_CLIENT_ENABLED", True)

    # 构造 manager 的最小形态
    servers = {"order-service": _FakeConn("order-service")}
    selector = SimpleNamespace(
        _tool_to_server={t: "order-service" for t in tools},
        _servers=servers,
        _initialized=True,
        has_tool=lambda a: a in tools,
    )

    async def _fake_call_tool(action, params=None, hardcode=None):
        sess = servers["order-service"].session
        args = dict(params or {})
        if hardcode:  # 模拟真实 MCPClientManager.call_tool 的硬强制合并
            for k, v in hardcode.items():
                args[k] = v
        return await sess.call_tool(action, arguments=args)

    selector.call_tool = _fake_call_tool
    for t in tools:
        selector._servers["order-service"].tools[t] = SimpleNamespace(name=t)

    async def fake_get():
        return selector

    import src.modules.chat.core.tool_registry as TR
    import src.modules.chat.core.mcp_client as MC

    monkeypatch.setattr(TR, "get_mcp_client", fake_get)
    monkeypatch.setattr(MC, "get_mcp_client", fake_get)
    return selector


@pytest.fixture
def mcp(monkeypatch):
    return _patch_mcp(monkeypatch, list(MCP_SENSITIVE_TOOLS) + ["query-order"])


def test_sensitive_tools_in_default_set():
    assert {"check-balance", "request-return", "refund-confirm"} <= MCP_SENSITIVE_TOOLS


@pytest.mark.asyncio
async def test_dispatch_sensitive_returns_waiting_and_no_remote_call(mcp):
    res = await ToolService._try_mcp_dispatch("request-return", {"order_id": "WB202405270001"})
    assert res is not None
    payload = json.loads(res)
    assert payload["status"] == "waiting_for_confirmation"
    assert "approval_id" in payload
    # 关键：dispatch 阶段**未**真实调用远程 MCP（先拦截后执行）
    assert mcp._servers["order-service"].session.calls == []


@pytest.mark.asyncio
async def test_approve_then_real_remote_call(mcp):
    res = await ToolService._try_mcp_dispatch("request-return", {"order_id": "WB202405270001"})
    approval_id = json.loads(res)["approval_id"]

    out = await ToolService.approve_mcp_tool(approval_id)
    assert out is not None
    # approve 后才真实调用远程 MCP
    calls = mcp._servers["order-service"].session.calls
    assert len(calls) == 1
    assert calls[0][0] == "request-return"
    # 高后果字段 order_id 由 hardcode 注入
    assert mcp._servers["order-service"].session.order_id_injected == "WB202405270001"


@pytest.mark.asyncio
async def test_reject_no_remote_call(mcp):
    res = await ToolService._try_mcp_dispatch("check-balance", {"order_id": "WB202405270001"})
    approval_id = json.loads(res)["approval_id"]
    out = await ToolService.reject_mcp_tool(approval_id)
    assert "拒绝" in out
    assert mcp._servers["order-service"].session.calls == []


@pytest.mark.asyncio
async def test_non_sensitive_bypasses_hitl(mcp):
    # query-order 非敏感，直接调用（且剥离 order_id 由 hardcode 注入）
    res = await ToolService._try_mcp_dispatch("query-order", {"order_id": "WB202405270001"})
    assert res is not None
    calls = mcp._servers["order-service"].session.calls
    assert len(calls) == 1
    assert calls[0][0] == "query-order"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
