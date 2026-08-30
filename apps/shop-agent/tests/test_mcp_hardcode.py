"""
Phase 2 端到端/单元测试：MCP 高后果字段硬强制（assert A/B/C/D）

不依赖真实 Rust server，用内存 fake session 验证「模型不可见 order_id、
但 tools/call 仍注入正确值并防篡改」的安全底线。

运行：pytest apps/shop-agent/tests/test_mcp_hardcode.py -q
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# 让 tests 能 import 到 src（与仓库其余测试保持一致）
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "shop-agent"))

from src.modules.chat.core.mcp_client import (  # noqa: E402
    HIGH_CONSEQUENCE_FIELDS,
    MCPToolInfo,
    MCPClientManager,
    apply_hardcode_policy,
)
from src.modules.chat.core.layered_param_extractor import (  # noqa: E402
    McpSchemaProvider,
)


class _FakeSession:
    """内存 session：记录最后一次 call_tool 的参数，验证 hardcode 生效。"""

    def __init__(self):
        self.last_action = None
        self.last_args = None

    async def call_tool(self, action, arguments=None, timeout=None):
        self.last_action = action
        self.last_args = dict(arguments or {})

        class _R:
            content = [type("T", (), {"text": '{"ok": true}'})()]

        return _R()


class _FakeConn:
    def __init__(self, name):
        self.name = name
        self.session = _FakeSession()


def _make_info(name, schema, server="order-service"):
    return MCPToolInfo(
        name=name,
        description="test",
        input_schema=schema,
        server_name=server,
        model_visible_schema=apply_hardcode_policy(schema, name),
    )


QUERY_ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string", "title": "订单号"},
        "detail": {"type": "boolean", "title": "是否含明细"},
    },
    "required": ["order_id"],
}


# ── 断言 A：模型可见 schema 剥离高后果字段 ──
def test_assert_a_model_visible_schema_strips_order_id():
    info = _make_info("query-order", QUERY_ORDER_SCHEMA)
    assert "order_id" not in info.model_visible_schema.get("properties", {})
    assert "order_id" not in info.model_visible_schema.get("required", [])
    # 其余字段保留
    assert "detail" in info.model_visible_schema["properties"]
    # 原始 input_schema 未被修改（仍含 order_id）
    assert "order_id" in info.input_schema["properties"]


def test_apply_hardcode_policy_noop_when_no_hc_fields():
    schema = {"type": "object", "properties": {"q": {}}, "required": ["q"]}
    out = apply_hardcode_policy(schema, "list-coupons")
    assert out == schema  # 无高后果字段则不改


# ── 断言 B：确定性来源注入 order_id ──
@pytest.mark.asyncio
async def test_assert_b_hardcode_injects_order_id():
    mgr = MCPClientManager.__new__(MCPClientManager)
    conn = _FakeConn("order-service")
    from types import SimpleNamespace

    mgr._servers = {"order-service": conn}
    mgr._tool_to_server = {"query-order": "order-service"}
    mgr._initialized = True

    # 模拟工具发现已就绪
    info = _make_info("query-order", QUERY_ORDER_SCHEMA)
    conn.tools = {"query-order": info}
    conn.connected = True

    # 模型只提供 detail，未提供 order_id
    res = await mgr.call_tool(
        "query-order", {"detail": True}, hardcode={"order_id": "WB202405270001"}
    )
    assert conn.session.last_args["order_id"] == "WB202405270001"
    assert conn.session.last_args["detail"] is True


# ── 断言 C：硬强制覆盖模型输入的篡改值 ──
@pytest.mark.asyncio
async def test_assert_c_hardcode_overrides_model_tamper():
    mgr = MCPClientManager.__new__(MCPClientManager)
    conn = _FakeConn("order-service")
    mgr._servers = {"order-service": conn}
    mgr._tool_to_server = {"query-order": "order-service"}
    mgr._initialized = True
    info = _make_info("query-order", QUERY_ORDER_SCHEMA)
    conn.tools = {"query-order": info}
    conn.connected = True

    # 模型试图注入他人 order_id，硬强制必须覆盖
    res = await mgr.call_tool(
        "query-order",
        {"order_id": "WB999999999999"},
        hardcode={"order_id": "WB202405270001"},
    )
    assert conn.session.last_args["order_id"] == "WB202405270001"  # 被覆盖


# ── 断言 D：McpSchemaProvider 返回 model_visible_schema ──
@pytest.mark.asyncio
async def test_assert_d_schema_provider_returns_model_visible(monkeypatch):
    info = _make_info("query-order", QUERY_ORDER_SCHEMA)
    fake_mgr = SimpleNamespace(get_tool_info=lambda n: info if n == "query-order" else None)
    import src.modules.chat.core.mcp_client as MC

    monkeypatch.setattr(MC, "mcp_manager", fake_mgr)

    provider = McpSchemaProvider()
    schema = await provider.get_schema("query-order")
    assert schema is not None
    assert "order_id" not in schema.get("properties", {})


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
