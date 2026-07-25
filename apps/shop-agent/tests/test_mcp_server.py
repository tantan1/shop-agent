"""
MCP Server 测试

覆盖场景：
  - Server 创建与工具注册
  - tools/list 返回正确的 tool 列表
  - tools/call 正确映射到 ToolService.dispatch()
  - 来自 SKILL.md 的 description 正确传递
  - SchemaDrivenExtractor 与 MCP tools/call 配合
"""

import asyncio
import json
import pytest

from src.modules.chat.core.mcp_server import (
    create_mcp_server,
    _build_input_schema,
    _normalize_params,
)
from src.modules.chat.core.tool_registry import ToolService
from src.modules.chat.core.schema_driven_extractor import SchemaDrivenExtractor


class TestInputSchemaBuilder:

    def test_query_order_schema(self):
        schema = _build_input_schema(
            type("Skill", (), {"name": "query-order"})()
        )
        assert schema["type"] == "object"
        assert "order_id" in schema["properties"]
        assert schema["properties"]["order_id"]["type"] == "string"

    def test_check_balance_schema(self):
        schema = _build_input_schema(
            type("Skill", (), {"name": "check-balance"})()
        )
        assert schema["properties"] == {}

    def test_unknown_skill_schema(self):
        schema = _build_input_schema(
            type("Skill", (), {"name": "nonexistent"})()
        )
        assert schema["properties"] == {}


class TestNormalizeParams:

    def test_normal_params(self):
        assert _normalize_params({"order_id": "123"}) == {"order_id": "123"}

    def test_filter_none(self):
        assert _normalize_params({"order_id": None, "phone": "138"}) == {"phone": "138"}

    def test_filter_empty_string(self):
        assert _normalize_params({"order_id": "", "phone": "138"}) == {"phone": "138"}

    def test_none_params(self):
        assert _normalize_params(None) == {}


class TestMCPServerCreation:

    def test_create_server_registers_all_skills(self):
        """创建 MCP Server 应注册所有 5 个 skill"""
        mcp = create_mcp_server()
        tools = asyncio.run(mcp.list_tools())

        tool_names = {t.name for t in tools}
        assert "query-order" in tool_names
        assert "check-shipping" in tool_names
        assert "request-return" in tool_names
        assert "check-balance" in tool_names
        assert "coupon-inquiry" in tool_names
        assert len(tools) == 5

    def test_tool_has_description(self):
        """每个 tool 应有来自 SKILL.md 的描述"""
        mcp = create_mcp_server()
        tools = asyncio.run(mcp.list_tools())

        for tool in tools:
            assert tool.description, f"Tool {tool.name} 缺少 description"


class TestMCPToolDispatch:

    @pytest.mark.asyncio
    async def test_dispatch_query_order(self):
        """tools/call → dispatch → 返回结果"""
        mcp = create_mcp_server()

        result = await mcp.call_tool(
            "query-order",
            arguments={"order_id": "WB202405270001"},
        )
        # call_tool 可能返回结构化对象或字符串
        result_str = str(result)
        assert "WB202405270001" in result_str

    @pytest.mark.asyncio
    async def test_dispatch_check_shipping(self):
        mcp = create_mcp_server()

        result = await mcp.call_tool(
            "check-shipping",
            arguments={"tracking_number": "SF1234567890"},
        )
        result_str = str(result)
        assert "SF1234567890" in result_str

    @pytest.mark.asyncio
    async def test_dispatch_check_balance(self):
        mcp = create_mcp_server()

        result = await mcp.call_tool("check-balance", arguments={})
        result_str = str(result)
        assert "balance" in result_str.lower() or "520" in result_str

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool(self):
        """未注册的 tool → 应报错"""
        mcp = create_mcp_server()

        with pytest.raises(Exception):
            await mcp.call_tool("nonexistent-tool", arguments={})


class TestMCPWithSchemaDrivenExtractor:

    def test_extractor_uses_schema_from_mcp(self):
        """SchemaDrivenExtractor 接收 mcp tool 的 inputSchema"""
        schema = _build_input_schema(
            type("Skill", (), {"name": "query-order"})()
        )

        result = SchemaDrivenExtractor.extract(
            "查订单 WB202405270001 手机 13800138000",
            mcp_schema=schema,
        )
        # schema 里只有 order_id, phone → 这些字段会被提取
        assert "order_id" in result
        assert result["order_id"] == "WB202405270001"
        assert "phone" in result
        assert result["phone"] == "13800138000"
        # schema 里没有 tracking_number → 不会被提取
        assert "tracking_number" not in result

    def test_full_flow_extract_then_dispatch(self):
        """完整流程：从消息提取参数 → MCP tools/call"""
        schema = _build_input_schema(
            type("Skill", (), {"name": "query-order"})()
        )

        # Step 1: 提取参数
        params = SchemaDrivenExtractor.extract(
            "我的订单 GD202405010099 到哪了",
            mcp_schema=schema,
        )
        assert "order_id" in params
        assert params["order_id"] == "GD202405010099"

        # Step 2: 直接 dispatch（模拟 MCP Server 内部流程）
        ts = ToolService()
        result = asyncio.run(ts.dispatch("query-order", params))
        assert "GD202405010099" in result


class TestToolServiceDirectDispatch:
    """验证 ToolService.dispatch() 直接调用仍正常工作（MCP 对外不对内原则）"""

    @pytest.mark.asyncio
    async def test_dispatch_internal_still_works(self):
        """内部 dispatch 不依赖 MCP Server"""
        ts = ToolService()
        result = await ts.dispatch("query-order", {"order_id": "TEST001"})
        assert "TEST001" in result

    @pytest.mark.asyncio
    async def test_all_tools_dispatch(self):
        """所有 5 个工具 dispatch 均正常"""
        ts = ToolService()
        for action in ["query-order", "check-shipping", "request-return",
                       "check-balance", "coupon-inquiry"]:
            result = await ts.dispatch(action, {})
            assert result, f"Tool {action} 返回空结果"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
