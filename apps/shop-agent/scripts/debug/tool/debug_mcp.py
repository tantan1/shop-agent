"""
MCP Server 调试脚本

用法:
    python scripts/debug_mcp.py                 # 编程方式调试
    python scripts/debug_mcp.py --sse           # SSE 模式启动（可接 HTTP 调试）
    python scripts/debug_mcp.py --inspect       # 用 MCP Inspector 调试
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.modules.chat.core.mcp_server import create_mcp_server


async def debug_programmatic():
    """编程方式：直接调 list_tools / call_tool"""
    mcp = create_mcp_server()

    # -- 1. tools/list --
    print("=" * 60)
    print("[tools/list]")
    print("=" * 60)
    tools = await mcp.list_tools()
    for t in tools:
        print(f"\n  Tool: {t.name}")
        print(f"  description: {t.description}")
        schema = t.inputSchema if hasattr(t, "inputSchema") else {}
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        print(f"  params: {list(props.keys())}")

    # -- 2. tools/call 逐个测试 --
    test_cases = [
        ("query-order", {"order_id": "WB202405270001", "phone": "13800138000"}),
        ("query-order", {"order_id": "WB202405270001"}),  # 仅 order_id
        ("check-shipping", {"tracking_number": "SF1234567890"}),
        ("request-return", {"order_id": "WB202405270001", "reason": "质量问题"}),
        ("check-balance", {}),
        ("coupon-inquiry", {"coupon_type": "满减券"}),
    ]

    print("\n" + "=" * 60)
    print("[tools/call] 逐个测试")
    print("=" * 60)
    for name, args in test_cases:
        print(f"\n  >> {name}({args})")
        try:
            result = await mcp.call_tool(name, arguments=args)
            result_str = str(result)
            if len(result_str) > 200:
                result_str = result_str[:200] + "..."
            print(f"     OK: {result_str}")
        except Exception as e:
            print(f"     FAIL [{type(e).__name__}]: {e}")

    # -- 3. 异常情况 --
    print("\n" + "=" * 60)
    print("[异常情况]")
    print("=" * 60)

    # 不存在的 tool
    print("\n  >> nonexistent-tool({})")
    try:
        await mcp.call_tool("nonexistent-tool", arguments={})
    except Exception as e:
        print(f"     (预期) [{type(e).__name__}]: {e}")

    # 空参数
    print("\n  >> query-order({})")
    try:
        result = await mcp.call_tool("query-order", arguments={})
        print(f"     OK (空参数不影响): {str(result)[:100]}")
    except Exception as e:
        print(f"     FAIL [{type(e).__name__}]: {e}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""

    if mode == "--sse":
        print("启动 MCP Server (SSE 模式, http://127.0.0.1:3001/sse)")
        server = create_mcp_server()
        server.run(transport="sse")

    elif mode == "--inspect":
        print("用 MCP Inspector 调试（将启动 stdio 模式）:")
        print("  npx @modelcontextprotocol/inspector python scripts/debug_mcp.py")
        server = create_mcp_server()
        server.run(transport="stdio")

    else:
        asyncio.run(debug_programmatic())


if __name__ == "__main__":
    main()
