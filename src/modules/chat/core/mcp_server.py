"""
MCP Server —— 将 ToolService 的 Skill 体系通过 MCP 协议对外暴露。

核心原则：MCP 对外，不对内。
  - 外部 Client（Claude Desktop / n8n / 其他 Agent）通过 MCP 调工具
  - 内部 ReActAgent ↔ ToolService.dispatch() 同进程直调，不走 JSON-RPC

协议能力：
  tools/list  —— 从 SkillRegistry 自动生成 tool 列表（schema 由函数类型注解推断）
  tools/call  —— 映射到 ToolService.dispatch(action, params)

使用方式：
  # 编程方式
  from src.modules.chat.core.mcp_server import create_mcp_server
  server = create_mcp_server()
  server.run(transport="stdio")  # 或 "sse"

  # CLI
  python -m src.modules.chat.core.mcp_server
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from src.modules.chat.agent.skill_loader import get_skill_registry, SkillDef
from src.modules.chat.core.tool_registry import ToolService
from src.core.permissions import (
    ClientInfo,
    get_client_accessible_tools,
    set_current_client,
    clear_current_client,
)
from src.core.config import config as app_config
from src.shared.logger import APILogger

logger = APILogger("mcp_server")


def _build_input_schema(skill: SkillDef) -> Dict[str, Any]:
    """从 SkillDef.params 生成 tool 的 inputSchema。

    参数定义唯一来源是 SKILL.md 的 frontmatter params 字段。
    无需单独维护硬编码的 param_schemas 字典。
    """
    properties: Dict[str, Dict[str, str]] = {}
    required: List[str] = []
    for name, meta in skill.params.items():
        if not isinstance(meta, dict):
            continue
        prop: Dict[str, str] = {"type": str(meta.get("type", "string"))}
        if meta.get("description"):
            prop["description"] = str(meta["description"])
        properties[name] = prop
        if meta.get("required") is True:
            required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _normalize_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """过滤 None 和空字符串，标准化为 dispatch 期望的格式。"""
    if not params:
        return {}
    return {k: v for k, v in params.items() if v not in (None, "")}


def _make_tool_fn(action: str, schema: dict, ts: ToolService, client: ClientInfo | None = None):
    """动态生成带类型注解的工具函数，使 FastMCP 能自动推断 inputSchema。

    原理：FastMCP.add_tool() 从函数签名的参数类型注解生成 JSON Schema。
    因此我们需要为每个 tool 动态创建一个签名为
        async def fn(param1: str|None = None, param2: str|None = None) -> str
    的函数。

    当 client 不为 None 且 PERMISSION_ENABLED 为 True 时，
    在 dispatch 前注入调用方上下文并在 dispatch 后清除。
    """
    props = schema.get("properties", {})

    if not props:
        # 无参工具（如 check-balance）
        async def _no_param_fn() -> str:
            logger.info("MCP tools/call", action=action)
            try:
                if client is not None and app_config.PERMISSION_ENABLED:
                    set_current_client(client)
                result = await ts.dispatch(action, {})
                return result
            except Exception as e:
                logger.error("MCP tool 执行失败", action=action, error=str(e))
                return json.dumps(
                    {"error": f"工具 {action} 执行失败: {str(e)}"},
                    ensure_ascii=False,
                )
            finally:
                if client is not None and app_config.PERMISSION_ENABLED:
                    clear_current_client()
        _no_param_fn.__name__ = action.replace("-", "_")
        return _no_param_fn

    # 构建带参数的工具函数
    param_parts = []
    body_parts = [
        "import json",
        f"action = {action!r}",
        "clean_params = {}",
    ]
    for param_name in props:
        safe_name = param_name.replace("-", "_")
        param_parts.append(f"{safe_name}: str|None = None")
        body_parts.append(
            f"if {safe_name} is not None: clean_params[{param_name!r}] = {safe_name}"
        )

    body_parts.extend([
        f"_logger.info('MCP tools/call', action=action, params=clean_params)",
        f"if _client is not None and _perm_enabled: _set_current_client(_client)",
        "try:",
        f"    result = await _ts.dispatch(action, clean_params)",
        f"    return result",
        "except Exception as _e:",
        f"    _logger.error('MCP tool 执行失败', action=action, error=str(_e))",
        f"    return json.dumps({{'error': f'工具 {{action}} 执行失败: {{str(_e)}}'}}, ensure_ascii=False)",
        "finally:",
        f"    if _client is not None and _perm_enabled: _clear_current_client()",
    ])

    body_source = "\n".join("    " + line for line in body_parts)
    fn_source = (
        f"async def _fn({', '.join(param_parts)}) -> str:\n"
        f"{body_source}\n"
    )

    namespace = {
        "_ts": ts,
        "_logger": logger,
        "_client": client,
        "_set_current_client": set_current_client,
        "_clear_current_client": clear_current_client,
        "_perm_enabled": app_config.PERMISSION_ENABLED,
    }
    exec(fn_source, namespace)
    fn = namespace["_fn"]
    fn.__name__ = action.replace("-", "_")
    return fn


def create_mcp_server(
    server_name: str = "shop-agent",
    tool_service: Optional[ToolService] = None,
    host: str = "127.0.0.1",
    port: int = 3001,
    streamable_http_path: str = "/",
    client_info: Optional[ClientInfo] = None,
) -> FastMCP:
    """创建 MCP Server 实例，自动注册所有 Skill 对应的 Tool。

    Args:
        server_name:           MCP Server 名称
        tool_service:          ToolService 实例，为 None 时自动创建
        host:                  SSE/HTTP 模式监听地址
        port:                  SSE/HTTP 模式监听端口
        streamable_http_path:  Streamable HTTP 端点路径（挂载时设 "/"，独立运行时用默认 "/mcp"）
        client_info:           调用方身份信息。传入时：
                               - tools/list 只返回该调用方有权访问的工具
                               - tools/call 注入 client 上下文以触发 dispatch 层权限校验
                               为 None 时行为与旧版完全一致（全量工具 + 无权限校验）

    Returns:
        FastMCP 实例，可直接 .run(transport="stdio"/"streamable-http") 或接入 FastAPI
    """
    mcp = FastMCP(server_name, host=host, port=port, streamable_http_path=streamable_http_path)
    ts = tool_service or ToolService()
    registry = get_skill_registry()
    registered_count = 0

    # ── 基于调用方身份过滤可注册的工具 ──
    if client_info is not None and app_config.PERMISSION_ENABLED:
        allowed = get_client_accessible_tools(client_info)
    else:
        allowed = None  # None = 不过滤，全量注册

    for skill in registry.skills:
        if allowed is not None and skill.name not in allowed:
            logger.info(f"MCP 工具过滤（权限不足）: {skill.name}", client=client_info.client_id if client_info else "N/A")
            continue

        input_schema = _build_input_schema(skill)
        tool_fn = _make_tool_fn(skill.name, input_schema, ts, client=client_info)

        mcp.add_tool(
            tool_fn,
            name=skill.name,
            description=skill.description,
        )
        registered_count += 1
        logger.info(f"注册 MCP Tool: {skill.name}", description=skill.description[:60])

    logger.info(
        f"MCP Server 初始化完成: {server_name}",
        tool_count=registered_count,
        client=client_info.client_id if client_info else "anonymous",
    )
    return mcp


# ── 单例 ──
_mcp_server: FastMCP | None = None


def get_mcp_server() -> FastMCP:
    """获取 MCP Server 单例（懒加载）。"""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = create_mcp_server()
    return _mcp_server


# ── CLI 入口 ──
if __name__ == "__main__":
    import sys

    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    print(f"启动 MCP Server (transport={transport})...")
    server = create_mcp_server()
    server.run(transport=transport)
