"""
MCP Client Manager —— 让 Agent 通过 MCP 协议发现并调用远程工具。

核心原则：MCP Client 是 ToolService 的远程后端之一，与现有的 HTTP REST 后端并列。
  - 优先级：MCP Client > HTTP REST > 本地 mock
  - 连接管理：Streamable HTTP 长连接，支持多 MCP Server
  - 工具发现：通过 tools/list 动态获取，替代硬编码 endpoint_map
  - 工具调用：通过 tools/call JSON-RPC，替代 httpx.post

使用方式：
    manager = MCPClientManager()
    await manager.connect_all()           # 连接所有配置的 MCP Server
    result = await manager.call_tool("query-order", {"order_id": "xxx"})
    await manager.disconnect_all()        # 关闭所有连接
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.core.config import config

logger = logging.getLogger("mcp_client")


@dataclass
class MCPToolInfo:
    """远程 MCP 工具的描述信息（从 tools/list 获取）"""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    server_name: str = ""


@dataclass
class MCPServerConnection:
    """单个 MCP Server 的连接状态"""
    url: str
    name: str
    headers: Dict[str, str] = field(default_factory=dict)
    session: Optional[ClientSession] = None
    tools: Dict[str, MCPToolInfo] = field(default_factory=dict)
    connected: bool = False


class MCPClientManager:
    """MCP 客户端管理器 —— 管理到多个远程 MCP Server 的连接。

    职责：
    1. 连接管理：建立/维持/关闭到多个 MCP Server 的 Streamable HTTP 连接
    2. 工具发现：从每个 server 的 tools/list 拉取工具列表并缓存
    3. 工具调用：根据 action 名称路由到对应的 MCP Server 并调用 tools/call
    4. 健康检查：检测连接断开并自动重连

    配置方式（.env）：
        MCP_CLIENT_SERVERS='[
            {"name":"order-system","url":"http://localhost:3002/mcp"},
            {"name":"shipping-system","url":"http://localhost:3003/mcp","headers":{"Authorization":"Bearer xxx"}}
        ]'
    """

    def __init__(self):
        self._servers: Dict[str, MCPServerConnection] = {}
        self._tool_to_server: Dict[str, str] = {}  # tool_name → server_name
        self._initialized = False

    # ── 生命周期 ──────────────────────────────────────────────────

    async def connect_all(self) -> None:
        """连接所有配置的 MCP Server 并发现工具。

        从 config.MCP_CLIENT_SERVERS 读取服务器列表，
        逐一建立 Streamable HTTP 连接，拉取 tools/list。
        """
        if self._initialized:
            return

        server_configs = self._parse_server_configs()
        if not server_configs:
            logger.info("MCP Client: 未配置远程 MCP Server，跳过连接")
            self._initialized = True
            return

        for cfg in server_configs:
            name = cfg.get("name", cfg.get("url", "unknown"))
            url = cfg.get("url", "")
            headers = cfg.get("headers", {})

            if not url:
                logger.warning(f"MCP Client: 跳过无效配置 (name={name}, url 为空)")
                continue

            conn = MCPServerConnection(url=url, name=name, headers=headers)
            try:
                await self._connect_server(conn)
                self._servers[name] = conn
                logger.info(
                    f"MCP Client: 已连接 {name} ({url})，发现 {len(conn.tools)} 个工具"
                )
            except Exception as e:
                logger.error(f"MCP Client: 连接 {name} ({url}) 失败: {e}")

        self._initialized = True

    async def disconnect_all(self) -> None:
        """关闭所有 MCP Server 连接。"""
        for name, conn in list(self._servers.items()):
            try:
                # 先退出 ClientSession（内层），再退出 streamablehttp_client（外层）
                session_ctx = getattr(conn, "_session_ctx", None)
                http_ctx = getattr(conn, "_http_ctx", None)

                if session_ctx is not None:
                    await session_ctx.__aexit__(None, None, None)
                if http_ctx is not None:
                    await http_ctx.__aexit__(None, None, None)

                conn.session = None
                conn.connected = False
                logger.info(f"MCP Client: 已断开 {name}")
            except Exception as e:
                logger.warning(f"MCP Client: 断开 {name} 时出错: {e}")

        self._servers.clear()
        self._tool_to_server.clear()
        self._initialized = False

    # ── 工具查询 ──────────────────────────────────────────────────

    def get_tool_names(self) -> List[str]:
        """返回所有远程 MCP 工具名称列表。"""
        names = []
        for conn in self._servers.values():
            names.extend(conn.tools.keys())
        return names

    def get_tool_info(self, action: str) -> Optional[MCPToolInfo]:
        """获取指定远程工具的描述信息。"""
        server_name = self._tool_to_server.get(action)
        if not server_name:
            return None
        conn = self._servers.get(server_name)
        if not conn:
            return None
        return conn.tools.get(action)

    def has_tool(self, action: str) -> bool:
        """检查指定工具是否在远程 MCP Server 中可用。"""
        return action in self._tool_to_server

    def get_all_tools(self) -> Dict[str, MCPToolInfo]:
        """返回所有已发现工具（名称 → 信息）。"""
        result: Dict[str, MCPToolInfo] = {}
        for conn in self._servers.values():
            result.update(conn.tools)
        return result

    # ── 工具调用 ──────────────────────────────────────────────────

    async def call_tool(self, action: str, params: Optional[Dict[str, Any]] = None) -> str:
        """通过 MCP 协议调用远程工具。

        Args:
            action: 工具名称（如 "query-order"）
            params: 调用参数

        Returns:
            工具执行结果字符串

        Raises:
            ValueError: 工具未在任何已连接的 MCP Server 中找到
            ConnectionError: 目标 MCP Server 连接不可用
        """
        if not self._initialized:
            await self.connect_all()

        server_name = self._tool_to_server.get(action)
        if not server_name:
            raise ValueError(
                f"MCP 工具 '{action}' 未在任何远程 MCP Server 中发现。"
                f"可用工具: {list(self._tool_to_server.keys())}"
            )

        conn = self._servers.get(server_name)
        if not conn or not conn.session:
            # 尝试重连
            logger.warning(f"MCP Client: {server_name} 连接不可用，尝试重连...")
            try:
                await self._connect_server(conn)
            except Exception as e:
                raise ConnectionError(
                    f"MCP Server '{server_name}' 重连失败: {e}"
                ) from e

        params = params or {}
        logger.info(f"MCP tools/call → {server_name}: {action}", extra={"params": params})

        try:
            result = await conn.session.call_tool(action, arguments=params)
        except Exception as e:
            logger.error(f"MCP tools/call 失败: {action} @ {server_name}: {e}")
            raise

        return self._format_tool_result(action, result)

    # ── 内部方法 ──────────────────────────────────────────────────

    @staticmethod
    def _parse_server_configs() -> List[Dict[str, Any]]:
        """解析 MCP_CLIENT_SERVERS 配置（JSON 字符串 → 列表）。"""
        raw = config.MCP_CLIENT_SERVERS
        if not raw:
            return []

        try:
            servers = json.loads(raw)
            if not isinstance(servers, list):
                logger.warning("MCP_CLIENT_SERVERS 格式错误，需要 JSON 数组")
                return []
            return servers
        except json.JSONDecodeError as e:
            logger.warning(f"MCP_CLIENT_SERVERS JSON 解析失败: {e}")
            return []

    async def _connect_server(self, conn: MCPServerConnection) -> None:
        """建立到单个 MCP Server 的 Streamable HTTP 持久连接并拉取工具列表。"""
        await self._persistent_connect(conn)

    async def _persistent_connect(self, conn: MCPServerConnection) -> None:
        """建立持久化的 MCP 连接（不随 async with 退出而断开）。

        使用 streamablehttp_client 的 __aenter__/__aexit__ 手动管理生命周期。
        """
        url = conn.url
        headers = conn.headers or None

        # 创建 streamablehttp_client 上下文管理器
        ctx = streamablehttp_client(url, headers=headers)
        read_stream, write_stream, _ = await ctx.__aenter__()

        # 创建 ClientSession
        session_ctx = ClientSession(read_stream, write_stream)
        session = await session_ctx.__aenter__()

        # 保存上下文管理器引用以便后续关闭
        conn._http_ctx = ctx  # type: ignore[attr-defined]
        conn._session_ctx = session_ctx  # type: ignore[attr-defined]
        conn.session = session

        # 初始化握手
        await session.initialize()

        # 发现工具
        tools_result = await session.list_tools()

        conn.tools.clear()
        for tool in tools_result.tools:
            conn.tools[tool.name] = MCPToolInfo(
                name=tool.name,
                description=getattr(tool, "description", "") or "",
                input_schema=getattr(tool, "inputSchema", {}) or {},
                server_name=conn.name,
            )
            self._tool_to_server[tool.name] = conn.name

        conn.connected = True
        logger.info(
            f"MCP Client: {conn.name} 工具列表: {list(conn.tools.keys())}"
        )

    @staticmethod
    def _format_tool_result(action: str, result: Any) -> str:
        """将 MCP tools/call 的返回值格式化为 Agent 可用的字符串。

        MCP call_tool 返回 CallToolResult，其中 content 是列表。
        """
        # 尝试提取 content
        if hasattr(result, "content"):
            content = result.content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                    elif hasattr(item, "data"):
                        parts.append(str(item.data))
                    else:
                        parts.append(str(item))
                return "\n".join(parts)
            return str(content)

        # 尝试提取 structuredContent
        if hasattr(result, "structuredContent") and result.structuredContent:
            return json.dumps(result.structuredContent, ensure_ascii=False)

        return str(result)


# ── 单例 ──

_mcp_client_manager: Optional[MCPClientManager] = None


async def get_mcp_client() -> MCPClientManager:
    """获取 MCP Client 单例（懒加载，自动连接）。"""
    global _mcp_client_manager
    if _mcp_client_manager is None:
        _mcp_client_manager = MCPClientManager()
        await _mcp_client_manager.connect_all()
    return _mcp_client_manager


async def shutdown_mcp_client() -> None:
    """关闭 MCP Client 所有连接。"""
    global _mcp_client_manager
    if _mcp_client_manager is not None:
        await _mcp_client_manager.disconnect_all()
        _mcp_client_manager = None
