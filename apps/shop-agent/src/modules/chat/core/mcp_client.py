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
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.core.config import config

logger = logging.getLogger("mcp_client")

try:
    from src.modules.monitoring.metrics import mcp_schema_mismatch_total
except Exception:  # 极端降级：避免 metrics 模块异常拖垮整个 client
    mcp_schema_mismatch_total = None


# ── 高后果字段（身份/资金类）：对模型不可见，由确定性来源注入 ──
# 复用 yaml_flow 既有语义集合，避免重复定义（见 validator._HIGH_CONSEQUENCE_SEMANTICS）
try:
    from src.modules.chat.agent.yaml_flow.validator import (
        _HIGH_CONSEQUENCE_SEMANTICS as HIGH_CONSEQUENCE_FIELDS,
    )
except Exception:  # 极端降级：保证模块可用
    HIGH_CONSEQUENCE_FIELDS = {"order_id", "phone", "tracking_number"}


def apply_hardcode_policy(schema: Dict[str, Any], action: str) -> Dict[str, Any]:
    """剥离高后果字段，生成「模型可见 schema」。

    安全底线（Phase 2 断言 A）：模型拿到的工具 schema 中不得含
    order_id / phone / tracking_number 等高后果字段，迫使其无法自由编造
    这些身份/资金类参数；真实值由上游确定性来源注入（见 call_tool 的 hardcode）。

    返回的是 *拷贝*，不修改原始 schema（原始 schema 仍用于真实 tools/call 校验）。
    """
    if not schema:
        return dict(schema or {})
    props = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []
    if not (props.keys() & HIGH_CONSEQUENCE_FIELDS):
        # 无需剥离，浅拷贝返回
        return dict(schema)
    new_props = {k: v for k, v in props.items() if k not in HIGH_CONSEQUENCE_FIELDS}
    new_required = [r for r in required if r not in HIGH_CONSEQUENCE_FIELDS]
    return {
        "type": schema.get("type", "object"),
        "properties": new_props,
        "required": new_required,
    }


# ── 档 B：Schema 契约（期望参数）声明与失配告警 ──
# 这是 shop-agent 侧对每个 MCP 工具「期望拿到什么参数」的声明（项目契约），
# 与远程真实 inputSchema 解耦。远程 schema 漂移时仅告警、不阻断调用（fail-open）。
#
# 字段语义：
#   properties: { 字段名: 期望类型 }  期望类型来源于 schema_driven_extractor.MCP_TOOLS_REQUEST_PARAMS
#   required:    [ 期望必填字段名 ]   项目业务逻辑强依赖、缺之不可的字段
#
# 注意：高后果字段（order_id/phone/tracking_number）由上游确定性注入，
# 不要求模型/远程 schema 暴露，因此 *不* 列入此处契约，避免误报。
MCP_EXPECTED_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "query-order": {
        "properties": {"order_id": "string"},
        "required": ["order_id"],
    },
    "list-coupons": {
        "properties": {"user_id": "string"},
        "required": ["user_id"],
    },
    "check-balance": {
        "properties": {"user_id": "string"},
        "required": ["user_id"],
    },
    "request-return": {
        "properties": {"order_id": "string", "reason": "string"},
        "required": ["order_id", "reason"],
    },
    "refund-confirm": {
        "properties": {"order_id": "string"},
        "required": ["order_id"],
    },
}


def _normalize_type(t: Any) -> str:
    """将 JSON Schema 的 type 字段归一化为小写字符串（兼容列表型 type）。"""
    if isinstance(t, list):
        # 取第一个即视为期望类型（MCP 工具 schema 一般为单类型）
        return str(t[0]).lower() if t else ""
    return str(t).lower() if t else ""


def validate_tool_contract(tool_name: str, remote_schema: Dict[str, Any]) -> List[str]:
    """比对单个远程工具 schema 与项目期望契约，返回失配类型列表。

    失配类型（与 mcp_schema_mismatch_total 的 mismatch_type 对齐）：
      - "field_missing"      : 远程工具缺少项目期望的字段（字段名漂移）
      - "type_drift"         : 远程字段类型与期望不符
      - "required_mismatch"  : 远程把项目期望的必填字段标为可选

    不抛异常；任何异常都视为「无失配」（fail-open，不阻断工具发现）。
    """
    expected = MCP_EXPECTED_SCHEMAS.get(tool_name)
    if not expected:
        # 项目未声明该工具契约 → 不检查（新增工具无需强制登记）
        return []

    mismatches: List[str] = []
    try:
        remote_props = remote_schema.get("properties", {}) or {}
        remote_required = set(remote_schema.get("required", []) or [])
        exp_props = expected.get("properties", {}) or {}
        exp_required = expected.get("required", []) or []

        # 1) 字段名漂移：项目期望的字段，远程必须有
        for field_name, exp_type in exp_props.items():
            if field_name not in remote_props:
                mismatches.append("field_missing")
                logger.warning(
                    f"[MCP schema 失配] {tool_name}: 远程缺少期望字段 '{field_name}' "
                    f"（字段名漂移）"
                )
                continue
            # 2) 类型漂移：远程字段类型需与期望一致
            remote_type = _normalize_type(remote_props[field_name].get("type"))
            if remote_type and exp_type and remote_type != exp_type.lower():
                mismatches.append("type_drift")
                logger.warning(
                    f"[MCP schema 失配] {tool_name}.{field_name}: 期望类型 "
                    f"'{exp_type}'，远程类型 '{remote_type}'（类型漂移）"
                )

        # 3) 必填缺漏：项目期望的必填，远程 schema 必填里必须含
        for field_name in exp_required:
            if field_name not in remote_required:
                mismatches.append("required_mismatch")
                logger.warning(
                    f"[MCP schema 失配] {tool_name}.{field_name}: 项目期望必填，"
                    f"但远程 schema 未将其标为 required（必填缺漏）"
                )

        # 去重（同一工具可能对多字段触发同类型）
        seen = set()
        unique = []
        for m in mismatches:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        return unique
    except Exception as e:
        logger.debug(f"[MCP schema 校验] {tool_name} 校验异常，跳过: {e}")
        return []


def report_schema_mismatches(tool_name: str, mismatches: List[str]) -> None:
    """将失配结果上报 Prometheus 指标（档 B 可观测性）。"""
    if not mismatches or mcp_schema_mismatch_total is None:
        return
    try:
        for m in mismatches:
            mcp_schema_mismatch_total.labels(tool=tool_name, mismatch_type=m).inc()
    except Exception:
        pass


@dataclass
class MCPToolInfo:
    """远程 MCP 工具的描述信息（从 tools/list 获取）

    - input_schema：远程下发的完整 schema（含高后果字段），用于真实 tools/call
    - model_visible_schema：剥离高后果字段后的 schema，仅供模型/参数抽取可见
    """

    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    server_name: str = ""
    model_visible_schema: Dict[str, Any] = field(default_factory=dict)


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
                _mcp_metrics_connection(name, True)
                logger.info(f"MCP Client: 已连接 {name} ({url})，发现 {len(conn.tools)} 个工具")
            except Exception as e:
                logger.error(f"MCP Client: 连接 {name} ({url}) 失败: {e}")
                _mcp_metrics_connection(name, False)

        self._initialized = True
        _mcp_metrics_summary(self)

    async def disconnect_all(self) -> None:
        """关闭所有 MCP Server 连接。"""
        for name, conn in list(self._servers.items()):
            try:
                # 先退出 ClientSession（内层），再退出 streamable_http_client（外层）
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

    async def call_tool(
        self,
        action: str,
        params: Optional[Dict[str, Any]] = None,
        hardcode: Optional[Dict[str, Any]] = None,
    ) -> str:
        """通过 MCP 协议调用远程工具。

        Args:
            action: 工具名称（如 "query-order"）
            params: 调用参数（通常来自模型/参数抽取）
            hardcode: 高后果字段确定性来源（如 order_id）。**硬强制覆盖** params 中的同名
                键，保证模型无法篡改身份/资金类参数（Phase 2 断言 B/C）。

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
                raise ConnectionError(f"MCP Server '{server_name}' 重连失败: {e}") from e

        call_args = dict(params or {})
        # ── 硬强制：高后果字段由确定性来源注入，覆盖模型输入（防篡改）──
        if hardcode:
            for k, v in hardcode.items():
                call_args[k] = v
        logger.info(f"MCP tools/call → {server_name}: {action}", extra={"params": call_args})

        _start = time.monotonic()
        _status = "success"
        try:
            result = await conn.session.call_tool(action, arguments=call_args)
        except Exception as e:
            _status = "error"
            logger.error(f"MCP tools/call 失败: {action} @ {server_name}: {e}")
            raise
        finally:
            # ── 可观测性：调用计数 + 耗时直方图 ──
            try:
                from src.modules.monitoring.metrics import (
                    mcp_call_duration,
                    mcp_call_total,
                )

                mcp_call_total.labels(tool=action, status=_status).inc()
                mcp_call_duration.labels(tool=action).observe(time.monotonic() - _start)
            except Exception:
                pass  # 埋点失败绝不影响主流程

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

        使用 streamable_http_client 的 __aenter__/__aexit__ 手动管理生命周期。
        """
        url = conn.url
        headers = conn.headers or None

        # 创建 streamable_http_client 上下文管理器
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
            raw_schema = getattr(tool, "inputSchema", {}) or {}
            model_visible = apply_hardcode_policy(raw_schema, tool.name)
            conn.tools[tool.name] = MCPToolInfo(
                name=tool.name,
                description=getattr(tool, "description", "") or "",
                input_schema=raw_schema,
                server_name=conn.name,
                model_visible_schema=model_visible,
            )
            self._tool_to_server[tool.name] = conn.name

            # 档 B：工具发现即校验 schema 契约，失配仅告警（fail-open）
            mismatches = validate_tool_contract(tool.name, raw_schema)
            if mismatches:
                report_schema_mismatches(tool.name, mismatches)

        conn.connected = True
        logger.info(f"MCP Client: {conn.name} 工具列表: {list(conn.tools.keys())}")

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
# 模块级单例桥接：供 layered_param_extractor.McpSchemaProvider 直接引用。
# 注意：需先调用 get_mcp_client() 才会被填充（懒加载）。
mcp_manager: Optional[MCPClientManager] = None


async def get_mcp_client() -> MCPClientManager:
    """获取 MCP Client 单例（懒加载，自动连接）。"""
    global _mcp_client_manager, mcp_manager
    if _mcp_client_manager is None:
        _mcp_client_manager = MCPClientManager()
        await _mcp_client_manager.connect_all()
    mcp_manager = _mcp_client_manager
    return _mcp_client_manager


async def shutdown_mcp_client() -> None:
    """关闭 MCP Client 所有连接。"""
    global _mcp_client_manager
    if _mcp_client_manager is not None:
        await _mcp_client_manager.disconnect_all()
        _mcp_client_manager = None


# ── 可观测性辅助（延迟导入 metrics，避免循环依赖）──


def _mcp_metrics_connection(server: str, connected: bool) -> None:
    """更新单个 MCP Server 的连接状态 Gauge。"""
    try:
        from src.modules.monitoring.metrics import mcp_connection_status

        mcp_connection_status.labels(server=server).set(1 if connected else 0)
    except Exception:
        pass


def _mcp_metrics_summary(manager: "MCPClientManager") -> None:
    """刷新 session 数 / 工具总数 Gauge。"""
    try:
        from src.modules.monitoring.metrics import (
            mcp_sessions_active,
            mcp_tools_total,
        )

        active = sum(1 for c in manager._servers.values() if getattr(c, "connected", False))
        total_tools = sum(len(c.tools) for c in manager._servers.values())
        mcp_sessions_active.set(active)
        mcp_tools_total.set(total_tools)
    except Exception:
        pass
