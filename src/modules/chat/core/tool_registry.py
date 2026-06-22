"""
Tool 服务 —— 意图命中后的业务函数定义、远程 API 调用（HTTP / MCP）、格式化

远程调用优先级：MCP Client > HTTP REST > 本地 mock
"""
from typing import Dict, Any, Optional
import json

import httpx

from src.shared.logger import APILogger
from src.core.config import config
from src.core.permissions import (
    get_current_client,
    check_tool_permission,
)
from src.modules.chat.core.mcp_client import get_mcp_client

logger = APILogger("tool_service")


class ToolPermissionError(PermissionError):
    """工具权限不足异常"""

    def __init__(self, tool_name: str, role: str, client_id: str):
        self.tool_name = tool_name
        self.role = role
        self.client_id = client_id
        super().__init__(
            f"权限不足：调用方 {client_id}（角色 {role}）无权调用工具 '{tool_name}'"
        )


class ToolService:
    """工具执行服务：Tool 定义 + 分发 + 远程 API 调用"""

    def __init__(self):
        self._registry: Dict[str, Any] = {}

    def _ensure_registry(self):
        """懒加载 tool 注册表"""
        if self._registry:
            return
        self._registry.update({
            "query-order":      self._tool_query_order,
            "check-shipping":   self._tool_check_shipping,
            "request-return":   self._tool_request_return,
            "check-balance":    self._tool_check_balance,
            "coupon-inquiry":   self._tool_coupon_inquiry,
        })

    # ── Tool 实现 ──────────────────────────────────────────────────

    @staticmethod
    async def _tool_query_order(params: Optional[Dict[str, Any]] = None) -> str:
        """查询订单"""
        params = params or {}
        order_id = params.get("order_id")
        if config.REMOTE_API_BASE_URL:
            return await ToolService._call_remote_api("query-order", params)
        if order_id:
            return json.dumps({
                "order": {"id": order_id, "status": "派送中", "total": 299.00},
                "note": f"已按 order_id={order_id} 查询"
            }, ensure_ascii=False)
        return json.dumps({
            "orders": [
                {"id": "202405270001001", "status": "已发货", "total": 299.00},
                {"id": "202405250016001", "status": "派送中", "total": 158.00},
            ],
            "note": "未指定 order_id，返回最近订单"
        }, ensure_ascii=False)

    @staticmethod
    async def _tool_check_shipping(params: Optional[Dict[str, Any]] = None) -> str:
        """查询物流"""
        params = params or {}
        tracking = params.get("tracking_number")
        if config.REMOTE_API_BASE_URL:
            return await ToolService._call_remote_api("check-shipping", params)
        if tracking:
            return json.dumps({
                "tracking_number": tracking,
                "tracking": [
                    {"time": "05-27 10:30", "status": "到达分拣中心"},
                    {"time": "05-27 08:15", "status": "已揽收"},
                ]
            }, ensure_ascii=False)
        return json.dumps({
            "tracking": [
                {"time": "05-27 10:30", "status": "到达分拣中心"},
                {"time": "05-27 08:15", "status": "已揽收"},
            ],
            "note": "未指定快递单号，返回最近物流"
        }, ensure_ascii=False)

    @staticmethod
    async def _tool_request_return(params: Optional[Dict[str, Any]] = None) -> str:
        """申请退货退款"""
        params = params or {}
        order_id = params.get("order_id", "未指定")
        reason = params.get("reason", "未说明")
        if config.REMOTE_API_BASE_URL:
            return await ToolService._call_remote_api("request-return", params)
        return (
            f"已为您提交退货申请（订单号: {order_id}，原因: {reason}），"
            f"退款将在 1-3 个工作日内原路返回。"
        )

    @staticmethod
    async def mock_refund_confirmation(order_id: str, reason: str, refund_amount: float = 0.0) -> None:
        """Mock API: 模拟调用退款确认外部服务，打印确认信息。

        在实际生产环境中，这里会调用真实的审批系统 API。
        当前仅打印退款确认信息到标准输出，模拟外部系统调用。
        """
        print(f"\n{'=' * 60}")
        print(f"[Mock API] 调用退款确认服务 —— refund_confirmation_endpoint")
        print(f"{'=' * 60}")
        print(f"  Action:          REFUND_CONFIRMATION")
        print(f"  Order ID:        {order_id}")
        print(f"  Reason:          {reason}")
        print(f"  Refund Amount:   ¥{refund_amount:.2f}")
        print(f"  Status:          ⚠ PENDING_HUMAN_APPROVAL")
        print(f"  Message:         退款申请需要人工审批确认")
        print(f"{'=' * 60}")
        print(f"")

    @staticmethod
    async def _tool_check_balance(params: Optional[Dict[str, Any]] = None) -> str:
        """查询余额/积分"""
        params = params or {}
        if config.REMOTE_API_BASE_URL:
            return await ToolService._call_remote_api("check-balance", params)
        return json.dumps({
            "balance": 520.00,
            "points": 1280,
        }, ensure_ascii=False)

    @staticmethod
    async def _tool_coupon_inquiry(params: Optional[Dict[str, Any]] = None) -> str:
        """查询优惠券"""
        params = params or {}
        if config.REMOTE_API_BASE_URL:
            return await ToolService._call_remote_api("coupon-inquiry", params)
        return json.dumps({
            "coupons": [
                {"name": "满200减30", "expire": "2026-06-30"},
                {"name": "新用户满100减15", "expire": "2026-06-15"},
            ]
        }, ensure_ascii=False)

    # ── 分发入口 ──────────────────────────────────────────────────

    async def dispatch(
        self,
        action: str,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """根据意图 action 路由到具体 tool 执行。

        优先级：MCP Client（远程 MCP 协议）> 本地注册表 > HTTP REST API > 本地 mock
        """
        params = params or {}

        # ── 权限检查：基于角色的工具访问控制 ──
        if config.PERMISSION_ENABLED:
            client = get_current_client()
            if client is not None and not check_tool_permission(client, action):
                raise ToolPermissionError(
                    tool_name=action,
                    role=client.role.value,
                    client_id=client.client_id,
                )

        # ── 优先级 1：MCP Client（通过 MCP 协议调用远程工具）──
        mcp_result = await self._try_mcp_dispatch(action, params)
        if mcp_result is not None:
            return mcp_result

        # ── 优先级 2：本地注册表（HTTP REST / mock）──
        self._ensure_registry()
        tool_fn = self._registry.get(action)
        if tool_fn is None:
            logger.warning(f"未注册的意图 action: {action}，回退远程API")
            return await ToolService._call_remote_api(action, params)

        logger.info(f"Tool调用", action=action, params=params)
        return await tool_fn(params)

    # ── MCP Client 集成 ──────────────────────────────────────────

    @staticmethod
    async def _try_mcp_dispatch(action: str, params: Dict[str, Any]) -> Optional[str]:
        """尝试通过 MCP Client 调用远程工具。

        Returns:
            工具执行结果字符串，如果 MCP 不可用或工具不存在则返回 None
        """
        if not config.MCP_CLIENT_ENABLED:
            return None

        try:
            client = await get_mcp_client()
        except Exception as e:
            logger.warning(f"MCP Client 初始化失败: {e}")
            return None

        if not client.has_tool(action):
            return None

        logger.info(f"MCP tools/call", action=action, params=params)
        try:
            return await client.call_tool(action, params)
        except Exception as e:
            logger.error(f"MCP tools/call 失败: {action}: {e}，回退到本地/HTTP")
            return None

    # ── 远程 API ──────────────────────────────────────────────────

    @staticmethod
    async def _call_remote_api(
        action: str,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """调用远程业务API。参数名按远端 API 契约映射。"""
        base_url = config.REMOTE_API_BASE_URL
        if not base_url:
            logger.warning(f"REMOTE_API_BASE_URL 未配置，无法调用远程API (action={action})")
            return "抱歉，远程服务暂未配置，请联系管理员。"

        endpoint_map = {
            "query-order": "/api/orders/query",
            "check-shipping": "/api/shipping/track",
            "request-return": "/api/returns/create",
            "check-balance": "/api/account/balance",
            "coupon-inquiry": "/api/coupons/list",
        }
        # 内部参数名 → 远端参数名映射（与远端 API 契约对齐）
        param_mapping: Dict[str, Dict[str, str]] = {
            "query-order": {"phone": "mobile"},
        }
        endpoint = endpoint_map.get(action, f"/api/{action}")
        url = f"{base_url.rstrip('/')}{endpoint}"
        params = params or {}

        # 参数名映射：内部名 → 远端名
        mapping = param_mapping.get(action, {})
        mapped_params = {mapping.get(k, k): v for k, v in params.items()}

        logger.info(f"调用远程API", url=url, action=action, params=mapped_params)

        async with httpx.AsyncClient(timeout=config.REMOTE_API_TIMEOUT) as client:
            response = await client.post(url, json={"action": action, **mapped_params})
            response.raise_for_status()
            data = response.json()

        return ToolService._format_remote_api_response(action, data)

    @staticmethod
    def _format_remote_api_response(action: str, data: Dict[str, Any]) -> str:
        """将远程API响应格式化为自然语言"""
        if isinstance(data, dict):
            if "message" in data:
                return data["message"]
            if "data" in data and isinstance(data["data"], str):
                return data["data"]

        formatters = {
            "query-order": lambda d: f"您的订单信息如下：\n{d.get('message', json.dumps(d, ensure_ascii=False))}",
            "check-shipping": lambda d: f"物流进度：\n{d.get('message', json.dumps(d, ensure_ascii=False))}",
            "request-return": lambda d: f"退货申请：\n{d.get('message', json.dumps(d, ensure_ascii=False))}",
            "check-balance": lambda d: f"账户信息：\n{d.get('message', json.dumps(d, ensure_ascii=False))}",
            "coupon-inquiry": lambda d: f"优惠券信息：\n{d.get('message', json.dumps(d, ensure_ascii=False))}",
        }
        if action in formatters:
            return formatters[action](data)

        return json.dumps(data, ensure_ascii=False, indent=2)
