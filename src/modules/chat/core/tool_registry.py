"""
Tool 服务 —— 意图命中后的业务函数定义、远程 API 调用、格式化
"""
from typing import Dict, Any, Optional
import json
import httpx

from src.shared.logger import APILogger
from src.core.config import config

logger = APILogger("tool_service")


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
        """根据意图 action 路由到具体 tool 执行"""
        self._ensure_registry()
        tool_fn = self._registry.get(action)
        if tool_fn is None:
            logger.warning(f"未注册的意图 action: {action}，回退远程API")
            return await ToolService._call_remote_api(action, params)

        logger.info(f"Tool调用", action=action, params=params)
        return await tool_fn(params)

    # ── 远程 API ──────────────────────────────────────────────────

    @staticmethod
    async def _call_remote_api(
        action: str,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """调用远程业务API"""
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
        endpoint = endpoint_map.get(action, f"/api/{action}")
        url = f"{base_url.rstrip('/')}{endpoint}"
        params = params or {}

        logger.info(f"调用远程API", url=url, action=action, params=params)

        async with httpx.AsyncClient(timeout=config.REMOTE_API_TIMEOUT) as client:
            response = await client.post(url, json={"action": action, **params})
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
