"""
Tool 服务 —— 意图命中后的业务函数定义、远程 API 调用（HTTP / MCP）、格式化

远程调用优先级：MCP Client > HTTP REST > 本地 mock
"""

import json  # noqa: E402
from typing import Any, Dict, Optional  # noqa: E402

import httpx  # noqa: E402

from src.core.config import config  # noqa: E402
from src.core.permissions import (  # noqa: E402
    check_tool_permission,
    get_current_client,
)


class OrderServiceError(Exception):
    """订单服务调用失败（未配置 / 网络不可达 / 非 2xx），供上层优雅降级。"""


from src.modules.chat.core.mcp_client import get_mcp_client  # noqa: E402
from src.shared.logger import APILogger  # noqa: E402

logger = APILogger("tool_service")


# ── 工具注册表（装饰器驱动的插件式注册，对标 Skills 的"只加文件不改代码"）──
# 新增业务工具：在方法上加 @register_tool("name") 即可，无需改 _ensure_registry 字典。
_TOOL_REGISTRY: Dict[str, Any] = {}


def register_tool(name: str):
    """工具注册装饰器：将业务函数登记到全局工具注册表。

    用法：
        @register_tool("query-order")
        @staticmethod
        async def _tool_query_order(params=None) -> str: ...
    """

    def _wrap(fn):
        _TOOL_REGISTRY[name] = fn
        return fn

    return _wrap


# 本地 mock 已知数据集合（用于区分「查无此单」与「命中数据」）
KNOWN_TRACKING_NUMBERS = {
    "SF1234567890",
}
KNOWN_ORDER_IDS = {
    "WB202405270001",
    "WB202405250016",
    "WB202405200088",
}


class ToolPermissionError(PermissionError):
    """工具权限不足异常"""

    def __init__(self, tool_name: str, role: str, client_id: str):
        self.tool_name = tool_name
        self.role = role
        self.client_id = client_id
        super().__init__(f"权限不足：调用方 {client_id}（角色 {role}）无权调用工具 '{tool_name}'")


# ── MCP 敏感工具 HITL 清单 ──
# 经 MCP 调用这些工具时需人工确认（先拦截生成审批单，approve 时才真实调用远程服务）。
# 设计来源：order-service MCP 工具表（risk=high, hitl=true）。可由运维按需在 config 扩展。
MCP_SENSITIVE_TOOLS = {
    "check-balance",
    "request-return",
    "refund-confirm",
}


class McpToolCommand:
    """MCP 远程工具的命令封装（供 ApprovalGate 管理 HITL）。

    采用「先拦截后执行」模型：dispatch 阶段只 create_pending_approval（不真实调用），
    人工 approve 时 ApprovalGate.approve → command.execute 才真正 call_tool，避免未授权副作用。
    """

    def __init__(self, action: str, params: Dict[str, Any]):
        self.command_name = action
        self._params = params

    async def execute(self, ctx) -> "ToolResult":
        from src.modules.chat.agent.tool_commands import ToolResult

        # 真实调用远程 MCP 工具（高后果字段由 hardcode 注入，详见 _try_mcp_dispatch）
        try:
            client = await get_mcp_client()
        except Exception as e:
            return ToolResult(status="failed", error=f"MCP 客户端不可用: {e}")

        from src.modules.chat.core.mcp_client import HIGH_CONSEQUENCE_FIELDS

        hardcode = {k: v for k, v in self._params.items() if k in HIGH_CONSEQUENCE_FIELDS}
        call_params = {k: v for k, v in self._params.items() if k not in HIGH_CONSEQUENCE_FIELDS}
        try:
            result = await client.call_tool(self.command_name, call_params, hardcode=hardcode or None)
            return ToolResult(status="success", data=result, message=result)
        except Exception as e:
            return ToolResult(status="failed", error=str(e))

    async def undo(self, ctx) -> "ToolResult":
        from src.modules.chat.agent.tool_commands import ToolResult

        # MCP 远程撤销需服务端支持，默认标记不支持
        return ToolResult(status="failed", error=f"{self.command_name} 不支持撤销")


class ToolService:
    """工具执行服务：Tool 定义 + 分发 + 远程 API 调用"""

    def __init__(self):
        self._registry: Dict[str, Any] = {}

    def _ensure_registry(self):
        """懒加载 tool 注册表（装饰器扫描结果，无需硬编码）"""
        if self._registry:
            return
        if not _TOOL_REGISTRY:
            logger.warning("工具注册表为空，未装饰任何 @register_tool")
            return
        self._registry.update(_TOOL_REGISTRY)
        logger.debug(f"工具注册表已加载，共 {len(self._registry)} 个工具")

    # ── Tool 实现 ──────────────────────────────────────────────────

    @register_tool("query-order")
    @staticmethod
    async def _tool_query_order(params: Optional[Dict[str, Any]] = None) -> str:
        """查询订单"""
        params = params or {}
        order_id = params.get("order_id")
        if config.REMOTE_API_BASE_URL:
            return await ToolService._call_remote_api("query-order", params)
        if order_id:
            if order_id not in KNOWN_ORDER_IDS:
                return json.dumps(
                    {
                        "order": None,
                        "found": False,
                        "note": f"未查询到订单 {order_id}，请核对订单号后重试",
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "order": {"id": order_id, "status": "派送中", "total": 299.00},
                    "found": True,
                    "note": f"已按 order_id={order_id} 查询",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "orders": [
                    {"id": "202405270001001", "status": "已发货", "total": 299.00},
                    {"id": "202405250016001", "status": "派送中", "total": 158.00},
                ],
                "note": "未指定 order_id，返回最近订单",
            },
            ensure_ascii=False,
        )

    @register_tool("check-shipping")
    @staticmethod
    async def _tool_check_shipping(params: Optional[Dict[str, Any]] = None) -> str:
        """查询物流"""
        params = params or {}
        tracking = params.get("tracking_number")
        if config.REMOTE_API_BASE_URL:
            return await ToolService._call_remote_api("check-shipping", params)
        if tracking:
            if tracking not in KNOWN_TRACKING_NUMBERS:
                return json.dumps(
                    {
                        "tracking_number": tracking,
                        "found": False,
                        "note": f"未查询到快递单号 {tracking} 的物流记录，请核对单号后重试",
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "tracking_number": tracking,
                    "found": True,
                    "tracking": [
                        {"time": "05-27 10:30", "status": "到达分拣中心"},
                        {"time": "05-27 08:15", "status": "已揽收"},
                    ],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "tracking": [
                    {"time": "05-27 10:30", "status": "到达分拣中心"},
                    {"time": "05-27 08:15", "status": "已揽收"},
                ],
                "note": "未指定快递单号，请提供快递单号后查询",
            },
            ensure_ascii=False,
        )

    @register_tool("request-return")
    @staticmethod
    async def _tool_request_return(params: Optional[Dict[str, Any]] = None) -> str:
        """申请退货退款。

        优先级：远程 Mock API > 订单服务 > 本地 Mock（订单服务不可用时优雅降级，
        保证 mockapi 演示链路可用，不会把「服务不可用」直接抛给用户）。
        """
        if config.REMOTE_API_BASE_URL:
            return await ToolService._call_remote_api("request-return", params)
        if getattr(config, "ORDER_SERVICE_URL", ""):
            try:
                return await ToolService._call_order_api("request-return", params)
            except OrderServiceError:
                logger.warning("订单服务不可用，回退本地 Mock (action=request-return)")
        params = params or {}
        order_id = params.get("order_id", "未指定")
        reason = params.get("reason", "未说明")
        return json.dumps(
            {
                "return_id": f"RT{str(order_id)[-6:]}",
                "order_id": order_id,
                "reason": reason,
                "status": "待审核",
                "refund_amount": 299.00,
                "expected_refund_time": "1-3个工作日",
                "found": True,
            },
            ensure_ascii=False,
        )

    @staticmethod
    async def refund_confirmation(order_id: str, reason: str, refund_amount: float = 0.0) -> None:
        """记录退款确认请求到订单服务（真实写库，待人工审批）。

        替代原 mock_refund_confirmation（仅打印）。订单服务不可用时仅记日志、不抛错，
        人工审批流程由 /agent/refund/confirm 独立驱动。

        ── Phase 5：REST 降级通道（直连 POST）──
        该退款确认当前经 REST 直连订单服务，尚未迁移到 MCP（order-service 的 MCP 写工具
        refund-confirm 尚未实现）。dispatch 优先级中 refund-confirm 若走 MCP 路径且已暴露，
        将经由 McpToolCommand + HITL；本直连通道作为该工具 MCP 化前的承载/兜底。
        下线条件：order-service 补齐 refund-confirm MCP 工具且生产验证通过后，移除本直连 POST。
        """
        base_url = getattr(config, "ORDER_SERVICE_URL", "")
        if not base_url:
            logger.warning("订单服务未配置，跳过退款确认记录")
            return
        url = f"{base_url.rstrip('/')}/api/refunds/confirm"
        try:
            async with httpx.AsyncClient(
                timeout=getattr(config, "ORDER_SERVICE_TIMEOUT", 5)
            ) as client:
                resp = await client.post(
                    url,
                    json={
                        "order_id": order_id,
                        "reason": reason,
                        "refund_amount": refund_amount,
                    },
                )
                resp.raise_for_status()
        except Exception as e:
            logger.error(f"记录退款确认失败: {e}")

    @register_tool("check-balance")
    @staticmethod
    async def _tool_check_balance(params: Optional[Dict[str, Any]] = None) -> str:
        """查询余额/积分。

        优先级：远程 Mock API > 订单服务 > 本地 Mock（订单服务不可用时优雅降级，
        保证 mockapi 演示链路可用，不会把「服务不可用」直接抛给用户）。
        """
        if config.REMOTE_API_BASE_URL:
            return await ToolService._call_remote_api("check-balance", params)
        if getattr(config, "ORDER_SERVICE_URL", ""):
            try:
                return await ToolService._call_order_api("check-balance", params)
            except OrderServiceError:
                logger.warning("订单服务不可用，回退本地 Mock (action=check-balance)")
        return json.dumps(
            {
                "balance": 520.00,
                "points": 1280,
                "coupons_count": 3,
                "found": True,
            },
            ensure_ascii=False,
        )

    @register_tool("coupon-inquiry")
    @staticmethod
    async def _tool_coupon_inquiry(params: Optional[Dict[str, Any]] = None) -> str:
        """查询优惠券。

        优先级：远程 Mock API > 订单服务 > 本地 Mock（订单服务不可用时优雅降级，
        保证 mockapi 演示链路可用，不会把「服务不可用」直接抛给用户）。
        """
        if config.REMOTE_API_BASE_URL:
            return await ToolService._call_remote_api("coupon-inquiry", params)
        if getattr(config, "ORDER_SERVICE_URL", ""):
            try:
                return await ToolService._call_order_api("coupon-inquiry", params)
            except OrderServiceError:
                logger.warning("订单服务不可用，回退本地 Mock (action=coupon-inquiry)")
        params = params or {}
        coupons = [
            {
                "name": "满200减30",
                "type": "满减券",
                "threshold": 200,
                "discount": 30,
                "expire": "2026-06-30",
            },
            {
                "name": "新用户满100减15",
                "type": "满减券",
                "threshold": 100,
                "discount": 15,
                "expire": "2026-06-15",
            },
            {"name": "全场9折", "type": "折扣券", "discount_rate": 0.9, "expire": "2026-06-10"},
            {"name": "免运费券", "type": "运费券", "expire": "2026-06-20"},
        ]
        coupon_type = params.get("coupon_type")
        if coupon_type:
            coupons = [c for c in coupons if coupon_type in c.get("type", "")]
        return json.dumps(
            {
                "coupons": coupons,
                "total": len(coupons),
                "found": True,
            },
            ensure_ascii=False,
        )

    @staticmethod
    async def _call_order_api(
        action: str,
        params: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """调用订单服务（Rust + PostgreSQL）的业务 API，替代原本地 Mock。

        契约与 _call_remote_api 兼容：POST {ORDER_SERVICE_URL}/api/<endpoint>，
        响应形如 {"message": ..., "data": {...}}，最终走 _format_remote_api_response 格式化。

        方案 A 配套：将调用方身份 user_id 注入请求体，由后端做数据级归属校验；
        后端返回 403/404 时转为对用户友好的反问话术（不泄漏原文）。

        ── Phase 5 评估结论（REST 降级通道）──
        dispatch 优先级为「MCP Client > 本地注册表/REST > mockapi」。当 MCP 已启用且对应
        工具存在时，调用走 MCP（含高后果字段硬强制 + HITL），本 REST 通道不会触发。
        当前 order-service 的 MCP Server 仅暴露只读工具（query-order / get-evidence /
        list-coupons），写操作与余额类工具（check-balance / request-return / refund-confirm
        / coupon-inquiry）仍需本 REST 通道承载。
        ⇒ 现阶段【保留】本 REST 降级通道作为过渡；下线条件：order-service 补齐上述 MCP 写工具
          且生产验证通过后，移除 _execute_* 中的 ORDER_SERVICE_URL 分支与 _record_refund_confirm
          的直连 POST。本通道当前是「MCP 不可用/未覆盖时的兜底」，非主路径。

        Raises:
            OrderServiceError: 订单服务未配置或调用失败（供上层优雅降级到本地 Mock）。
        """
        base_url = getattr(config, "ORDER_SERVICE_URL", "")
        if not base_url:
            raise OrderServiceError("订单服务未配置")
        endpoint_map = {
            "check-balance": "/api/account/balance",
            "coupon-inquiry": "/api/coupons/list",
            "request-return": "/api/returns/create",
            "refund-confirm": "/api/refunds/confirm",
        }
        endpoint = endpoint_map.get(action, f"/api/{action}")
        url = f"{base_url.rstrip('/')}{endpoint}"
        params = params or {}
        # 注入调用方身份，供后端归属校验（方案 A）
        # 未显式传入时，从当前会话调用方自动取 client_id 作为 user_id
        if not user_id:
            try:
                cur = get_current_client()
                if cur is not None:
                    user_id = cur.client_id
            except Exception:
                user_id = None
        if user_id:
            params = {**params, "user_id": user_id}
        try:
            async with httpx.AsyncClient(
                timeout=getattr(config, "ORDER_SERVICE_TIMEOUT", 5)
            ) as client:
                resp = await client.post(url, json={"action": action, **params})
                if resp.status_code in (403, 404):
                    # 后端拒绝（无权限 / 订单不存在）→ 友好反问，不泄漏 403 原文
                    logger.warning(f"订单服务拒绝 ({action}): {resp.status_code}")
                    return (
                        "抱歉，您提供的订单号似乎不存在，或不属于当前账号。"
                        "请核对订单号后重试，或联系客服协助处理。"
                    )
                resp.raise_for_status()
                data = resp.json()
            return ToolService._format_remote_api_response(action, data)
        except httpx.HTTPStatusError as e:
            # 其他 4xx/5xx 也走友好兜底，避免把后端错误原文透传给用户
            if e.response.status_code in (403, 404):
                return (
                    "抱歉，您提供的订单号似乎不存在，或不属于当前账号。"
                    "请核对订单号后重试，或联系客服协助处理。"
                )
            logger.error(f"订单服务调用失败 ({action}): {e}")
            raise OrderServiceError(str(e)) from e
        except Exception as e:
            logger.error(f"订单服务调用失败 ({action}): {e}")
            raise OrderServiceError(str(e)) from e

    # ── 分发入口 ──────────────────────────────────────────────────

    async def dispatch(self, action: str, params: Optional[Dict[str, Any]] = None) -> str:
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

        logger.info("Tool调用", action=action, params=params)
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

        # ── 硬强制：高后果字段（order_id 等）由确定性来源注入，覆盖模型输入（防篡改）──
        # params 中的高后果字段来自上游 _prepare_intent_params 的 extra_required 正则提取，
        # 此处显式作为 hardcode 传入 call_tool，确保即便模型侧传入也以确定性值为准。
        from src.modules.chat.core.mcp_client import HIGH_CONSEQUENCE_FIELDS

        hardcode = {k: v for k, v in params.items() if k in HIGH_CONSEQUENCE_FIELDS}
        call_params = {k: v for k, v in params.items() if k not in HIGH_CONSEQUENCE_FIELDS}

        # ── 敏感工具 HITL：先拦截生成审批单，approve 时才真实调用远程服务 ──
        if action in MCP_SENSITIVE_TOOLS:
            from src.modules.chat.agent.tool_commands import (
                ApprovalGate,
                ToolContext,
            )

            ctx = ToolContext(
                action=action,
                params=params,
                conversation_id=str(params.get("conversation_id", "")),
            )
            command = McpToolCommand(action, params)
            gate = ApprovalGate()
            approval_id = await gate.create_pending_approval(
                command,
                ctx,
                message=f"该操作（{action}）需人工确认后执行。",
            )
            logger.info("MCP 敏感工具进入 HITL", action=action, approval_id=approval_id)
            return json.dumps(
                {
                    "status": "waiting_for_confirmation",
                    "approval_id": approval_id,
                    "action": action,
                    "message": f"该操作（{action}）需人工确认后执行。请回复「确认」以继续。",
                },
                ensure_ascii=False,
            )

        logger.info("MCP tools/call", action=action, params=call_params, hardcode=hardcode)
        try:
            return await client.call_tool(action, call_params, hardcode=hardcode or None)
        except Exception as e:
            logger.error(f"MCP tools/call 失败: {action}: {e}，回退到本地/HTTP")
            return None

    # ── MCP HITL 确认入口 ────────────────────────────────────────
    # 供对话层在用户「确认」后调用：approve 时 ApprovalGate.approve → McpToolCommand.execute
    # 才真实 call_tool（先拦截后执行模型）。

    @staticmethod
    async def approve_mcp_tool(approval_id: str) -> str:
        """审批通过 MCP 敏感工具：真实调用远程服务。"""
        from src.modules.chat.agent.tool_commands import ApprovalGate

        gate = ApprovalGate()
        result = await gate.approve(approval_id)
        if result.status == "success":
            return result.data if isinstance(result.data, str) else str(result.data)
        return f"审批执行失败：{result.error or result.message}"

    @staticmethod
    async def reject_mcp_tool(approval_id: str) -> str:
        """审批拒绝 MCP 敏感工具：撤销（远程不支持时为 no-op）。"""
        from src.modules.chat.agent.tool_commands import ApprovalGate

        gate = ApprovalGate()
        result = await gate.reject(approval_id)
        return result.message or "已拒绝该操作。"

    # ── 远程 API ──────────────────────────────────────────────────

    @staticmethod
    async def _call_remote_api(action: str, params: Optional[Dict[str, Any]] = None) -> str:
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

        logger.info("调用远程API", url=url, action=action, params=mapped_params)

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
            "query-order": lambda d: (
                f"您的订单信息如下：\n{d.get('message', json.dumps(d, ensure_ascii=False))}"
            ),
            "check-shipping": lambda d: (
                f"物流进度：\n{d.get('message', json.dumps(d, ensure_ascii=False))}"
            ),
            "request-return": lambda d: (
                f"退货申请：\n{d.get('message', json.dumps(d, ensure_ascii=False))}"
            ),
            "check-balance": lambda d: (
                f"账户信息：\n{d.get('message', json.dumps(d, ensure_ascii=False))}"
            ),
            "coupon-inquiry": lambda d: (
                f"优惠券信息：\n{d.get('message', json.dumps(d, ensure_ascii=False))}"
            ),
        }
        if action in formatters:
            return formatters[action](data)

        return json.dumps(data, ensure_ascii=False, indent=2)
