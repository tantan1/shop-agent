"""
Mock 远程业务 API —— 模拟订单/物流/退货/余额/优惠券等远程接口

使用方式：
1. 在 .env 中设置 REMOTE_API_BASE_URL=http://localhost:8000/api/mockapi
2. 发送意图请求（如"查一下我的订单"），Agent 会走完整的意图识别→参数抽取→远程API调用链路
3. 本 router 模拟远程 API 响应，无需部署真实业务后端
"""
import json
from typing import Optional
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from src.shared.responses import success_response

router = APIRouter(prefix="/mockapi", tags=["Mock远程API测试"])


class MockApiRequest(BaseModel):
    action: str = Field(..., description="操作类型: query-order | check-shipping | request-return | check-balance | coupon-inquiry")
    order_id: Optional[str] = Field(default=None, description="订单号")
    phone: Optional[str] = Field(default=None, description="手机号后四位")
    status_filter: Optional[str] = Field(default=None, description="订单状态筛选")
    tracking_number: Optional[str] = Field(default=None, description="快递单号")
    reason: Optional[str] = Field(default=None, description="退货原因")
    coupon_type: Optional[str] = Field(default=None, description="优惠券类型")


# =============================================================================
# 模拟数据
# =============================================================================

MOCK_ORDERS = [
    {"id": "WB202405270001", "status": "已发货", "total": 299.00, "items": ["无线蓝牙耳机 Pro", "充电线"], "create_time": "2026-05-25 14:30"},
    {"id": "WB202405250016", "status": "派送中", "total": 158.00, "items": ["手机壳 透明款"], "create_time": "2026-05-23 09:15"},
    {"id": "WB202405200088", "status": "已签收", "total": 1280.00, "items": ["智能手表 S3"], "create_time": "2026-05-18 16:00"},
]

MOCK_TRACKING = {
    "SF1234567890": [
        {"time": "2026-05-27 10:30", "status": "到达目的地分拣中心", "location": "北京市朝阳区"},
        {"time": "2026-05-27 08:15", "status": "已揽收", "location": "上海市浦东新区"},
    ],
    "default": [
        {"time": "2026-05-27 11:00", "status": "运输中", "location": "转运中心"},
        {"time": "2026-05-27 06:30", "status": "已揽收", "location": "发货仓"},
    ],
}

MOCK_COUPONS = [
    {"name": "满200减30", "type": "满减券", "threshold": 200, "discount": 30, "expire": "2026-06-30"},
    {"name": "新用户满100减15", "type": "满减券", "threshold": 100, "discount": 15, "expire": "2026-06-15"},
    {"name": "全场9折", "type": "折扣券", "discount_rate": 0.9, "expire": "2026-06-10"},
    {"name": "免运费券", "type": "运费券", "expire": "2026-06-20"},
]


# =============================================================================
# Mock API 路由
# =============================================================================

@router.post("/api/orders/query", summary="[Mock] 查询订单")
async def mock_query_order(payload: dict):
    """模拟查询订单接口"""
    order_id = payload.get("order_id")
    if order_id:
        matched = [o for o in MOCK_ORDERS if o["id"] == order_id]
        if matched:
            return success_response(data={"order": matched[0]}, message="订单查询成功")
        return success_response(data={"order": None}, message=f"未找到订单 {order_id}", code=404)

    status_filter = payload.get("status_filter")
    if status_filter:
        filtered = [o for o in MOCK_ORDERS if status_filter in o.get("status", "")]
        return success_response(data={"orders": filtered, "total": len(filtered)}, message=f"状态={status_filter} 的订单")

    return success_response(data={"orders": MOCK_ORDERS, "total": len(MOCK_ORDERS)}, message="最近订单列表")


@router.post("/api/shipping/track", summary="[Mock] 查询物流")
async def mock_check_shipping(payload: dict):
    """模拟查询物流接口"""
    tracking_number = payload.get("tracking_number", "")
    order_id = payload.get("order_id", "")

    tracking_data = MOCK_TRACKING.get(tracking_number, MOCK_TRACKING["default"])

    result = {
        "tracking_number": tracking_number or "SF1234567890",
        "order_id": order_id or "WB202405270001",
        "current_status": tracking_data[0]["status"],
        "details": tracking_data,
    }
    return success_response(data=result, message="物流查询成功")


@router.post("/api/returns/create", summary="[Mock] 申请退货")
async def mock_request_return(payload: dict):
    """模拟退货申请接口"""
    order_id = payload.get("order_id", "未指定")
    reason = payload.get("reason", "未说明")

    result = {
        "return_id": f"RT{order_id[-6:]}",
        "order_id": order_id,
        "reason": reason,
        "status": "待审核",
        "refund_amount": 299.00,
        "expected_refund_time": "1-3个工作日",
    }
    return success_response(data=result, message="退货申请已提交")


@router.post("/api/account/balance", summary="[Mock] 查询余额/积分")
async def mock_check_balance(payload: dict):
    """模拟账户查询接口"""
    result = {
        "balance": 520.00,
        "points": 1280,
        "coupons_count": 3,
    }
    return success_response(data=result, message="账户查询成功")


@router.post("/api/coupons/list", summary="[Mock] 查询优惠券")
async def mock_coupon_inquiry(payload: dict):
    """模拟优惠券查询接口"""
    coupon_type = payload.get("coupon_type")
    if coupon_type:
        filtered = [c for c in MOCK_COUPONS if coupon_type in c.get("type", "")]
        return success_response(data={"coupons": filtered, "total": len(filtered)}, message=f"{coupon_type} 列表")

    return success_response(data={"coupons": MOCK_COUPONS, "total": len(MOCK_COUPONS)}, message="全部优惠券")


# =============================================================================
# 快捷测试接口（一条链走通意图→参数→MockAPI）
# =============================================================================

@router.get("/health", summary="[Mock] MockAPI 健康检查")
async def mockapi_health():
    """检查 mock API 是否可用"""
    return success_response(data={
        "status": "healthy",
        "endpoints": [
            "POST /api/orders/query       → 查询订单",
            "POST /api/shipping/track     → 查询物流",
            "POST /api/returns/create     → 申请退货",
            "POST /api/account/balance    → 查询余额",
            "POST /api/coupons/list       → 查询优惠券",
        ],
        "usage": "在 .env 中设置 REMOTE_API_BASE_URL=http://localhost:8000/api/mockapi 即可启用 Mock API",
    })
