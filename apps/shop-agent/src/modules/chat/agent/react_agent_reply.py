"""
ReAct Agent 回复构建模块（从 react_agent.py 拆分的场景化回复逻辑）。
"""
from __future__ import annotations

import json
import re as _re

from langchain_core.messages import AIMessage, ToolMessage


def is_scenario_covered(final_output: str) -> bool:
    """检查最终答复是否已覆盖查无/缺参数场景。"""
    _NOT_FOUND = (
        "未查询到", "未找到", "查无", "不存在", "已过期",
        "没有找到", "无此单号", "暂无该",
    )
    _MISSING_ID = (
        "未指定快递单号", "未指定订单号", "请提供快递单号", "请提供订单号",
        "缺少快递单号", "缺少订单号", "未提供快递单号", "未提供订单号",
    )
    return any(p in final_output for p in _NOT_FOUND + _MISSING_ID)


def extract_identifier(tool_input: dict, observation: str) -> str:
    """从工具输入/观测中取关键标识符。"""
    for key in ("tracking_number", "order_id", "phone"):
        val = (tool_input or {}).get(key)
        if val:
            return str(val)
    try:
        data = json.loads(observation)
        for key in ("tracking_number", "order_id", "phone"):
            val = data.get(key)
            if val:
                return str(val)
    except Exception:
        pass
    m = _re.search(r"[A-Z]{2}\d{6,}", observation)
    return m.group(0) if m else ""


def build_missing_id_reply(tool_name: str) -> str:
    """构建缺参数回复。"""
    if tool_name == "check-shipping":
        return "请提供您的快递单号，我来帮您查询物流进度。"
    if tool_name == "query-order":
        return "请提供您的订单号，我来帮您查询订单信息。"
    return "您提供的信息可能不完整，请补充订单号或快递单号后重试。"


def build_not_found_reply(tool_name: str, ident: str) -> str:
    """构建查无结果回复。"""
    ident_text = f"「{ident}」" if ident else ""
    if tool_name == "check-shipping":
        return (
            f"抱歉，未查询到快递单号{ident_text}的物流信息，"
            "该单号可能不存在或已过期，请核对后重试。"
        )
    if tool_name == "query-order":
        return f"抱歉，未查询到订单号{ident_text}的订单信息，请核对订单号后重试。"
    return "抱歉，暂时未查询到相关信息，请核对您提供的信息后重试。"


def build_shipping_summary(data: dict, tool_input: dict, observation: str) -> str | None:
    """构建物流查询概要。"""
    tn = data.get("tracking_number") or extract_identifier(tool_input, observation)
    tracks = data.get("tracking") or []
    lines = [
        f"- {t.get('time', '')} {t.get('status', '')}"
        for t in tracks
        if isinstance(t, dict) and (t.get("time") or t.get("status"))
    ]
    if lines:
        head = f"快递{(' ' + tn) if tn else ''}的物流进度如下："
        return head + "\n" + "\n".join(lines)
    return None


def build_order_summary(data: dict) -> str | None:
    """构建订单查询概要。"""
    order = data.get("order")
    if isinstance(order, dict):
        return (
            f"订单号 {order.get('id', '')} 当前状态：{order.get('status', '')}，"
            f"支付金额 {order.get('total', '')} 元。"
        )
    return None


def build_balance_summary(data: dict) -> str | None:
    """构建余额查询概要。"""
    if data.get("balance") is not None:
        return (
            f"您的账户余额为 {data.get('balance')} 元，"
            f"积分 {data.get('points', 0)}，"
            f"可用优惠券 {data.get('coupons_count', 0)} 张。"
        )
    return None


def build_coupon_summary(data: dict) -> str | None:
    """构建优惠券查询概要。"""
    coupons = data.get("coupons") or []
    lines = []
    for c in coupons[:6]:
        if not isinstance(c, dict):
            continue
        name = c.get("name", "")
        expire = c.get("expire", "")
        lines.append(f"- {name}（{expire}前有效）" if expire else f"- {name}")
    if lines:
        return "您可用的优惠券如下：\n" + "\n".join(lines)
    return None


def build_summary_from_observation(tool_name: str, tool_input: dict, observation: str) -> str | None:
    """从工具 observation 确定性生成概要。"""
    try:
        data = json.loads(str(observation))
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("found") is False:
        return None

    builders = {
        "check-shipping": lambda: build_shipping_summary(data, tool_input, observation),
        "query-order": lambda: build_order_summary(data),
        "check-balance": lambda: build_balance_summary(data),
        "coupon-inquiry": lambda: build_coupon_summary(data),
    }
    builder = builders.get(tool_name)
    return builder() if builder else None


def apply_scenario_reply(final_output: str, intermediate_steps: list) -> str:
    """场景化回退回复。"""
    if is_scenario_covered(final_output):
        return final_output

    for tool_name, tool_input, observation in intermediate_steps:
        obs = str(observation)
        if any(p in obs for p in (
            "未指定快递单号", "未指定订单号", "请提供快递单号", "请提供订单号",
            "缺少快递单号", "缺少订单号", "未提供快递单号", "未提供订单号",
        )):
            return build_missing_id_reply(tool_name)

        if any(p in obs for p in (
            "未查询到", "未找到", "查无", "不存在", "已过期",
            "没有找到", "无此单号", "暂无该",
        )):
            ident = extract_identifier(tool_input, obs)
            return build_not_found_reply(tool_name, ident)

        from src.modules.chat.agent.react_agent_utils import _looks_like_ack
        if _looks_like_ack(final_output):
            summary = build_summary_from_observation(tool_name, tool_input, obs)
            if summary:
                return summary

    return final_output


def parse_messages(messages: list) -> tuple[str, list]:
    """从 LangGraph messages 列表中提取最终回复和中间步骤。"""
    final_output = ""
    intermediate_steps = []

    pending_calls = {}

    for msg in messages:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    pending_calls[tc["id"]] = (
                        tc.get("name", "unknown"),
                        tc.get("args", {}),
                    )
            elif msg.content and not msg.tool_calls:
                final_output = str(msg.content)
            elif msg.content:
                final_output = str(msg.content)
        elif isinstance(msg, ToolMessage):
            tc_id = getattr(msg, "tool_call_id", "")
            if tc_id in pending_calls:
                tool_name, tool_input = pending_calls.pop(tc_id)
                intermediate_steps.append((tool_name, tool_input, str(msg.content)))
            else:
                tool_name = getattr(msg, "name", "unknown")
                intermediate_steps.append((tool_name, {}, str(msg.content)))

    return final_output, intermediate_steps


def format_intermediate_steps(
    intermediate_steps: list,
    total_elapsed_ms: int,
) -> list:
    """将 (tool_name, tool_input, observation) 元组列表转成统一 step 格式"""
    steps = []
    for idx, (tool_name, tool_input, observation) in enumerate(intermediate_steps):
        obs_str = str(observation)[:300]
        name = str(tool_name)

        steps.append(
            {
                "step_name": f"ReAct-{name}",
                "step_order": idx + 1,
                "status": "success",
                "output_data": {
                    "tool": name,
                    "tool_input": tool_input,
                    "observation": obs_str,
                },
            }
        )

    steps.append(
        {
            "step_name": "ReAct-总结",
            "step_order": len(steps) + 1,
            "status": "success",
            "output_data": {
                "total_iterations": len(intermediate_steps),
                "duration_ms": total_elapsed_ms,
            },
        }
    )

    return steps
