"""
观察脚本：展示纠纷协调最典型流程的完整输入输出。

运行: python tests/inspect_dispute_flow.py
"""

import sys
import io
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock

# 强制 UTF-8 输出，避免 Windows GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.modules.chat.agent.dispute_coordinator import (
    DisputeCoordinator,
    BUYER_AGENT_PROMPT,
    SELLER_AGENT_PROMPT,
    MEDIATOR_AGENT_PROMPT,
    PLATFORM_RULING_POLICY,
)
from src.modules.chat.schemas import ChatRequest
from src.modules.chat.core.sentiment_service import EmotionResult, EmotionLevel


# ═══════════════════════════════════════════════════════════════════════
# Mock 数据：三个 Agent 的受控输出
# ═══════════════════════════════════════════════════════════════════════

BUYER_AGENT_OUTPUT = json.dumps({
    "core_issue": "买家收到洗衣机时玻璃面板已碎裂，外包装有撞击痕迹，认为快递运输造成",
    "buyer_demands": ["全额退款 ¥3299", "或换货处理"],
    "emotion_intensity": "severe",
    "mentioned_evidence": [
        "破损照片4张（面板碎裂特写、外包装撞击凹陷、物流面单）",
        "签收后2小时内即发起投诉",
        "历史购物12单无退款记录"
    ],
    "compensation_expectation": "全额退款 ¥3299 或免费换新",
    "buyer_summary": "买家证据充分：4张破损照片清晰显示面板碎裂和外包装撞击痕迹，签收后2小时内发起投诉时效合理。用户历史12单无退款记录可排除恶意投诉。买家诉求合理，应退款或换货。"
}, ensure_ascii=False)

SELLER_AGENT_OUTPUT = json.dumps({
    "rule_assessment": "卖家已提供发货前完整验机视频（58秒，无剪辑），面板完好。根据平台规则，卖家已完成发货义务，本次破损由快递运输造成。",
    "seller_faults": [],
    "seller_defenses": [
        "发货前58秒验机视频全程无剪辑，玻璃面板完好",
        "已购买顺丰保价 ¥3000",
        "商家已按平台要求履行发货验机义务"
    ],
    "acceptable_solutions": [
        "由快递公司承担保价理赔",
        "平台信用垫付退款后向快递追偿",
        "卖家可协助提供验机视频作为理赔证据"
    ],
    "escalation_risk": "low",
    "seller_summary": "卖家已履行发货义务，验机视频证据完整。物流保价¥3000已购买。卖家无责，破损由快递运输造成，应由快递公司赔付。"
}, ensure_ascii=False)

MEDIATOR_AGENT_OUTPUT = json.dumps({
    "verdict": "双方证据有效，责任方为快递公司。卖家先行退款，平台发起保价理赔，差额由平台垫付。",
    "suggested_solution": (
        "① 卖家先行退款 ¥3299（平台信用垫付，T+1到账至您原支付方式）\n"
        "② 平台方向顺丰速运发起保价理赔（保价金额 ¥3000）\n"
        "③ 差额 ¥299 由平台纠纷基金垫付（用户体验优先，无需您承担）\n"
        "④ 退款到账后，如需重新购买同款商品可享 95 折优惠"
    ),
    "responsibility_split": {"buyer_percent": 0, "seller_percent": 0, "courier_percent": 100},
    "responsible_party": "courier",
    "matched_rule": "R1, C1",
    "compensation": {
        "type": "insurance_claim",
        "amount_yuan": 3299,
        "detail": "卖家先行退款¥3299，平台向顺丰发起保价理赔¥3000，差额¥299由平台纠纷基金垫付"
    },
    "escalate_to_human": False,
    "escalate_reason": "",
    "mediator_summary": "根据平台规则R1（快递破损判定）：卖家提供了发货前验机视频（面板完好），买家在签收后2小时内提供了4张破损照片，时效合理。判定快递公司承担100%责任。赔付按规则C1执行：卖家先行退款¥3299（平台信用垫付），平台向顺丰发起保价理赔¥3000，差额¥299由平台纠纷基金垫付。"
}, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════════════

_SEP = "=" * 70
_SEP2 = "-" * 70


def _make_mock_llm(responses: list[str]):
    """返回 Mock LLMService，按顺序返回固定响应"""
    mock = MagicMock()
    prompts_received = []

    async def mock_chat_qwen(prompt: str = "", system_prompt: str = "", **kwargs):
        idx = mock._call_count
        result = responses[min(idx, len(responses) - 1)]
        prompts_received.append(prompt)
        mock._call_count = idx + 1
        return result

    mock._call_count = 0
    mock.chat_qwen_with_prompt = AsyncMock(side_effect=mock_chat_qwen)
    mock._prompts = prompts_received
    return mock


def _make_mock_tool(order_data: str = "", shipping_data: str = ""):
    mock = MagicMock()

    async def mock_dispatch(action: str, params: dict = None):
        if action == "query-order":
            if order_data:
                return order_data
            return json.dumps({
                "orders": [{
                    "order_id": "ORDER_WM20240601_001",
                    "product": "全自动滚筒洗衣机 XQG100-2024",
                    "amount_yuan": 3299,
                    "status": "已完成",
                    "created_at": "2024-06-01 10:00:00"
                }]
            }, ensure_ascii=False)
        elif action == "check-shipping":
            if shipping_data:
                return shipping_data
            return json.dumps({
                "tracking": {
                    "tracking_number": "SF123456789",
                    "carrier": "顺丰速运",
                    "status": "已签收",
                    "delivered_at": "2024-06-03 12:15:00",
                    "signed_by": "本人签收"
                }
            }, ensure_ascii=False)
        elif action == "check-balance":
            return json.dumps({"balance": 520.00, "points": 1280}, ensure_ascii=False)
        elif action == "coupon-inquiry":
            return json.dumps({"coupons": [{"name": "满200减30", "expire": "2024-12-31"}]}, ensure_ascii=False)
        return json.dumps({"error": f"未知 action: {action}"}, ensure_ascii=False)

    mock.dispatch = AsyncMock(side_effect=mock_dispatch)
    return mock


async def main():
    # ────────────────────────────────────────────────────────
    # 场景说明
    # ────────────────────────────────────────────────────────
    print(_SEP)
    print("场景：洗衣机面板碎裂纠纷")
    print(_SEP)
    print("""
用户投诉: "收到洗衣机的玻璃面板碎了，申请退款"
卖家回复: "发货时有验机视频，面板完好，是快递造成的"

订单信息:
  - 订单号: ORDER_WM20240601_001
  - 商品: 全自动滚筒洗衣机 XQG100-2024
  - 金额: ¥3,299
  - 物流: 顺丰速运 SF123456789（已保价 ¥3,000）
  - 签收时间: 2024-06-03 12:15
  - 投诉时间: 2024-06-03 14:22（签收后 2 小时）
""")

    # ── 构造 Mock ──
    order_json = json.dumps({
        "orders": [{
            "order_id": "ORDER_WM20240601_001",
            "product": "全自动滚筒洗衣机 XQG100-2024",
            "amount_yuan": 3299,
            "status": "已完成",
            "created_at": "2024-06-01 10:00:00"
        }]
    }, ensure_ascii=False)

    shipping_json = json.dumps({
        "tracking": {
            "tracking_number": "SF123456789",
            "carrier": "顺丰速运",
            "status": "已签收",
            "shipped_at": "2024-06-01 15:00:00",
            "delivered_at": "2024-06-03 12:15:00",
            "signed_by": "本人签收"
        }
    }, ensure_ascii=False)

    mock_llm = _make_mock_llm([
        BUYER_AGENT_OUTPUT,
        SELLER_AGENT_OUTPUT,
        MEDIATOR_AGENT_OUTPUT,
    ])
    mock_tool = _make_mock_tool(order_json, shipping_json)

    coordinator = DisputeCoordinator(llm=mock_llm, tool_service=mock_tool)
    request = ChatRequest(message="收到洗衣机的玻璃面板碎了，申请退款", conversation_id="test_conv_001")
    emotion = EmotionResult(
        level=EmotionLevel.ANGRY,
        confidence=0.95,
        escalate=True,
        is_emergency=False,
        keywords=["退款", "投诉"],
    )

    # ── 执行 ──
    response = await coordinator.resolve(
        request=request,
        emotion_result=emotion,
        conversation_id="test_conv_001",
        order_id="ORDER_WM20240601_001",
    )

    # ═══════════════════════════════════════════════════════════════
    # 输出：按真实流程，展示每个 Agent 的输入输出
    # ═══════════════════════════════════════════════════════════════

    print(_SEP)
    print("STEP 1: 事实收集 (FactCollector)")
    print(_SEP)
    print("调用 5 个数据源，并行收集：")
    print("  ① query-order     → 订单系统")
    print("  ② check-shipping  → 物流系统")
    print("  ③ after_sale_evidence → 售后举证（Mock API）")
    print("  ④ check-balance   → 账户余额")
    print("  ⑤ coupon-inquiry  → 优惠券")
    print()

    facts_step = next((s for s in response.steps if s["step_name"] == "纠纷协调-事实收集"), None)
    if facts_step:
        print("收集结果 (keys):", facts_step["output_data"]["keys"])
        print(f"耗时: {facts_step['output_data']['duration_ms']}ms")

    # ═══════════════════════════════════════════════════════════════
    print()
    print(_SEP)
    print("STEP 2a: BuyerAgent — 买家立场分析 (与 SellerAgent 并行)")
    print(_SEP)

    # 这里 prompt 来自第一次 LLM 调用
    buyer_prompt = mock_llm._prompts[0] if len(mock_llm._prompts) > 0 else "(未捕获)"
    print()
    print("【输入 — System Prompt】")
    print(BUYER_AGENT_PROMPT)
    print()
    print(f"【输入 — 拼接上下文】（长度 {len(buyer_prompt)} 字符）")
    # 只展示 prompt 中 BUYER_AGENT_PROMPT 之后的部分（事实 + 情绪 + 用户消息）
    buyer_context = buyer_prompt.replace(BUYER_AGENT_PROMPT, "").strip()
    # 截取前 2000 字符
    if len(buyer_context) > 2000:
        buyer_context = buyer_context[:2000] + "\n... [截断]"
    print(buyer_context)
    print()
    print("【输出 — BuyerAgent JSON】")
    print(BUYER_AGENT_OUTPUT)
    print()
    print("【解析 — AgentPerspective】")
    print(f"  summary         : 买家证据充分：4张破损照片清晰显示面板碎裂...")
    print(f"  demands         : ['全额退款 ¥3299', '或换货处理']")
    print(f"  evidence        : [破损照片4张, 签收后2小时内投诉, 历史12单无退款]")
    print(f"  proposed_solution: 全额退款 ¥3299 或免费换新")
    print(f"  confidence      : 0.85")

    # ═══════════════════════════════════════════════════════════════
    print()
    print(_SEP)
    print("STEP 2b: SellerAgent — 卖家合规评估 (与 BuyerAgent 并行)")
    print(_SEP)

    seller_prompt = mock_llm._prompts[1] if len(mock_llm._prompts) > 1 else "(未捕获)"
    print()
    print("【输入 — System Prompt】")
    print(SELLER_AGENT_PROMPT)
    print()
    print(f"【输入 — 拼接上下文】（长度 {len(seller_prompt)} 字符）")
    seller_context = seller_prompt.replace(SELLER_AGENT_PROMPT, "").strip()
    if len(seller_context) > 2000:
        seller_context = seller_context[:2000] + "\n... [截断]"
    print(seller_context)
    print()
    print("【输出 — SellerAgent JSON】")
    print(SELLER_AGENT_OUTPUT)
    print()
    print("【解析 — AgentPerspective】")
    print(f"  summary         : 卖家已履行发货义务，验机视频证据完整...")
    print(f"  demands (辩解)  : [验机视频全程无剪辑, 已购顺丰保价¥3000, 已履行发货义务]")
    print(f"  evidence (过失) : []  (无过失)")
    print(f"  proposed_solution: 快递理赔 / 平台垫付退款 / 协助证据")
    print(f"  confidence      : 0.8")

    # ═══════════════════════════════════════════════════════════════
    print()
    print(_SEP)
    print("STEP 3: MediatorAgent — 调停裁决")
    print(_SEP)

    mediator_prompt = mock_llm._prompts[2] if len(mock_llm._prompts) > 2 else "(未捕获)"
    print()
    print("【输入 — System Prompt (含平台裁决规则注入)】")
    # 展示 MEDIATOR_AGENT_PROMPT 摘要 + PLATFORM_RULING_POLICY 摘要
    print(f"  MEDIATOR_AGENT_PROMPT 长度: {len(MEDIATOR_AGENT_PROMPT)} 字符")
    print(f"  PLATFORM_RULING_POLICY 长度: {len(PLATFORM_RULING_POLICY)} 字符")
    print()
    print("  --- 平台裁决规则（注入内容）---")
    print(PLATFORM_RULING_POLICY)
    print()
    print("  --- 双方分析摘要（注入内容）---")
    # 从 mediator prompt 中提取"买家立场分析"和"卖家立场分析"部分
    buyer_start = mediator_prompt.find("## 买家立场分析")
    if buyer_start > 0:
        analysis_section = mediator_prompt[buyer_start:]
        if len(analysis_section) > 2500:
            analysis_section = analysis_section[:2500] + "\n... [截断]"
        print(analysis_section)
    print()
    print("【输出 — MediatorAgent JSON】")
    print(MEDIATOR_AGENT_OUTPUT)
    print()
    print("【解析 — AgentPerspective + 裁决关键字段】")
    print(f"  summary                  : {json.loads(MEDIATOR_AGENT_OUTPUT)['mediator_summary'][:80]}...")
    print(f"  responsible_party        : courier (快递公司)")
    print(f"  matched_rule             : R1, C1")
    print(f"  responsibility_split     : 买家 0% / 卖家 0% / 快递 100%")
    print(f"  compensation type        : insurance_claim")
    print(f"  compensation amount      : ¥3,299")
    print(f"  escalate_to_human        : False")
    print(f"  third_party_responsibility: True")

    # ═══════════════════════════════════════════════════════════════
    print()
    print(_SEP)
    print("STEP 4: 最终回复")
    print(_SEP)
    print()
    print(response.message)
    print()

    # ═══════════════════════════════════════════════════════════════
    print(_SEP)
    print("测试断言结果")
    print(_SEP)

    checks = []

    # 1. 基本响应
    checks.append(("回复非空", bool(response.message)))
    checks.append(("status=resolved", response.status == "resolved"))

    # 2. 裁决元数据
    mediator_step = next(
        (s for s in response.steps if s["step_name"] == "纠纷协调-调停裁决"), None
    )
    if mediator_step:
        od = mediator_step["output_data"]
        checks.append(("责任方=courier", od.get("responsible_party") == "courier"))
        checks.append(("命中规则含R1", "R1" in od.get("matched_rule", "")))
        checks.append(("命中规则含C1", "C1" in od.get("matched_rule", "")))
        checks.append(("第三方责任=True", od.get("third_party_responsibility") is True))
        checks.append(("未升级人工", od.get("escalate") is False))
    else:
        checks.append(("裁决步骤存在", False))

    # 3. 回复内容
    checks.append(("回复含'卖家先行退款'", "卖家先行退款" in response.message))
    checks.append(("回复含金额3299", "3299" in response.message))
    checks.append(("回复含'保价理赔'", "保价理赔" in response.message))
    checks.append(("回复含差额299", "299" in response.message))

    # 4. LLM 调用次数
    checks.append(("LLM调用3次(Buyer+Seller+Mediator)", mock_llm.chat_qwen_with_prompt.call_count == 3))

    # 5. 事实收集包含售后举证
    facts_step2 = next(
        (s for s in response.steps if s["step_name"] == "纠纷协调-事实收集"), None
    )
    if facts_step2:
        checks.append(("事实收集含after_sale_evidence", "after_sale_evidence" in facts_step2["output_data"].get("keys", [])))
    else:
        checks.append(("事实收集步骤存在", False))

    # 6. Mediator prompt 注入了平台规则
    checks.append(("Mediator prompt含R1", "R1 快递破损判定" in mediator_prompt))
    checks.append(("Mediator prompt含C1", "C1 快递责任赔付" in mediator_prompt))

    all_pass = True
    for name, result in checks:
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{status}] {name}")

    print()
    print(f"  总计: {sum(1 for _, r in checks if r)}/{len(checks)} 通过")

    # ═══════════════════════════════════════════════════════════════
    print()
    print(_SEP)
    print("流程总结")
    print(_SEP)
    print("""
  ┌──────────── 用户投诉 ────────────┐
  │ "收到洗衣机的玻璃面板碎了"      │
  └────────────┬─────────────────────┘
               │
  ┌────────────▼─────────────────────┐
  │ FactCollector                    │
  │ 5 个数据源并行拉取 ~0ms (mock)   │
  │ · 订单 ¥3,299                   │
  │ · 物流 顺丰 SF123456789 已签收   │
  │ · 举证 买家4张破损照 + 卖家验机视频 │
  │ · 余额 ¥520 · 优惠券 满200-30   │
  └────────────┬─────────────────────┘
               │
     ┌─────────┴─────────┐
     ▼                   ▼
  ┌──────────┐      ┌──────────┐
  │BuyerAgent│      │SellerAgent│  ← 并行！
  │买家诉求  │      │合规评估   │
  │全额退款  │      │卖家无责   │
  │破损证据  │      │快递赔付   │
  └────┬─────┘      └─────┬─────┘
       └────────┬──────────┘
                ▼
  ┌─────────────────────────────┐
  │ MediatorAgent               │
  │ 裁决框架: 平台规则 R1 + C1  │
  │ · 责任方: 快递公司          │
  │ · 卖家垫付退款 ¥3,299      │
  │ · 平台发起保价理赔 ¥3,000  │
  │ · 差额 ¥299 平台纠纷基金垫付│
  └────────────┬────────────────┘
               ▼
  ┌─────────────────────────────┐
  │ 最终回复 → 用户             │
  │ status: resolved            │
  │ 三方闭环: 用户退款 / 卖家无责 / 快递赔付 │
  └─────────────────────────────┘
""")


if __name__ == "__main__":
    asyncio.run(main())
