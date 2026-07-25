"""
纠纷协调器单元测试 —— 破损纠纷场景（洗衣机面板碎裂）

测试目标：验证 DisputeCoordinator 在「买家举证破损、卖家举证验机视频」
的场景下，MediatorAgent 能否：
1. 命中平台规则 R1（快递责任）和 C1（保价赔付）
2. 正确判定责任方为 courier
3. 生成包含三方赔付方案（卖家垫付 → 保价理赔 → 差额垫付）的裁决
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.modules.chat.agent.dispute_coordinator import (
    DisputeCoordinator,
    should_use_dispute_coordinator,
    _mock_get_after_sale_evidence,
    PLATFORM_RULING_POLICY,
)
from src.modules.chat.schemas import ChatRequest
from src.modules.chat.core.sentiment_service import EmotionResult, EmotionLevel


# ═══════════════════════════════════════════════════════════════════════
# Mock 数据：LLM 返回的受控 JSON
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

# ── 兜底/降级 LLM 输出（用于边界测试）──
BUYER_AGENT_FAIL_OUTPUT = "```json\n" + json.dumps({
    "core_issue": "无法解析",
    "buyer_demands": [],
    "emotion_intensity": "mild",
    "mentioned_evidence": [],
    "compensation_expectation": "",
    "buyer_summary": "买家分析失败",
}, ensure_ascii=False) + "\n```"

MEDIATOR_LOW_CONF_OUTPUT = json.dumps({
    "verdict": "证据矛盾，无法做出可靠裁决",
    "suggested_solution": "升级人工处理",
    "responsibility_split": {"buyer_percent": 50, "seller_percent": 50, "courier_percent": 0},
    "responsible_party": "shared",
    "matched_rule": "R4",
    "compensation": {"type": "refund", "amount_yuan": 0, "detail": "待人工裁决"},
    "escalate_to_human": True,
    "escalate_reason": "双方证据冲突，需要平台介入调查",
    "mediator_summary": "买家提供了破损照片，但卖家也提供了验机视频，双方证据充足但结论冲突，需人工进一步核查物流环节。"
}, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════

def _make_mock_llm_service(responses: list[str]):
    """构造带受控返回值的 Mock LLMService。
    
    Args:
        responses: 按调用顺序返回的字符串列表
    """
    mock = MagicMock()
    call_count = [0]

    async def mock_chat_qwen(prompt: str = "", system_prompt: str = "", **kwargs):
        idx = min(call_count[0], len(responses) - 1)
        result = responses[idx]
        call_count[0] += 1
        return result

    mock.chat_qwen_with_prompt = AsyncMock(side_effect=mock_chat_qwen)
    return mock


def _make_mock_tool_service(order_data: str = "", shipping_data: str = ""):
    """构造带受控返回值的 Mock ToolService。"""
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


def _make_emotion_result(level: EmotionLevel) -> EmotionResult:
    """构造 EmotionResult 实例。"""
    return EmotionResult(
        level=level,
        confidence=0.95,
        escalate=(level >= EmotionLevel.DISAPPOINTED),
        is_emergency=(level == EmotionLevel.EMERGENCY),
        keywords=["退款", "投诉"] if level >= EmotionLevel.DISAPPOINTED else [],
    )


def _make_chat_request(message: str) -> ChatRequest:
    """构造 ChatRequest 实例。"""
    return ChatRequest(message=message, conversation_id="test_conv_001")


# ═══════════════════════════════════════════════════════════════════════
# 测试类 1: 售后举证 Mock API
# ═══════════════════════════════════════════════════════════════════════

class TestMockAfterSaleEvidence:
    """售后举证 Mock API 单元测试"""

    @pytest.mark.asyncio
    async def test_evidence_returns_for_known_order(self):
        """已知订单返回完整的双方举证数据"""
        result = await _mock_get_after_sale_evidence("ORDER_WM20240601_001")
        data = json.loads(result)
        assert data["order_id"] == "ORDER_WM20240601_001"
        assert data["order_amount_yuan"] == 3299
        assert data["buyer_evidence"]["photo_count"] == 4
        assert "玻璃面板碎裂" in data["buyer_evidence"]["photo_descriptions"][0]
        assert data["seller_evidence"]["has_verification_video"] is True
        assert data["seller_evidence"]["shipping_insured"] is True
        assert data["seller_evidence"]["insured_amount_yuan"] == 3000
        assert data["logistics"]["tracking_number"] == "SF123456789"

    @pytest.mark.asyncio
    async def test_evidence_returns_error_for_unknown_order(self):
        """未知订单返回错误信息"""
        result = await _mock_get_after_sale_evidence("ORDER_UNKNOWN")
        data = json.loads(result)
        assert "error" in data
        assert "ORDER_UNKNOWN" in data["error"]

    @pytest.mark.asyncio
    async def test_evidence_returns_error_for_none_order(self):
        """未提供订单号返回错误信息"""
        result = await _mock_get_after_sale_evidence(None)
        data = json.loads(result)
        assert "error" in data
        assert "未提供订单号" in data["error"]


# ═══════════════════════════════════════════════════════════════════════
# 测试类 2: 平台裁决规则存在性
# ═══════════════════════════════════════════════════════════════════════

class TestPlatformRulingPolicy:
    """验证平台裁决规则文本的完整性和可注入性"""

    def test_policy_contains_responsibility_rules(self):
        """裁决规则包含 R1-R4 四条责任判定规则"""
        assert "R1 快递破损判定" in PLATFORM_RULING_POLICY
        assert "R2 卖家质量责任" in PLATFORM_RULING_POLICY
        assert "R3 买家责任" in PLATFORM_RULING_POLICY
        assert "R4 证据矛盾" in PLATFORM_RULING_POLICY

    def test_policy_contains_compensation_rules(self):
        """裁决规则包含 C1-C4 四条赔付规则"""
        assert "C1 快递责任赔付" in PLATFORM_RULING_POLICY
        assert "C2 快递责任赔付" in PLATFORM_RULING_POLICY
        assert "未保价" in PLATFORM_RULING_POLICY
        assert "C3 卖家责任赔付" in PLATFORM_RULING_POLICY
        assert "C4 买家责任" in PLATFORM_RULING_POLICY

    def test_policy_covers_insurance_gap(self):
        """C1 规则覆盖保价差额处理逻辑"""
        assert "保价金额 < 订单金额" in PLATFORM_RULING_POLICY
        assert "差额" in PLATFORM_RULING_POLICY
        assert "500" in PLATFORM_RULING_POLICY  # 垫付上限

    def test_policy_injectable_into_prompt(self):
        """裁决规则可以注入 Mediator prompt 且不破坏 JSON 格式"""
        from src.modules.chat.agent.dispute_coordinator import MEDIATOR_AGENT_PROMPT
        combined = MEDIATOR_AGENT_PROMPT + "\n\n" + PLATFORM_RULING_POLICY
        # 确保 JSON 格式关键字段仍然存在
        assert '"verdict"' in combined
        assert '"matched_rule"' in combined
        assert '"responsible_party"' in combined
        assert '"courier_percent"' in combined


# ═══════════════════════════════════════════════════════════════════════
# 测试类 3: 纠纷触发判断
# ═══════════════════════════════════════════════════════════════════════

class TestShouldUseDisputeCoordinator:
    """纠纷路由判断单元测试"""

    def test_angry_emotion_triggers(self):
        """ANGRY 情绪直接触发纠纷协调"""
        emotion = _make_emotion_result(EmotionLevel.ANGRY)
        assert should_use_dispute_coordinator("我的订单到哪了", emotion) is True

    def test_emergency_triggers(self):
        """EMERGENCY 情绪直接触发"""
        emotion = _make_emotion_result(EmotionLevel.EMERGENCY)
        assert should_use_dispute_coordinator("我要打12315", emotion) is True

    def test_disappointed_plus_return_triggers(self):
        """DISAPPOINTED + request-return 触发"""
        emotion = _make_emotion_result(EmotionLevel.DISAPPOINTED)
        assert should_use_dispute_coordinator(
            "这个商品质量太差了，我要退货",
            emotion,
            intent_action="request-return"
        ) is True

    def test_disappointed_without_return_no_trigger(self):
        """仅有 DISAPPOINTED 但没有退货意图，不触发"""
        emotion = _make_emotion_result(EmotionLevel.DISAPPOINTED)
        assert should_use_dispute_coordinator(
            "物流太慢了",
            emotion,
            intent_action="check-shipping"
        ) is False

    def test_keywords_trigger(self):
        """纠纷关键词匹配触发"""
        assert should_use_dispute_coordinator("你们是骗子，我要投诉到12315") is True

    def test_neutral_no_trigger(self):
        """中性消息不触发"""
        emotion = _make_emotion_result(EmotionLevel.NEUTRAL)
        assert should_use_dispute_coordinator("帮我查下订单", emotion) is False


# ═══════════════════════════════════════════════════════════════════════
# 测试类 4: 纠纷协调端到端 —— 破损纠纷主场景
# ═══════════════════════════════════════════════════════════════════════

class TestDisputeCoordinatorEndToEnd:
    """DisputeCoordinator.resolve() 端到端测试 —— 洗衣机面板碎裂场景"""

    @pytest.mark.asyncio
    async def test_broken_washing_machine_full_pipeline(self):
        """
        测试场景：洗衣机面板碎裂
        - 买家：收到货面板碎了，有4张破损照片
        - 卖家：发货前有58秒验机视频，面板完好，已买顺丰保价¥3000
        - 预期裁决：快递责任(R1)，保价赔付(C1)，卖家垫付退款
        """
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

        mock_llm = _make_mock_llm_service([
            BUYER_AGENT_OUTPUT,      # BuyerAgent
            SELLER_AGENT_OUTPUT,     # SellerAgent
            MEDIATOR_AGENT_OUTPUT,   # MediatorAgent
        ])
        mock_tool = _make_mock_tool_service(order_json, shipping_json)

        coordinator = DisputeCoordinator(llm=mock_llm, tool_service=mock_tool)
        request = _make_chat_request("收到洗衣机的玻璃面板碎了，申请退款")
        emotion = _make_emotion_result(EmotionLevel.ANGRY)

        # ── 执行 ──
        response = await coordinator.resolve(
            request=request,
            emotion_result=emotion,
            conversation_id="test_conv_001",
            order_id="ORDER_WM20240601_001",
        )

        # ── 验证 ──

        # 1. 基本响应存在
        assert response is not None
        assert response.message, "最终回复不应为空"
        assert response.status == "resolved", f"预期 resolved，实际 {response.status}"

        # 2. 步骤中的裁决元数据
        mediator_step = None
        for step in response.steps:
            if step["step_name"] == "纠纷协调-调停裁决":
                mediator_step = step
                break
        assert mediator_step is not None, "应存在调停裁决步骤"

        output = mediator_step["output_data"]
        assert output["responsible_party"] == "courier", (
            f"责任方应为 courier，实际 {output['responsible_party']}"
        )
        assert "R1" in output.get("matched_rule", ""), (
            f"应命中规则 R1，实际命中 {output.get('matched_rule')}"
        )
        assert output["third_party_responsibility"] is True, "应标记为第三方（快递）责任"
        assert output["escalate"] is False, "不应建议升级人工"

        # 3. 裁决结果包含关键信息
        assert "卖家先行退款" in response.message, f"回复应提及卖家先行退款: {response.message[:200]}"
        assert "3299" in response.message, f"回复应包含退款金额 ¥3299: {response.message[:200]}"

        # 4. 事实数据包含售后举证
        facts_step = None
        for step in response.steps:
            if step["step_name"] == "纠纷协调-事实收集":
                facts_step = step
                break
        assert facts_step is not None, "应存在事实收集步骤"
        # FactCollector 应包含 after_sale_evidence
        assert "after_sale_evidence" in facts_step["output_data"].get("keys", []), (
            "事实收集应包含 after_sale_evidence 数据源"
        )

        # 5. LLM 被调用了 3 次（BuyerAgent + SellerAgent + MediatorAgent）
        assert mock_llm.chat_qwen_with_prompt.call_count == 3, (
            f"LLM 应被调用 3 次，实际 {mock_llm.chat_qwen_with_prompt.call_count}"
        )

    @pytest.mark.asyncio
    async def test_mediator_prompt_includes_platform_policy(self):
        """验证 MediatorAgent 的 prompt 确实注入了平台裁决规则"""
        mock_llm = _make_mock_llm_service([
            BUYER_AGENT_OUTPUT,
            SELLER_AGENT_OUTPUT,
            MEDIATOR_AGENT_OUTPUT,
        ])
        mock_tool = _make_mock_tool_service()
        coordinator = DisputeCoordinator(llm=mock_llm, tool_service=mock_tool)

        await coordinator.resolve(
            request=_make_chat_request("收到洗衣机的玻璃面板碎了，申请退款"),
            emotion_result=_make_emotion_result(EmotionLevel.ANGRY),
            order_id="ORDER_WM20240601_001",
        )

        # 检查第三次 LLM 调用（MediatorAgent）的 prompt 参数
        mediator_call = mock_llm.chat_qwen_with_prompt.call_args_list[2]
        prompt_arg = mediator_call.kwargs.get("prompt", "")
        assert "R1 快递破损判定" in prompt_arg, "Mediator prompt 应包含平台规则 R1"
        assert "C1 快递责任赔付" in prompt_arg, "Mediator prompt 应包含赔付规则 C1"
        assert "保价金额 < 订单金额" in prompt_arg, "Mediator prompt 应包含差额处理规则"


# ═══════════════════════════════════════════════════════════════════════
# 测试类 5: 边界场景
# ═══════════════════════════════════════════════════════════════════════

class TestDisputeCoordinatorEdgeCases:
    """纠纷协调边界场景测试"""

    @pytest.mark.asyncio
    async def test_buyer_agent_failure_graceful_degradation(self):
        """BuyerAgent 返回无效数据 → Mediator 兜底升级"""
        mock_llm = _make_mock_llm_service([
            BUYER_AGENT_FAIL_OUTPUT,  # 买家分析失败
            SELLER_AGENT_OUTPUT,
            MEDIATOR_LOW_CONF_OUTPUT,
        ])
        mock_tool = _make_mock_tool_service()
        coordinator = DisputeCoordinator(llm=mock_llm, tool_service=mock_tool)

        response = await coordinator.resolve(
            request=_make_chat_request("退款"),
            emotion_result=_make_emotion_result(EmotionLevel.DISAPPOINTED),
        )

        assert response is not None
        # 买家属地失败 → 应升级或给出兜底回复
        assert response.status in ("escalated", "resolved")

    @pytest.mark.asyncio
    async def test_mediator_escalate_on_low_confidence(self):
        """MediatorAgent 置信度低 → status = escalated"""
        mock_llm = _make_mock_llm_service([
            BUYER_AGENT_OUTPUT,
            SELLER_AGENT_OUTPUT,
            MEDIATOR_LOW_CONF_OUTPUT,  # escalate_to_human=True
        ])
        mock_tool = _make_mock_tool_service()
        coordinator = DisputeCoordinator(llm=mock_llm, tool_service=mock_tool)

        response = await coordinator.resolve(
            request=_make_chat_request("收到货坏了"),
            emotion_result=_make_emotion_result(EmotionLevel.ANGRY),
            order_id="ORDER_WM20240601_001",
        )

        assert response.status == "escalated", (
            f"低置信度裁决应升级，实际 status={response.status}"
        )
        assert "升级" in response.message or "专员" in response.message, (
            f"升级回复应提及升级/专员: {response.message[:200]}"
        )

    @pytest.mark.asyncio
    async def test_fact_collection_partial_failure(self):
        """事实收集部分失败 → 不中断流程"""
        mock_llm = _make_mock_llm_service([
            BUYER_AGENT_OUTPUT,
            SELLER_AGENT_OUTPUT,
            MEDIATOR_AGENT_OUTPUT,
        ])

        # ToolService 中 check-balance 抛异常
        mock_tool = MagicMock()

        async def mock_dispatch(action: str, params: dict = None):
            if action == "check-balance":
                raise RuntimeError("余额服务不可用")
            if action == "query-order":
                return json.dumps({"orders": [{"order_id": "X", "amount_yuan": 100}]})
            if action == "check-shipping":
                return json.dumps({"tracking": {"status": "ok"}})
            if action == "coupon-inquiry":
                return json.dumps({"coupons": []})
            return "{}"

        mock_tool.dispatch = AsyncMock(side_effect=mock_dispatch)
        coordinator = DisputeCoordinator(llm=mock_llm, tool_service=mock_tool)

        response = await coordinator.resolve(
            request=_make_chat_request("收到货坏了，退款"),
            emotion_result=_make_emotion_result(EmotionLevel.ANGRY),
            order_id="ORDER_WM20240601_001",
        )

        # 不应崩溃
        assert response is not None
        assert response.message

    @pytest.mark.asyncio
    async def test_emotion_neutral_no_escalation(self):
        """NEUTRAL 情绪 → Mediator 正常裁决，不送升级"""
        mock_llm = _make_mock_llm_service([
            BUYER_AGENT_OUTPUT,
            SELLER_AGENT_OUTPUT,
            MEDIATOR_AGENT_OUTPUT,  # escalate=False
        ])
        mock_tool = _make_mock_tool_service()
        coordinator = DisputeCoordinator(llm=mock_llm, tool_service=mock_tool)

        response = await coordinator.resolve(
            request=_make_chat_request("洗衣机面板碎了，退款"),
            emotion_result=_make_emotion_result(EmotionLevel.NEUTRAL),
            order_id="ORDER_WM20240601_001",
        )

        assert response.status == "resolved"
        # 中性情绪回复不应以道歉开头
        # (最终回复格式化在 _format_final_reply 中会根据情绪等级调整语气)

    @pytest.mark.asyncio
    async def test_resolve_no_order_id_still_works(self):
        """无订单号时纠纷协调不应崩溃（FactCollector 降级到查最近订单）"""
        mock_llm = _make_mock_llm_service([
            BUYER_AGENT_OUTPUT,
            SELLER_AGENT_OUTPUT,
            MEDIATOR_AGENT_OUTPUT,
        ])
        mock_tool = _make_mock_tool_service()
        coordinator = DisputeCoordinator(llm=mock_llm, tool_service=mock_tool)

        response = await coordinator.resolve(
            request=_make_chat_request("你们是骗子，我要投诉"),
            emotion_result=_make_emotion_result(EmotionLevel.ANGRY),
            # 不传 order_id
        )

        assert response is not None
        assert response.message


# ═══════════════════════════════════════════════════════════════════════
# 测试类 6: 裁决结果内容验证
# ═══════════════════════════════════════════════════════════════════════

class TestRulingOutputContent:
    """验证裁决输出的具体内容质量"""

    @pytest.mark.asyncio
    async def test_ruling_includes_compensation_split(self):
        """裁决回复应包含多方赔付方案"""
        mock_llm = _make_mock_llm_service([
            BUYER_AGENT_OUTPUT,
            SELLER_AGENT_OUTPUT,
            MEDIATOR_AGENT_OUTPUT,
        ])
        mock_tool = _make_mock_tool_service()
        coordinator = DisputeCoordinator(llm=mock_llm, tool_service=mock_tool)

        response = await coordinator.resolve(
            request=_make_chat_request("收到洗衣机的玻璃面板碎了，申请退款"),
            emotion_result=_make_emotion_result(EmotionLevel.ANGRY),
            order_id="ORDER_WM20240601_001",
        )

        # 回复应包含三要素
        reply = response.message
        assert "退款" in reply, f"回复应提及退款: {reply[:300]}"
        assert "3299" in reply, f"回复应包含金额: {reply[:300]}"

        # 验证 steps 中的证据包含 courier_percent
        mediator_step = next(
            (s for s in response.steps if s["step_name"] == "纠纷协调-调停裁决"),
            None
        )
        assert mediator_step is not None
        assert mediator_step["output_data"]["third_party_responsibility"] is True
