"""纠纷协调共享常量、Prompt 与工具函数。"""
from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.modules.chat.core.sentiment_service import EmotionLevel  # noqa: F401

# ═══════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class AgentPerspective:
    """单个 Agent 的分析结果"""

    role: str  # "buyer" | "seller" | "mediator"
    summary: str  # 核心观点摘要
    demands: List[str] = field(default_factory=list)  # 诉求/立场
    evidence: List[str] = field(default_factory=list)  # 提及的证据
    proposed_solution: str = ""  # 提出的解决方案
    confidence: float = 0.8  # 置信度
    raw_output: str = ""  # LLM 原始输出
    third_party_responsibility: bool = False  # 是否有第三方（快递等）责任
    escalate: bool = False  # 是否需升级人工（LLM 显式要求或置信度过低)


# ═══════════════════════════════════════════════════════════════════════
# Agent Prompts
# ═══════════════════════════════════════════════════════════════════════

BUYER_AGENT_PROMPT = """你是一个电商消费者权益分析专家。你的任务是站在买家角度，客观、全面地分析用户在投诉中表达的诉求和情绪。  # noqa: E501

分析框架：
1. **核心诉求**：买家想要什么？（退款/换货/赔偿/道歉/改善服务）
2. **问题定性**：买家认为出了什么问题？（质量问题/发错货/延期/虚假宣传/态度差）
3. **情绪信号**：买家的情绪有多强烈？有没有威胁行为（打12315/报警/曝光）？
4. **证据主张**：买家提到了哪些证据？（照片/聊天记录/订单号）
5. **期望补偿**：买家是否提出了具体的赔偿金额或优惠要求？

请严格按以下 JSON 格式输出，不要包含任何额外文字：
{
  "core_issue": "问题的核心是什么（一句话）",
  "buyer_demands": ["诉求1", "诉求2"],
  "emotion_intensity": "mild|moderate|severe|extreme",
  "mentioned_evidence": ["证据1", "证据2"],
  "compensation_expectation": "买家期望的具体补偿",
  "buyer_summary": "从买家角度的完整分析（2-3句话）"
}"""


SELLER_AGENT_PROMPT = """你是一个电商平台合规与卖家权益分析专家。你的任务是从平台规则和卖家立场出发，对买家的投诉进行合规性评估。  # noqa: E501

评估框架：
1. **平台规则对照**：根据7天无理由退货、质量问题退货、发货时效等平台规则，评估卖家责任范围
2. **订单事实检查**：查看订单状态、物流节点、签收时间等客观事实
3. **卖家合理立场**：卖家在哪些方面有合理的辩解？哪些方面确实存在过失？
4. **解决问题选项**：卖家可以接受哪些方案？（补发/部分退款/全额退款+退货/赔偿优惠券）
5. **升级风险评估**：如果问题不解决，走平台介入或法律途径，卖家可能面临什么？

请严格按以下 JSON 格式输出，不要包含任何额外文字：
{
  "rule_assessment": "根据平台规则的责任判定（一句话）",
  "seller_faults": ["卖家过失1", "卖家过失2"],
  "seller_defenses": ["卖家合理辩解1", "卖家合理辩解2"],
  "acceptable_solutions": ["可接受方案1", "可接受方案2", "可接受方案3"],
  "escalation_risk": "low|medium|high|critical",
  "seller_summary": "从卖家角度的完整分析（2-3句话）"
}"""


MEDIATOR_AGENT_PROMPT = """你是一个电商售后纠纷调停专家。你的任务是基于买家分析和卖家分析的完整上下文，结合平台规则和行业最佳实践，给出一份公正、可执行的裁决意见。  # noqa: E501

裁决原则：
1. **规则优先**：平台规定明确的情况下，按规则执行；规则模糊的情况下，偏向保护消费者权益
2. **实质公平**：不只看法条，还要考虑实际情况（用户等待时间、之前的购物体验、问题严重程度）
3. **可行方案**：给出的方案必须是当前情况下可执行的（退款金额/补发方式/优惠券面额）
4. **情绪安抚**：措辞要真诚、有温度，让用户感受到被重视
5. **风险管理**：如果存在舆情或法律风险，要在裁决中明确指出升级建议

请严格按以下 JSON 格式输出，不要包含任何额外文字：
{
  "verdict": "裁决结论（一句话）",
  "suggested_solution": "建议的具体解决方案（可包含多个步骤）",
  "responsibility_split": {"buyer_percent": 数字, "seller_percent": 数字, "courier_percent": 数字},
  "responsible_party": "buyer|seller|courier|platform|shared",
  "matched_rule": "命中的平台规则编号（如 R1, R3a 等）",
  "compensation": {"type": "refund|resend|coupon|return_refund|combination|insurance_claim", "amount_yuan": 数字或0, "detail": "详细说明"},  # noqa: E501
  "escalate_to_human": true或false,
  "escalate_reason": "如果需要升级，说明原因；否则为空字符串",
  "mediator_summary": "完整的裁决意见（3-5句话）"
}

重要提醒：
- 必须严格按照下方「平台裁决规则」进行责任判定和赔付方案制定
- 如果订单事实不足，建议先补全信息再裁决
- 如果买家情绪达到 EXTREME 级别（威胁法律/舆情），必须建议升级
- 责任占比必须为整数，且 buyer_percent + seller_percent + courier_percent = 100"""


PLATFORM_RULING_POLICY = """
## 平台裁决规则（必须严格遵守，以下规则优先级从高到低）

### 一、责任判定规则

**R1 快递破损判定**（发货前有验货证据 + 签收后有破损证据 → 快递责任）
- 卖家提供了发货前验货证据（视频/照片，时间戳在发货前）
- 买家在签收后 24 小时内提供了破损照片/视频
- 判定结果：卖家已履约，快递公司承担赔付责任

**R2 卖家质量责任**（无验货证据或买家举证质量问题 → 卖家责任）
- 卖家未提供有效发货前验货证据
- 或买家举证的质量问题与物流无关（如功能故障、型号不符）
- 判定结果：卖家承担退货退款责任

**R3 买家责任**（签收超时未投诉或无有效证据 → 买家责任）
- 签收超过 24 小时未发起投诉，且无有效破损证据
- 或买家无法提供任何破损/瑕疵证据
- 判定结果：驳回退款诉求，提供自行联系快递理赔指引

**R4 证据矛盾**（双方证据充足但互相矛盾 → 平台调查）
- 双方均提供有效证据但结论冲突
- 判定结果：建议升级人工，平台介入调查取证

### 二、赔付规则

**C1 快递责任赔付**（已保价）
- 卖家先行退款（平台信用垫付，T+1 到账）→ 确保用户体验
- 平台方向快递公司发起保价理赔，按保价金额赔付上限
- 保价金额 < 订单金额 → 差额 ≤500 元由平台纠纷基金垫付；>500 元建议升级人工

**C2 快递责任赔付**（未保价）
- 卖家先行退款（平台信用垫付）→ 快递按行业标准赔付（通常为运费的 3-7 倍）
- 差额由平台纠纷基金垫付（上限 500 元/单）

**C3 卖家责任赔付**
- 卖家全额退款 + 退货包运费（7 天无理由）
- 质量问题额外补偿优惠券（面额 = 订单金额 × 5%，上限 50 元）
- 同一卖家月纠纷 >3 单 → 平台扣信用分

**C4 买家责任**
- 驳回退款申请，提供自行联系快递理赔指引
- 平台不垫付、不补偿

### 三、输出格式要求

每份裁决必须包含：
1. 责任方（从 R1-R4 中选出匹配的规则编号）
2. 赔付方案（从 C1-C4 中选出匹配的赔付规则，填入具体金额）
3. 差额处理（如适用）
4. 是否需要升级人工（confidence < 0.5 或证据矛盾时）
"""


# ═══════════════════════════════════════════════════════════════════════
# 置信度融合工具函数
# ═══════════════════════════════════════════════════════════════════════


def _fuse_confidence(
    logprob: float,
    tool_confidence: float,
    format_validity: float,
    refusal_score: float,
) -> float:
    """按博客 07 篇融合公式合成置信度（各输入归一化到 [0,1]）。"""
    return 0.4 * logprob + 0.3 * tool_confidence + 0.2 * format_validity + 0.1 * refusal_score


def _logprob_proxy() -> float:
    """logprob 占位（当前 LLM 服务未暴露 logprobs）。"""
    return 0.5


def _format_validity(data: Dict[str, Any], required_keys: List[str]) -> float:
    """JSON 解析与关键字段完整度 → [0,1]。"""
    if not isinstance(data, dict):
        return 0.0
    if not required_keys:
        return 1.0
    present = sum(1 for k in required_keys if data.get(k))
    return min(1.0, 0.3 + 0.7 * (present / len(required_keys)))


def _tool_confidence(facts: Dict[str, str]) -> float:
    """事实收集成功率 → [0,1]：成功条数 / 总条数（错误标记视为失败）。"""
    if not facts:
        return 0.0
    ok = sum(
        1
        for v in facts.values()
        if v and "[查询失败" not in v and "[收集失败" not in v and "失败" not in v
    )
    return ok / len(facts)


def _refusal_score(raw: str) -> float:
    """拒绝 / 非承诺信号 → [0,1]（1=无拒绝，0.2=存在拒绝标记）。"""
    if not raw:
        return 0.0
    markers = ["无法做出裁决", "建议升级人工", "I cannot", "无法完成", "暂时无法", "建议人工"]
    return 0.2 if any(m in raw for m in markers) else 1.0


# ═══════════════════════════════════════════════════════════════════════
# 售后举证 API
# ═══════════════════════════════════════════════════════════════════════

_MOCK_AFTERSALE_DB = {
    "ORDER_WM20240601_001": {
        "order_id": "ORDER_WM20240601_001",
        "buyer_evidence": {
            "photos": [
                {
                    "url": "https://cdn.shop.example/evidence/photos/buyer_001_01.jpg",
                    "description": "洗衣机玻璃面板碎裂特写，裂纹从左上角延伸至右下角",
                },
                {
                    "url": "https://cdn.shop.example/evidence/photos/buyer_001_02.jpg",
                    "description": "外包装纸箱侧面有撞击凹陷痕迹",
                },
                {
                    "url": "https://cdn.shop.example/evidence/photos/buyer_001_03.jpg",
                    "description": "洗衣机整体外观，面板碎裂部位全景",
                },
                {
                    "url": "https://cdn.shop.example/evidence/photos/buyer_001_04.jpg",
                    "description": "物流面单特写，单号 SF123456789，收件人信息完整",
                },
            ],
            "complaint_time": "2024-06-03 14:22:00",
            "complaint_channel": "在线客服",
            "buyer_note": "收到货打开就发现玻璃面板碎了，外包装也有撞击痕迹",
        },
        "seller_evidence": {
            "verification_video": {
                "url": "https://cdn.shop.example/evidence/videos/seller_001.mp4",
                "duration_seconds": 58,
                "recorded_at": "2024-06-01 14:30:00",
                "description": "发货前验机视频：全程无剪辑，面板完好，通电测试正常",
                "key_frames": [
                    "00:00-00:15: 完整外观展示，玻璃面板无任何裂纹",
                    "00:16-00:35: 通电启动，显示屏正常亮起",
                    "00:36-00:58: 各功能旋钮测试，进水排水正常",
                ],
            },
            "shipping_insurance": {
                "insured": True,
                "insured_amount_yuan": 3000,
                "insurance_company": "顺丰保价",
            },
            "seller_note": "发货前已录制完整验机视频，面板完好，快递运输中造成的破损应由快递公司负责",
        },
        "order_amount_yuan": 3299,
        "logistics": {
            "tracking_number": "SF123456789",
            "carrier": "顺丰速运",
            "shipped_at": "2024-06-01 15:00:00",
            "delivered_at": "2024-06-03 12:15:00",
            "signed_by": "本人签收",
        },
    },
}


async def _mock_get_after_sale_evidence(order_id: str | None = None) -> str:
    """Mock 售后举证 API —— 返回订单的双方举证数据。"""
    import json as _json_lib

    if not order_id:
        return _json_lib.dumps({"error": "未提供订单号，无法查询售后举证"}, ensure_ascii=False)

    data = _MOCK_AFTERSALE_DB.get(order_id)
    if not data:
        return _json_lib.dumps(
            {
                "error": f"订单 {order_id} 无售后举证记录",
                "hint": "可能是新订单或举证尚未上传",
            },
            ensure_ascii=False,
        )

    summary = {
        "order_id": data["order_id"],
        "order_amount_yuan": data["order_amount_yuan"],
        "buyer_evidence": {
            "photo_count": len(data["buyer_evidence"]["photos"]),
            "photo_descriptions": [p["description"] for p in data["buyer_evidence"]["photos"]],
            "complaint_time": data["buyer_evidence"]["complaint_time"],
            "complaint_note": data["buyer_evidence"]["buyer_note"],
        },
        "seller_evidence": {
            "has_verification_video": True,
            "video_recorded_at": data["seller_evidence"]["verification_video"]["recorded_at"],
            "video_description": data["seller_evidence"]["verification_video"]["description"],
            "shipping_insured": True,
            "insured_amount_yuan": data["seller_evidence"]["shipping_insurance"][
                "insured_amount_yuan"
            ],
            "seller_note": data["seller_evidence"]["seller_note"],
        },
        "logistics": data["logistics"],
    }
    return _json_lib.dumps(summary, ensure_ascii=False, indent=2)


async def get_after_sale_evidence(order_id: str | None = None) -> str:
    """查询售后举证（真实订单服务，Rust 实现，数据来自 PostgreSQL）。"""
    if not order_id:
        return _json.dumps({"error": "未提供订单号，无法查询售后举证"}, ensure_ascii=False)

    from src.core.config import config

    base_url = getattr(config, "ORDER_SERVICE_URL", "")
    if not base_url:
        return _json.dumps(
            {
                "error": "订单服务未配置 (ORDER_SERVICE_URL 为空)",
                "hint": "请部署 apps/order-service 并配置 ORDER_SERVICE_URL",
            },
            ensure_ascii=False,
        )

    import httpx

    url = f"{base_url.rstrip('/')}/orders/{order_id}/evidence"
    try:
        async with httpx.AsyncClient(timeout=getattr(config, "ORDER_SERVICE_TIMEOUT", 5)) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        return _json.dumps(data, ensure_ascii=False, indent=2)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return _json.dumps(
                {
                    "error": f"订单 {order_id} 无售后举证记录",
                    "hint": "可能是新订单或举证尚未上传",
                },
                ensure_ascii=False,
            )
        return _json.dumps(
            {"error": f"订单服务查询失败: HTTP {e.response.status_code}"}, ensure_ascii=False
        )
    except Exception as e:
        return _json.dumps({"error": f"订单服务不可用: {e}"}, ensure_ascii=False)
