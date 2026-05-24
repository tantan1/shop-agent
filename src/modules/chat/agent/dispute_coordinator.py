"""
纠纷协调器 (Dispute Coordinator) —— 多 Agent 售后纠纷三方协调

架构（三个专职 Agent + 事实收集层）：
  ┌─────────── 用户投诉消息 ───────────┐
  │              │                     │
  │    ┌─────────┴──────────┐          │
  │    │  FactCollector     │          │
  │    │ (查订单/物流/政策)  │          │
  │    └─────────┬──────────┘          │
  │              │                     │
  │    ┌─────────┼──────────┐          │
  │    ▼         ▼          ▼          │
  │ BuyerAgent SellerAgent (并行调用)    │
  │ (诉求提取) (卖家立场评估)           │
  │    │         │                     │
  │    └────┬────┘                     │
  │         ▼                          │
  │   MediatorAgent                    │
  │   (调停裁决 + 平台规则)              │
  │         │                          │
  │         ▼                          │
  │   DisputeResult → ChatResponse     │
  └────────────────────────────────────┘

触发条件（在编排器中判断）：
  - 情绪等级 >= ANGRY 或 DISAPPOINTED
  - 意图为 request-return 或包含纠纷关键词
  - 用户明确表达"投诉""退款纠纷""不同意""卖家不"等

设计原则：
  - BuyerAgent 和 SellerAgent 并行执行（互不依赖，降低延迟）
  - MediatorAgent 串行等待两者结果（需要完整上下文）
  - 所有 Agent 使用同一 LLM（复用已有服务），仅 prompt 不同
  - 支持 Mock 事实数据（无远程 API 环境时自动降级）
"""
from __future__ import annotations

import json as _json
import time as _time
import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from src.modules.chat.schemas import ChatRequest, ChatResponse
from src.modules.chat.core.sentiment_service import EmotionResult, EmotionLevel
from src.shared.logger import APILogger

if TYPE_CHECKING:
    from src.modules.chat.core.llm_service import LLMService
    from src.modules.chat.core.tool_registry import ToolService

logger = APILogger("dispute_coordinator")


# ═══════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AgentPerspective:
    """单个 Agent 的分析结果"""
    role: str                                    # "buyer" | "seller" | "mediator"
    summary: str                                 # 核心观点摘要
    demands: List[str] = field(default_factory=list)    # 诉求/立场
    evidence: List[str] = field(default_factory=list)   # 提及的证据
    proposed_solution: str = ""                  # 提出的解决方案
    confidence: float = 0.8                      # 置信度
    raw_output: str = ""                         # LLM 原始输出


@dataclass
class DisputeResult:
    """纠纷协调的完整结果"""
    buyer: AgentPerspective
    seller: AgentPerspective
    mediator: AgentPerspective
    facts: Dict[str, str] = field(default_factory=dict)  # {tool_name: result}
    resolution: str = ""                          # 最终给用户的回复
    escalate: bool = False                        # 是否需要人工升级
    duration_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Agent System Prompts
# ═══════════════════════════════════════════════════════════════════════

BUYER_AGENT_PROMPT = """你是一个电商消费者权益分析专家。你的任务是站在买家角度，客观、全面地分析用户在投诉中表达的诉求和情绪。

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


SELLER_AGENT_PROMPT = """你是一个电商平台合规与卖家权益分析专家。你的任务是从平台规则和卖家立场出发，对买家的投诉进行合规性评估。

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


MEDIATOR_AGENT_PROMPT = """你是一个电商售后纠纷调停专家。你的任务是基于买家分析和卖家分析的完整上下文，结合平台规则和行业最佳实践，给出一份公正、可执行的裁决意见。

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
  "responsibility_split": {"buyer_percent": 数字, "seller_percent": 数字},
  "compensation": {"type": "refund|resend|coupon|return_refund|combination", "amount_yuan": 数字或0, "detail": "详细说明"},
  "escalate_to_human": true或false,
  "escalate_reason": "如果需要升级，说明原因；否则为空字符串",
  "mediator_summary": "完整的裁决意见（3-5句话）"
}

重要提醒：
- 如果订单事实不足，建议先补全信息再裁决
- 如果买家情绪达到 EXTREME 级别（威胁法律/舆情），必须建议升级
- 责任占比必须为整数，且 buyer_percent + seller_percent = 100"""


# ═══════════════════════════════════════════════════════════════════════
# 事实收集层
# ═══════════════════════════════════════════════════════════════════════

class FactCollector:
    """从工具服务中收集纠纷相关的客观事实。
    
    并行调用多个工具，将结果汇总为结构化的事实字典，
    供 BuyerAgent / SellerAgent / MediatorAgent 使用。
    """

    def __init__(self, tool_service: "ToolService"):
        self._tool_service = tool_service

    async def collect(
        self,
        order_id: Optional[str] = None,
        tracking_number: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """并行收集订单、物流、余额、优惠券等事实数据。"""
        tasks = []

        # 如果有订单号，查订单 + 查物流
        if order_id:
            tasks.append(("order_info", self._tool_service.dispatch("query-order", {"order_id": order_id})))
            tasks.append(("order_shipping", self._tool_service.dispatch("check-shipping", {"order_id": order_id})))
        elif tracking_number:
            tasks.append(("shipping_info", self._tool_service.dispatch("check-shipping", {"tracking_number": tracking_number})))
        else:
            # 没有订单号，查最近订单
            tasks.append(("recent_orders", self._tool_service.dispatch("query-order", {})))

        # 总是查余额和优惠券（赔偿方案参考）
        tasks.append(("balance_info", self._tool_service.dispatch("check-balance", {})))
        tasks.append(("coupon_info", self._tool_service.dispatch("coupon-inquiry", {})))

        # 并行执行
        results = await asyncio.gather(
            *(self._execute_one(name, fn) for name, fn in tasks),
            return_exceptions=True,
        )

        facts: Dict[str, str] = {}
        for i, (name, _) in enumerate(tasks):
            r = results[i]
            if isinstance(r, Exception):
                facts[name] = f"[查询失败: {str(r)[:100]}]"
            else:
                facts[name] = str(r)

        return facts

    async def _execute_one(self, name: str, coro) -> str:
        """执行单个工具调用，失败时返回错误信息不中断流程。"""
        try:
            return await coro
        except Exception as e:
            logger.warning(f"事实收集 {name} 失败: {str(e)[:80]}")
            return f"[收集失败]"


# ═══════════════════════════════════════════════════════════════════════
# 纠纷协调器主体
# ═══════════════════════════════════════════════════════════════════════

class DisputeCoordinator:
    """多 Agent 纠纷协调器。

    使用方式:
        coordinator = DisputeCoordinator(llm=llm_svc, tool_service=tool_svc)
        result = await coordinator.resolve(request, emotion_result, conversation_id, domain)
        # → ChatResponse
    """

    def __init__(
        self,
        *,
        llm: "LLMService",
        tool_service: "ToolService",
        domain: str = "ecommerce",
    ):
        self._llm = llm
        self._tool_service = tool_service
        self._domain = domain
        self._fact_collector = FactCollector(tool_service)

    # ── 主入口 ──────────────────────────────────────────────────────────

    async def resolve(
        self,
        request: ChatRequest,
        emotion_result: EmotionResult | None = None,
        *,
        conversation_id: str = "",
        domain: str = "ecommerce",
        intent_steps: list | None = None,
        order_id: str | None = None,
        langfuse_handler=None,
    ) -> ChatResponse:
        """
        执行纠纷协调流程。

        Args:
            request: 用户请求
            emotion_result: 情绪检测结果（可选，用于调整策略）
            conversation_id: 会话 ID
            domain: 业务领域
            intent_steps: 已有步骤列表
            order_id: 涉及的订单号（可选，从意图参数中提取）
            langfuse_handler: Langfuse 回调

        Returns:
            ChatResponse
        """
        t_start = _time.perf_counter()
        steps = list(intent_steps) if intent_steps else []

        emotion_level = emotion_result.level if emotion_result else EmotionLevel.NEUTRAL
        emotion_keywords = emotion_result.keywords if emotion_result else []

        logger.info(
            "纠纷协调启动",
            conversation_id=conversation_id,
            emotion=emotion_level.name,
            order_id=order_id or "未提供",
            message_preview=request.message[:100],
        )

        # ── Step 1: 事实收集（并行查订单/物流/余额/优惠券）──
        t_facts_start = _time.perf_counter()
        facts = {}
        try:
            facts = await self._fact_collector.collect(order_id=order_id)
        except Exception as e:
            logger.warning(f"事实收集失败: {str(e)[:100]}")

        t_facts_ms = (_time.perf_counter() - t_facts_start) * 1000
        steps.append({
            "step_name": "纠纷协调-事实收集",
            "step_order": len(steps),
            "status": "success" if facts else "partial",
            "output_data": {
                "facts_count": len(facts),
                "keys": list(facts.keys()),
                "duration_ms": round(t_facts_ms, 1),
            },
        })
        logger.debug("事实收集完成", facts_keys=list(facts.keys()), duration_ms=round(t_facts_ms, 1))

        # ── Step 2: BuyerAgent + SellerAgent 并行分析 ──
        t_analysis_start = _time.perf_counter()
        buyer_task = self._run_buyer_agent(request.message, facts, emotion_level)
        seller_task = self._run_seller_agent(request.message, facts, emotion_level)

        buyer, seller = await asyncio.gather(buyer_task, seller_task, return_exceptions=True)

        if isinstance(buyer, Exception):
            buyer = AgentPerspective(
                role="buyer", summary="买家分析失败",
                demands=["信息不足，需人工介入"], raw_output=str(buyer),
            )
        if isinstance(seller, Exception):
            seller = AgentPerspective(
                role="seller", summary="卖家分析失败",
                proposed_solution="信息不足", raw_output=str(seller),
            )

        t_analysis_ms = (_time.perf_counter() - t_analysis_start) * 1000
        steps.append({
            "step_name": "纠纷协调-双方分析",
            "step_order": len(steps),
            "status": "success",
            "output_data": {
                "buyer_demands": buyer.demands[:3],
                "seller_solutions": seller.proposed_solution[:100],
                "duration_ms": round(t_analysis_ms, 1),
            },
        })
        logger.info(
            "双方分析完成",
            buyer_demands=buyer.demands[:2],
            duration_ms=round(t_analysis_ms, 1),
        )

        # ── Step 3: MediatorAgent 裁决 ──
        t_mediator_start = _time.perf_counter()
        mediator = await self._run_mediator_agent(
            request.message, facts, buyer, seller, emotion_level
        )
        t_mediator_ms = (_time.perf_counter() - t_mediator_start) * 1000
        steps.append({
            "step_name": "纠纷协调-调停裁决",
            "step_order": len(steps),
            "status": "success",
            "output_data": {
                "verdict_short": mediator.summary[:100],
                "escalate": mediator.confidence < 0.5,
                "duration_ms": round(t_mediator_ms, 1),
            },
        })

        # ── Step 4: 生成最终回复 ──
        final_reply = self._format_final_reply(buyer, seller, mediator, facts, emotion_level)

        t_total = (_time.perf_counter() - t_start) * 1000
        logger.info(
            "纠纷协调完成",
            escalate=mediator.confidence < 0.5,
            total_ms=round(t_total, 1),
            facts_ms=round(t_facts_ms, 1),
            analysis_ms=round(t_analysis_ms, 1),
            mediator_ms=round(t_mediator_ms, 1),
        )

        return ChatResponse(
            message=final_reply,
            conversation_id=conversation_id,
            steps=steps,
            documents_used=[],
            safety_passed=True,
            stream_available=True,
            domain=domain,
            status="escalated" if mediator.confidence < 0.5 else "resolved",
        )

    # ── BuyerAgent ─────────────────────────────────────────────────────

    async def _run_buyer_agent(
        self,
        message: str,
        facts: Dict[str, str],
        emotion_level: EmotionLevel,
    ) -> AgentPerspective:
        """运行买家 Agent：提取投诉中的诉求和证据。"""
        facts_text = self._format_facts(facts)
        emotion_label = self._emotion_context(emotion_level)

        prompt = (
            f"{BUYER_AGENT_PROMPT}\n\n"
            f"## 订单事实数据\n{facts_text}\n\n"
            f"## 用户情绪等级\n{emotion_label}\n\n"
            f"## 用户投诉消息\n{message}"
        )

        try:
            raw = await self._llm.chat_qwen_with_prompt(
                prompt=prompt,
                system_prompt="你是一个电商消费者权益分析专家。只输出JSON，不要解释。",
            )
            data = self._safe_json_parse(raw)
            return AgentPerspective(
                role="buyer",
                summary=data.get("buyer_summary", data.get("core_issue", "无法解析买家分析")),
                demands=data.get("buyer_demands", [data.get("core_issue", "")]),
                evidence=data.get("mentioned_evidence", []),
                proposed_solution=data.get("compensation_expectation", ""),
                confidence=0.85,
                raw_output=raw,
            )
        except Exception as e:
            logger.warning(f"BuyerAgent 失败: {str(e)[:100]}")
            return AgentPerspective(
                role="buyer",
                summary=f"买家投诉: {message[:100]}...",
                demands=["退款/退货"],
                raw_output=str(e),
                confidence=0.3,
            )

    # ── SellerAgent ────────────────────────────────────────────────────

    async def _run_seller_agent(
        self,
        message: str,
        facts: Dict[str, str],
        emotion_level: EmotionLevel,
    ) -> AgentPerspective:
        """运行卖家 Agent：从平台规则和卖家立场评估投诉。"""
        facts_text = self._format_facts(facts)
        emotion_label = self._emotion_context(emotion_level)

        prompt = (
            f"{SELLER_AGENT_PROMPT}\n\n"
            f"## 订单事实数据\n{facts_text}\n\n"
            f"## 用户情绪等级\n{emotion_label}\n\n"
            f"## 用户投诉消息\n{message}"
        )

        try:
            raw = await self._llm.chat_qwen_with_prompt(
                prompt=prompt,
                system_prompt="你是一个电商平台合规与卖家权益分析专家。只输出JSON。",
            )
            data = self._safe_json_parse(raw)
            solutions = data.get("acceptable_solutions", [])
            return AgentPerspective(
                role="seller",
                summary=data.get("seller_summary", data.get("rule_assessment", "无法解析卖家分析")),
                demands=data.get("seller_defenses", []),        # 卖家立场 = 辩解论点
                evidence=data.get("seller_faults", []),         # 事实证据 = 确认的过失
                proposed_solution="; ".join(solutions) if solutions else data.get("rule_assessment", ""),
                confidence=0.8,
                raw_output=raw,
            )
        except Exception as e:
            logger.warning(f"SellerAgent 失败: {str(e)[:100]}")
            return AgentPerspective(
                role="seller",
                summary="无法完成卖家分析",
                proposed_solution="建议人工审核",
                raw_output=str(e),
                confidence=0.3,
            )

    # ── MediatorAgent ──────────────────────────────────────────────────

    async def _run_mediator_agent(
        self,
        message: str,
        facts: Dict[str, str],
        buyer: AgentPerspective,
        seller: AgentPerspective,
        emotion_level: EmotionLevel,
    ) -> AgentPerspective:
        """运行调停 Agent：综合双方观点 + 平台规则做出裁决。"""
        # ── 参数校验：双方分析质量不足 → 直接兜底 ──
        buyer_ok = self._is_perspective_viable(buyer)
        seller_ok = self._is_perspective_viable(seller)
        if not buyer_ok or not seller_ok:
            missing = []
            if not buyer_ok:
                missing.append("买家分析")
            if not seller_ok:
                missing.append("卖家分析")
            logger.warning(
                "MediatorAgent 输入质量不足，直接兜底",
                missing=missing,
                buyer_summary_len=len(buyer.summary) if buyer.summary else 0,
                seller_summary_len=len(seller.summary) if seller.summary else 0,
            )
            return AgentPerspective(
                role="mediator",
                summary="前置分析数据不足，无法做出可靠裁决",
                proposed_solution="建议升级人工处理，并重新收集订单及双方信息",
                confidence=0.0,
                evidence=[f"缺失分析: {', '.join(missing)}"],
            )

        facts_text = self._format_facts(facts)
        emotion_label = self._emotion_context(emotion_level)

        prompt = (
            f"{MEDIATOR_AGENT_PROMPT}\n\n"
            f"## 订单事实数据\n{facts_text}\n\n"
            f"## 用户情绪等级\n{emotion_label}\n\n"
            f"## 买家立场分析\n- 核心诉求: {buyer.summary}\n"
            f"- 具体要求: {', '.join(buyer.demands) if buyer.demands else '未明确'}\n"
            f"- 证据主张: {', '.join(buyer.evidence) if buyer.evidence else '未提供'}\n"
            f"- 期望补偿: {buyer.proposed_solution}\n\n"
            f"## 卖家立场分析\n- 规则评估: {seller.summary}\n"
            f"- 卖家辩解: {', '.join(seller.demands) if seller.demands else '暂未确认'}\n"
            f"- 确认过失: {', '.join(seller.evidence) if seller.evidence else '无'}\n"
            f"- 可接受方案: {seller.proposed_solution}\n\n"
            f"## 用户原始投诉\n{message}"
        )

        try:
            raw = await self._llm.chat_qwen_with_prompt(
                prompt=prompt,
                system_prompt="你是一个电商售后纠纷调停专家。只输出JSON，不要解释。",
            )
            data = self._safe_json_parse(raw)
            escalate = data.get("escalate_to_human", False)

            # 组装 evidence：责任占比 + 补偿方案 + 升级原因（供日志/调试）
            evidence_parts = []
            resp = data.get("responsibility_split")
            if resp and isinstance(resp, dict):
                evidence_parts.append(
                    f"责任占比: 买家{resp.get('buyer_percent', '?')}% / "
                    f"卖家{resp.get('seller_percent', '?')}%"
                )
            comp = data.get("compensation")
            if comp and isinstance(comp, dict):
                evidence_parts.append(
                    f"补偿方案: {comp.get('type', '?')} {comp.get('amount_yuan', 0)}元 "
                    f"({comp.get('detail', '无详情')})"
                )
            if escalate and data.get("escalate_reason"):
                evidence_parts.append(f"升级原因: {data['escalate_reason']}")

            # summary 优先用完整的 mediator_summary，退化到 verdict
            full_summary = data.get("mediator_summary") or data.get("verdict") or "无法做出裁决"

            return AgentPerspective(
                role="mediator",
                summary=full_summary,
                demands=[data.get("suggested_solution", "")],
                evidence=evidence_parts,
                proposed_solution=data.get("suggested_solution", ""),
                confidence=0.3 if escalate else 0.85,
                raw_output=raw,
            )
        except Exception as e:
            logger.warning(f"MediatorAgent 失败: {str(e)[:100]}")
            return AgentPerspective(
                role="mediator",
                summary="自动裁决暂时无法完成，建议升级人工处理",
                proposed_solution="升级到高级专员处理",
                confidence=0.0,
                raw_output=str(e),
            )

    # ── 工具函数 ──────────────────────────────────────────────────────

    @staticmethod
    def _is_perspective_viable(p: AgentPerspective) -> bool:
        """检查 Agent 分析结果是否可被 Mediator 使用。

        条件：summary 非空且非兜底语，且 demands 或 proposed_solution 有实质内容。
        """
        if not p:
            return False
        summary_ok = bool(p.summary) and p.summary not in (
            "买家分析失败", "无法解析买家分析", "卖家分析失败",
            "无法解析卖家分析", "无法完成卖家分析",
        )
        if not summary_ok:
            return False
        has_demands = bool(p.demands) and any(d for d in p.demands if d)
        has_solution = bool(p.proposed_solution) and p.proposed_solution not in ("", "信息不足")
        return has_demands or has_solution

    @staticmethod
    def _format_facts(facts: Dict[str, str]) -> str:
        """格式化事实数据为 LLM 可读文本。"""
        if not facts:
            return "（未获取到事实数据）"
        lines = []
        for key, value in facts.items():
            # 截断过长的事实数据
            value_short = value[:500] + "..." if len(value) > 500 else value
            lines.append(f"- {key}: {value_short}")
        return "\n".join(lines)

    @staticmethod
    def _emotion_context(level: EmotionLevel) -> str:
        """将情绪等级转为上下文描述。"""
        labels = {
            EmotionLevel.EMERGENCY: "极端（涉及法律/舆情威胁，需立即升级）",
            EmotionLevel.ANGRY: "非常愤怒（强烈不满、质疑诚信）",
            EmotionLevel.DISAPPOINTED: "失望（对服务体验不满）",
            EmotionLevel.ANXIOUS: "焦急（希望尽快解决）",
            EmotionLevel.NEUTRAL: "中性（普通投诉）",
            EmotionLevel.SATISFIED: "满意",
            EmotionLevel.GRATEFUL: "感激",
        }
        return labels.get(level, "中性")

    @staticmethod
    def _safe_json_parse(raw: str) -> dict:
        """安全解析 LLM 输出的 JSON。"""
        if not raw:
            return {}
        text = raw.strip()
        # 去除 markdown 代码块
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:]) if len(lines) > 1 else text
        if text.endswith("```"):
            text = text[:-3].strip()
        if text.startswith("```"):
            text = text[3:].strip()
        try:
            return _json.loads(text)
        except _json.JSONDecodeError:
            # 尝试从文本中提取 JSON 部分
            import re
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                try:
                    return _json.loads(match.group())
                except _json.JSONDecodeError:
                    pass
            return {"raw_text": raw}

    @staticmethod
    def _format_final_reply(
        buyer: AgentPerspective,
        seller: AgentPerspective,
        mediator: AgentPerspective,
        facts: Dict[str, str],
        emotion_level: EmotionLevel,
    ) -> str:
        """将裁决结果格式化为面向用户的最终回复。"""
        # 裁决心意度低 → 升级提示
        if mediator.confidence < 0.3:
            return (
                "非常抱歉给您带来了不便，您的问题我们已经详细记录。"
                "由于情况较为复杂，我们已将其升级给高级专员处理，"
                "专员将在 2 小时内通过电话或在线客服与您联系。"
                "如有紧急问题，请拨打客服热线 400-XXX-XXXX。"
            )

        # 正常裁决
        parts = []
        # 开头共情（根据情绪等级调整语气）
        if emotion_level >= EmotionLevel.ANGRY:
            parts.append("非常理解您的心情，对于这次不愉快的购物体验我们深表歉意。")
        elif emotion_level >= EmotionLevel.DISAPPOINTED:
            parts.append("感谢您的耐心反馈，我们非常重视您提出的问题。")
        else:
            parts.append("感谢您的反馈，我们已经仔细核实了相关情况。")

        # 裁决结论
        parts.append(f"\n经过核查，{mediator.summary}")

        # 具体方案
        if mediator.proposed_solution:
            parts.append(f"\n我们为您提供的解决方案如下：\n{mediator.proposed_solution}")

        # 如果提到升级
        if mediator.confidence < 0.5:
            parts.append(
                "\n\n由于该问题需要进一步核查，我们同时已将其升级给高级专员，"
                "专员将与您联系确认后续处理。"
            )

        parts.append("\n\n如您还有其他疑问，随时可以联系我们。再次为给您带来的不便表示歉意。")

        return "".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# 纠纷触发判断工具函数（供编排器使用）
# ═══════════════════════════════════════════════════════════════════════

# 纠纷特征关键词 —— 匹配到其中 2 个以上认为需要走纠纷协调
_DISPUTE_KEYWORDS = [
    "骗子", "骗钱", "垃圾", "曝光", "举报", "投诉",
    "卖家不", "不同意", "拒绝退款", "不退款", "不退",
    "发错货", "质量问题", "与描述不符", "假货",
    "找你们领导", "投诉到", "消费者协会", "12315",
    "赔偿", "三倍", "假一赔", "补偿",
]


def should_use_dispute_coordinator(
    message: str,
    emotion_result: EmotionResult | None = None,
    intent_action: str | None = None,
) -> bool:
    """判断当前请求是否适合路由到纠纷协调器。

    触发条件（满足任一即触发）：
    1. 情绪 ANGRY 或 EMERGENCY
    2. 情绪 DISAPPOINTED + 意图为 request-return
    3. 消息中匹配到 2 个以上纠纷关键词
    """
    msg_lower = message.lower()

    # 条件 1: 强烈负面情绪
    if emotion_result and emotion_result.level >= EmotionLevel.ANGRY:
        return True

    # 条件 2: 失望 + 退货意图
    if (
        emotion_result
        and emotion_result.level == EmotionLevel.DISAPPOINTED
        and intent_action == "request-return"
    ):
        return True

    # 条件 3: 纠纷关键词匹配
    match_count = sum(1 for kw in _DISPUTE_KEYWORDS if kw in msg_lower)
    if match_count >= 2:
        return True

    return False
