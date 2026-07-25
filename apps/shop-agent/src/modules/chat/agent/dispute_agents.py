"""纠纷协调 Agent 树。

将 BuyerAgent / SellerAgent / MediatorAgent / FactCollector 实现为
`AgentNode` 接口的叶子节点，通过 `ParallelComposite` / `SerialComposite`
组合为 Agent 树。组合模式基础设施见 `composite.py`。
"""
from __future__ import annotations

import asyncio
import json as _json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.modules.chat.agent.composite import (
    AgentContext,
    AgentNode,
)
from src.modules.chat.agent.dispute_common import (
    BUYER_AGENT_PROMPT,
    MEDIATOR_AGENT_PROMPT,
    PLATFORM_RULING_POLICY,
    SELLER_AGENT_PROMPT,
    AgentPerspective,
    EmotionLevel,
    _format_validity,
    _fuse_confidence,
    _logprob_proxy,
    _refusal_score,
    _tool_confidence,
    get_after_sale_evidence,
)
from src.shared.logger import APILogger

logger = APILogger("dispute_agent")


# ── 数据结构 ──────────────────────────────────────────────────────────


@dataclass
# ── 叶子节点 ─────────────────────────────────────────────────────────


class FactCollectorAgent(AgentNode):
    """事实收集 Agent（叶子节点）。"""

    name = "fact_collector"

    def __init__(self, tool_service: Any):
        self._tool_service = tool_service

    async def execute(self, ctx: AgentContext) -> AgentPerspective:
        """并行收集订单、物流、余额、优惠券等事实数据。"""
        tasks = []
        order_id = ctx.order_id

        if order_id:
            tasks.append(("order_info", self._tool_service.dispatch("query-order", {"order_id": order_id})))
            tasks.append(("order_shipping", self._tool_service.dispatch("check-shipping", {"order_id": order_id})))
            tasks.append(("after_sale_evidence", get_after_sale_evidence(order_id)))
        else:
            tasks.append(("recent_orders", self._tool_service.dispatch("query-order", {})))

        tasks.append(("balance_info", self._tool_service.dispatch("check-balance", {})))
        tasks.append(("coupon_info", self._tool_service.dispatch("coupon-inquiry", {})))

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

        ctx.facts.update(facts)

        return AgentPerspective(
            role="fact_collector",
            summary=f"收集到 {len(facts)} 条事实数据",
            demands=list(facts.keys()),
            evidence=list(facts.values()),
            confidence=self._tool_confidence(facts),
            raw_output=str(facts),
        )

    async def _execute_one(self, name: str, coro) -> str:
        try:
            return await coro
        except Exception as e:
            logger.warning(f"事实收集 {name} 失败: {str(e)[:80]}")
            return "[收集失败]"

    @staticmethod
    def _tool_confidence(facts: Dict[str, str]) -> float:
        if not facts:
            return 0.0
        ok = sum(
            1
            for v in facts.values()
            if v and "[查询失败" not in v and "[收集失败" not in v and "失败" not in v
        )
        return ok / len(facts)


class BuyerAnalysisAgent(AgentNode):
    """买家分析 Agent（叶子节点）。"""

    name = "buyer_analysis"

    def __init__(self, llm_service: Any):
        self._llm = llm_service

    async def execute(self, ctx: AgentContext) -> AgentPerspective:
        """站在买家角度分析诉求和证据。"""
        facts_text = self._format_facts(ctx.facts)
        emotion_label = self._emotion_context(ctx.emotion_level)

        prompt = (
            f"{BUYER_AGENT_PROMPT}\n\n"
            f"## 订单事实数据\n{facts_text}\n\n"
            f"## 用户情绪等级\n{emotion_label}\n\n"
            f"## 用户投诉消息\n{ctx.request_message}"
        )

        try:
            raw = await self._llm.chat_qwen_with_prompt(
                prompt=prompt,
                system_prompt="你是一个电商消费者权益分析专家。只输出JSON，不要解释。",
            )
            data = self._safe_json_parse(raw)
            fused_conf = _fuse_confidence(
                _logprob_proxy(),
                _tool_confidence(ctx.facts),
                _format_validity(data, ["buyer_summary", "core_issue", "buyer_demands"]),
                _refusal_score(raw),
            )
            return AgentPerspective(
                role="buyer",
                summary=data.get("buyer_summary", data.get("core_issue", "无法解析买家分析")),
                demands=data.get("buyer_demands", [data.get("core_issue", "")]),
                evidence=data.get("mentioned_evidence", []),
                proposed_solution=data.get("compensation_expectation", ""),
                confidence=fused_conf,
                raw_output=raw,
            )
        except Exception as e:
            logger.warning(f"BuyerAgent 失败: {str(e)[:100]}")
            return AgentPerspective(
                role="buyer",
                summary=f"买家投诉: {ctx.request_message[:100]}...",
                demands=["退款/退货"],
                raw_output=str(e),
                confidence=0.3,
            )

    @staticmethod
    def _format_facts(facts: Dict[str, str]) -> str:
        if not facts:
            return "（未获取到事实数据）"
        lines = []
        for key, value in facts.items():
            value_short = value[:500] + "..." if len(value) > 500 else value
            lines.append(f"- {key}: {value_short}")
        return "\n".join(lines)

    @staticmethod
    def _emotion_context(level: Optional[EmotionLevel]) -> str:
        labels = {
            EmotionLevel.EMERGENCY: "极端（涉及法律/舆情威胁，需立即升级）",
            EmotionLevel.ANGRY: "非常愤怒（强烈不满、质疑诚信）",
            EmotionLevel.DISAPPOINTED: "失望（对服务体验不满）",
            EmotionLevel.ANXIOUS: "焦急（希望尽快解决）",
            EmotionLevel.NEUTRAL: "中性（普通投诉）",
            EmotionLevel.SATISFIED: "满意",
            EmotionLevel.GRATEFUL: "感激",
        }
        return labels.get(level, "中性") if level else "中性"

    @staticmethod
    def _safe_json_parse(raw: str) -> dict:
        if not raw:
            return {}
        text = raw.strip()
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
            import re
            match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if match:
                try:
                    return _json.loads(match.group())
                except _json.JSONDecodeError:
                    pass
            return {"raw_text": raw}


class SellerAnalysisAgent(AgentNode):
    """卖家分析 Agent（叶子节点）。"""

    name = "seller_analysis"

    def __init__(self, llm_service: Any):
        self._llm = llm_service

    async def execute(self, ctx: AgentContext) -> AgentPerspective:
        """从平台规则和卖家立场评估投诉。"""
        facts_text = self._format_facts(ctx.facts)
        emotion_label = self._emotion_context(ctx.emotion_level)

        prompt = (
            f"{SELLER_AGENT_PROMPT}\n\n"
            f"## 订单事实数据\n{facts_text}\n\n"
            f"## 用户情绪等级\n{emotion_label}\n\n"
            f"## 用户投诉消息\n{ctx.request_message}"
        )

        try:
            raw = await self._llm.chat_qwen_with_prompt(
                prompt=prompt,
                system_prompt="你是一个电商平台合规与卖家权益分析专家。只输出JSON。",
            )
            data = self._safe_json_parse(raw)
            solutions = data.get("acceptable_solutions", [])
            fused_conf = _fuse_confidence(
                _logprob_proxy(),
                _tool_confidence(ctx.facts),
                _format_validity(data, ["seller_summary", "rule_assessment", "seller_defenses"]),
                _refusal_score(raw),
            )
            return AgentPerspective(
                role="seller",
                summary=data.get("seller_summary", data.get("rule_assessment", "无法解析卖家分析")),
                demands=data.get("seller_defenses", []),
                evidence=data.get("seller_faults", []),
                proposed_solution="; ".join(solutions) if solutions else data.get("rule_assessment", ""),
                confidence=fused_conf,
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

    @staticmethod
    def _format_facts(facts: Dict[str, str]) -> str:
        if not facts:
            return "（未获取到事实数据）"
        lines = []
        for key, value in facts.items():
            value_short = value[:500] + "..." if len(value) > 500 else value
            lines.append(f"- {key}: {value_short}")
        return "\n".join(lines)

    @staticmethod
    def _emotion_context(level: Optional[EmotionLevel]) -> str:
        labels = {
            EmotionLevel.EMERGENCY: "极端（涉及法律/舆情威胁，需立即升级）",
            EmotionLevel.ANGRY: "非常愤怒（强烈不满、质疑诚信）",
            EmotionLevel.DISAPPOINTED: "失望（对服务体验不满）",
            EmotionLevel.ANXIOUS: "焦急（希望尽快解决）",
            EmotionLevel.NEUTRAL: "中性（普通投诉）",
            EmotionLevel.SATISFIED: "满意",
            EmotionLevel.GRATEFUL: "感激",
        }
        return labels.get(level, "中性") if level else "中性"

    @staticmethod
    def _safe_json_parse(raw: str) -> dict:
        if not raw:
            return {}
        text = raw.strip()
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
            import re
            match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if match:
                try:
                    return _json.loads(match.group())
                except _json.JSONDecodeError:
                    pass
            return {"raw_text": raw}


class MediatorAgent(AgentNode):
    """调停裁决 Agent（叶子节点，需要 buyer + seller 输入）。"""

    name = "mediator"

    def __init__(self, llm_service: Any):
        self._llm = llm_service

    async def execute(self, ctx: AgentContext) -> AgentPerspective:
        """综合双方观点 + 平台规则做出裁决。"""
        buyer = ctx.metadata.get("buyer_analysis_result")
        seller = ctx.metadata.get("seller_analysis_result")

        if not self._is_perspective_viable(buyer) or not self._is_perspective_viable(seller):
            missing = []
            if not self._is_perspective_viable(buyer):
                missing.append("买家分析")
            if not self._is_perspective_viable(seller):
                missing.append("卖家分析")
            return AgentPerspective(
                role="mediator",
                summary="前置分析数据不足，无法做出可靠裁决",
                proposed_solution="建议升级人工处理，并重新收集订单及双方信息",
                confidence=0.0,
                escalate=True,
                evidence=[f"缺失分析: {', '.join(missing)}"],
            )

        facts_text = self._format_facts(ctx.facts)
        emotion_label = self._emotion_context(ctx.emotion_level)

        prompt = (
            f"{MEDIATOR_AGENT_PROMPT}\n\n"
            f"{PLATFORM_RULING_POLICY}\n\n"
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
            f"## 用户原始投诉\n{ctx.request_message}"
        )

        try:
            raw = await self._llm.chat_qwen_with_prompt(
                prompt=prompt,
                system_prompt="你是一个电商售后纠纷调停专家。只输出JSON，不要解释。",
            )
            data = self._safe_json_parse(raw)

            fused_conf = _fuse_confidence(
                _logprob_proxy(),
                _tool_confidence(ctx.facts),
                _format_validity(data, ["mediator_summary", "verdict", "suggested_solution"]),
                _refusal_score(raw),
            )
            escalate = data.get("escalate_to_human", False) or fused_conf < 0.5

            evidence_parts = []
            resp = data.get("responsibility_split")
            if resp and isinstance(resp, dict):
                evidence_parts.append(
                    f"责任占比: 买家{resp.get('buyer_percent', '?')}% / "
                    f"卖家{resp.get('seller_percent', '?')}% / "
                    f"快递{resp.get('courier_percent', 0)}%"
                )
            comp = data.get("compensation")
            if comp and isinstance(comp, dict):
                evidence_parts.append(
                    f"补偿方案: {comp.get('type', '?')} {comp.get('amount_yuan', 0)}元 "
                    f"({comp.get('detail', '无详情')})"
                )
            matched_rule = data.get("matched_rule", "")
            if matched_rule:
                evidence_parts.append(f"命中规则: {matched_rule}")
            responsible_party = data.get("responsible_party", "")
            if responsible_party:
                evidence_parts.append(f"责任方: {responsible_party}")
            if escalate and data.get("escalate_reason"):
                evidence_parts.append(f"升级原因: {data['escalate_reason']}")

            full_summary = data.get("mediator_summary") or data.get("verdict") or "无法做出裁决"
            has_third_party = responsible_party == "courier" or (
                resp and isinstance(resp, dict) and resp.get("courier_percent", 0) > 0
            )

            return AgentPerspective(
                role="mediator",
                summary=full_summary,
                demands=[data.get("suggested_solution", "")],
                evidence=evidence_parts,
                proposed_solution=data.get("suggested_solution", ""),
                confidence=fused_conf,
                raw_output=raw,
                third_party_responsibility=has_third_party,
                escalate=escalate,
            )
        except Exception as e:
            logger.warning(f"MediatorAgent 失败: {str(e)[:100]}")
            return AgentPerspective(
                role="mediator",
                summary="自动裁决暂时无法完成，建议升级人工处理",
                proposed_solution="升级到高级专员处理",
                confidence=0.0,
                escalate=True,
                raw_output=str(e),
            )

    @staticmethod
    def _is_perspective_viable(p: Optional[AgentPerspective]) -> bool:
        if not p:
            return False
        summary_ok = bool(p.summary) and p.summary not in (
            "买家分析失败",
            "无法解析买家分析",
            "卖家分析失败",
            "无法解析卖家分析",
            "无法完成卖家分析",
        )
        if not summary_ok:
            return False
        has_demands = bool(p.demands) and any(d for d in p.demands if d)
        has_solution = bool(p.proposed_solution) and p.proposed_solution not in ("", "信息不足")
        return has_demands or has_solution

    @staticmethod
    def _format_facts(facts: Dict[str, str]) -> str:
        if not facts:
            return "（未获取到事实数据）"
        lines = []
        for key, value in facts.items():
            value_short = value[:500] + "..." if len(value) > 500 else value
            lines.append(f"- {key}: {value_short}")
        return "\n".join(lines)

    @staticmethod
    def _emotion_context(level: Optional[EmotionLevel]) -> str:
        labels = {
            EmotionLevel.EMERGENCY: "极端（涉及法律/舆情威胁，需立即升级）",
            EmotionLevel.ANGRY: "非常愤怒（强烈不满、质疑诚信）",
            EmotionLevel.DISAPPOINTED: "失望（对服务体验不满）",
            EmotionLevel.ANXIOUS: "焦急（希望尽快解决）",
            EmotionLevel.NEUTRAL: "中性（普通投诉）",
            EmotionLevel.SATISFIED: "满意",
            EmotionLevel.GRATEFUL: "感激",
        }
        return labels.get(level, "中性") if level else "中性"

    @staticmethod
    def _safe_json_parse(raw: str) -> dict:
        if not raw:
            return {}
        text = raw.strip()
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
            import re
            match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if match:
                try:
                    return _json.loads(match.group())
                except _json.JSONDecodeError:
                    pass
            return {"raw_text": raw}
