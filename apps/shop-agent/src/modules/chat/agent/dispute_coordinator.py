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

import time as _time
from typing import TYPE_CHECKING, Dict

from src.modules.chat.agent.composite import (
    AgentContext,
    AgentNode,
    ParallelComposite,
    ResultMerger,
    SerialComposite,
)
from src.modules.chat.agent.dispute_agents import (
    BuyerAnalysisAgent,
    FactCollectorAgent,
    MediatorAgent,
    SellerAnalysisAgent,
)
from src.modules.chat.agent.dispute_common import (
    MEDIATOR_AGENT_PROMPT,  # noqa: F401
    PLATFORM_RULING_POLICY,  # noqa: F401
    AgentPerspective,
    EmotionLevel,
    _mock_get_after_sale_evidence,  # noqa: F401
    get_after_sale_evidence,  # noqa: F401
)
from src.modules.chat.agent.dispute_graph import DisputeLangGraph
from src.modules.chat.core.sentiment_service import EmotionResult
from src.modules.chat.schemas import ChatRequest, ChatResponse
from src.shared.logger import APILogger

if TYPE_CHECKING:
    from src.modules.chat.core.llm_service import LLMService
    from src.modules.chat.core.tool_registry import ToolService

logger = APILogger("dispute_coordinator")


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
        use_langgraph: bool = False,
    ):
        self._llm = llm
        self._tool_service = tool_service
        self._domain = domain
        self._use_langgraph = use_langgraph
        self._langgraph: DisputeLangGraph | None = None

    def _ensure_langgraph(self) -> DisputeLangGraph:
        """懒初始化 LangGraph 实例。"""
        if self._langgraph is None:
            self._langgraph = DisputeLangGraph(
                tool_service=self._tool_service,
                llm_service=self._llm,
            )
        return self._langgraph

    def _build_agent_tree(self) -> AgentNode:
        """构建纠纷协调 Agent 树（组合模式）。

        结构：
            SerialComposite([
                FactCollectorAgent,
                ParallelComposite(
                    [BuyerAnalysisAgent, SellerAnalysisAgent],
                    result_keys=["buyer_analysis", "seller_analysis"],
                ),
                MediatorAgent,
            ])
        """
        fact_collector = FactCollectorAgent(self._tool_service)
        buyer = BuyerAnalysisAgent(self._llm)
        seller = SellerAnalysisAgent(self._llm)
        mediator = MediatorAgent(self._llm)

        analysis = ParallelComposite(
            [buyer, seller],
            merger=ResultMerger(),
            result_keys=["buyer_analysis", "seller_analysis"],
        )
        return SerialComposite([fact_collector, analysis, mediator])

    # ── 主入口 ──────────────────────────────────────────────────────────

    async def resolve(  # noqa: PLR0913
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
        """执行纠纷协调流程（组合模式）。

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

        logger.info(
            "纠纷协调启动",
            conversation_id=conversation_id,
            emotion=emotion_level.name,
            order_id=order_id or "未提供",
            message_preview=request.message[:100],
            backend="langgraph" if self._use_langgraph else "composite",
        )

        # ── 执行 Agent 树 ────────────────────────────────────────────────
        if self._use_langgraph:
            return await self._resolve_with_langgraph(
                request=request,
                emotion_level=emotion_level,
                order_id=order_id,
                conversation_id=conversation_id,
                domain=domain,
                intent_steps=steps,
                t_start=t_start,
            )

        # 默认：使用 composite.py 组合模式
        agent_tree = self._build_agent_tree()
        ctx = AgentContext(
            request_message=request.message,
            facts={},
            emotion_level=emotion_level,
            order_id=order_id,
            conversation_id=conversation_id,
            domain=domain,
        )

        mediator = await agent_tree.execute(ctx)
        facts = ctx.facts

        # ── 步骤记录（保持与旧实现兼容）──
        t_facts_ms = 0.0
        t_analysis_ms = 0.0
        t_mediator_ms = 0.0

        steps.append(
            {
                "step_name": "纠纷协调-事实收集",
                "step_order": len(steps),
                "status": "success" if facts else "partial",
                "output_data": {
                    "facts_count": len(facts),
                    "keys": list(facts.keys()),
                    "duration_ms": round(t_facts_ms, 1),
                },
            }
        )

        buyer = ctx.metadata.get("buyer_analysis_result")
        seller = ctx.metadata.get("seller_analysis_result")

        if isinstance(buyer, Exception):
            buyer = AgentPerspective(
                role="buyer", summary="买家分析失败", demands=["信息不足，需人工介入"], raw_output=str(buyer)
            )
        if isinstance(seller, Exception):
            seller = AgentPerspective(
                role="seller", summary="卖家分析失败", proposed_solution="信息不足", raw_output=str(seller)
            )

        steps.append(
            {
                "step_name": "纠纷协调-双方分析",
                "step_order": len(steps),
                "status": "success",
                "output_data": {
                    "buyer_demands": buyer.demands[:3] if buyer else [],
                    "seller_solutions": seller.proposed_solution[:100] if seller else "",
                    "duration_ms": round(t_analysis_ms, 1),
                },
            }
        )

        responsible_party = ""
        matched_rule = ""
        if mediator.evidence:
            for ev in mediator.evidence:
                if ev.startswith("责任方: "):
                    responsible_party = ev[len("责任方: ") :]
                elif ev.startswith("命中规则: "):
                    matched_rule = ev[len("命中规则: ") :]

        steps.append(
            {
                "step_name": "纠纷协调-调停裁决",
                "step_order": len(steps),
                "status": "success",
                "output_data": {
                    "verdict_short": mediator.summary[:100],
                    "escalate": mediator.escalate,
                    "responsible_party": responsible_party,
                    "matched_rule": matched_rule,
                    "third_party_responsibility": mediator.third_party_responsibility,
                    "duration_ms": round(t_mediator_ms, 1),
                },
            }
        )

        final_reply = self._format_final_reply(buyer, seller, mediator, facts, emotion_level)

        t_total = (_time.perf_counter() - t_start) * 1000
        logger.info(
            "纠纷协调完成",
            escalate=mediator.escalate,
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
            status="escalated" if mediator.escalate else "resolved",
        )

    @staticmethod
    def _format_final_reply(
        buyer: AgentPerspective,
        seller: AgentPerspective,
        mediator: AgentPerspective,
        facts: Dict[str, str],
        emotion_level: EmotionLevel,
    ) -> str:
        """将裁决结果格式化为面向用户的最终回复。"""
        if mediator.confidence < 0.3:
            return (
                "非常抱歉给您带来了不便，您的问题我们已经详细记录。"
                "由于情况较为复杂，我们已将其升级给高级专员处理，"
                "专员将在 2 小时内通过电话或在线客服与您联系。"
                "如有紧急问题，请拨打客服热线 400-XXX-XXXX。"
            )

        parts = []
        if emotion_level >= EmotionLevel.ANGRY:
            parts.append("非常理解您的心情，对于这次不愉快的购物体验我们深表歉意。")
        elif emotion_level >= EmotionLevel.DISAPPOINTED:
            parts.append("感谢您的耐心反馈，我们非常重视您提出的问题。")
        else:
            parts.append("感谢您的反馈，我们已经仔细核实了相关情况。")

        parts.append(f"\n经过核查，{mediator.summary}")

        if mediator.proposed_solution:
            parts.append(f"\n我们为您提供的解决方案如下：\n{mediator.proposed_solution}")

        if mediator.confidence < 0.5:
            parts.append(
                "\n\n由于该问题需要进一步核查，我们同时已将其升级给高级专员，"
                "专员将与您联系确认后续处理。"
            )

        parts.append("\n\n如您还有其他疑问，随时可以联系我们。再次为给您带来的不便表示歉意。")
        return "".join(parts)

    async def _resolve_with_langgraph(  # noqa: PLR0913
        self,
        request: ChatRequest,
        emotion_level: EmotionLevel,
        order_id: str | None,
        conversation_id: str,
        domain: str,
        intent_steps: list,
        t_start: float,
    ) -> ChatResponse:
        """使用 LangGraph 执行纠纷协调流程。"""
        langgraph = self._ensure_langgraph()
        result = await langgraph.resolve(
            request_message=request.message,
            emotion_level=emotion_level,
            order_id=order_id,
            conversation_id=conversation_id,
            domain=domain,
        )

        facts = result.get("facts", {})
        buyer = result.get("buyer_result")
        seller = result.get("seller_result")
        mediator = result.get("mediator_result")
        error = result.get("error")

        if error:
            logger.error(f"LangGraph 纠纷协调异常: {error}")

        # 容错：确保 mediator 存在
        if mediator is None:
            mediator = AgentPerspective(
                role="mediator",
                summary="自动裁决暂时无法完成，建议升级人工处理",
                proposed_solution="升级到高级专员处理",
                confidence=0.0,
                escalate=True,
            )

        steps = list(intent_steps)
        steps.append(
            {
                "step_name": "纠纷协调-事实收集",
                "step_order": len(steps),
                "status": "success" if facts else "partial",
                "output_data": {
                    "facts_count": len(facts),
                    "keys": list(facts.keys()),
                    "duration_ms": 0.0,
                },
            }
        )

        steps.append(
            {
                "step_name": "纠纷协调-双方分析",
                "step_order": len(steps),
                "status": "success",
                "output_data": {
                    "buyer_demands": buyer.demands[:3] if buyer else [],
                    "seller_solutions": seller.proposed_solution[:100] if seller else "",
                    "duration_ms": 0.0,
                },
            }
        )

        responsible_party = ""
        matched_rule = ""
        if mediator and mediator.evidence:
            for ev in mediator.evidence:
                if ev.startswith("责任方: "):
                    responsible_party = ev[len("责任方: ") :]
                elif ev.startswith("命中规则: "):
                    matched_rule = ev[len("命中规则: ") :]

        steps.append(
            {
                "step_name": "纠纷协调-调停裁决",
                "step_order": len(steps),
                "status": "success",
                "output_data": {
                    "verdict_short": mediator.summary[:100] if mediator else "",
                    "escalate": mediator.escalate if mediator else True,
                    "responsible_party": responsible_party,
                    "matched_rule": matched_rule,
                    "third_party_responsibility": getattr(mediator, "third_party_responsibility", False),
                    "duration_ms": 0.0,
                },
            }
        )

        final_reply = self._format_final_reply(
            buyer or AgentPerspective(role="buyer", summary="买家分析失败"),
            seller or AgentPerspective(role="seller", summary="卖家分析失败"),
            mediator,
            facts,
            emotion_level,
        )

        t_total = (_time.perf_counter() - t_start) * 1000
        logger.info(
            "纠纷协调完成(langgraph)",
            escalate=mediator.escalate if mediator else True,
            total_ms=round(t_total, 1),
        )

        return ChatResponse(
            message=final_reply,
            conversation_id=conversation_id,
            steps=steps,
            documents_used=[],
            safety_passed=True,
            stream_available=True,
            domain=domain,
            status="escalated" if mediator and mediator.escalate else "resolved",
        )


# ═══════════════════════════════════════════════════════════════════════
# 纠纷触发判断工具函数（供编排器使用）
# ═══════════════════════════════════════════════════════════════════════

# 纠纷特征关键词 —— 匹配到其中 2 个以上认为需要走纠纷协调
_DISPUTE_KEYWORDS = [
    "骗子",
    "骗钱",
    "垃圾",
    "曝光",
    "举报",
    "投诉",
    "卖家不",
    "不同意",
    "拒绝退款",
    "不退款",
    "不退",
    "发错货",
    "质量问题",
    "与描述不符",
    "假货",
    "找你们领导",
    "投诉到",
    "消费者协会",
    "12315",
    "赔偿",
    "三倍",
    "假一赔",
    "补偿",
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
