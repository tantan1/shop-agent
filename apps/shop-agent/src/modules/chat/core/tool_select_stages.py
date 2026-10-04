"""工具选择四层 Stage 实现（P0 规则 / P1 FAISS / P2 线性头 / P3 LLM 兜底）。

各 Stage 忠实复用现有逻辑：
- P0 RuleFilterStage  → 复用 ``SkillRegistry.intent_tool_map``（意图→工具规则）
- P1 FaissRecallStage → 复用 ``react_agent_selection.EmbeddingToolMatcher``（FAISS 语义重排）
- P2 LinearHeadStage  → 复用训练好的 PyTorch 线性头（``ToolHeadClassifier`` / ``outputs/tool_head_M.pt``）
- P3 LlmFallbackStage → 复用 ``LLMService.tool_selector_llm``（LLM 兜底收窄）

设计约束（见 tool-select-pipeline-design §4、implementation-plan Phase 4）：
- 每层只回答"本层对候选的相关度"，不自行融合；Pipeline 统一做候选传递与降级。
- P0/P1/P2 不早停（``tool=None``），仅通过 ``scored_candidates`` 把候选集收窄后传给下一层；
  最终由 ``ToolSelectPipeline.emit_final_scope_as_plan`` 把末级收窄集组装成 ``ToolPlan``，
  与旧 ``ReActAgent._select_tools_for_intent`` 的输出语义一致。
- 模块导入时通过 ``register_stage`` 登记，供配置按 name 解析（不依赖 importlib）。

依赖通过 ``context``（= Pipeline 构造时的 ``deps``）传入，Stage 不持久持有重对象。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.modules.chat.agent.skill_loader import SkillRegistry
from src.modules.chat.core.tool_head_classifier import ToolHeadClassifier
from src.modules.chat.core.tool_select_stage import (
    CandidateScore,
    StageResult,
    ToolSelectStage,
    register_stage,
)
from src.modules.chat.schemas import (
    STAGE_P0_RULE,
    STAGE_P1_FAISS,
    STAGE_P2_LINEAR,
    STAGE_P3_LLM,
)
from src.shared.logger import get_logger

logger = get_logger(__name__)


class RuleFilterStage(ToolSelectStage):
    """P0：意图规则过滤。依 ``intent_action`` 从意图→工具映射取候选集。

    不受入参 ``scope`` 约束（与旧 ``_select_tools_for_intent`` 一致：P0 由意图直接决定候选）。
    不早停（``tool=None``），候选集经 ``scored_candidates`` 传给 P1。
    """

    def __init__(self, name: str = STAGE_P0_RULE) -> None:
        super().__init__(name)

    async def run(
        self, query: str, scope: List[str], context: Dict[str, Any]
    ) -> StageResult:
        skill_registry: Optional[SkillRegistry] = context.get("skill_registry")
        # 多意图（治本）：intent_actions 为意图动作列表，对每条意图各取工具集再取并集；
        # 兼容旧单次调用仍可能传入单个字符串。
        intent_actions_raw = context.get("intent_actions")
        if isinstance(intent_actions_raw, str):
            intent_actions_raw = [intent_actions_raw]
        intent_actions: List[str] = list(intent_actions_raw) if intent_actions_raw else []
        if skill_registry is None:
            logger.warning("P0 skill_registry 缺失")
            return StageResult(tool=None, source=self.name, error="skill_registry 缺失")

        try:
            intent_map = skill_registry.intent_tool_map
            # 对每条意图各取工具集，取并集（多意图 → 多工具）
            tool_names: set = set()
            for a in intent_actions:
                tool_names |= intent_map.get(
                    a or "unknown", intent_map.get("unknown", set())
                )
        except Exception as e:
            logger.error(f"P0 意图规则查询失败: {e}", exc_info=True)
            return StageResult(tool=None, source=self.name, error=f"intent_tool_map error: {e}")

        if not tool_names:
            logger.info(
            "P0 意图规则无命中，候选集为空",
            intent_actions=intent_actions,
            )
            return StageResult(tool=None, source=self.name, scored_candidates=[])

        # 真实打分（设计 1）：弱/无意图（unknown）→ 低分 0.0；明确命中 → 高 1.0
        is_weak = not intent_actions or all(
            (a in (None, "unknown")) for a in intent_actions
        )
        rule_score = 0.0 if is_weak else 1.0
        scored = [
            CandidateScore(tool=n, score=rule_score, source=self.name)
            for n in sorted(tool_names)
        ]
        logger.info(
            "P0 意图过滤完成",
            intent_actions=intent_actions,
            weak=is_weak,
            candidates=len(scored),
            tool_names=sorted(tool_names),
        )
        # 唯一强意图（确定性规则，且为单意图）→ 允许早停（设计 §2 line 49，更早收敛、省成本）。
        # 多意图（>1 意图动作）不早停：P0 对每条意图各取【候选工具集】取并集，得到覆盖两个意图组的
        # 候选作用域；精确工具由后续 P1/P2/P3 收窄（intent_tool_map 是宽集合，不应在 P0 直接收敛），
        # 从而稳定保证「多意图 → 多工具」且不漏掉任一意图组。
        if len(intent_actions) == 1 and (not is_weak) and len(tool_names) == 1:
            only = sorted(tool_names)[0]
            return StageResult(
                tool=only, confidence=1.0, source=self.name, scored_candidates=scored
            )
        return StageResult(
            tool=None, confidence=1.0, source=self.name, scored_candidates=scored
        )


class FaissRecallStage(ToolSelectStage):
    """P1：FAISS 语义重排。复用 ``EmbeddingToolMatcher`` 对 ``scope`` 做 Top-K 重排。

    不早停（``tool=None``）；重排后的 Top-K 作为 ``scored_candidates`` 传给 P2。
    matcher 不可用 / 重排失败时原样透传 ``scope``，不缩候选（与旧逻辑回退行为一致）。
    """

    def __init__(self, name: str = STAGE_P1_FAISS, top_k: int = 5) -> None:
        super().__init__(name)
        self._top_k = top_k

    async def run(
        self, query: str, scope: List[str], context: Dict[str, Any]
    ) -> StageResult:
        tool_matcher = context.get("tool_matcher")
        intent_actions = context.get("intent_actions") or []
        intent_action: Optional[str] = intent_actions[0] if intent_actions else None

        if tool_matcher is None or not scope:
            # 无 matcher → 无真实语义信号，透传且 score=None（不覆盖上层）
            return StageResult(
                tool=None,
                source=self.name,
                scored_candidates=[CandidateScore(t, source=self.name) for t in scope],
            )

        try:
            ranked = await tool_matcher.rank_with_scores(
                user_query=query,
                candidate_names=set(scope),
                intent_action=intent_action,
                top_k=self._top_k,
                query_embedding=context.get("query_embedding"),
            )
        except Exception as e:
            logger.warning(f"P1 FAISS 重排失败，回退全量候选: {e}")
            ranked = [(t, None) for t in scope]

        if not ranked:
            ranked = [(t, None) for t in scope]

        # 透传分支（失败/无索引）score=None；正常分支为真实余弦相似度
        scored = [CandidateScore(tool=t, score=s, source=self.name) for t, s in ranked]
        return StageResult(
            tool=None, confidence=None, source=self.name, scored_candidates=scored
        )


class LinearHeadStage(ToolSelectStage):
    """P2：PyTorch 线性头确认。复用训练好的 ``ToolHead`` 权重（``outputs/tool_head_prod.pt``）
    对候选工具做 ``embedding -> 线性头`` 打分，取 top-k。

    仅当候选数 > 2 时介入（与迁移前的 P2 本地模型分类器行为一致）；否则原样透传。
    不早停（``tool=None``）；选中子集作为 ``scored_candidates`` 传给 P3 / 末级。

    词表安全：head 只在训练工具集上有效。若 scope 中无候选落在 head 词表内
    （如生产 skill 名与训练词表 0 重合），``ToolHeadClassifier.score`` 返回 ``None``，
    本层退化为透传，绝不误删可能正确的工具。
    """

    def __init__(self, name: str = STAGE_P2_LINEAR, top_k: Optional[int] = None) -> None:
        super().__init__(name)
        self._top_k = top_k

    async def run(
        self, query: str, scope: List[str], context: Dict[str, Any]
    ) -> StageResult:
        if len(scope) <= 2:
            # 未介入：不产分，透传（score=None，沿用上游真实分数）
            return StageResult(
                tool=None,
                source=self.name,
                scored_candidates=[CandidateScore(t, source=self.name) for t in scope],
            )

        try:
            clf: Optional[ToolHeadClassifier] = context.get("tool_head_classifier")
            if clf is None:
                clf = ToolHeadClassifier.get_instance()
            selected = clf.score_with_probs(
                query, list(scope), top_k=self._top_k,
                query_embedding=context.get("query_embedding"),
            )
        except Exception as e:
            logger.warning(f"P2 线性头调用失败，保留 P1 结果: {e}")
            selected = None

        if not selected:
            # 无法评分（加载失败/候选全在词表外）：透传且不产分
            return StageResult(
                tool=None,
                source=self.name,
                scored_candidates=[CandidateScore(t, source=self.name) for t in scope],
            )

        in_scope = {n for n in scope}
        selected = [(n, p) for n, p in selected if n in in_scope]
        if not selected:
            selected = [(t, None) for t in scope]
        scored = [CandidateScore(tool=n, score=p, source=self.name) for n, p in selected]
        return StageResult(
            tool=None, confidence=None, source=self.name, scored_candidates=scored
        )


class LlmFallbackStage(ToolSelectStage):
    """P3：LLM 兜底收窄。复用 ``LLMService.tool_selector_llm`` 从候选中选最相关（可多选）。

    仅当候选数 > 1 时介入；否则原样透传。不早停（``tool=None``）：
    选中的子集作为 ``scored_candidates`` 传给末级，由 Pipeline 组装最终 ToolPlan。
    与旧流程差异：旧 P3 是运行时 LangChain 中间件（在 Agent 执行期选工具），
    此处为规划期一次性 LLM 收窄；两者角色一致（LLM 兜底），接入口不同。
    """

    def __init__(self, name: str = STAGE_P3_LLM) -> None:
        super().__init__(name)

    async def run(
        self, query: str, scope: List[str], context: Dict[str, Any]
    ) -> StageResult:
        if len(scope) <= 1:
            # 选择器：不产分（score=None），沿用上游真实分数
            return StageResult(
                tool=None,
                source=self.name,
                scored_candidates=[CandidateScore(t, source=self.name) for t in scope],
            )

        llm_service = context.get("llm_service")
        if llm_service is None:
            return StageResult(
                tool=None,
                source=self.name,
                scored_candidates=[CandidateScore(t, source=self.name) for t in scope],
            )

        try:
            model = llm_service.tool_selector_llm
            prompt = (
                "你是一个工具路由器。从候选工具中选出与用户问题最相关的工具"
                "（可多选，每行一个工具名，不要任何解释）。\n"
                f"候选工具: {sorted(scope)}\n用户问题: {query}"
            )
            resp = await model.ainvoke(prompt)
            text = getattr(resp, "content", str(resp))
            chosen = [t for t in scope if t in text]
        except Exception as e:
            logger.warning(f"P3 LLM 选择失败，保留上游结果: {e}")
            return StageResult(
                tool=None,
                source=self.name,
                error=f"llm_failed: {e}",
                scored_candidates=[CandidateScore(t, source=self.name) for t in scope],
            )

        if not chosen:
            chosen = list(scope)

        # P3 是离散选择器，不输出置信度分数（设计 §1.1），仅收窄候选
        scored = [CandidateScore(tool=t, source=self.name) for t in chosen]
        return StageResult(
            tool=None, confidence=None, source=self.name, scored_candidates=scored
        )


# ── 模块导入时登记（设计 §6：显式注册表，不依赖 importlib）──
register_stage(STAGE_P0_RULE, RuleFilterStage)
register_stage(STAGE_P1_FAISS, FaissRecallStage)
register_stage(STAGE_P2_LINEAR, LinearHeadStage)
register_stage(STAGE_P3_LLM, LlmFallbackStage)
