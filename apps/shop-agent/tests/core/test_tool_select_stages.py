"""工具选择四级 Stage + 新 Pipeline 集成回归测试（仅新实现）。

覆盖：
- 四层 Stage 经 import 时 register_stage 登记（STAGE_REGISTRY）
- 每个 Stage 独立 mock 依赖（SkillRegistry / EmbeddingToolMatcher / ToolHeadClassifier / LLMService）
- ToolSelectPipeline.emit_final_scope_as_plan：末级收窄集组装为 ToolPlan（与旧 _select_tools_for_intent 输出一致）
- 默认关闭 emit_final_scope_as_plan 时保持原 need_llm 降级契约（不破坏既有框架测试）

运行：cd apps/shop-agent && python -m pytest tests/core/test_tool_select_stages.py -q
"""
import asyncio
from types import SimpleNamespace

from src.modules.chat.core.tool_select_pipeline import PipelinePolicy, ToolSelectPipeline
from src.modules.chat.core.tool_select_stage import (
    STAGE_REGISTRY,
    CandidateScore,
    StageResult,
)
from src.modules.chat.core.tool_select_stages import (
    FaissRecallStage,
    LinearHeadStage,
    LlmFallbackStage,
    RuleFilterStage,
)
from src.modules.chat.schemas import (
    STAGE_P0_RULE,
    STAGE_P1_FAISS,
    STAGE_P2_LINEAR,
    STAGE_P3_LLM,
    ToolPlan,
)


# ── 轻量 fake 依赖（不拉起真实 Embedding / 本地模型 / LLM）── #
class FakeSkillRegistry:
    def __init__(self, intent_map: dict):
        self._m = intent_map

    @property
    def intent_tool_map(self):
        return self._m

    @property
    def tool_descriptions(self):
        return {}


class FakeMatcher:
    def __init__(self, ranked):
        self._ranked = ranked

    async def rank(self, user_query, candidate_names, intent_action=None, top_k=5, query_embedding=None):
        return [t for t in self._ranked if t in candidate_names][:top_k]


class FakeToolHead:
    """受控 P2 线性头：从候选中保留 selected 交集（模拟「线性头确认」）。"""

    def __init__(self, selected):
        self._selected = selected

    def score(self, query, candidates, top_k=3, query_embedding=None):
        return [t for t in self._selected if t in set(candidates)]


class FakeLLM:
    def __init__(self, text):
        self._text = text

    async def ainvoke(self, prompt):
        return SimpleNamespace(content=self._text)


class FakeLLMService:
    """fake LLMService：tool_selector_llm 暴露一个 FakeLLM（带 ainvoke）。"""

    def __init__(self, text):
        self.tool_selector_llm = FakeLLM(text)


def _ctx(**over):
    base = {
        "intent_action": "query_order",
        "tool_matcher": None,
        "local_model_service": None,
        "llm_service": None,
        "skill_registry": FakeSkillRegistry(
            {"query_order": {"query-order", "check-shipping"}}
        ),
        "tool_descriptions": {},
    }
    base.update(over)
    return base


def _run(stage, query, scope, ctx):
    return asyncio.run(stage.run(query, scope, ctx))


# ── 登记校验 ── #
def test_stages_registered():
    assert STAGE_REGISTRY[STAGE_P0_RULE] is RuleFilterStage
    assert STAGE_REGISTRY[STAGE_P1_FAISS] is FaissRecallStage
    assert STAGE_REGISTRY[STAGE_P2_LINEAR] is LinearHeadStage
    assert STAGE_REGISTRY[STAGE_P3_LLM] is LlmFallbackStage


# ── P0 规则过滤 ── #
def test_p0_rule_filter_narrows_by_intent():
    st = RuleFilterStage()
    res = _run(st, "查订单", ["query-order", "check-shipping", "request-return"], _ctx())
    assert res.tool is None  # P0 是过滤层，不早停
    assert {c.tool for c in res.scored_candidates} == {"query-order", "check-shipping"}
    assert res.source == STAGE_P0_RULE


def test_p0_unknown_falls_to_default():
    ctx = _ctx(
        intent_action="zzz",
        skill_registry=FakeSkillRegistry({"unknown": {"knowledge_search"}}),
    )
    st = RuleFilterStage()
    res = _run(st, "随便问问", [], ctx)
    assert {c.tool for c in res.scored_candidates} == {"knowledge_search"}


def test_p0_missing_skill_registry_is_error():
    st = RuleFilterStage()
    res = _run(st, "x", [], {k: v for k, v in _ctx().items() if k != "skill_registry"})
    assert res.error is not None


# ── P1 FAISS 重排 ── #
def test_p1_faiss_rerank():
    ctx = _ctx(tool_matcher=FakeMatcher(["check-shipping", "query-order", "request-return"]))
    st = FaissRecallStage(top_k=2)
    res = _run(st, "查物流", ["query-order", "check-shipping", "request-return"], ctx)
    assert [c.tool for c in res.scored_candidates] == ["check-shipping", "query-order"]


def test_p1_no_matcher_passthrough():
    st = FaissRecallStage()
    res = _run(st, "x", ["a", "b"], _ctx(tool_matcher=None))
    assert {c.tool for c in res.scored_candidates} == {"a", "b"}


# ── P2 线性头（PyTorch 前向确认）── #
def test_p2_local_model_confirms_subset():
    ctx = _ctx(tool_head_classifier=FakeToolHead(["query-order"]))
    st = LinearHeadStage()
    res = _run(st, "查订单", ["query-order", "check-shipping", "request-return"], ctx)
    assert {c.tool for c in res.scored_candidates} == {"query-order"}


def test_p2_skip_when_le2():
    st = LinearHeadStage()
    res = _run(st, "x", ["a", "b"], _ctx())
    assert {c.tool for c in res.scored_candidates} == {"a", "b"}


# ── P3 LLM 兜底 ── #
def test_p3_llm_fallback_narrows():
    ctx = _ctx(llm_service=FakeLLMService("query-order\ncheck-shipping"))
    st = LlmFallbackStage()
    res = _run(st, "查订单和物流", ["query-order", "check-shipping", "request-return"], ctx)
    assert {c.tool for c in res.scored_candidates} == {"query-order", "check-shipping"}


def test_p3_skip_when_le1():
    st = LlmFallbackStage()
    res = _run(st, "x", ["a"], _ctx())
    assert {c.tool for c in res.scored_candidates} == {"a"}


# ── Pipeline 集成：emit_final_scope_as_plan ── #
def test_pipeline_emit_final_scope_as_plan():
    pipeline = ToolSelectPipeline(
        stages=[
            RuleFilterStage(),
            FaissRecallStage(top_k=2),
            LinearHeadStage(),
            LlmFallbackStage(),
        ],
        all_tools=["query-order", "check-shipping", "request-return", "knowledge_search"],
        deps=_ctx(
            intent_action="query_order",
            tool_matcher=FakeMatcher(["check-shipping", "query-order"]),
            tool_head_classifier=FakeToolHead(["query-order"]),
            llm_service=FakeLLMService("query-order"),
        ),
        policy=PipelinePolicy(
            thresholds={
                STAGE_P0_RULE: 1.0,
                STAGE_P1_FAISS: 1.0,
                STAGE_P2_LINEAR: 1.0,
                STAGE_P3_LLM: 1.0,
            },
            fallback={"on_stage_failure": "skip"},
            emit_final_scope_as_plan=True,
            convergence_threshold=0,  # 隔离：本测试仅验证 emit 路径，关闭收敛早停
        ),
    )
    plan = asyncio.run(pipeline.select("查订单"))
    assert isinstance(plan, ToolPlan)
    assert plan.source == STAGE_P3_LLM  # 末级生效阶段
    assert plan.to_tool_names() == {"query-order"}
    assert plan.stop_condition == "plan_complete"


def test_pipeline_emit_preserves_multi_tool_set():
    # P0 给出 2 个候选，P1/P2/P3 均不改写 → 末级集为 2 个 → plan_complete（<=2 确定性）
    pipeline = ToolSelectPipeline(
        stages=[
            RuleFilterStage(),
            FaissRecallStage(top_k=5),
            LinearHeadStage(),
            LlmFallbackStage(),
        ],
        all_tools=["query-order", "check-shipping", "request-return", "knowledge_search"],
        deps=_ctx(
            intent_action="query_order",
            tool_matcher=FakeMatcher(["query-order", "check-shipping", "request-return"]),
            tool_head_classifier=FakeToolHead(["query-order", "check-shipping"]),
            llm_service=FakeLLMService("query-order\ncheck-shipping"),
        ),
        policy=PipelinePolicy(
            thresholds={STAGE_P0_RULE: 1.0, STAGE_P1_FAISS: 1.0, STAGE_P2_LINEAR: 1.0, STAGE_P3_LLM: 1.0},
            fallback={"on_stage_failure": "skip"},
            emit_final_scope_as_plan=True,
        ),
    )
    plan = asyncio.run(pipeline.select("查订单和物流"))
    assert plan.to_tool_names() == {"query-order", "check-shipping"}
    assert plan.stop_condition == "plan_complete"


def test_pipeline_default_no_emit_returns_need_llm():
    # emit_final_scope_as_plan 默认关闭：保持原 need_llm 降级契约（既有框架测试依赖）
    pipeline = ToolSelectPipeline(
        stages=[RuleFilterStage(), FaissRecallStage(top_k=2)],
        all_tools=["query-order", "check-shipping"],
        deps=_ctx(
            intent_action="query_order",
            tool_matcher=FakeMatcher(["query-order", "check-shipping"]),
        ),
        policy=PipelinePolicy(thresholds={}, fallback={}),
    )
    plan = asyncio.run(pipeline.select("查订单"))
    assert plan.stop_condition == "need_llm"
    assert plan.actions == []
