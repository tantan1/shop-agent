"""阶段 2（解析器 v1）单元测试：图构建与条件路由（阶段 6.2）。

不真实执行 LLM/Tool（compile 仅构建图，handler 不触发），聚焦：
- 图节点/边结构正确
- 条件路由函数按 intent/emergency 正确分流
"""

from __future__ import annotations

from pathlib import Path

from src.modules.chat.agent.yaml_flow import (
    CompiledFlow,
    Condition,
    ConditionOp,
    FlowCompiler,
    _compile_router,
    load_flow_file,
)

_EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "modules"
    / "chat"
    / "agent"
    / "yaml_flow"
    / "examples"
    / "return_flow.yaml"
)


def _edges_of(flow: CompiledFlow):
    g = flow.graph.get_graph()
    return [(e.source, e.target) for e in g.edges]


# ───────────────────────────── 图结构 ─────────────────────────────

def test_compile_example_succeeds():
    flow_file, _ = load_flow_file(_EXAMPLE)
    compiler = FlowCompiler()
    compiled = compiler.compile(flow_file)
    assert isinstance(compiled, CompiledFlow)


def test_example_node_count():
    flow_file, _ = load_flow_file(_EXAMPLE)
    compiled = FlowCompiler().compile(flow_file)
    edge_tuples = _edges_of(compiled)
    sources = {s for s, _ in edge_tuples}
    targets = {t for _, t in edge_tuples}
    # 业务节点都应出现在边的端点里（或作为 entry 经 START 连接）
    for nid in ["normalize", "intent", "query_order", "validate", "confirm", "execute_return", "reply"]:
        assert (nid in sources) or (nid in targets), f"节点 {nid} 未出现在图中"


def test_conditional_edges_from_confirm():
    flow_file, _ = load_flow_file(_EXAMPLE)
    compiled = FlowCompiler().compile(flow_file)
    edge_tuples = _edges_of(compiled)
    confirm_targets = {t for s, t in edge_tuples if s == "confirm"}
    # confirm 条件边：approve→execute_return, reject→reply, + END 兜底
    assert "execute_return" in confirm_targets
    assert "reply" in confirm_targets
    assert any(t in ("__end__", "END") for t in confirm_targets)


# ───────────────────────────── 条件路由逻辑 ─────────────────────────────

def _router_with(conditions, state):
    return _compile_router(conditions)(state)


def test_router_on_intent_eq():
    conds = [
        Condition(op=ConditionOp.ON_INTENT_EQ, value="approve", target="execute_return"),
        Condition(op=ConditionOp.ON_INTENT_EQ, value="reject", target="reply"),
        Condition(op=ConditionOp.ALWAYS, target="fallback"),
    ]
    assert _router_with(conds, {"intent": {"action": "approve"}}) == "execute_return"
    assert _router_with(conds, {"intent": {"action": "reject"}}) == "reply"
    assert _router_with(conds, {"intent": {"action": "other"}}) == "fallback"


def test_router_emergency_shortcut():
    conds = [
        Condition(op=ConditionOp.ON_EMERGENCY, target="escalate"),
        Condition(op=ConditionOp.ON_INTENT_EQ, value="approve", target="execute_return"),
    ]
    assert _router_with(conds, {"emergency": True}) == "escalate"
    # 非紧急时 emergency 不命中，落到 intent 分支
    assert _router_with(conds, {"emergency": False, "intent": {"action": "approve"}}) == "execute_return"


def test_router_no_match_returns_end():
    conds = [Condition(op=ConditionOp.ON_INTENT_EQ, value="approve", target="execute_return")]
    assert _router_with(conds, {"intent": {"action": "reject"}}) == "__end__" or _router_with(conds, {"intent": {"action": "reject"}}) == "END"


def test_router_complexity_eq():
    conds = [Condition(op=ConditionOp.ON_COMPLEXITY_EQ, value="multi_step", target="react_path")]
    assert _router_with(conds, {"intent": {"complexity": "multi_step"}}) == "react_path"
    assert _router_with(conds, {"intent": {"complexity": "single"}}) in ("__end__", "END")
