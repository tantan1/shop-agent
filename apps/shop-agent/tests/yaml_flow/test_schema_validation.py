"""阶段 1（YAML Schema 设计）单元测试。

覆盖：协议解析 + 校验器（阶段 1.9 全规则）+ 示例加载（阶段 6.1）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.modules.chat.agent.yaml_flow import (
    EdgeDef,
    FlowFile,
    FlowValidationError,
    GraphState,
    NodeConfig,
    NodeDef,
    NodeType,
    ReadFrom,
    StateField,
    load_flow_file,
    validate_flow,
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


def _minimal_flow(**overrides) -> FlowFile:
    """构造一个最小合法 flow，便于单测定向破坏。"""
    state = GraphState(fields=[StateField(name="params", type="dict", required=True)])
    nodes = [
        NodeDef(id="normalize", type="normalize"),
        NodeDef(id="react_node", type="react", config=NodeConfig(skill="request-return")),
    ]
    base = dict(
        entry="normalize",
        state=state,
        nodes=nodes,
        edges=[EdgeDef(from_="normalize", to="react_node")],
        guards=[{"type": "input_filter"}, {"type": "output_filter"}],
    )
    base.update(overrides)
    return FlowFile.model_validate(base)


# ───────────────────────────── 示例加载 ─────────────────────────────

def test_example_loads_and_validates():
    flow, warnings = load_flow_file(_EXAMPLE)
    assert flow.entry == "normalize"
    assert len(flow.nodes) == 7
    assert len(flow.edges) == 6
    # 守卫齐备（输入/输出过滤 + 锁 + 持久化 + 埋点）
    guard_types = {g.type for g in flow.guards}
    assert {"input_filter", "output_filter", "lock", "persist", "observe"} <= guard_types
    # 示例无阻断级错误
    validate_flow(flow)
    # 敏感节点存在时，应给出 lock 建议告警（本示例已挂 lock，故无该告警）
    assert not any("lock" in w for w in warnings)


# ───────────────────────────── 校验规则 ─────────────────────────────

def test_entry_must_exist():
    with pytest.raises(FlowValidationError, match="entry"):
        validate_flow(_minimal_flow(entry="ghost"))


def test_edge_from_unknown_node():
    bad = _minimal_flow(edges=[EdgeDef(from_="ghost", to="react_node")])
    with pytest.raises(FlowValidationError, match="edge.from"):
        validate_flow(bad)


def test_edge_to_unknown_node():
    bad = _minimal_flow(edges=[EdgeDef(from_="normalize", to="ghost")])
    with pytest.raises(FlowValidationError, match="edge.to"):
        validate_flow(bad)


def test_condition_target_unknown_node():
    bad = _minimal_flow(
        edges=[
            EdgeDef(
                from_="normalize",
                conditions=[{"op": "on_intent==", "value": "x", "target": "ghost"}],
            )
        ]
    )
    with pytest.raises(FlowValidationError, match="condition target"):
        validate_flow(bad)


def test_read_from_undeclared_state_field():
    node = NodeDef(
        id="react_node",
        type="react",
        config=NodeConfig(
            skill="request-return",
            read_from=[ReadFrom(state="nonexistent")],
        ),
    )
    bad = _minimal_flow(nodes=[NodeDef(id="normalize", type="normalize"), node])
    with pytest.raises(FlowValidationError, match="read_from.state"):
        validate_flow(bad)


def test_read_from_required_but_state_not_required():
    state = GraphState(fields=[StateField(name="params", type="dict", required=False)])
    node = NodeDef(
        id="react_node",
        type="react",
        config=NodeConfig(
            skill="request-return",
            read_from=[ReadFrom(state="params", required=True)],
        ),
    )
    bad = _minimal_flow(
        state=state,
        nodes=[NodeDef(id="normalize", type="normalize"), node],
    )
    with pytest.raises(FlowValidationError, match="required=False"):
        validate_flow(bad)


def test_missing_input_filter_guard_fails():
    bad = _minimal_flow(
        guards=[{"type": "output_filter"}],  # 缺 input_filter
    )
    with pytest.raises(FlowValidationError, match="输入安全过滤"):
        validate_flow(bad)


def test_missing_output_filter_guard_fails():
    bad = _minimal_flow(
        guards=[{"type": "input_filter"}],  # 缺 output_filter
    )
    with pytest.raises(FlowValidationError, match="输出安全过滤"):
        validate_flow(bad)


def test_primitive_node_cannot_reference_skill():
    bad = _minimal_flow(
        nodes=[
            NodeDef(id="normalize", type="normalize", config=NodeConfig(skill="query-order")),
            NodeDef(id="react_node", type="react", config=NodeConfig(skill="request-return")),
        ]
    )
    with pytest.raises(FlowValidationError, match="不应引用 skill"):
        validate_flow(bad)


def test_sensitive_skill_inline_override_rejected():
    bad = _minimal_flow(
        nodes=[
            NodeDef(id="normalize", type="normalize"),
            NodeDef(
                id="react_node",
                type="react",
                config=NodeConfig(skill="request-return", extra={"sop_override": "..."}),
            ),
        ]
    )
    with pytest.raises(FlowValidationError, match="禁止的安全覆盖键"):
        validate_flow(bad)


def test_edge_requires_to_or_conditions():
    with pytest.raises(Exception):
        EdgeDef(from_="a")  # 既无 to 也无 conditions
