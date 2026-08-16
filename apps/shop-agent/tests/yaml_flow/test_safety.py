"""阶段 5.2 / 5.3 / 5.5 测试：硬强制与安全的 YAML 固化（fail-closed 校验）。

验证（需 skill_registry 进入编译期强校验）：
- 5.2：敏感 Skill 的高后果字段（order_id/phone 等身份-资金类）必须在节点中被
       hardcode 定稿或 read_from 提供确定性来源，否则编译失败（业务不可关闭强制）。
- 5.3：敏感 Skill（risk=high / hitl=true）被引用时必须有 SOP 操作指南（body 非空），
       否则编译失败（校验拒绝发布）。
- 5.5：敏感 Skill 只能被引用，节点不得携带 sop_override / skip_validation / disable_hitl
       等安全覆盖键，否则编译失败。
- 5.4：缺少 input_filter / output_filter 守卫则编译失败（已有，这里回归确认）。
"""

from __future__ import annotations

import types

import pytest

from src.modules.chat.agent.yaml_flow.compiler import FlowCompiler
from src.modules.chat.agent.yaml_flow.schema import (
    EdgeDef,
    FlowFile,
    GraphState,
    GuardDef,
    HardcodeDecl,
    NodeConfig,
    NodeDef,
    StateField,
)
from src.modules.chat.agent.yaml_flow.validator import (
    FlowValidationError,
    validate_flow,
)


# ───────────────────────────── mock 依赖 ─────────────────────────────

class _Skill:
    def __init__(self, name, *, risk="low", hitl=False, body="", params=None):
        self.name = name
        self.risk = risk
        self.hitl = hitl
        self.body = body
        self.params = params or {}


class _FakeSkillRegistry:
    """skills 为 dict（兼容 validator 取值）。"""

    def __init__(self, skills):
        self.skills = skills


def _registry_with_sensitive():
    """含一个敏感 Skill：request-return（risk=high, hitl=true），高后果字段 order_id。"""
    return _FakeSkillRegistry({
        "request-return": _Skill(
            "request-return",
            risk="high",
            hitl=True,
            body="1. 校验订单 2. 发起退款 3. 回告单号",  # 有 SOP
            params={
                "order_id": {"type": "string", "required": False, "semantic": "order_id"},
                "reason": {"type": "string", "required": False, "semantic": "return_reason"},
            },
        ),
    })


def _base_flow(node: NodeDef, *, edges=None, guards=None):
    # 提供最小 GraphState 契约（params / tool_result 等字段），满足 1.9 read_from 校验
    state = GraphState(fields=[
        StateField(name="params", type="dict", required=False),
        StateField(name="tool_result", type="any", required=False),
    ])
    # 所有合法流程的基线：input + output 守卫（阶段 5.4）。
    # 仅 5.4 专门测缺失的用例通过 guards=[] 显式覆盖。
    base_guards = guards if guards is not None else [
        GuardDef(type="input_filter"),
        GuardDef(type="output_filter"),
    ]
    return FlowFile(
        version="1.0",
        entry=node.id,
        state=state,
        nodes=[node],
        edges=edges or [],
        guards=base_guards,
    )


# ───────────────────────────── 5.2：高后果字段强制绑定 ─────────────────────────────

def _sensitive_node(**overrides) -> NodeDef:
    cfg = dict(skill="request-return")
    cfg.update(overrides)
    return NodeDef(
        id="exec",
        type="direct_tool",
        config=NodeConfig(**cfg),
    )


def test_5_2_rejects_unbound_high_consequence_field():
    """敏感 Skill 的 order_id 既未 hardcode 也未 read_from → 编译失败（阶段 5.2）。"""
    # 节点只 read_from tool_result，未绑定 order_id
    node = _sensitive_node(read_from=[{"state": "tool_result"}])
    flow = _base_flow(node)
    with pytest.raises(FlowValidationError):
        validate_flow(flow, skill_registry=_registry_with_sensitive())


def test_5_2_accepts_hardcode_binding():
    """order_id 经 hardcode 定稿注入 → 通过（阶段 5.2）。"""
    node = _sensitive_node(
        hardcode=[HardcodeDecl(field="order_id", value="WB123")],
        read_from=[{"state": "tool_result"}],
    )
    flow = _base_flow(node)
    # 不应抛错
    assert validate_flow(flow, skill_registry=_registry_with_sensitive()) is not None


def test_5_2_accepts_read_from_binding():
    """order_id 经 read_from: state.params 提供确定性来源 → 通过（阶段 5.2）。"""
    node = _sensitive_node(read_from=[{"state": "params"}])
    flow = _base_flow(node)
    assert validate_flow(flow, skill_registry=_registry_with_sensitive()) is not None


def test_5_2_compiler_blocks_unbound_on_compile():
    """FlowCompiler.compile 也执行强校验，未绑定高后果字段则编译失败（阶段 5.2）。"""
    node = _sensitive_node(read_from=[{"state": "tool_result"}])
    flow = _base_flow(node)
    compiler = FlowCompiler(skill_registry=_registry_with_sensitive())
    with pytest.raises(FlowValidationError):
        compiler.compile(flow)


# ───────────────────────────── 5.3：敏感 Skill 必须有 SOP ─────────────────────────────

def test_5_3_rejects_sensitive_skill_without_sop():
    """敏感 Skill 被引用但 body 为空（无操作指南）→ 编译失败（阶段 5.3）。"""
    reg = _FakeSkillRegistry({
        "request-return": _Skill(
            "request-return",
            risk="high",
            hitl=True,
            body="",  # 缺 SOP
            params={"order_id": {"semantic": "order_id"}},
        ),
    })
    # 即便 order_id 已绑定，仍因缺 SOP 被拒
    node = _sensitive_node(
        hardcode=[HardcodeDecl(field="order_id", value="WB123")],
    )
    flow = _base_flow(node)
    with pytest.raises(FlowValidationError):
        validate_flow(flow, skill_registry=reg)


def test_5_3_low_risk_skill_exempt_from_sop_requirement():
    """非敏感 Skill（risk=low, hitl=false）无 SOP 不触发 5.3 拒绝。"""
    reg = _FakeSkillRegistry({
        "coupon-inquiry": _Skill("coupon-inquiry", risk="low", hitl=False, body=""),
    })
    node = NodeDef(
        id="exec",
        type="direct_tool",
        config=NodeConfig(skill="coupon-inquiry", read_from=[{"state": "params"}]),
    )
    flow = _base_flow(node)
    # 非敏感 Skill 不受 5.2/5.3 约束（无高后果字段、无需 SOP）
    assert validate_flow(flow, skill_registry=reg) is not None


# ───────────────────────────── 5.5：敏感 Skill 只能引用不可改写 ─────────────────────────────

def test_5_5_rejects_sop_override_key():
    """节点引用敏感 Skill 同时携带 sop_override → 编译失败（阶段 5.5）。"""
    node = _sensitive_node(
        read_from=[{"state": "params"}],
        extra={"sop_override": "改掉安全边界"},
    )
    flow = _base_flow(node)
    with pytest.raises(FlowValidationError):
        validate_flow(flow, skill_registry=_registry_with_sensitive())


def test_5_5_rejects_disable_hitl_key():
    """节点引用敏感 Skill 携带 disable_hitl → 编译失败（阶段 5.5）。"""
    node = _sensitive_node(
        read_from=[{"state": "params"}],
        extra={"disable_hitl": True},
    )
    flow = _base_flow(node)
    with pytest.raises(FlowValidationError):
        validate_flow(flow, skill_registry=_registry_with_sensitive())


# ───────────────────────────── 5.4：守卫强制存在 ─────────────────────────────

def test_5_4_rejects_missing_output_filter():
    """缺 output_filter 守卫 → 编译失败（阶段 5.4，回归）。"""
    node = _sensitive_node(read_from=[{"state": "params"}])
    flow = _base_flow(node, guards=[GuardDef(type="input_filter")])  # 仅 input，缺 output
    with pytest.raises(FlowValidationError):
        validate_flow(flow, skill_registry=_registry_with_sensitive())


def test_5_4_accepts_with_both_filters():
    """input + output 守卫齐备 → 通过（阶段 5.4，回归）。"""
    node = _sensitive_node(read_from=[{"state": "params"}])
    flow = _base_flow(node)  # 默认已带 input + output 守卫
    assert validate_flow(flow, skill_registry=_registry_with_sensitive()) is not None
