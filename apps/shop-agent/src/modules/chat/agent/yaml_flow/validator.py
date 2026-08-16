"""YAML 编排校验器（阶段 1.9）。

在解析器编译成图之前，对 FlowFile 做静态校验，给出明确错误。
校验失败直接抛 ``FlowValidationError``（fail-closed：不合法的流程不允许发布/执行）。

校验项（对应任务文件 1.9）：
1. entry 节点存在
2. 所有 edge 的 from/to 引用合法（指向已声明节点）
3. 条件边的 target 引用合法
4. read_from.state 路径必须存在于 GraphState.fields（阶段 1.5 + 1.9）
5. read_from 声明 required=True 的字段，必须在 GraphState.fields 标记为 required=True（否则 fail-closed 无意义）
6. 敏感 Skill（request-return / check-balance 等）只能被节点 config.skill 引用为黑盒，
   不允许内联改写（阶段 5.5，此处仅校验「无内联 SOP 覆盖字段」的约定位）
7. 守卫强制存在性：input_filter / output_filter 守卫（或对应节点类型）必须存在，缺失则编译失败（阶段 5.4）
8. react / direct_tool / rag_pipeline 类节点若声明 skill，skill 必须存在（运行时由 SkillRegistry 校验，此处只报结构性缺失）
"""

from __future__ import annotations

from typing import List

from .schema import (
    EdgeDef,
    FlowFile,
    GuardDef,
    GuardType,
    NodeDef,
    NodeType,
)

# 必须存在输入/输出过滤守卫的节点类型（阶段 5.4 强制）
# 注：llm_call 为单步生成、无副作用写库，不强制挂锁（但 input/output_filter 仍全局强制）。
_SENSITIVE_NODE_TYPES: set[NodeType] = {
    "react", "direct_tool", "dispute", "human_approval",
    # 阶段 4.2：RAG 四步拆节点后，检索/生成涉及知识库与生成内容，同样需守卫包裹
    "rag_rewrite", "rag_review", "rag_retrieve", "rag_generate",
}
# 引用 Skill 的节点类型（阶段 1.2b）；llm_call 也可引用 Skill 注入 SOP 上下文（黑盒，不可改写）
_SKILL_NODE_TYPES: set[NodeType] = {"react", "direct_tool", "rag_pipeline", "llm_call"}


class FlowValidationError(ValueError):
    """流程校验失败（fail-closed 信号）。"""


def validate_flow(flow: FlowFile, skill_registry=None) -> List[str]:
    """校验 FlowFile，返回告警列表（非空不阻断）；非法则抛 FlowValidationError。

    返回值：非阻断级告警（如「节点无出边」提示），供开发者参考。

    ``skill_registry``：可选（阶段 5.2 / 5.3）。编译期传入时可做
    「敏感 Skill 安全边界」的语义强校验（操作指南必须存在、高后果字段必须强制绑定）；
    文件加载期（loader）可不传，仅做纯结构校验。
    """
    warnings: List[str] = []
    node_ids = set(flow.node_ids())

    # 1. entry 存在
    if flow.entry not in node_ids:
        raise FlowValidationError(f"entry 节点 '{flow.entry}' 未在 nodes 中声明")

    # 2/3. edge 引用合法性
    _validate_edges(flow.edges, node_ids)

    # 4/5. read_from 字段合法性
    _validate_read_from(flow)

    # 6. 敏感 Skill 黑盒引用约束（阶段 5.5 结构性检查）
    _validate_skill_references(flow)

    # 7. 守卫强制存在性（阶段 5.4）
    _validate_required_guards(flow, warnings)

    # 8. skill 节点类型约束
    _validate_skill_node_types(flow)

    # 9. 阶段 4.1 完整版：subgraph 引用合法性 + 递归校验
    _validate_subgraphs(flow, skill_registry)

    # 10. 阶段 5.2 / 5.3：敏感 Skill 安全边界语义强校验（需 registry）
    if skill_registry is not None:
        _validate_sensitive_skill_safety(flow, skill_registry)

    return warnings


def _validate_subgraphs(flow: FlowFile, skill_registry=None) -> None:
    """阶段 4.1 完整版：react 节点引用的 subgraph 必须存在，且子图本身合法。

    - react 节点声明 config.subgraph 时，对应的 key 须在 flow.subgraphs 中。
    - 子图须通过 validate_flow 递归校验（结构与守卫强制同主图）。
    - 仅 react 节点允许引用 subgraph（其他节点类型引用视为非法）。
    """
    declared = set(getattr(flow, "subgraphs", {}).keys())
    for node in flow.nodes:
        sub = node.config.subgraph
        if sub is None:
            continue
        if node.type != "react":
            raise FlowValidationError(
                f"节点 '{node.id}' type='{node.type}' 不应引用 subgraph"
                f"（仅 react 节点可引用 config.subgraph）"
            )
        if sub not in declared:
            raise FlowValidationError(
                f"react 节点 '{node.id}' 引用的 subgraph '{sub}' 未在 flow.subgraphs 中定义"
            )
        # 递归校验子图（fail-closed：子图非法则主图不可发布）
        validate_flow(flow.subgraphs[sub], skill_registry=skill_registry)


def _validate_edges(edges: List[EdgeDef], node_ids: set[str]) -> None:
    for e in edges:
        if e.from_ not in node_ids:
            raise FlowValidationError(f"edge.from '{e.from_}' 指向未声明节点")
        if e.to is not None and e.to not in node_ids:
            raise FlowValidationError(f"edge.to '{e.to}' 指向未声明节点")
        for cond in e.conditions:
            if cond.target not in node_ids:
                raise FlowValidationError(
                    f"condition target '{cond.target}'（op={cond.op.value}）指向未声明节点"
                )


def _validate_read_from(flow: FlowFile) -> None:
    """校验 read_from 引用的 state 字段存在、required 语义自洽（阶段 1.5c + 1.9）。"""
    declared_fields = {f.name: f for f in flow.state.fields}
    for node in flow.nodes:
        for rf in node.config.read_from:
            if rf.state not in declared_fields:
                raise FlowValidationError(
                    f"节点 '{node.id}' 的 read_from.state='{rf.state}' 未在 GraphState 中声明"
                )
            field_def = declared_fields[rf.state]
            if rf.required and not field_def.required:
                raise FlowValidationError(
                    f"节点 '{node.id}' 声明 read_from='{rf.state}' required=True，"
                    f"但 GraphState.fields 中该字段 required=False —— fail-closed 无意义"
                )


def _validate_skill_references(flow: FlowFile) -> None:
    """阶段 5.5：敏感 Skill 只能被引用，不允许内联改写其 SOP/校验。

    此处结构性检查：若节点声明了 skill，不得同时通过 config.extra 携带
    'sop_override' / 'skip_validation' 等会绕过安全边界的键。
    """
    BLOCKED_OVERRIDE_KEYS = {"sop_override", "skip_validation", "disable_hitl"}
    for node in flow.nodes:
        if node.config.skill and node.config.extra:
            hit = BLOCKED_OVERRIDE_KEYS & set(node.config.extra.keys())
            if hit:
                raise FlowValidationError(
                    f"节点 '{node.id}' 引用 Skill '{node.config.skill}' 的同时"
                    f"携带了禁止的安全覆盖键 {sorted(hit)}（阶段 5.5：敏感 Skill 只能引用不可改写）"
                )


def _validate_required_guards(flow: FlowFile, warnings: List[str]) -> None:
    """阶段 5.4：输入/输出过滤守卫必须存在，缺失则编译失败。

    允许两种形态：① guards 列表里有 input_filter / output_filter 守卫；
    ② 图中存在 type=input_filter / output_filter 的节点。两者任一即满足。
    """
    guard_types: set[GuardType] = {g.type for g in flow.guards}
    node_types: set[NodeType] = {n.type for n in flow.nodes}

    if "input_filter" not in guard_types and "input_filter" not in node_types:
        raise FlowValidationError("缺少输入安全过滤（guard input_filter 或 input_filter 节点），编译失败（阶段 5.4）")
    if "output_filter" not in guard_types and "output_filter" not in node_types:
        raise FlowValidationError("缺少输出安全过滤（guard output_filter 或 output_filter 节点），编译失败（阶段 5.4）")

    # 提示：敏感节点未挂对应守卫则不阻断，但给出告警
    _sensitive_present = _SKILL_NODE_TYPES & node_types
    if _sensitive_present and "lock" not in guard_types and "lock" not in node_types:
        warnings.append("存在敏感节点（react/direct_tool/rag_pipeline）但未声明 lock 守卫，建议加分布式锁")


def _validate_skill_node_types(flow: FlowFile) -> None:
    """阶段 1.2b：仅 react / direct_tool / rag_pipeline 类节点可声明 skill；引擎原语节点不可。"""
    for node in flow.nodes:
        if node.config.skill is None:
            continue
        if node.type not in _SKILL_NODE_TYPES:
            raise FlowValidationError(
                f"节点 '{node.id}' type='{node.type}' 不应引用 skill"
                f"（仅 {sorted(_SKILL_NODE_TYPES)} 可引用 Skill）"
            )


# 高后果/身份-资金类语义字段：敏感 Skill 必须强制绑定来源（read_from/hardcode），
# 业务不可在 YAML 中省略让模型自由填空（阶段 5.2「引擎自动强制，业务不可关闭」）。
_HIGH_CONSEQUENCE_SEMANTICS = {"order_id", "phone", "tracking_number"}


def _is_skill_sensitive(skill_def) -> bool:
    """Skill 是否为敏感（资金/隐私）：risk=high 或 hitl=true。"""
    if skill_def is None:
        return False
    risk = str(getattr(skill_def, "risk", "low") or "low").strip().lower()
    hitl = bool(getattr(skill_def, "hitl", False))
    return risk == "high" or hitl


def _validate_sensitive_skill_safety(flow: FlowFile, skill_registry) -> None:
    """阶段 5.2 / 5.3：敏感 Skill 安全边界语义强校验（需 skill_registry）。

    - 5.3：敏感 Skill（risk=high / hitl=true）被节点引用时，其 SOP（body）必须存在，
      否则 YAML 校验拒绝发布（fail-closed）——操作指南/校验规则缺失则不允许上线。
    - 5.2：敏感 Skill 的高后果字段（params 中 required=True，或 semantic 属于
      身份/资金类 {order_id, phone, tracking_number}）必须在节点中被强制绑定：
      要么 config.hardcode 声明定稿注入，要么 read_from 提供确定性来源。
      两者皆无 → 编译失败，防止高后果字段缺失让模型自由编造（引擎不可关闭的强制）。
    """
    # 兼容 skills 为 List[SkillDef]（真实）或 Dict（测试 fake）
    skills = getattr(skill_registry, "skills", None)
    if skills is None:
        return
    if isinstance(skills, dict):
        registry_map = skills
    else:
        registry_map = {getattr(s, "name", None): s for s in skills}

    for node in flow.nodes:
        skill_name = node.config.skill
        if not skill_name or node.type not in _SKILL_NODE_TYPES:
            continue
        skill_def = registry_map.get(skill_name)
        if skill_def is None:
            continue  # 结构上由 _validate_skill_node_types 不校验存在性，运行期再报
        if not _is_skill_sensitive(skill_def):
            continue

        # 5.3：敏感 Skill 必须有操作指南（SOP 正文）
        if not getattr(skill_def, "body", None):
            raise FlowValidationError(
                f"敏感 Skill '{skill_name}'（risk={getattr(skill_def,'risk','low')}, "
                f"hitl={getattr(skill_def,'hitl',False)}）被节点 '{node.id}' 引用，"
                f"但其缺少 SOP 操作指南（SKILL.md 正文为空）——校验拒绝发布（阶段 5.3）"
            )

        # 5.2：高后果字段必须被强制绑定（hardcode 或 read_from）
        params = getattr(skill_def, "params", {}) or {}
        required_fields = {
            name for name, meta in params.items()
            if isinstance(meta, dict) and meta.get("required")
        }
        high_consequence_fields = {
            name for name, meta in params.items()
            if isinstance(meta, dict) and meta.get("semantic") in _HIGH_CONSEQUENCE_SEMANTICS
        }
        must_bind = required_fields | high_consequence_fields
        if not must_bind:
            continue

        hardcode_fields = {hc.field for hc in node.config.hardcode if getattr(hc, "field", None)}
        # read_from 提供的 state 路径（保留全路径以便判定 params 子字段可达性）
        read_from_paths = {rf.state for rf in node.config.read_from}
        read_from_leaves = {p.split(".")[-1] for p in read_from_paths}

        def _is_bound(field: str) -> bool:
            # 命中 hardcode 定稿
            if field in hardcode_fields:
                return True
            # 直接 read_from 该字段（如 state.params.order_id）
            if field in read_from_leaves:
                return True
            # 读取整个 params（state.params）→ params 下所有字段可达
            if "params" in read_from_leaves:
                return True
            if f"params.{field}" in read_from_paths:
                return True
            return False

        unbound = [f for f in must_bind if not _is_bound(f)]
        if unbound:
            raise FlowValidationError(
                f"敏感 Skill '{skill_name}' 的高后果字段 {sorted(unbound)} 在节点 '{node.id}' 中"
                f"既未被 config.hardcode 定稿注入，也未通过 read_from 提供确定性来源——"
                f"YAML 校验拒绝发布（阶段 5.2：高后果字段强制注入，业务不可关闭）"
            )


__all__ = ["FlowValidationError", "validate_flow"]
