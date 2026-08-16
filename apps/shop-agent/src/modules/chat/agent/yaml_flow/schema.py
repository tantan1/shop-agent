"""YAML 编排协议 —— 数据模型（阶段 1：Schema 设计）。

对应 docs/yaml-orchestration-tasks.md 阶段 1.1-1.9。

设计原则（见 29-从ReAct黑盒到可视化编排）：
- 控制流外移：步骤顺序由图定义，模型只做节点内局部决策。
- 数据流外移：节点间参数走显式 GraphState 契约（read_from / write_to），
  不靠 messages / 闭包 / system prompt 隐式拼接（阶段 1.5）。
- 安全底座不下放：硬强制（hardcode）、守卫（guards）由引擎固化，业务不可关闭。

注意：本模块只定义「协议与校验」，不负责编译成可执行图（那是阶段 2 的事）。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

# ═══════════════════════════════════════════════════════════════════════
# 枚举
# ═══════════════════════════════════════════════════════════════════════

# 14 种节点类型（阶段 1.2）
NodeType = Literal[
    "normalize",       # 归一化
    "truncate",        # 截断
    "sentiment",       # 情绪/舆情
    "intent",          # 意图识别
    "rag_pipeline",    # RAG 固定四步
    "direct_tool",     # 直接 Tool（无 LLM 循环）
    "llm_call",        # 单步 LLM 调用（无工具循环，仅生成/校验文案，阶段 4.1 轻量版）
    "rag_rewrite",     # RAG 步骤1：问题理解/改写（阶段 4.2 拆节点）
    "rag_review",      # RAG 步骤2：安全审查（阶段 4.2 拆节点）
    "rag_retrieve",    # RAG 步骤3：知识检索（阶段 4.2 拆节点）
    "rag_generate",    # RAG 步骤4：回答生成+质量评估（阶段 4.2 拆节点）
    "react",           # ReAct 循环（v1 黑盒节点）
    "dispute",         # 纠纷协调
    "human_approval",  # 人在回路
    "input_filter",    # 输入安全过滤
    "output_filter",   # 输出安全过滤
    "lock",            # 分布式锁
    "persist",         # 历史持久化
    "observe",         # 可观测埋点
]

GuardType = Literal[
    "input_filter",
    "output_filter",
    "lock",
    "persist",
    "observe",
]


class ConditionOp(str, Enum):
    """条件边算子（阶段 1.3）。"""

    ON_INTENT_EQ = "on_intent=="
    ON_COMPLEXITY_EQ = "on_complexity=="
    ON_EMERGENCY = "on_emergency"
    ALWAYS = "always"
    EXPR = "expr"  # 自定义表达式（由解析器评估 state）
    # 阶段 4.2：RAG 安全审查结果路由（state['rag']['can_proceed'] / ['safety']）
    ON_RAG_SAFE = "on_rag_safe"        # rag_review 通过（can_proceed=True）走该边
    ON_RAG_UNSAFE = "on_rag_unsafe"    # rag_review 不通过（can_proceed=False）走该边


# ═══════════════════════════════════════════════════════════════════════
# 显式 State 契约（阶段 1.5）
# ═══════════════════════════════════════════════════════════════════════

class StateField(BaseModel):
    """GraphState 中的一个共享字段（阶段 1.5a）。

    节点不感知上下游名字，只通过 read_from / write_to 引用这些字段路径。
    """

    name: str = Field(..., description="字段路径，如 'params.order_id' / 'tool_result' / 'messages'")
    type: str = Field(default="any", description="JSON 类型提示：any/string/int/dict/list/bool")
    required: bool = Field(default=False, description="为 True 时，声明 read_from 该字段的节点若拿不到则 fail-closed")


class GraphState(BaseModel):
    """流程共享状态契约（阶段 1.5a）。

    至少含 messages / intent / params / tool_result / hitl_pending / thread_id，
    替代现状隐式三通道（messages 列表 + ReActRunContext 闭包 + system prompt 拼接）。
    """

    fields: List[StateField] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# 声明式参数传递（阶段 1.5b/c/d/e）
# ═══════════════════════════════════════════════════════════════════════

class ReadFrom(BaseModel):
    """节点从 GraphState 读取的字段（阶段 1.5b）。

    state 指向 GraphState.fields[].name，如 'params' / 'intent.action'。
    required=True 且上游未写该字段 → 解析/运行期 fail-closed（阶段 1.5c）。
    """

    state: str = Field(..., description="GraphState 字段路径")
    required: bool = Field(default=False, description="缺失是否 fail-closed")
    # 节点函数内部使用的本地参数名（缺省等于 state 末段）
    as_: Optional[str] = Field(default=None, alias="as", description="节点内别名")


class WriteTo(BaseModel):
    """节点写回 GraphState 的字段（阶段 1.5b）。

    节点产出（如工具结果）经此落回 state，供下游 read_from 消费。
    """

    state: str = Field(..., description="GraphState 字段路径，如 'tool_result' / 'params'")
    # 节点函数返回的哪个 key 映射到该 state 字段
    from_: Optional[str] = Field(default=None, alias="from", description="节点返回值中的 key")


# ═══════════════════════════════════════════════════════════════════════
# 硬强制声明（阶段 1.4 / 5.1）
# ═══════════════════════════════════════════════════════════════════════

class HardcodeDecl(BaseModel):
    """节点 config 中的硬强制字段声明（阶段 1.4）。

    对应 27-参数硬强制注入 的「闭包覆盖 + schema 字段切除」。
    - field：要强制注入的参数名（须是 Skill 的 params 之一）
    - value：确定性来源，引擎在节点执行时定稿注入（如从已认证身份查询补全）
    - strip_from_schema：是否从工具 schema 切除该字段（剥夺模型写入权限），默认 True
    """

    field: str = Field(..., description="参数名，须属于所引用 Skill 的 params")
    value: Any = Field(..., description="确定性注入值（来自上游抽取/查询，非模型生成）")
    strip_from_schema: bool = Field(default=True, description="是否从工具 schema 切除该字段")


# ═══════════════════════════════════════════════════════════════════════
# 节点配置（阶段 1.2 / 1.2b / 1.5 / 1.7）
# ═══════════════════════════════════════════════════════════════════════

class NodeConfig(BaseModel):
    """节点 config（按节点类型可选字段）。

    设计说明：
    - skill / hardcode / read_from / write_to / human_approval / subgraph
      为可选声明；校验器按节点 type 检查「该填的是否填了」（见 validator.py）。
    - 引擎原语节点（normalize / input_filter / lock 等）不指向任何 Skill，
      是内置原子能力，故 skill 字段仅 react / direct_tool / rag_pipeline 类节点使用。
    """

    # —— 1.2b：Skill 引用（仅 react / direct_tool / rag_pipeline）——
    skill: Optional[str] = Field(default=None, description="引用的 Skill ID（从 SkillRegistry 取 SOP 与工具）")

    # —— 1.4：硬强制声明 ——
    hardcode: List[HardcodeDecl] = Field(default_factory=list)

    # —— 1.5b/c：声明式参数传递 ——
    read_from: List[ReadFrom] = Field(default_factory=list)
    write_to: List[WriteTo] = Field(default_factory=list)

    # —— 1.7：react 子图钩子（为阶段 4 预留）——
    subgraph: Optional[str] = Field(default=None, description="react 节点的子图引用（v1 留空）")

    # —— 1.6：human_approval 元信息 ——
    approval_type: Optional[str] = Field(default=None, description="审批类型，如 'refund' / 'cancel'")
    thread_id_ref: Optional[str] = Field(default=None, description="跨请求恢复用的 thread_id 来源字段（默认 'thread_id'）")

    # —— 阶段 4.1 轻量版：llm_call 节点提示模板 ——
    # 支持 {state.xxx} / {params.xxx} 占位，由引擎在调用前用 GraphState 渲染。
    # 不填则回落到「读取 messages[-1] 直接生成」。
    prompt_template: Optional[str] = Field(default=None, description="llm_call 节点的提示模板（阶段 4.1）")

    # —— 阶段 4.2：RAG 四步拆节点专用字段 ——
    # 四步节点（rag_rewrite/rag_review/rag_retrieve/rag_generate）复用现有 steps 实现，
    # 通过以下字段做轻量配置；依赖（embedding/milvus/llm）由 compiler 统一注入。
    rag_top_k: Optional[int] = Field(default=None, description="检索 top_k（覆盖 config.top_k）")
    rag_domain: Optional[str] = Field(default=None, description="领域覆盖（默认取 intent.domain）")
    # rag_generate 是否启用输出过滤 + 质量评估（默认 true，对应步骤4 的 filter_and_evaluate）
    rag_output_filter: bool = Field(default=True, description="rag_generate 是否做输出安全过滤")
    # rag_review 不通过（can_proceed=False）时，是否短路到 fallback 节点（默认 true）
    rag_fallback_on_review_fail: bool = Field(default=True, description="安全审查不通过时短路到 fallback 节点")

    # —— 通用透传：节点专属的可选键值（如 react 的 max_iterations）——
    extra: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


# ═══════════════════════════════════════════════════════════════════════
# 条件边（阶段 1.3）
# ═══════════════════════════════════════════════════════════════════════

class Condition(BaseModel):
    """条件边路由条件（阶段 1.3）。

    op 决定语义：
    - on_intent==：value 为意图名，state.intent == value 时走该边
    - on_complexity==：value 为复杂度档位
    - on_emergency：舆情升级短路
    - always：兜底边（每个条件出口至多一条）
    - expr：自定义表达式（由解析器在 state 上评估）
    """

    op: ConditionOp
    value: Optional[str] = None
    target: str = Field(..., description="命中条件时跳转的节点 id")


# ═══════════════════════════════════════════════════════════════════════
# 守卫（阶段 1.6 / 2.6）
# ═══════════════════════════════════════════════════════════════════════

class GuardDef(BaseModel):
    """横切守卫（输入过滤/输出过滤/锁/持久化/埋点）。

    可声明为图级 guards（对所有节点生效）或挂在某节点前后置钩子。
    """

    type: GuardType
    # 作用目标：空表示全图；否则为节点 id 列表（针对特定节点）
    target_nodes: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
# 顶层结构（阶段 1.1）
# ═══════════════════════════════════════════════════════════════════════

class NodeDef(BaseModel):
    id: str = Field(..., description="节点唯一 id（edges 引用）")
    type: NodeType
    config: NodeConfig = Field(default_factory=NodeConfig)


class EdgeDef(BaseModel):
    from_: str = Field(..., alias="from", description="源节点 id")
    to: Optional[str] = Field(default=None, description="固定目标节点 id（无条件边用）")
    conditions: List[Condition] = Field(default_factory=list, description="条件边（from 到多个 target）")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _check_target_or_conditions(self) -> "EdgeDef":
        if self.to is None and not self.conditions:
            raise ValueError("edge 必须提供 to（固定边）或 conditions（条件边）之一")
        if self.to is not None and self.conditions:
            raise ValueError("edge 不能同时提供 to 与 conditions")
        return self


class SubgraphFlow(BaseModel):
    """子流程定义（阶段 4.1 完整版）：被 react 节点的 config.subgraph 引用。

    结构与 FlowFile 一致（entry/nodes/edges/state/guards），但无嵌套 subgraph，
    且子图内节点类型应当收敛为单步原子节点（direct_tool / llm_call / human_approval 等），
    不鼓励在子图内再放 react 黑盒（否则退化为原样）。
    """

    name: str = Field(..., description="子图名称（flow.subgraphs 的 key）")
    entry: str = Field(..., description="子图入口节点 id")
    state: GraphState = Field(default_factory=GraphState, description="子图 State 契约（与父图对齐或子集）")
    nodes: List[NodeDef] = Field(..., min_length=1)
    edges: List[EdgeDef] = Field(default_factory=list)
    guards: List[GuardDef] = Field(default_factory=list)

    # —— 便捷访问（与 FlowFile 对齐，供 validator 复用）——
    def node_ids(self) -> List[str]:
        return [n.id for n in self.nodes]

    def get_node(self, node_id: str) -> Optional[NodeDef]:
        return next((n for n in self.nodes if n.id == node_id), None)


class FlowFile(BaseModel):
    """YAML 编排文件顶层模型（阶段 1.1）。"""

    version: str = Field(default="1.0", description="协议版本")
    entry: str = Field(..., description="入口节点 id（必须存在于 nodes）")
    state: GraphState = Field(default_factory=GraphState, description="显式 State 契约（阶段 1.5）")
    nodes: List[NodeDef] = Field(..., min_length=1)
    edges: List[EdgeDef] = Field(default_factory=list)
    guards: List[GuardDef] = Field(default_factory=list)

    # —— 便捷访问（与 FlowFile 对齐，供 validator 复用）——
    def node_ids(self) -> List[str]:
        return [n.id for n in self.nodes]

    def get_node(self, node_id: str) -> Optional[NodeDef]:
        return next((n for n in self.nodes if n.id == node_id), None)
    # —— 阶段 4.1 完整版：可被 react 节点 config.subgraph 引用的子流程集合 ——
    subgraphs: Dict[str, SubgraphFlow] = Field(default_factory=dict)

    # —— 便捷访问 ——
    def node_ids(self) -> List[str]:
        return [n.id for n in self.nodes]

    def get_node(self, node_id: str) -> Optional[NodeDef]:
        return next((n for n in self.nodes if n.id == node_id), None)


__all__ = [
    "NodeType",
    "GuardType",
    "ConditionOp",
    "StateField",
    "GraphState",
    "ReadFrom",
    "WriteTo",
    "HardcodeDecl",
    "NodeConfig",
    "Condition",
    "GuardDef",
    "NodeDef",
    "EdgeDef",
    "FlowFile",
    "SubgraphFlow",
]
