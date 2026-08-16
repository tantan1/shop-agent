"""yaml_flow —— YAML 驱动的多步 Agent 流程编排。

阶段 1（Schema 设计）已落地：协议模型 + 校验器 + 加载器 + 示例。
阶段 2（解析器）起将 FlowFile 编译成 LangGraph 可执行图。
"""

from .compiler import CompiledFlow, FlowCompiler, _compile_router
from .loader import FlowValidationError, load_flow_file
from .runtime import GraphState, new_graph_state
from .schema import (
    Condition,
    ConditionOp,
    EdgeDef,
    FlowFile,
    GraphState,
    GuardDef,
    GuardType,
    HardcodeDecl,
    NodeConfig,
    NodeDef,
    NodeType,
    ReadFrom,
    StateField,
    WriteTo,
)
from .validator import validate_flow

__all__ = [
    "FlowFile",
    "NodeDef",
    "EdgeDef",
    "NodeType",
    "NodeConfig",
    "GraphState",
    "StateField",
    "ReadFrom",
    "WriteTo",
    "HardcodeDecl",
    "Condition",
    "ConditionOp",
    "GuardDef",
    "GuardType",
    "validate_flow",
    "FlowValidationError",
    "load_flow_file",
    "FlowCompiler",
    "CompiledFlow",
    "new_graph_state",
    "_compile_router",
]
