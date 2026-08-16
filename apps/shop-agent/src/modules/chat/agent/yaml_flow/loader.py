"""YAML 流程加载器（阶段 2.1 前置，最小可用封装）。

职责：读本地 YAML → 解析为 FlowFile → 跑校验器 → 返回 (flow, warnings)。
编译成可执行 LangGraph（阶段 2.2-2.9）后续落地。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import yaml

from .schema import FlowFile
from .validator import FlowValidationError, validate_flow


def load_flow_file(path: str | Path) -> Tuple[FlowFile, List[str]]:
    """加载并校验一个 YAML 编排文件。

    返回值：(FlowFile, 非阻断告警列表)。
    校验失败抛 FlowValidationError（fail-closed）。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"流程文件不存在: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise FlowValidationError(f"YAML 解析失败: {p}") from e

    flow = FlowFile.model_validate(raw)
    warnings = validate_flow(flow)
    return flow, warnings


__all__ = ["load_flow_file", "FlowValidationError"]
