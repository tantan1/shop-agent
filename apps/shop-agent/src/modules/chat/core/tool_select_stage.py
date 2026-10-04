"""工具选择统一 Stage 接口与数据结构。

见 docs/architecture/tool-select-pipeline-design.md §3。
每层只回答"本层对候选的相关度"，不做最终决策；Pipeline 统一做阈值判断、
候选传递与降级。

设计要点：
- `name` 必须在实例化时由配置注入（= 配置中 stage 的 name，如 "p0_rule"），
  Pipeline 用 `thresholds[stage.name]` / `timeouts[stage.name]` 查找本层阈值与超时。
- 各层从 `run()` 的 `context` 参数读取自身依赖（embedding_model / llm_service 等），
  不通过构造参数注入。
- `STAGE_REGISTRY` 为显式注册表（name→类），不使用 importlib 动态加载；
  具体 Stage 在其模块导入时通过 `register_stage` 登记（见 Phase 4 迁移）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CandidateScore:
    """单层对某工具的打分证据，供观测与下一层 scope 缩小使用。

    ``score`` 为真实打分（P1 余弦相似度 / P2 softmax 概率 / P0 规则强弱分）；
    选择器层（P3）与"未实际产分的透传路径"置 ``None``，表示本层不产出分数。
    Pipeline 组装 score_map 时跳过 None，避免透传占位覆盖上层真实分数（设计 1）。
    """

    tool: str
    score: Optional[float] = None
    source: str = ""


@dataclass
class StageResult:
    """单层的输出。

    - tool / confidence：单层最优（供 Pipeline 早停判断）
    - scored_candidates：本层对 scope 内候选的打分（供观测与缩小下一层 scope），
      而非仅 top-1（P1 为 Top-K 子集，P2 为全量 softmax 分布）
    - error：非空表示该层执行失败，交由 Pipeline 按 on_stage_failure 处理
    """

    tool: Optional[str] = None
    confidence: float = 0.0
    source: str = ""
    scored_candidates: List[CandidateScore] = field(default_factory=list)
    error: Optional[str] = None


class ToolSelectStage(ABC):
    """每层的统一接口。"""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    async def run(self, query: str, scope: List[str], context: Dict) -> StageResult:
        """返回该层的选择结果与置信度；scope 为本层待评候选（上一层软过滤后的工具名）。

        context 携带各层依赖（thresholds / tool_registry / embedding_model / llm_service 等），
        Stage 应从中读取自身依赖，不通过构造参数注入。
        """
        raise NotImplementedError


# 显式注册表：name -> Stage 类。新增 Stage 在其模块导入时调用 register_stage 登记，
# 配置加载即按 name 解析，无需改 ToolSelectPipeline（见 §6）。
STAGE_REGISTRY: Dict[str, type[ToolSelectStage]] = {}


def register_stage(name: str, stage_cls: type[ToolSelectStage]) -> None:
    """登记一个 Stage 类到注册表（name 必须与配置中的 stage.name 一致）。"""
    STAGE_REGISTRY[name] = stage_cls
