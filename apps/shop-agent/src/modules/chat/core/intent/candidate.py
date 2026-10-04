"""意图识别的纯数据结构 —— 分类与路由解耦的契约。

设计要点：
- ``IntentClassifier`` 只产出 ``IntentCandidate``（是什么意图 + 多自信）；
- ``ExecutionRouter`` 只产出 ``ExecutionPlan``（怎么执行）；
- 两者互不感知，由编排层组合，避免「分类里塞路由、路由里读阈值」的耦合。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

# ── 执行模式 ──
# 命名直接对齐 yaml_flow 的 NodeType（见 yaml_flow/schema.py:26-46），
# 便于后续把路由职责整体移交 yaml_flow 时零改名。
# 当前意图识别只产出这三种：
#   rag_pipeline : 走 RAG 知识库回答（原 intent="rag_answer"）
#   direct_tool  : 单次工具调用即可（原 intent="call_remote_api" + complexity="simple"）
#   react        : 需多步/组合能力，交 Agent（原 complexity="multi_step"）
# llm_call（单步生成、无工具）为预留值，编排层全量接管后启用。
ExecutionMode = Literal["rag_pipeline", "direct_tool", "react"]

EXECUTION_MODES: List[str] = ["rag_pipeline", "direct_tool", "react"]

# ── 分类来源 ──
ClassifierSource = Literal["negation", "faiss", "llm", "fallback"]


@dataclass(frozen=True)
class IntentCandidate:
    """分类结果：只回答「是什么意图 + 多自信」。

    Attributes:
        action: 命中的 skill/工具名；``None`` 表示无工具意图（如政策咨询，应走 RAG）。
        score: 置信度 0~1。
        source: 由哪一层判出来的（negation/faiss/llm/fallback）。
        matched: 命中的示例或规则原文，便于排查与可观测。
    """

    action: Optional[str]
    score: float
    source: ClassifierSource
    matched: Optional[str] = None


@dataclass(frozen=True)
class ExecutionPlan:
    """路由结果：只回答「怎么执行」。

    ``mode`` 取值对齐 yaml_flow 的 ``node.type``，见 ``ExecutionMode``。
    """

    mode: ExecutionMode
    skill: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""
