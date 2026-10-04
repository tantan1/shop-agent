"""意图识别分层模块（关注点分离重构）。

分层职责：
- ``candidate``    : 纯数据结构 —— ``IntentCandidate``（分类结果）/
                     ``ExecutionPlan``（路由结果），两者互不感知。
- ``index_builder``: 基础设施 —— FAISS 索引构建与全局缓存。
                     唯一数据源 = ``SkillRegistry``（SKILL.md frontmatter 的 ``examples``）。
- ``classifiers``  : 分类层 —— 只回答「是什么意图 + 多自信」，不含执行决策。
- ``router``       : 路由层 —— 只回答「怎么执行」，产出 ``ExecutionPlan``，
                     ``mode`` 对齐 yaml_flow 的 ``node.type``。
"""

from .candidate import (
    EXECUTION_MODES,
    ClassifierSource,
    ExecutionMode,
    ExecutionPlan,
    IntentCandidate,
)
from .classifiers import (
    FaissIntentClassifier,
    IntentClassifier,
    NegationRuleClassifier,
)
from .index_builder import (
    IntentIndex,
    IntentMatch,
    ensure_intent_index_async,
    ensure_intent_index_sync,
    examples_from_registry,
    flatten_examples,
    get_cached_index,
    reset_intent_index_cache,
)
from .router import ExecutionRouter, RoutingPolicy, is_sensitive_skill

__all__ = [
    "EXECUTION_MODES",
    "ClassifierSource",
    "ExecutionMode",
    "ExecutionPlan",
    "IntentCandidate",
    "IntentClassifier",
    "NegationRuleClassifier",
    "FaissIntentClassifier",
    "IntentIndex",
    "IntentMatch",
    "ensure_intent_index_async",
    "ensure_intent_index_sync",
    "examples_from_registry",
    "flatten_examples",
    "get_cached_index",
    "reset_intent_index_cache",
    "ExecutionRouter",
    "RoutingPolicy",
    "is_sensitive_skill",
]
