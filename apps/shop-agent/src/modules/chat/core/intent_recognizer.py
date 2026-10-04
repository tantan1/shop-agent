"""
意图识别器 —— 编排层（关注点分离重构 · 第 1 步）

职责划分：
- 分类：委托 ``core/intent/classifiers.py``（NegationRuleClassifier / FaissIntentClassifier）
- 索引：委托 ``core/intent/index_builder.py``，数据源 = SkillRegistry（SKILL.md 的 examples）
- 路由：委托 ``core/intent/router.py`` 的 ExecutionRouter，产出 ExecutionPlan

注意：意图示例的唯一数据源是各 SKILL.md frontmatter 的 ``examples`` 字段，
本文件不再硬编码 INTENT_EXAMPLES（消灭双份事实源）。
"""

import time as _perf_time
from typing import List, Optional, Tuple

from src.core.config import config
from src.modules.chat.agent.skill_loader import get_skill_registry
from src.modules.chat.core.intent.candidate import ExecutionPlan, IntentCandidate
from src.modules.chat.core.intent.classifiers import (
    FaissIntentClassifier,
    NegationRuleClassifier,
)
from src.modules.chat.core.intent.index_builder import (
    ExamplesProvider,
    ensure_intent_index_async,
    ensure_intent_index_sync,
    examples_from_registry,
)
from src.modules.chat.core.intent.router import ExecutionRouter, RoutingPolicy
from src.modules.chat.schemas import IntentResult
from src.shared.logger import APILogger

logger = APILogger("intent_recognizer")

# ═══════════════════════════════════════════════════════════════════════════════
# 常量 & 模式定义
# ═══════════════════════════════════════════════════════════════════════════════

# 否定词：问的是"政策/流程/规则"，走 RAG 而非远程 API
NEGATION_PATTERNS: List[str] = [
    "退货政策",
    "退款流程",
    "退货流程",
    "怎么退",
    "退货条件",
    "退款规则",
    "退换政策",
    "什么是",
    "请说明政策",
    "服务说明",
]

# 信号1：问题里包含推理/多步关键词 → 不是一次 tool 调用能搞定的
REACT_TRIGGER_PATTERNS: List[str] = [
    "为什么",
    "怎么办",
    "怎么处理",
    "怎么操作",
    "怎么解决",
    "帮我处理",
    "帮我操作",
    "帮我解决",
    "同时",
    "并且",
    "还要",
    "另外",
    "然后再",
    "能不能",
    "可以吗",
    "行不行",  # 需要判断 → 可能需要 RAG 查政策
    "哪里有问题",
    "什么原因",
    "怎么回事",  # 需要诊断
]

# 多意图连接词：用于把一条消息拆分为多个意图子句，分别分类后取并集。
# 例：「我上周的商品破损想退货退款，另外有没有优惠可以领？」→ 退货 + 领券。
MULTI_INTENT_CONNECTORS: List[str] = [
    "另外",
    "还有",
    "顺便",
    "以及",
    "同时",
    "也",
    "和",
    "与",
    "再",
    "一起",
    "此外",
    "并且",
    "然后再",
    "还要",
    "还想",
]

# 信号2（多步骤意图）与「写类意图」判定已移入路由层：由 SKILL.md 的
# risk / hitl 推导（见 core/intent/router.py::is_sensitive_skill）。
# 本模块不再维护 ALWAYS_AGENT_ACTIONS / WRITE_ACTIONS 这类硬编码集合。

# 信号3：FAISS 相似分在歧义带（疑似歧义查询），交给 Agent 处理更稳妥。
# 仅作为 RoutingPolicy.ambiguity_zone 的兜底默认值，实际取值来自 config。
AMBIGUITY_SIMILARITY_THRESHOLD: float = 0.72

# 意图示例的唯一数据源：各 SKILL.md frontmatter 的 examples 字段，
# 经 SkillRegistry 抽取（见 core/intent/index_builder.py::examples_from_registry）。
# 本模块不再维护硬编码示例，避免「改了 SKILL.md 但意图识别不跟随」。


class IntentRecognizer:
    """意图识别器 —— 编排「分类 → 路由」，产出 IntentResult。

    分类委托 ``core/intent/classifiers``，索引委托 ``core/intent/index_builder``，
    路由委托 ``core/intent/router`` 的 ``ExecutionRouter``；
    本类只负责编排，以及把 ExecutionPlan 映射为兼容期 IntentResult。
    """

    def __init__(self, embedding_service, llm_service=None, skill_registry=None):
        """
        Args:
            embedding_service: EmbeddingService 实例（必需，用于向量化）
            llm_service: LLMService 实例（可选，用于 LLM 模式意图识别/参数抽取）
            skill_registry: SkillRegistry 实例（可选，意图示例数据源；
                缺省使用全局单例 get_skill_registry()）
        """
        self._embedding_service = embedding_service
        self._llm_service = llm_service
        self._skill_registry = skill_registry

        def _examples_provider():
            return examples_from_registry(
                self._skill_registry or get_skill_registry()
            )

        self._examples_provider: ExamplesProvider = _examples_provider
        self._negation: NegationRuleClassifier = NegationRuleClassifier(
            NEGATION_PATTERNS
        )
        self._faiss: FaissIntentClassifier = FaissIntentClassifier(
            embedding_service, _examples_provider
        )

        self._policy: RoutingPolicy = RoutingPolicy(
            min_confidence=float(
                getattr(config, "INTENT_VECTOR_SIMILARITY_THRESHOLD", 0.65)
            ),
            write_min_confidence=float(
                getattr(config, "INTENT_WRITE_THRESHOLD", 0.78)
            ),
            ambiguity_zone=float(
                getattr(
                    config,
                    "AMBIGUITY_SIMILARITY_THRESHOLD",
                    AMBIGUITY_SIMILARITY_THRESHOLD,
                )
            ),
            react_triggers=list(REACT_TRIGGER_PATTERNS),
        )
        self._router: ExecutionRouter = ExecutionRouter(
            self._policy, skill_lookup=self._skill_of
        )

    def _skill_of(self, action: str):
        """按意图名取 SkillDef，供路由层判定 risk / hitl。"""
        registry = self._skill_registry or get_skill_registry()
        for s in getattr(registry, "skills", None) or []:
            if getattr(s, "name", None) == action:
                return s
        return None

    # ════════════════════════════════════════════════════════════════════════
    # 多意图识别（治本：让意图识别产出意图列表，下游 P0 取并集 → 多工具）
    # ════════════════════════════════════════════════════════════════════════

    def _split_multi_intent(self, message: str) -> List[str]:
        """按多意图连接词把一条消息切成多个子句。

        逐连接词切分（连接词本身丢弃），返回去空白后的非空子句列表。
        无连接词时返回 [整句]（单意图退化）。
        """
        parts: List[str] = [message]
        for conn in MULTI_INTENT_CONNECTORS:
            new_parts: List[str] = []
            for p in parts:
                if conn in p:
                    new_parts.extend(
                        seg.strip() for seg in p.split(conn) if seg.strip()
                    )
                else:
                    new_parts.append(p)
            parts = new_parts
        return [p for p in parts if p]

    async def _classify_action(self, text: str) -> Optional[Tuple[str, float]]:
        """对单条文本用 FAISS 取最优意图动作（action, score）。失败时返回 None。

        注意 ``FaissIntentClassifier.classify`` 是 async 方法（见 classifiers.py），
        ``recognize`` 本身是 async 上下文，故此处用 await 取真实 Candidate，避免拿到协程。
        """
        try:
            cand = await self._faiss.classify(text)
        except Exception:
            return None
        if cand and cand.action and cand.score is not None:
            return (cand.action, cand.score)
        return None

    async def _extract_intent_actions(
        self, message: str, primary: Optional[IntentCandidate] = None
    ) -> List[str]:
        """从一条消息中识别多个意图动作，返回有序、去重的动作列表。

        做法：
        1. 整句动作优先 —— 复用 ``recognize`` 已算出的 ``primary``（避免重复 FAISS 调用）；
           否则对整句做一次 FAISS 分类。
        2. 子句补充 —— 按连接词切分后，对每条子句分类，把额外意图并入。
        3. 过滤低于最小置信阈值的弱信号、去重（保留首次出现顺序）。

        单意图消息（无连接词）退化为 [整句动作]。
        """
        if not message or not message.strip():
            return []

        collected: List[Tuple[str, float]] = []
        # 整句动作：优先复用 primary，避免重复 classify
        if primary is not None and primary.action and primary.score is not None:
            collected.append((primary.action, primary.score))
        else:
            whole = await self._classify_action(message)
            if whole:
                collected.append(whole)

        # 子句动作（复用整句结果已计入，这里跳过等于整句的子句）
        for seg in self._split_multi_intent(message):
            if seg == message.strip():
                continue
            sub = await self._classify_action(seg)
            if sub:
                collected.append(sub)

        seen: set = set()
        actions: List[str] = []
        for action, score in collected:
            if action in seen:
                continue
            if score < self._policy.min_confidence:
                continue
            seen.add(action)
            actions.append(action)
        return actions

    # ════════════════════════════════════════════════════════════════════════
    # FAISS 意图索引
    # ════════════════════════════════════════════════════════════════════════

    async def warmup(self):
        """预热 FAISS 意图索引（在启动时调用，避免首次请求等待）"""
        _ = await ensure_intent_index_async(
            self._embedding_service, self._examples_provider
        )

    @classmethod
    def warmup_sync(cls, embedding_service):
        """同步预热 FAISS 意图索引（用于 lifespan 中，不依赖事件循环）"""
        _ = ensure_intent_index_sync(
            embedding_service,
            lambda: examples_from_registry(get_skill_registry()),
        )

    # 注：复杂性检测已移入路由层 ``ExecutionRouter._assess``。
    # 其「多步骤意图」信号来自 SKILL.md 的 risk / hitl，
    # 关键词与阈值来自 ``RoutingPolicy``，不再硬编码在本模块。

    async def _llm_fallback_classify(
        self, message: str, fallback_action: str, fallback_score: float
    ) -> Optional[Tuple[str, str, str]]:
        """LLM 二次分类：在歧义带内用 LLM 精判意图和复杂性。

        Args:
            message: 用户消息
            fallback_action: FAISS 返回的候选意图
            fallback_score: FAISS 相似分

        Returns:
            (action, complexity, reason) 或 None（LLM 调用失败时）
        """
        if not self._llm_service:
            return None

        prompt = (
            "你是一个意图识别助手。请判断用户消息的意图和复杂性。\n\n"
            f"用户消息: {message}\n\n"
            f"候选意图: {fallback_action} (相似度: {fallback_score:.3f})\n\n"
            "可选意图: query-order, check-shipping, request-return, check-balance, coupon-inquiry, rag_answer\n\n"
            "请返回 JSON 格式:\n"
            '{"action": "意图名称", "complexity": "simple 或 multi_step", "reason": "判断理由"}'
        )

        try:
            response = await self._llm_service.chat(prompt)
            import json
            data = json.loads(response.strip())
            action = data.get("action", fallback_action)
            complexity = data.get("complexity", "simple")
            reason = data.get("reason", "LLM 二次分类")
            return action, complexity, reason
        except Exception as e:
            logger.warning(f"LLM 二次分类解析失败: {e}")
            return None

    # ════════════════════════════════════════════════════════════════════════
    # 统一入口
    # ════════════════════════════════════════════════════════════════════════
    # 统一入口
    # ════════════════════════════════════════════════════════════════════════

    async def recognize(self, message: str, langfuse_handler=None) -> IntentResult:
        """本地意图识别（否定过滤 + FAISS 向量匹配）。无 LLM 调用，延迟 < 5ms"""
        t_total_start = _perf_time.perf_counter()

        # ---- 第一层：否定模式过滤（咨询类问题走 RAG，委托 NegationRuleClassifier）----
        neg = await self._negation.classify(message)

        # ---- 第二层：FAISS 向量语义匹配（委托 FaissIntentClassifier）----
        # 否定词命中时直接采用其候选（action=None 表示无工具意图），不再提前 return，
        # 使「分类 → 路由」成为单一路径，路由结果统一由 ExecutionRouter 产出。
        candidate = neg if neg is not None else await self._faiss.classify(message)

        # ---- 第三层（可选）：歧义带内用 LLM 精分类 ----
        complexity_override: Optional[str] = None
        reason_override: Optional[str] = None
        if (
            neg is None
            and candidate is not None
            and candidate.action
            and self._llm_service
            and getattr(config, "INTENT_RECOGNITION_MODE", "local") != "local"
            and self._policy.min_confidence
            < candidate.score
            < self._policy.ambiguity_zone
        ):
            try:
                llm = await self._llm_fallback_classify(
                    message, candidate.action, candidate.score
                )
                if llm:
                    llm_action, complexity_override, reason_override = llm
                    # 修复：原实现接住了 LLM 返回的 action 却未使用，导致其无法覆盖意图
                    if llm_action:
                        candidate = IntentCandidate(
                            action=llm_action,
                            score=candidate.score,
                            source="llm",
                            matched=candidate.matched,
                        )
            except Exception as e:
                logger.warning(f"LLM 二次分类失败: {e}")

        # ---- 路由：由 ExecutionRouter 决定「怎么执行」 ----
        plan = self._router.route(
            candidate, message, complexity_override, reason_override
        )

        # ── 多意图识别（治本）── 让意图识别产出意图列表 ──
        # 对整句及各连接词子句分别分类后取并集，得到 actions；单意图退化为 [整句动作]。
        # 多意图（>1 条意图动作）→ 强制走 react 多工具路径，由 _select_tools_for_intent
        # 对每条意图各取工具集、取并集，保证「多意图 → 多工具」。
        actions = await self._extract_intent_actions(message, primary=candidate)
        if len(actions) > 1:
            plan = ExecutionPlan(
                mode="react",
                skill=actions[0],
                confidence=plan.confidence,
                reason=f"multi_intent({'+'.join(actions)})",
            )
            logger.info("多意图命中，转 react 多工具路径", actions=actions)
        elif not actions and candidate and candidate.action:
            actions = [candidate.action]
        elif not actions:
            actions = [plan.skill] if plan.skill else []

        t_total = (_perf_time.perf_counter() - t_total_start) * 1000
        logger.info(
            "意图识别完成",
            mode=plan.mode,
            action=plan.skill,
            confidence=round(plan.confidence, 3),
            total_ms=round(t_total, 1),
        )
        if langfuse_handler:
            try:
                src = neg or candidate
                matched = src.matched if src is not None else None
                langfuse_handler.trace(
                    name="intent_recognizer",
                    metadata={
                        "path": "negation" if neg is not None else plan.mode,
                        "mode": plan.mode,
                        "action": plan.skill,
                        "actions": actions,
                        "score": round(candidate.score, 3) if candidate else None,
                        "matched": matched,
                        "reason": plan.reason,
                        "message": message[:100],
                    },
                )
            except Exception:
                pass

        return self._to_result(plan, candidate.score if candidate else None, actions=actions)

    @staticmethod
    def _to_result(
        plan: ExecutionPlan,
        score: Optional[float],
        actions: Optional[List[str]] = None,
    ) -> IntentResult:
        """``ExecutionPlan`` → ``IntentResult``。

        唯一事实源是 ``plan``；``mode`` / ``skill`` / ``reason`` 直接来自它，
        不再维护 deprecated 的 ``intent`` / ``complexity`` 双写字段。
        ``actions`` 承载多意图列表（治本：多意图→多工具），单意图退化为 [action]。
        """
        if not actions:
            actions = [plan.skill] if plan.skill else []
        return IntentResult(
            plan=plan,
            action=plan.skill,
            actions=actions,
            similarity_score=round(score, 4) if score is not None else None,
        )
