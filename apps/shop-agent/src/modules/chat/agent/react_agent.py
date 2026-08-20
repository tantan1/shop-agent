"""
ReAct Agent —— 基于 LangChain create_agent 的 ReAct 循环

Agent 自主决策：
1. 是否需要先查 RAG（政策/规则类知识）
2. 调用对应的业务 tool（查订单/查物流/退货/余额/优惠券）
3. 将工具结果与 RAG 结果融合成最终回复

工具选择策略（P0 + P1 + P2 三层过滤）：
- P0 意图前置过滤：IntentResult.action → 缩减候选池到 2-5 个
- P1 Embedding 语义重排（FAISS HNSW）：user query × tool description 余弦相似度 + 意图加权 → Top-3/5
- P2 本地模型最终确认：用本地 Qwen2.5-1.5B 从 Top-3/5 中选出最相关的

Skill SOP 注入：
- 所有 skill 定义在 skills/*/SKILL.md 文件中
- 启动时：SkillLoader 读取 frontmatter + 正文存入 SkillRegistry
- 运行时：P0/P1/P2 确定工具后，反向查找命中 skill，将正文内联注入 system prompt

Agent Rules 注入（仿 Claude Code rules/ 机制）：
- 所有规则定义在 agent-rules/*.md 文件中，支持 YAML frontmatter 声明作用域
- 每次请求按意图 + 工具过滤，只注入匹配的规则
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from src.modules.chat.agent.react_agent_interrupt import (
    InterruptContext,
    _INTERRUPT_MEM,
    _pop_interrupt,
    _store_interrupt,
)
from src.modules.chat.agent.react_agent_prompts import (
    _REACT_SYSTEM_PROMPT,
    _emotion_to_tone_mode,
)
from src.modules.chat.agent.react_agent_reply import (
    apply_scenario_reply,
    format_intermediate_steps,
    parse_messages,
)
from src.modules.chat.agent.react_agent_rules import _filter_rules, _load_rules
from src.modules.chat.agent.react_agent_selection import (
    EmbeddingToolMatcher,
    _get_tool_selector_middleware,
    _local_p1_tool_select,
    _make_business_args_schema,
)
from src.modules.chat.agent.skill_loader import (
    SkillRegistry,
    get_skill_registry,
)
from src.modules.chat.core.content_filter import ContentFilterService
from src.modules.chat.core.sentiment_service import (
    EMOTION_TONE_PROMPTS,
    EmotionLevel,
)
from src.modules.chat.schemas import (
    ChatRequest,
    ChatResponse,
    IntentResult,
    PlannedAction,
    ToolPlan,
)
from src.modules.monitoring.langfuse_callback import create_langfuse_handler
from src.shared.logger import APILogger

if TYPE_CHECKING:
    from typing import Union

    from src.modules.chat.core.embedding_service import EmbeddingService
    from src.modules.chat.core.llm_service import LLMService
    from src.modules.chat.core.milvus_service import MilvusService
    from src.modules.chat.core.pgvector_service import PgVectorService
    from src.modules.chat.core.tool_registry import ToolService

    VectorStoreService = Union[MilvusService, PgVectorService]

logger = APILogger("react_agent")

# 全局懒加载单例
_SKILL_REGISTRY: SkillRegistry | None = None


def _skill_registry() -> SkillRegistry:
    global _SKILL_REGISTRY
    if _SKILL_REGISTRY is None:
        _SKILL_REGISTRY = get_skill_registry()
    return _SKILL_REGISTRY


_ORDER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{3,31}$")


def _is_valid_order_id(order_id: str | None) -> bool:
    """轻量订单号格式校验（硬注入前最后一道闸）。

    仅做格式约束，不做业务存在性判断：非空、长度 4-32、字符集为字母/数字/连字符/下划线。
    返回 False 时调用方应提示用户而非静默下发，避免前置抽取错误导致资损。
    """
    if not order_id or not isinstance(order_id, str):
        return False
    return bool(_ORDER_ID_RE.match(order_id.strip()))


def _get_intent_tool_map() -> dict[str, set[str]]:
    return _skill_registry().intent_tool_map


@dataclass
class ReActRunContext:
    """ReAct Agent 执行上下文，封装 run 方法所需参数。"""
    request: ChatRequest
    intent_result: IntentResult
    conversation_id: str
    domain: str
    intent_steps: list
    langfuse_handler: Any = None


@dataclass
class PendingApprovalContext:
    """人在回路上下文，封装 _handle_pending_approval 所需参数。"""
    config: dict
    agent_graph: Any
    conversation_id: str
    intent_steps: list
    domain: str
    langfuse_ctx: Any


@dataclass
class SuccessResponseContext:
    """成功响应构建上下文，封装 _build_success_response 所需参数。"""
    result: dict
    request: ChatRequest
    intent_result: IntentResult
    intent_steps: list
    domain: str
    conversation_id: str
    react_start: float


class ReActAgent:
    """真正的 ReAct Agent —— 具备 tool-calling 迭代循环 + 意图前置工具过滤"""

    def __init__(
        self,
        *,
        llm_service: LLMService,
        tool_service: ToolService,
        embedding_service: EmbeddingService | None = None,
        milvus_service: VectorStoreService | None = None,
        max_iterations: int = 5,
        emotion_result: Any = None,
        input_truncated: bool = False,
        skill_filter: str | None = None,
    ):
        self._llm_service = llm_service
        self._tool_service = tool_service
        self._embedding_service = embedding_service
        self._milvus_service = milvus_service
        self._max_iterations = max_iterations
        self._emotion_result = emotion_result
        self._input_truncated = input_truncated
        # 单 skill 收敛：指定后仅构建该 skill 工具集，绕过节点内 P0/P1/P2 精选
        self._skill_filter = skill_filter

        self._skill_registry = _skill_registry()

        from src.modules.chat.agent.command_tool_service import CommandToolService
        self._command_tool_service = CommandToolService(tool_service=self._tool_service)

        self._all_tools = self._build_tools()

        # 单 skill 收敛场景下工具集已确定，无需构建 Embedding 精选器（省一次初始化）
        self._tool_matcher: EmbeddingToolMatcher | None = None
        if self._embedding_service and not self._skill_filter:
            self._tool_matcher = EmbeddingToolMatcher(
                self._skill_registry.tool_descriptions, self._embedding_service
            )

        self._pending_approval: tuple[str, str, str] | None = None
        # T5: 最近一次 P0/P1/P2 规划收敛出的结构化 ToolPlan（确定性终止信号载体）
        self._last_tool_plan: ToolPlan | None = None

    def _intent_candidate_union(self) -> set[str]:
        """所有 skill 的 ``allowed_tools`` 并集（意图候选全集，含关联工具）。

        与 ``SkillRegistry.intent_tool_map`` 同源：把每个 skill 的 ``allowed-tools``
        候选集汇总去重，得到"任意意图可能调用到的全部业务 action"。通用入口据此构建
        tool，而非盲目注册整张 skill 注册表（后者可能包含未在任何意图候选集中的悬空工具）。
        """
        union: set[str] = set()
        for s in self._skill_registry.skills:
            union.add(s.name)
            union.update(s.allowed_tools)
        # 剔除注册表中不存在的悬空引用，仅保留真实可 dispatch 的 action
        valid = {s.name for s in self._skill_registry.skills}
        return {n for n in union if n in valid}

    def _build_tools(self):
        """构建 LangChain tool 列表。

        通用入口与单 skill 收敛入口的工具名集合**均源自 ``allowed-tools``（意图候选集，
        含关联工具）**，与 P0 的 ``INTENT_TOOL_MAP`` 保持同源，避免把未在任何意图候选集中
        出现的悬空工具暴露给模型。

        - 未指定 ``skill_filter``（通用入口）：工具集 = 所有 skill 的 ``allowed_tools`` 并集
          （意图候选全集，含关联工具），节点内再由 P0/P1/P2 精选进一步收敛。
        - 指定 ``skill_filter``（单 skill 收敛）：工具集 = 该 skill 自身 + 其 ``allowed_tools``
          声明的关联工具（如 ``query-order`` 的 ``allowed-tools: query-order check-shipping``
          会同时注册两个 tool，使单 skill 模式也能调用关联工具）。
        """
        skills = self._skill_registry.skills
        if self._skill_filter:
            matched = [s for s in skills if s.name == self._skill_filter]
            if not matched:
                logger.warning(
                    "skill_filter 未命中任何 skill，回退意图候选全集",
                    skill_filter=self._skill_filter,
                )
                tool_names = self._intent_candidate_union()
            else:
                # 收敛工具名集合：skill 自身 + allowed_tools 声明的关联工具
                skill = matched[0]
                tool_names = set(skill.allowed_tools) | {skill.name}
        else:
            # 通用入口：意图候选全集 = 所有 skill 的 allowed_tools 并集
            tool_names = self._intent_candidate_union()

        # 仅保留注册表中真实存在的业务 action，避免引用悬空工具
        skills = [s for s in skills if s.name in tool_names]

        tools = []

        for skill in skills:
            action = skill.name
            if skill.hitl:
                tools.append(self._make_refund_tool_with_confirmation())
            else:
                tools.append(self._make_business_tool(action))

        if self._milvus_service and self._embedding_service:
            tools.append(self._make_knowledge_search_tool())

        return tools

    def _make_business_tool(self, action: str, preset_params: dict | None = None):
        """为一个业务 action 创建 LangChain tool（描述来自 SKILL.md）。

        硬强制（方式 2）：``preset_params`` 为前置意图识别确定性抽取的参数。
        通过闭包捕获，在最终 dispatch 时**预设值优先覆盖**模型生成的同名参数；
        配合 ``_make_business_args_schema`` 已从 schema 剔除该字段，模型既看不到
        也填不进，最终进后端的参数 100% 来自确定性抽取。
        """
        desc = self._skill_registry.tool_descriptions.get(action, f"执行{action}操作")
        _action = action
        _dispatch = self._tool_service.dispatch
        _preset = {k: v for k, v in (preset_params or {}).items() if v not in (None, "")}
        _schema = _make_business_args_schema(action, preset_params)

        @tool(_action, description=desc, args_schema=_schema)
        async def business_tool(**kwargs: Any) -> str:
            model_filled = kwargs.get("kwargs") if isinstance(kwargs.get("kwargs"), dict) else kwargs
            # 预设优先：模型生成的同名键被覆盖，关键字段无法被模型篡改
            params = {**model_filled, **_preset}
            logger.info(
                "dispatch 参数合并(硬强制)",
                action=_action,
                preset_fields=list(_preset.keys()),
                model_fields=list(model_filled.keys()),
                final_params=params,
            )
            return await _dispatch(_action, params or None)

        return business_tool

    def _make_refund_tool_with_confirmation(self, preset_params: dict | None = None):
        """创建带人在回路确认的退款工具（命令模式）。

        硬强制（方式 2）：``order_id`` 等高后果字段由前置确定性抽取，经闭包预设，
        模型生成的 order_id 会被覆盖，且 schema 中不暴露该字段（见下方 args_schema）。
        """
        desc = self._skill_registry.tool_descriptions.get("request-return", "申请退货退款")
        _agent = self
        _preset = {k: v for k, v in (preset_params or {}).items() if v not in (None, "")}
        _schema = _make_business_args_schema("request-return", preset_params)

        @tool("request-return", description=desc, args_schema=_schema)
        async def request_return_with_confirm(
            order_id: str = "",
            reason: str = "未说明",
            **kwargs: Any,
        ) -> str:
            """退款工具 —— 带人在回路确认（命令模式）"""
            # 预设优先：确定性抽出的 order_id 覆盖模型生成的
            effective_order = _preset.get("order_id") or order_id
            effective_reason = _preset.get("reason") or reason
            # 轻量格式校验：硬注入的值若非法，不下发而是提示用户确认，避免资损
            if not _is_valid_order_id(effective_order):
                logger.warning(
                    "退货订单号格式校验未通过",
                    action="request-return",
                    order_id=effective_order,
                )
                return (
                    f"您提供的订单号「{effective_order}」格式不正确，无法发起退货申请。"
                    "请核对后提供正确的订单号（通常为数字或字母组合）。"
                )
            result = await _agent._command_tool_service.dispatch(
                action="request-return",
                params={"order_id": effective_order, "reason": effective_reason, **kwargs},
                conversation_id=getattr(_agent, "_current_conversation_id", ""),
                domain=getattr(_agent, "_current_domain", ""),
            )

            if "等待人工审批" in result or "进入人工审批队列" in result:
                _agent._pending_approval = (effective_order, effective_reason, "")
                return result

            return result

        return request_return_with_confirm

    def _make_knowledge_search_tool(self):
        """创建 RAG 知识库检索工具"""

        @tool(description="搜索知识库，查询公司政策、退货规则、产品信息等。参数: query(搜索内容)")
        async def knowledge_search(query: str) -> str:
            """从 Milvus + embedding 检索知识库"""
            try:
                emb = self._embedding_service.get_embeddings()
                query_vec = await emb.aembed_query(query)
                docs = self._milvus_service.search_similar(query_vec, top_k=3)

                if not docs:
                    return "知识库中未找到相关信息。"

                results = []
                for i, doc in enumerate(docs, 1):
                    content = doc.page_content[:600]
                    results.append(f"[文档{i}] {content}")
                return "\n\n".join(results)
            except Exception as e:
                logger.error(f"RAG检索失败: {e}")
                return "知识库检索暂时不可用。"

        return knowledge_search

    async def _select_tools_for_intent(self, action: str | None, user_query: str) -> list:
        """三层工具精选流水线：P0 意图规则过滤 → P1 Embedding 语义重排 → P2 本地模型确认。

        T5 think/act 解耦：三层规划只负责「选工具」（thinking），收敛为结构化
        ToolPlan；执行（act）由下游消费 ToolPlan 驱动。本方法兼容旧下游——仍返回
        LangChain tool 对象列表，但内部统一以 ToolPlan 组织，并据置信度决定是否
        给出 plan_complete 终止信号（供后续执行器直接 dispatch 使用）。
        """
        intent_map = _get_intent_tool_map()
        tool_names: set[str] = intent_map.get(
            action or "unknown",
            intent_map["unknown"],
        )

        # ── P0 规划：意图规则直接产出 PlannedAction(source="p0") ──
        plan = ToolPlan(
            actions=[PlannedAction(name=n, source="p0", confidence=1.0) for n in tool_names],
            source="p0",
        )
        logger.info(
            "P0 意图过滤完成",
            action=action,
            candidates=len(tool_names),
            tool_names=sorted(tool_names),
        )

        # ── P1 规划：Embedding 语义重排，更新 plan（保留 P0 中落选工具的来源）──
        p2_ranked: list[str] = []
        if self._tool_matcher and len(tool_names) > 1:
            try:
                p2_top_k = 5 if (action is None or action == "unknown") else 4
                ranked_names = await self._tool_matcher.rank(
                    user_query=user_query,
                    candidate_names=tool_names,
                    intent_action=action,
                    top_k=p2_top_k,
                )
                if ranked_names:
                    p2_ranked = list(ranked_names)
                    # P1 重排结果覆盖为更高优先级规划（置信度统一 1.0，细节分由 matcher 内部使用）
                    plan = ToolPlan(
                        actions=[PlannedAction(name=n, source="p1", confidence=1.0) for n in p2_ranked],
                        source="p1",
                    )
                else:
                    logger.warning("P1 重排返回空结果，保持 P0 候选集")
            except Exception as e:
                logger.warning(f"Embedding 重排失败，回退到 P0 结果: {e}")

        # ── P2 规划：本地小模型确认，产出 source="p2" 的 ToolPlan ──
        current_names = plan.to_tool_names()
        if self._tool_matcher and len(current_names) > 2:
            tool_descs = self._skill_registry.tool_descriptions
            try:
                p2_plan = await _local_p1_tool_select(
                    user_query=user_query,
                    tool_names=current_names,
                    tool_descriptions=tool_descs,
                    p2_ranked=p2_ranked,
                )
                if p2_plan.actions:
                    plan = p2_plan
                    logger.info(
                        "P2 本地模型工具选择完成",
                        before=sorted(current_names),
                        after=sorted(plan.to_tool_names()),
                    )
            except Exception as e:
                logger.warning(f"P2 本地模型工具选择失败，保留 P1 结果: {e}")

        # ── 确定性终止信号：高置信度规划 → plan_complete（执行器可直接 dispatch）──
        stop_condition = "plan_complete" if plan.is_confident else "need_llm"
        plan = plan.with_stop_condition(stop_condition)

        # ── 兼容消费：将 ToolPlan 还原为 LangChain tool 对象列表 ──
        selected_names = plan.to_tool_names()
        filtered: list = []
        for t in self._all_tools:
            name = getattr(t, "name", "")
            if name in selected_names:
                filtered.append(t)
            elif name == "knowledge_search":
                # knowledge_search 为 always_include，不计入规划但始终可用
                filtered.append(t)

        logger.info(
            "P0+P1+P2 工具过滤最终结果",
            action=action,
            total_tools=len(self._all_tools),
            filtered=len(filtered),
            source=plan.source,
            stop_condition=plan.stop_condition,
            tool_names=sorted([getattr(t, "name", "") for t in filtered]),
        )
        # 附加 ToolPlan 到返回值（透传给需要确定性执行的下层，向后兼容：list 仍可直接遍历）
        self._last_tool_plan = plan
        return filtered

    def _build_graph(
        self,
        tools: list | None = None,
        checkpointer=None,
        intent: str | None = None,
        extra_system_context: str | None = None,
    ):
        """构建 LangChain create_agent。"""
        from langchain.agents import create_agent

        llm = self._llm_service.qwen_llm
        if llm is None:
            raise RuntimeError("LLM 未初始化，无法构建 Agent")
        selected_tools = tools if tools is not None else self._all_tools

        middleware = []
        tool_count = len([t for t in selected_tools if not isinstance(t, dict)])
        if tool_count > 3:
            middleware = [_get_tool_selector_middleware(self._llm_service)]

        tool_names = {getattr(t, "name", "") for t in selected_tools}
        prompt = self._build_system_prompt(
            tool_names, intent=intent, extra_context=extra_system_context
        )

        graph = create_agent(
            model=llm,
            tools=selected_tools,
            system_prompt=prompt,
            middleware=middleware,
            checkpointer=checkpointer,
        )
        return graph.with_config({"recursion_limit": self._max_iterations * 2 + 2})

    def _build_system_prompt(
        self, tool_names: set[str], intent: str | None = None, extra_context: str | None = None
    ) -> str:
        """动态拼装 system prompt：base 核心指令 + Agent Rules + 情绪 tone + 命中 skill 的 SOP 正文。"""
        prompt = _REACT_SYSTEM_PROMPT

        rules_text = _filter_rules(
            _load_rules(),
            intent=intent,
            tool_names=tool_names,
        )
        if rules_text:
            prompt += "\n\n## 行为规则（来自 agent-rules/）\n" + rules_text

        if self._emotion_result and self._emotion_result.level != EmotionLevel.NEUTRAL:
            tone_mode = _emotion_to_tone_mode(self._emotion_result.level)
            tone_text = EMOTION_TONE_PROMPTS.get(tone_mode, "")
            if tone_text:
                prompt += tone_text
                logger.info(
                    "系统提示注入情绪感知",
                    level=self._emotion_result.level.name,
                    mode=tone_mode,
                )

        biz_tool_names = tool_names - {"knowledge_search"}

        matched_skills = [
            s for s in self._skill_registry.skills if set(s.allowed_tools) & biz_tool_names
        ]
        if matched_skills:
            bodies = [s.body for s in matched_skills if s.body]
            if bodies:
                prompt += "\n\n## 当前场景操作指南\n" + "\n\n".join(bodies)

        if self._input_truncated:
            prompt += (
                "\n\n## 输入截断提醒\n"
                "用户本次输入较长，已被自动精简，订单号、手机号等关键信息可能丢失。\n"
                "回复时如发现处理请求所必需的信息不完整，请主动、礼貌地向用户询问补充，"
                "例如：'请问您的订单号是多少呢？'、'您能提供一下手机号码吗？'"
            )

        if extra_context:
            prompt += "\n\n## 意图上下文（由意图识别阶段注入，供你理解本次请求）\n" + extra_context

        return prompt

    async def _execute_plan_directly(self, ctx: ReActRunContext, plan: "ToolPlan") -> ChatResponse:
        """T5 执行侧：确定性消费 ToolPlan，直接 dispatch 并润色回复（跳过 ReAct 循环）。

        执行（act）完全由规划（think）产出的 ToolPlan 驱动：
        - 对每个 PlannedAction 用确定性抽取的参数（intent_result.params）直接 dispatch；
        - 模型仅负责把工具结果合成自然语言（不再决定调什么、何时停）；
        - 输出经内容安全过滤后返回。
        调用方已保证 plan.stop_condition=="plan_complete" 且不含 hitl 工具。
        """
        from src.modules.chat.core.content_filter import ContentFilterService
        from src.modules.chat.agent.react_agent_reply import apply_scenario_reply

        react_start = time.monotonic()
        intent_result = ctx.intent_result
        preset_params = intent_result.params or {}

        # 1) 确定性逐个 dispatch（模型不介入选工具/填参）
        tool_outputs: list[dict] = []
        for action in plan.actions:
            name = action.name
            try:
                t0 = time.monotonic()
                raw = await self._tool_service.dispatch(name, preset_params or None)
                t_dur = int((time.monotonic() - t0) * 1000)
                logger.info(
                    "确定性执行 ToolPlan 动作",
                    action=name,
                    source=action.source,
                    duration_ms=t_dur,
                )
                tool_outputs.append({
                    "action": name,
                    "output": raw,
                    "status": "success",
                })
            except Exception as e:
                logger.error(f"确定性执行工具 {name} 失败: {str(e)[:200]}")
                tool_outputs.append({
                    "action": name,
                    "output": f"工具 {name} 执行失败：{str(e)[:120]}",
                    "status": "failed",
                })

        combined = "\n\n".join(
            f"[{o['action']}]\n{o['output']}" for o in tool_outputs
        )

        # 2) 模型仅做结果润色（自然语言合成），不决定工具调用
        langfuse_handler, langfuse_ctx = self._init_langfuse(
            ctx.conversation_id, ctx.domain, intent_result, ctx.langfuse_handler
        )
        final_output = combined
        try:
            try:
                llm = self._llm_service.qwen_llm
                if llm is not None:
                    polish_prompt = (
                        "你是电商客服助手。下面是为用户查询得到的工具返回结果，"
                        "请用简洁、自然的中文口语化转述给用户，不要输出工具名或原始 JSON 标记。"
                        "若结果提示信息不足，可友好地补充询问。\n\n"
                        f"用户问题：{ctx.request.message}\n\n工具结果：\n{combined}"
                    )
                    polish = await llm.ainvoke(polish_prompt)
                    final_output = getattr(polish, "content", None) or combined
            except Exception as e:
                logger.warning(f"ToolPlan 结果润色失败，回退原始拼接: {str(e)[:120]}")
                final_output = combined

            final_output = apply_scenario_reply(final_output, tool_outputs)
        finally:
            if langfuse_ctx:
                langfuse_ctx.__exit__(None, None, None)

        # 3) 输出安全过滤
        output_filter_safe = True
        cf = ContentFilterService.get_instance()
        output_check = cf.filter_output(final_output, ctx.domain)
        if not output_check.is_safe:
            output_filter_safe = False
            logger.warning(
                "确定性执行输出安全检查未通过",
                domain=ctx.domain,
                risk_categories=output_check.risk_categories,
            )
            final_output = output_check.filtered_text or "抱歉，当前无法处理您的请求，请稍后重试。"

        elapsed_ms = int((time.monotonic() - react_start) * 1000)
        logger.info(
            "确定性执行完成（跳过 ReAct 循环）",
            source=plan.source,
            actions=sorted(plan.to_tool_names()),
            duration_ms=elapsed_ms,
        )
        return ChatResponse(
            message=final_output,
            conversation_id=ctx.conversation_id,
            steps=ctx.intent_steps
            + [
                {
                    "step_name": "ToolPlan确定性执行",
                    "step_order": len(ctx.intent_steps),
                    "status": "success",
                    "output_data": {
                        "mode": "plan_complete_direct",
                        "source": plan.source,
                        "actions": [o["action"] for o in tool_outputs],
                    },
                }
            ],
            documents_used=[],
            safety_passed=output_filter_safe,
            stream_available=True,
            domain=ctx.domain,
        )

    async def run(self, ctx: ReActRunContext) -> ChatResponse:
        """执行 ReAct 循环。

        T5 执行侧解耦：规划（_select_tools_for_intent）已产出 ToolPlan 并存于
        self._last_tool_plan。若规划置信度足够高（stop_condition=="plan_complete"）
        且不涉及人在回路（hitl）工具，则**跳过 ReAct 循环**，由确定性逻辑直接
        dispatch 计划中的工具、再用 LLM 仅做结果润色——执行（act）完全由 plan 驱动，
        模型不再决定「何时停」。否则回退到原 ReAct 自主循环。
        """
        selected_tools = await self._select_tools_for_intent(
            ctx.intent_result.action, user_query=ctx.request.message
        )

        blocked = self._check_input_safety(ctx.request, ctx.domain, ctx.intent_steps, ctx.conversation_id)
        if blocked:
            return blocked

        # ── 执行侧解耦：高置信规划 → 确定性直接执行，绕过 ReAct 循环 ──
        plan = self._last_tool_plan
        if plan is not None and plan.stop_condition == "plan_complete":
            hitl_actions = {s.name for s in self._skill_registry.skills if s.hitl}
            if not (plan.to_tool_names() & hitl_actions):
                logger.info(
                    "规划置信度高，走确定性直接执行（跳过 ReAct 循环）",
                    source=plan.source,
                    actions=sorted(plan.to_tool_names()),
                )
                return await self._execute_plan_directly(ctx, plan)
            logger.info(
                "规划含人在回路工具，仍走 ReAct 循环以触发确认",
                actions=sorted(plan.to_tool_names()),
            )

        agent_graph, enhanced_message = self._build_react_graph(
            selected_tools, ctx.conversation_id, ctx.intent_result, ctx.request
        )

        langfuse_handler, langfuse_ctx = self._init_langfuse(
            ctx.conversation_id, ctx.domain, ctx.intent_result, ctx.langfuse_handler
        )

        react_start = time.monotonic()

        try:
            config = {
                "recursion_limit": self._max_iterations * 2 + 2,
                "configurable": {"thread_id": ctx.conversation_id},
            }
            if langfuse_handler:
                config["callbacks"] = [langfuse_handler]

            result = await agent_graph.ainvoke(
                {"messages": [HumanMessage(content=enhanced_message)]},
                config=config,
            )

            interrupt_response = self._handle_pending_approval(
                PendingApprovalContext(
                    config=config,
                    agent_graph=agent_graph,
                    conversation_id=ctx.conversation_id,
                    intent_steps=ctx.intent_steps,
                    domain=ctx.domain,
                    langfuse_ctx=langfuse_ctx,
                )
            )
            if interrupt_response:
                return interrupt_response

        except Exception as e:
            self._clear_pending_approval()
            return self._build_error_response(e, ctx.intent_steps, ctx.domain, ctx.conversation_id)
        finally:
            if langfuse_ctx:
                langfuse_ctx.__exit__(None, None, None)

        return self._build_success_response(
            SuccessResponseContext(
                result=result,
                request=ctx.request,
                intent_result=ctx.intent_result,
                intent_steps=ctx.intent_steps,
                domain=ctx.domain,
                conversation_id=ctx.conversation_id,
                react_start=react_start,
            )
        )

    def _check_input_safety(
        self, request: ChatRequest, domain: str, intent_steps: list, conversation_id: str
    ) -> ChatResponse | None:
        """输入内容安全过滤。"""
        cf = ContentFilterService.get_instance()
        input_check = cf.filter_input(request.message, domain)
        if input_check.is_safe:
            return None
        logger.warning(
            "ReAct Agent 输入安全检查未通过",
            domain=domain,
            risk_categories=input_check.risk_categories,
        )
        return ChatResponse(
            message="抱歉，您的请求包含不安全内容，无法处理。",
            conversation_id=conversation_id,
            steps=intent_steps
            + [
                {
                    "step_name": "输入安全过滤",
                    "step_order": len(intent_steps),
                    "status": "blocked",
                    "output_data": {"reason": input_check.reason},
                }
            ],
            documents_used=[],
            safety_passed=False,
            stream_available=True,
            domain=domain,
        )

    def _apply_preset_to_tools(self, selected_tools: list, preset_params: dict | None) -> list:
        """用确定性抽取的参数（preset）重建工具，实现硬强制。

        仅对 action 名命中的工具重建（带 preset 闭包 + 从 schema 剔除已预设字段），
        其余工具原样保留。``knowledge_search`` 等非业务工具不受影响。
        hitl（人在回路）类 Skill 走带确认的工具构建，不再特判 action 名。
        """
        if not preset_params:
            return selected_tools
        hitl_actions = {s.name for s in self._skill_registry.skills if s.hitl}
        rebuilt = []
        for t in selected_tools:
            name = getattr(t, "name", "")
            if name in hitl_actions:
                rebuilt.append(self._make_refund_tool_with_confirmation(preset_params))
            elif name in self._skill_registry.tool_descriptions:
                rebuilt.append(self._make_business_tool(name, preset_params))
            else:
                rebuilt.append(t)
        return rebuilt

    def _build_react_graph(
        self,
        selected_tools: list,
        conversation_id: str,
        intent_result: IntentResult,
        request: ChatRequest,
    ):
        """构建 ReAct Agent 图并构造增强消息。"""
        preset_params = intent_result.params or {}
        # 硬强制：用确定性抽取的参数重建工具，模型无法篡改关键字段
        preset_tools = self._apply_preset_to_tools(selected_tools, preset_params)

        memory_saver = MemorySaver()
        agent_graph = self._build_graph(
            tools=preset_tools, checkpointer=memory_saver, intent=intent_result.action
        )

        params_str = ""
        confirmed_order_id = None
        if intent_result.params:
            params_str = f"已抽取参数: {json.dumps(intent_result.params, ensure_ascii=False)}. "
            confirmed_order_id = intent_result.params.get("order_id") or None

        confirm_str = ""
        if confirmed_order_id:
            confirm_str = (
                f"\n【强制约束】本次请求已确认订单号为 {confirmed_order_id}。"
                f"调用 query-order / check-shipping 等工具时，order_id 必须且只能是 "
                f"{confirmed_order_id}，严禁使用历史对话中出现的任何其他订单号。"
            )

        intent_context = (
            f"用户意图: {intent_result.action}, "
            f"复杂程度: {intent_result.complexity}. "
            f"{params_str}{confirm_str}"
        )
        agent_graph = self._build_graph(
            tools=preset_tools,
            checkpointer=memory_saver,
            intent=intent_result.action,
            extra_system_context=intent_context,
        )

        enhanced_message = f"用户问题: {request.message}"
        if confirmed_order_id:
            enhanced_message += (
                f"\n\n(本次已确认订单号={confirmed_order_id}，请务必使用此订单号，"
                f"不要使用聊天历史中的其他订单号)"
            )

        logger.info(
            "ReAct Agent 开始执行",
            action=intent_result.action,
            params=intent_result.params,
            message_length=len(request.message),
            tools_count=len(selected_tools),
        )

        return agent_graph, enhanced_message

    def _init_langfuse(
        self,
        conversation_id: str,
        domain: str,
        intent_result: IntentResult,
        langfuse_handler,
    ):
        """初始化 Langfuse。外部传入时复用，否则内部创建。"""
        langfuse_ctx = None
        if langfuse_handler is None:
            result = create_langfuse_handler(
                session_id=conversation_id,
                tags=[domain, "react-agent"],
                trace_name=f"{domain}-react-{intent_result.action}",
                metadata={
                    "domain": domain,
                    "action": intent_result.action,
                    "complexity": intent_result.complexity,
                },
            )
            if result:
                langfuse_handler, langfuse_ctx = result
                langfuse_ctx.__enter__()
        return langfuse_handler, langfuse_ctx

    def _handle_pending_approval(
        self,
        approval_ctx: PendingApprovalContext,
    ) -> ChatResponse | None:
        """处理人在回路（request-return 工具）。"""
        if not self._command_tool_service.has_pending_approval:
            return None
        action, approval_id, params = self._command_tool_service.pop_pending_approval()
        order_id = params.get("order_id", "")
        reason = params.get("reason", "")

        logger.info(
            "ReAct Agent 进入人在回路等待",
            conversation_id=approval_ctx.conversation_id,
            action=action,
            order_id=order_id,
            approval_id=approval_id,
        )
        _store_interrupt(
            InterruptContext(
                thread_id=approval_ctx.conversation_id,
                graph=approval_ctx.agent_graph,
                config=approval_ctx.config,
                conversation_id=approval_ctx.conversation_id,
                intent_steps=approval_ctx.intent_steps,
                domain=approval_ctx.domain,
                order_id=order_id,
                reason=reason,
            )
        )

        if approval_ctx.langfuse_ctx:
            approval_ctx.langfuse_ctx.__exit__(None, None, None)

        return ChatResponse(
            message="退款申请需要人工确认，请通过审批系统进行操作。",
            conversation_id=approval_ctx.conversation_id,
            steps=approval_ctx.intent_steps
            + [
                {
                    "step_name": "人在回路-退款确认",
                    "step_order": len(approval_ctx.intent_steps),
                    "status": "waiting",
                    "output_data": {
                        "action": action,
                        "order_id": order_id,
                        "reason": reason,
                        "approval_id": approval_id,
                        "message": "等待人工审批确认退款",
                    },
                }
            ],
            documents_used=[],
            safety_passed=True,
            stream_available=True,
            domain=approval_ctx.domain,
            status="waiting_for_confirmation",
            interrupt_data={
                "type": "refund_confirmation",
                "action": action,
                "conversation_id": approval_ctx.conversation_id,
                "order_id": order_id,
                "reason": reason,
                "approval_id": approval_id,
            },
        )

    def _clear_pending_approval(self):
        """清除 pending approval 状态，避免状态残留。"""
        if self._command_tool_service.has_pending_approval:
            self._command_tool_service.pop_pending_approval()

    def _build_error_response(
        self, e: Exception, intent_steps: list, domain: str, conversation_id: str
    ) -> ChatResponse:
        """构建执行错误响应。"""
        logger.error(f"ReAct Agent 执行失败: {e}")
        return ChatResponse(
            message="抱歉，处理您的请求时遇到了问题，请稍后重试。",
            conversation_id=conversation_id,
            steps=intent_steps
            + [
                {
                    "step_name": "ReAct执行",
                    "step_order": len(intent_steps),
                    "status": "failed",
                    "error_message": str(e)[:200],
                }
            ],
            documents_used=[],
            safety_passed=False,
            stream_available=True,
            domain=domain,
        )

    def _build_success_response(
        self,
        success_ctx: SuccessResponseContext,
    ) -> ChatResponse:
        """构建成功响应（解析结果 + 安全过滤 + 日志）。"""
        elapsed_ms = int((time.monotonic() - success_ctx.react_start) * 1000)

        messages: list = success_ctx.result.get("messages", [])
        final_output, intermediate_steps = parse_messages(messages)
        if not final_output:
            final_output = "抱歉，我暂时无法处理这个请求。"
        final_output = apply_scenario_reply(final_output, intermediate_steps)

        output_filter_safe = True
        cf = ContentFilterService.get_instance()
        output_check = cf.filter_output(final_output, success_ctx.domain)
        if not output_check.is_safe:
            output_filter_safe = False
            logger.warning(
                "ReAct Agent 输出安全检查未通过",
                domain=success_ctx.domain,
                risk_categories=output_check.risk_categories,
            )
            if output_check.filtered_text:
                final_output = output_check.filtered_text
            else:
                final_output = "抱歉，当前无法处理您的请求，请稍后重试。"

        react_steps = format_intermediate_steps(intermediate_steps, elapsed_ms)

        logger.log_business_event(
            "电商Agent ReAct调用",
            success=True,
            domain=success_ctx.domain,
            action=success_ctx.intent_result.action,
            complexity=success_ctx.intent_result.complexity,
            conversation_id=success_ctx.conversation_id,
            message_length=len(success_ctx.request.message),
            response_length=len(final_output),
            react_iterations=len(intermediate_steps),
            duration_ms=elapsed_ms,
        )

        return ChatResponse(
            message=final_output,
            conversation_id=success_ctx.conversation_id,
            steps=success_ctx.intent_steps + react_steps,
            documents_used=[],
            safety_passed=output_filter_safe,
            stream_available=True,
            domain=success_ctx.domain,
        )

    @staticmethod
    async def resume_execution(
        thread_id: str,
        confirm: bool,
        tool_service: "ToolService | None" = None,
    ) -> "ChatResponse | None":
        """恢复被人在回路中断的退款执行（命令模式）。"""
        from src.modules.chat.agent.command_tool_service import CommandToolService

        stored = _pop_interrupt(thread_id)
        if stored is None:
            logger.warning(f"未找到待恢复的中断: thread_id={thread_id}")
            return None

        _, __, conversation_id, intent_steps, domain, order_id, reason = stored

        if not confirm:
            logger.log_business_event(
                "退款审批-拒绝", conversation_id=conversation_id, order_id=order_id
            )
            return ChatResponse(
                message=f"退款申请已被取消（订单号: {order_id}）。如有需要，请重新发起申请。",
                conversation_id=conversation_id,
                steps=intent_steps
                + [
                    {
                        "step_name": "人在回路-退款拒绝",
                        "step_order": len(intent_steps),
                        "status": "success",
                        "output_data": {"order_id": order_id, "confirm": False},
                    }
                ],
                documents_used=[],
                safety_passed=True,
                stream_available=True,
                domain=domain,
                status="completed",
            )

        logger.info("退款审批通过，执行退款", conversation_id=conversation_id, order_id=order_id)
        react_start = time.monotonic()
        try:
            command_service = CommandToolService(tool_service=tool_service)
            # ReAct 退款链路未走 ApprovalGate.execute_with_approval，故 ApprovalGate 内
            # 不存在 approval_id 记录，command_service.approve 会返回 failed。此处不再
            # 依赖 ApprovalGate：先尝试 approve（兼容未来接通 ApprovalGate 的场景），
            # 失败则降级为用中断上下文直接 dispatch 退款「确认执行」，避免 fail-closed
            # 误报「执行失败」导致审批通过却无反馈。
            dispatch_result = None
            try:
                result = await command_service.approve(approval_id=thread_id)
                if result.status == "success":
                    dispatch_result = result.message
            except Exception as e:
                logger.warning(
                    "ApprovalGate.approve 未命中记录，降级直接 dispatch 执行退款",
                    thread_id=thread_id,
                    error=str(e)[:200],
                )

            if dispatch_result is None:
                # 降级路径：直接重新 dispatch 退款确认执行（带 confirm 标记，供后端幂等去重）。
                # 幂等保证：_pop_interrupt 在取回上下文时已删除 Redis 中断记录，
                # 同一 conversation_id 二次 resume 会返回 None（services 抛「未找到」），
                # 因此本分支天然不会重复执行。
                exec_result = await tool_service.dispatch(
                    "request-return",
                    {
                        "order_id": order_id,
                        "reason": reason,
                        "confirm": True,
                        "approval_id": thread_id,
                    },
                )
                # 若后端仍返回「等待审批」，说明该申请已被处理过（幂等命中），提示而非报错
                if isinstance(exec_result, str) and "等待" in exec_result:
                    dispatch_result = f"退款申请（订单号: {order_id}）已处理，无需重复确认。"
                else:
                    dispatch_result = (
                        exec_result
                        if isinstance(exec_result, str)
                        else "退款申请已批准并执行。"
                    )
        except Exception as e:
            logger.error(f"退款执行失败: {e}", thread_id=thread_id, order_id=order_id)
            return ChatResponse(
                message="审批已通过，但退款操作执行失败，请稍后重试。",
                conversation_id=conversation_id,
                status="completed",
                safety_passed=False,
                domain=domain,
            )

        elapsed_ms = int((time.monotonic() - react_start) * 1000)

        logger.info(
            "人在回路恢复执行完成", thread_id=thread_id, confirm=True, elapsed_ms=elapsed_ms
        )

        output_filter_safe = True
        cf = ContentFilterService.get_instance()
        output_check = cf.filter_output(dispatch_result, domain)
        if not output_check.is_safe:
            output_filter_safe = False
            logger.warning(
                "人在回路 输出安全检查未通过",
                domain=domain,
                risk_categories=output_check.risk_categories,
            )
            if output_check.filtered_text:
                dispatch_result = output_check.filtered_text
            else:
                dispatch_result = "操作已执行，但结果包含无法展示的内容。"

        return ChatResponse(
            message=dispatch_result,
            conversation_id=conversation_id,
            steps=intent_steps
            + [
                {
                    "step_name": "人在回路-退款确认执行",
                    "step_order": len(intent_steps),
                    "status": "success",
                    "output_data": {
                        "order_id": order_id,
                        "reason": reason,
                        "confirm": True,
                        "result": dispatch_result[:300],
                        "duration_ms": elapsed_ms,
                    },
                }
            ],
            documents_used=[],
            safety_passed=output_filter_safe,
            stream_available=True,
            domain=domain,
            status="completed",
        )

    async def approve_tool(self, approval_id: str) -> dict:
        """审批通过工具调用（命令模式）。"""
        result = await self._command_tool_service.approve(approval_id)
        return {
            "status": result.status,
            "message": result.message or "审批通过，操作已执行。",
            "error": result.error,
        }

    async def reject_tool(self, approval_id: str) -> dict:
        """审批拒绝工具调用（命令模式，触发 undo）。"""
        result = await self._command_tool_service.reject(approval_id)
        return {
            "status": result.status,
            "message": result.message or "审批已拒绝，操作已撤销。",
            "error": result.error,
        }


# ════════════════════════════════════════════════════════════════════════
# LangGraph Studio 导出函数 —— Time Travel Web 调试入口
# ════════════════════════════════════════════════════════════════════════


def get_graph():
    """LangGraph Studio 入口：返回带 MemorySaver 的编译后 graph。"""
    from langchain_core.tools import tool

    from src.modules.chat.core.llm_service import LLMService

    llm_svc = LLMService()
    llm_svc.initialize()
    llm = llm_svc.qwen_llm
    if llm is None:
        raise RuntimeError("LLM 未初始化，请检查 TONGYI_API_KEY 配置")

    registry = _skill_registry()
    stub_tools = []

    def _make_stub(tool_name: str, tool_desc: str):
        @tool(tool_name, description=tool_desc)
        def stub(**kwargs: Any) -> str:
            return f"[Studio] 工具 {tool_name} 被调用（LangGraph Studio 调试模式，未接入真实后端）"

        return stub

    for name, desc in registry.tool_descriptions.items():
        stub_tools.append(_make_stub(name, desc))

    @tool(description="搜索知识库，查询公司政策、退货规则、产品信息等。参数: query(搜索内容)")
    def knowledge_search_stub(query: str = "") -> str:
        return f"[Studio] 知识库检索: {query}（调试模式）"

    stub_tools.append(knowledge_search_stub)

    graph = create_agent(
        model=llm,
        tools=stub_tools,
        system_prompt=_REACT_SYSTEM_PROMPT,
        middleware=[],
        checkpointer=MemorySaver(),
    )
    return graph.with_config({"recursion_limit": 24})
