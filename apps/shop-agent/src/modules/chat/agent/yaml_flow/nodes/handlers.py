"""YAML 编排节点处理器（阶段 2.4 / 2.5）。

每个 handler 只关心「本节点的业务逻辑」，不感知上下游节点名。
参数输入（read_from 映射结果）由编译器在调用前组装为 ``inputs`` 注入；
产出由编译器按 write_to 写回 GraphState。

handler 统一签名：``async def run(self, state, inputs) -> Any``。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.shared.logger import APILogger

from .rag_handlers import (
    RagGenerateHandler,
    RagRetrieveHandler,
    RagReviewHandler,
    RagRewriteHandler,
)

logger = APILogger("yaml_flow_nodes")


def _build_hardcode_preset(hardcode) -> Dict[str, Any]:
    """把节点 config.hardcode 声明编译成「闭包预设」dict（阶段 1.4 / 5.1）。

    - 兼容 schema.HardcodeDecl 对象与 dict（测试中 fake）。
    - 只保留 value 非 None/空串的字段；空值视为「不强制」（沿用上游）。
    - 返回 {field: value}，供节点在 dispatch 时定稿注入、值优先于模型/上游输入。
    """
    preset: Dict[str, Any] = {}
    if not hardcode:
        return preset
    for hc in hardcode:
        if isinstance(hc, dict):
            field = hc.get("field")
            value = hc.get("value")
        else:
            field = getattr(hc, "field", None)
            value = getattr(hc, "value", None)
        if not field:
            continue
        if value is None or value == "":
            # 空值不强制：业务可能依赖上游抽取
            continue
        preset[field] = value
    return preset


def _nested_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """把扁平 dict 包成支持 `{params.xxx}` 点访问的包装（用于 str.format）。

    str.format 的 `{params.xxx}` 要求 params 对象有 `.xxx` 属性或支持 `__getitem__`。
    直接用 dict 无法点访问，这里用 types.SimpleNamespace 包裹（一层）。
    """
    import types
    return types.SimpleNamespace(**{k: (v if not isinstance(v, dict) else _nested_dict(v))
                                    for k, v in (d or {}).items()})


params_nested = _nested_dict


class NodeHandler:
    """节点处理器基类。"""

    # 子类可声明是否需要 LLM / tool_service
    needs_llm: bool = False
    needs_tool_service: bool = False

    async def run(self, state: Dict[str, Any], inputs: Dict[str, Any]) -> Any:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════
# 前置节点（阶段 2.5）
# ═══════════════════════════════════════════════════════════════════════

class NormalizeHandler(NodeHandler):
    """归一化节点：复用现有 InputNormalizer。

    此处为 v1 占位实现（直接透传 message）；真实归一化接入
    src.modules.chat.core.synonym_normalizer.InputNormalizer（见阶段 0 现状梳理）。
    """

    async def run(self, state, inputs):
        return {"message": state.get("messages", [{}])[-1].get("content") if state.get("messages") else None}


class IntentHandler(NodeHandler):
    """意图识别节点：v1 占位，直接沿用上游已写入 state.intent。

    真实意图识别接入 IntentRecognizer（见阶段 0 现状梳理）。
    """

    async def run(self, state, inputs):
        return {"intent": state.get("intent", {})}


# ═══════════════════════════════════════════════════════════════════════
# 守卫节点（阶段 2.6）
# ═══════════════════════════════════════════════════════════════════════

class GuardHandler(NodeHandler):
    """横切守卫节点（输入/输出过滤、锁、持久化、埋点）的 v1 占位包装。

    守卫在 v1 不串入主链路，仅登记存在（由节点函数内部调用接口预留），
    校验器已保证 input/output 过滤类守卫必须存在（阶段 1.9 / 5.4）。
    """

    def __init__(self, guard_type: str):
        self._guard_type = guard_type

    async def run(self, state, inputs):
        # v1 守卫透传：真实过滤/锁/持久化逻辑在阶段 5 固化
        logger.info("guard 节点占位执行（v1 透传）", guard_type=self._guard_type)
        return {}


# ═══════════════════════════════════════════════════════════════════════
# 执行节点（阶段 2.4）
# ═══════════════════════════════════════════════════════════════════════

class DirectToolHandler(NodeHandler):
    """直接 Tool 节点（无 LLM 循环）：dispatch 到业务工具。

    对应 2.4.2：tool_service.dispatch（无 LLM 循环）。

    硬强制（阶段 5.1）：节点 config.hardcode 声明的字段由确定性来源定稿注入，
    值优先于上游/模型填入的同名参数（闭包预设）。对应 29 篇「高后果字段强制注入、
    业务不可关闭」——业务在 YAML 里声明即生效，模型无法改写这些字段。
    （ReactHandler 走 ReActAgent 内部 _apply_preset_to_tools 做 schema 剔除；
    direct_tool 无工具调用 schema，故以「参数合并时硬强制值优先 + 审计留痕」等价落地。）
    """

    needs_tool_service = True

    def __init__(self, tool_service: Any, skill: Optional[str] = None, hardcode=None):
        self._tool_service = tool_service
        self._skill = skill
        self._hardcode = hardcode or []

    async def run(self, state, inputs):
        if self._tool_service is None:
            raise ValueError("direct_tool 节点需要 tool_service（运行期缺失）")
        action = self._skill or state.get("intent", {}).get("action")
        if not action:
            raise ValueError("direct_tool 节点缺少 action（config.skill 或 intent.action）")
        params = dict(inputs.get("params", {}) or {})
        # ── 5.1 硬强制（闭包预设 + 值优先）──
        preset = _build_hardcode_preset(self._hardcode)
        if preset:
            overridden = [k for k in preset if k in params and params[k] != preset[k]]
            params = {**params, **preset}  # 硬强制值优先覆盖
            logger.info(
                "direct_tool 硬强制注入（阶段 5.1）",
                action=action,
                preset_fields=list(preset.keys()),
                overridden_upstream=overridden,
            )
        logger.info("direct_tool dispatch", action=action, params=params)
        result = await self._tool_service.dispatch(action, params)
        return {"result": result}


class ReactHandler(NodeHandler):
    """ReAct 节点（v1 黑盒循环）：调用现有 ReActAgent。

    对应 2.4.3：v1 内部仍黑盒循环，但作为图节点；v2 阶段 4.1 拆子图后此节点被子图替代。
    """

    needs_llm = True
    needs_tool_service = True

    def __init__(self, llm_service, tool_service, skill: Optional[str] = None, hardcode: Optional[list] = None):
        self._llm = llm_service
        self._tool_service = tool_service
        self._skill = skill
        self._hardcode = hardcode or []

    async def run(self, state, inputs):
        if self._llm is None or self._tool_service is None:
            raise ValueError("react 节点需要 llm_service 与 tool_service（运行期缺失）")
        from src.modules.chat.agent.react_agent import ReActAgent, ReActRunContext
        from src.modules.chat.core.intent.candidate import ExecutionPlan
        from src.modules.chat.schemas import ChatRequest, IntentResult

        action = self._skill or state.get("intent", {}).get("action")
        # 组装 preset（硬强制）：read_from 的 params + hardcode 声明（阶段 1.4 / 1.5d）
        # 硬强制值优先于模型抽取；落点复用 intent_result.params（ReActAgent 内部 _apply_preset_to_tools 读取，
        # 并据非空 preset 字段从工具 schema 剔除——模型既看不到也填不进）。
        preset = dict(inputs.get("params", {}))
        preset.update(_build_hardcode_preset(self._hardcode))

        intent_state = state.get("intent", {})
        message = state.get("messages", [{}])[-1].get("content", "") if state.get("messages") else ""
        domain = intent_state.get("domain", "ecommerce")
        request = ChatRequest(
            message=message,
            conversation_id=state.get("thread_id", ""),
            domain=domain,
        )
        # 与识别器保持一致：plan 为唯一路由结果，下游统一读 intent_result.mode。
        # 这里把 yaml_flow 旧的 state.intent / state.complexity 适配为 plan.mode。
        raw_intent = intent_state.get("intent", "rag_answer")
        raw_complexity = intent_state.get("complexity")
        if raw_intent != "call_remote_api":
            flow_mode = "rag_pipeline"
        else:
            flow_mode = "react" if raw_complexity == "multi_step" else "direct_tool"
        intent_result = IntentResult(
            plan=ExecutionPlan(mode=flow_mode, skill=action, reason="yaml_flow 节点构造"),
            action=action,
            params=preset or None,
        )
        context = ReActRunContext(
            request=request,
            intent_result=intent_result,
            conversation_id=state.get("thread_id", ""),
            domain=domain,
            intent_steps=[],
        )
        agent = ReActAgent(
            llm_service=self._llm,
            tool_service=self._tool_service,
            embedding_service=None,
            milvus_service=None,
            # 单 skill 收敛（阶段 2.4.7）：绑定 config.skill 时仅构建该 skill 工具集，
            # 绕过节点内 P0/P1/P2 精选；未绑定（多 skill 通用入口）则传 None 走全量精选。
            skill_filter=self._skill,
        )
        response = await agent.run(context)
        return {"result": response.message}


class LLMCallHandler(NodeHandler):
    """单步 LLM 调用节点（阶段 4.1 轻量版）：替代不需要工具循环的 react 节点。

    与 react 节点的区别：
    - 不构建工具、不做 ReAct 循环、不注入 P0/P1/P2 精选；
    - 只做一次 LLM 生成（校验文案 / 回复话术 / 简单判断）；
    - 提示由 ``config.prompt_template`` 渲染（支持 {params.xxx} / {tool_result} / {message}），
      并可注入所引用 Skill 的 SOP 作为 system 上下文（黑盒引用，不可改写）。

    典型用法：退货流程的 validate（校验话术）/ reply（最终回复）节点。
    仍受 guards（input/output_filter）包裹，安全底座不下放。
    """

    needs_llm = True

    def __init__(self, llm_service, skill: Optional[str] = None,
                 prompt_template: Optional[str] = None, skill_registry=None):
        self._llm = llm_service
        self._skill = skill
        self._prompt_template = prompt_template
        self._skill_registry = skill_registry

    @staticmethod
    def _render(template: str, state: Dict[str, Any], inputs: Dict[str, Any]) -> str:
        """把 {params.xxx} / {tool_result} / {message} / {rag.xxx} 渲染为字符串（阶段 4.1/4.2）。"""
        params = state.get("params") or inputs.get("params") or {}
        last_msg = ""
        msgs = state.get("messages") or []
        if msgs:
            last_msg = msgs[-1].get("content", "") if isinstance(msgs[-1], dict) else str(msgs[-1])
        tool_result = state.get("tool_result")
        if tool_result is None:
            tool_result = inputs.get("tool_result")
        ctx = {
            "params": params_nested(params),
            "tool_result": tool_result,
            "message": last_msg,
            "rag": params_nested(state.get("rag") or {}),
        }
        try:
            return template.format(**ctx)
        except (KeyError, IndexError):
            # 占位缺失：用简单拼接兜底，避免硬失败
            return template

    def _build_messages(self, state, inputs) -> List[Dict[str, str]]:
        # system：引用 Skill 的 SOP（黑盒，不可改写——阶段 5.5）
        system_parts: List[str] = []
        if self._skill and self._skill_registry is not None:
            # 兼容 skills 为 List[SkillDef]（真实 SkillRegistry）或 Dict（测试中 fake）。
            skills = self._skill_registry.skills
            skill_def = None
            if isinstance(skills, dict):
                skill_def = skills.get(self._skill)
            else:
                for s in skills or []:
                    if getattr(s, "name", None) == self._skill:
                        skill_def = s
                        break
            if skill_def is not None:
                # SOP 正文：真实 SkillDef 在 body；fake 可能提供 sop
                sop = getattr(skill_def, "sop", None) or getattr(skill_def, "body", None) \
                    or getattr(skill_def, "description", None)
                if sop:
                    system_parts.append(f"[Skill {self._skill} SOP]\n{sop}")
        system_parts.append(
            "你是电商客服助手，请基于给定上下文用简洁中文回答。不要编造订单/退款信息。"
        )

        if self._prompt_template:
            user_text = self._render(self._prompt_template, state, inputs)
        else:
            msgs = state.get("messages") or []
            user_text = msgs[-1].get("content", "") if msgs and isinstance(msgs[-1], dict) else ""

        messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
        if user_text:
            messages.append({"role": "user", "content": user_text})
        return messages

    async def run(self, state, inputs):
        if self._llm is None:
            raise ValueError("llm_call 节点需要 llm_service，请在 FlowCompiler 注入")
        messages = self._build_messages(state, inputs)
        logger.info("llm_call 执行", skill=self._skill, n_messages=len(messages))
        text = await self._llm.chat_qwen(messages)
        return {"result": text}


class RagPipelineHandler(NodeHandler):
    """RAG 固定四步节点（2.4.1）：接入现有 GeneralAgentExecutor / Pipeline。

    v1 仍走 executor.execute 的黑盒四步（改写→审查→检索→生成评估），
    v2 阶段 4.2 再拆成 YAML 子图节点。此处等价于现状 _chat_with_rag_agent。
    """

    needs_llm = True

    def __init__(
        self,
        *,
        llm_service=None,
        embedding_service=None,
        milvus_service=None,
        redis_cache_service=None,
        skill: Optional[str] = None,
        skill_registry=None,
    ):
        self._llm = llm_service
        self._embedding = embedding_service
        self._milvus = milvus_service
        self._redis = redis_cache_service
        self._skill = skill
        self._skill_registry = skill_registry

    async def run(self, state, inputs):
        from src.modules.chat.agent.executor import GeneralAgentExecutor
        from src.modules.chat.schemas import ChatRequest

        # fail-closed：RAG 必须依赖向量检索底座
        if self._embedding is None or self._milvus is None:
            raise ValueError(
                "rag_pipeline handler 需要 embedding_service 与 milvus_service，"
                "请在 FlowCompiler 注入（缺失会导致无检索能力的盲生成）"
            )

        message = state.get("messages", [{}])[-1].get("content", "") if state.get("messages") else ""
        domain = state.get("intent", {}).get("domain", "ecommerce")
        request = ChatRequest(
            message=message,
            conversation_id=state.get("thread_id", ""),
            domain=domain,
        )
        executor = GeneralAgentExecutor(
            domain=domain,
            llm_service=self._llm,
            embedding_service=self._embedding,
            milvus_service=self._milvus,
            redis_cache_service=self._redis,
        )
        response = await executor.execute(request, langfuse_handler=None)
        return {"result": response.message}


class DisputeHandler(NodeHandler):
    """纠纷协调节点（2.4.4）：接入现有 DisputeCoordinator。

    分布式锁语义（现状在 orchestrator_remote.try_dispute_flow 内实现）不在 v1 handler
    内复刻——锁与流程层安全边界固化留待阶段 5，避免与 orchestrator 双写锁逻辑。
    """

    needs_llm = True
    needs_tool_service = True

    def __init__(
        self,
        *,
        llm_service=None,
        tool_service=None,
        redis_cache_service=None,
        skill: Optional[str] = None,
        skill_registry=None,
    ):
        self._llm = llm_service
        self._tool_service = tool_service
        self._redis = redis_cache_service
        self._skill = skill
        self._skill_registry = skill_registry

    async def run(self, state, inputs):
        from src.modules.chat.agent.dispute_coordinator import DisputeCoordinator
        from src.modules.chat.schemas import ChatRequest

        if self._llm is None or self._tool_service is None:
            raise ValueError(
                "dispute handler 需要 llm_service 与 tool_service，请在 FlowCompiler 注入"
            )

        message = state.get("messages", [{}])[-1].get("content", "") if state.get("messages") else ""
        domain = state.get("intent", {}).get("domain", "ecommerce")
        order_id = (inputs.get("params") or state.get("params") or {}).get("order_id")
        request = ChatRequest(
            message=message,
            conversation_id=state.get("thread_id", ""),
            domain=domain,
        )
        coordinator = DisputeCoordinator(
            llm=self._llm,
            tool_service=self._tool_service,
            domain=domain,
        )
        response = await coordinator.resolve(
            request,
            emotion_result=None,
            conversation_id=state.get("thread_id", ""),
            domain=domain,
            order_id=order_id,
            langfuse_handler=None,
        )
        return {"result": response.message}


# ═══════════════════════════════════════════════════════════════════════
# 人在回路节点（阶段 3.1 / 3.4 / 3.5）
# ═══════════════════════════════════════════════════════════════════════

class _ApprovalCommand:
    """human_approval 节点专用的轻量命令：仅登记 pending 审批记录 + 提供 undo。

    真实副作用（如退款写库）由 ApprovalGate.approve / reject 在审批后执行
    （复用 tool_commands 里已注册的 RefundCommand 等业务命令）。本命令仅负责
    「挂起前登记 + 拒绝时撤销登记」。
    """

    def __init__(self, command_name: str):
        self.command_name = command_name

    def requires_approval(self) -> bool:
        return True

    async def execute(self, ctx) -> "ToolResult":
        from src.modules.chat.agent.tool_commands import ToolResult
        return ToolResult(
            status="pending_approval",
            message=f"待人工审批：{self.command_name}",
            undo_data={"command_name": self.command_name, "params": ctx.params},
        )

    async def undo(self, ctx) -> "ToolResult":
        from src.modules.chat.agent.tool_commands import ToolResult
        return ToolResult(
            status="success",
            message=f"{self.command_name} 的审批申请已取消。",
        )


class HumanApprovalHandler(NodeHandler):
    """人在回路节点：LangGraph interrupt 真挂起 + ApprovalGate 审批双分支（阶段 3）。

    执行流：
      1. 挂起前：用 ApprovalGate.execute_with_approval 登记 pending 审批记录，
         拿到 approval_id（形如 approval:xxxx），存入 state 供跨请求恢复。
      2. 对展示给用户的「待审批摘要」做输出安全过滤（阶段 3.5 前）。
      3. 调用 LangGraph ``interrupt(payload)`` 挂起，state 写入 checkpointer。
      4. resume 时 ``interrupt`` 返回 confirm(bool)：
         - True  → ApprovalGate.approve(approval_id)（阶段 3.4 通过分支）
         - False → ApprovalGate.reject(approval_id)（阶段 3.4 拒绝分支）
      5. 对审批结果 message 做输出安全过滤后返回（阶段 3.5 后，fail-closed）。
    """

    needs_tool_service = True

    def __init__(
        self,
        approval_type: str = "confirm",
        *,
        tool_service=None,
        approval_store=None,
        execution_store=None,
    ):
        self._approval_type = approval_type
        self._tool_service = tool_service
        self._approval_store = approval_store
        self._execution_store = execution_store

    async def run(self, state, inputs):
        from langgraph.types import interrupt

        from src.modules.chat.agent.tool_commands import ApprovalGate, ToolContext
        from src.modules.chat.core.content_filter import ContentFilterService

        if self._tool_service is None:
            raise ValueError("human_approval 节点需要 tool_service（运行期缺失）")

        params = dict(inputs.get("params", {}) or {})
        order_id = params.get("order_id")
        conversation_id = state.get("thread_id", "")
        domain = state.get("intent", {}).get("domain", "ecommerce")

        # ── 1. 登记 pending 审批记录（PostgreSQL / Redis / 内存）──
        command_name = self._approval_type or state.get("intent", {}).get("action") or "manual_approval"
        command = _ApprovalCommand(command_name)
        ctx = ToolContext(
            action=command_name,
            params=params,
            conversation_id=conversation_id,
            domain=domain,
        )
        gate = ApprovalGate(approval_store=self._approval_store)
        pending = await gate.execute_with_approval(command, ctx)
        approval_id = pending.approval_id
        if not approval_id:
            raise RuntimeError("human_approval 创建审批记录失败（approval_id 为空）")

        # ── 1.5 写入执行事件（审批创建）──
        if self._execution_store is not None:
            await self._execution_store.append_event(
                thread_id=conversation_id,
                event_type="approval_created",
                payload={
                    "approval_id": approval_id,
                    "command_name": command_name,
                    "action": command_name,
                    "params_masked": params,
                    "status": "pending_approval",
                },
                node_name="human_approval",
            )

        # ── 2. 挂起前输出安全过滤（阶段 3.5 前）──
        cf = ContentFilterService.get_instance()
        summary = pending.message or f"待审批：{command_name}"
        pre_check = cf.filter_output(summary, domain)
        if pre_check.filtered_text is not None:
            summary = pre_check.filtered_text
        elif not pre_check.is_safe:
            summary = "您有一项操作待人工审批，详情暂无法展示。"

        # ── 3. LangGraph interrupt 挂起（state 已写入 checkpointer）──
        payload = {
            "approval_id": approval_id,
            "type": self._approval_type,
            "order_id": order_id,
            "summary": summary,
        }
        confirm = interrupt(payload)  # ← 此处挂起，resume 时返回 confirm(bool)

        # ── 4. 审批双分支（阶段 3.4）──
        if confirm:
            result = await gate.approve(approval_id)
            event_type = "approval_approved"
        else:
            result = await gate.reject(approval_id)
            event_type = "approval_rejected"

        if self._execution_store is not None:
            await self._execution_store.append_event(
                thread_id=conversation_id,
                event_type=event_type,
                payload={
                    "approval_id": approval_id,
                    "confirm": confirm,
                    "result_status": result.status,
                },
                node_name="human_approval",
            )

        # ── 5. 审批后输出安全过滤（阶段 3.5 后，fail-closed）──
        message = result.message or ("操作已执行。" if confirm else "操作已取消。")
        post_check = cf.filter_output(message, domain)
        if post_check.filtered_text is not None:
            message = post_check.filtered_text
        elif not post_check.is_safe:
            message = "操作已执行，但结果包含无法展示的内容。" if confirm else "操作已取消。"

        logger.info(
            "human_approval 审批完成",
            approval_id=approval_id,
            confirm=bool(confirm),
            status=result.status,
        )
        # 审批结论写回 intent.action（approve/reject），供下游条件边（on_intent==）
        # 路由到 execute_return / reply。原 intent.action（如 request-return）在此
        # 被审批结论覆盖，不影响 direct_tool 节点（其 skill 来自 config 而非 intent）。
        intent = dict(state.get("intent") or {})
        intent["action"] = "approve" if confirm else "reject"
        return {
            "result": message,
            "hitl_pending": None,
            "approval_id": approval_id,
            "intent": intent,
        }


# ═══════════════════════════════════════════════════════════════════════
# 工厂（build_handler）
# ═══════════════════════════════════════════════════════════════════════

def build_handler(
    node_type: str,
    config,
    *,
    llm_service=None,
    tool_service=None,
    embedding_service=None,
    milvus_service=None,
    redis_cache_service=None,
    skill_registry=None,
    approval_store=None,
    execution_store=None,
):
    """节点类型 → handler 实例（阶段 2.4 / 2.5 / 2.6）。

    ``config`` 为 schema.NodeConfig，提供 skill / hardcode / approval_type 等字段。
    """
    cfg = config or {}
    skill = getattr(cfg, "skill", None) or (cfg.get("skill") if isinstance(cfg, dict) else None)
    hardcode = getattr(cfg, "hardcode", None) or (cfg.get("hardcode") if isinstance(cfg, dict) else None)
    approval_type = getattr(cfg, "approval_type", None) or (cfg.get("approval_type") if isinstance(cfg, dict) else None)

    if node_type == "normalize":
        return NormalizeHandler()
    if node_type == "truncate":
        return NormalizeHandler()  # v1 截断复用归一化占位
    if node_type == "sentiment":
        return NormalizeHandler()  # v1 情绪复用占位（真实接入 SentimentService）
    if node_type == "intent":
        return IntentHandler()
    if node_type == "direct_tool":
        # 编译期不强制 tool_service（图可构建）；运行期 run 再 fail-closed 校验
        # hardcode 声明透传：运行期 run 应用闭包预设（阶段 5.1）
        return DirectToolHandler(tool_service, skill=skill, hardcode=hardcode)
    if node_type == "react":
        # 编译期不强制依赖（图可构建）；运行期 run 再 fail-closed 校验
        return ReactHandler(llm_service, tool_service, skill=skill, hardcode=hardcode)
    if node_type == "llm_call":
        # 编译期不强制 llm_service（图可构建）；运行期 run 再 fail-closed 校验
        prompt_template = getattr(cfg, "prompt_template", None) or (cfg.get("prompt_template") if isinstance(cfg, dict) else None)
        return LLMCallHandler(
            llm_service,
            skill=skill,
            prompt_template=prompt_template,
            skill_registry=skill_registry,
        )
    if node_type == "rag_pipeline":
        if embedding_service is None or milvus_service is None:
            raise ValueError(
                "rag_pipeline 节点需要 embedding_service 与 milvus_service，"
                "请在 FlowCompiler 注入（缺失会导致无检索能力的盲生成）"
            )
        return RagPipelineHandler(
            llm_service=llm_service,
            embedding_service=embedding_service,
            milvus_service=milvus_service,
            redis_cache_service=redis_cache_service,
            skill=skill,
            skill_registry=skill_registry,
        )
    # ── 阶段 4.2：RAG 固定四步拆节点 ──
    domain = getattr(cfg, "rag_domain", None) or (
        cfg.get("rag_domain") if isinstance(cfg, dict) else None
    )
    if node_type == "rag_rewrite":
        # 编译期不强制 llm_service（图可构建）；运行期 run 再 fail-closed 校验
        return RagRewriteHandler(
            llm_service=llm_service,
            embedding_service=embedding_service,
            milvus_service=milvus_service,
            redis_cache_service=redis_cache_service,
            domain=domain or "ecommerce",
        )
    if node_type == "rag_review":
        # 编译期不强制 llm_service（图可构建）；运行期 run 再 fail-closed 校验
        return RagReviewHandler(
            llm_service=llm_service,
            embedding_service=embedding_service,
            milvus_service=milvus_service,
            redis_cache_service=redis_cache_service,
            domain=domain or "ecommerce",
        )
    if node_type == "rag_retrieve":
        if embedding_service is None or milvus_service is None:
            raise ValueError(
                "rag_retrieve 节点需要 embedding_service 与 milvus_service，"
                "请在 FlowCompiler 注入（缺失会导致无检索能力的盲生成）"
            )
        top_k = getattr(cfg, "rag_top_k", None) or (cfg.get("rag_top_k") if isinstance(cfg, dict) else None)
        return RagRetrieveHandler(
            llm_service=llm_service,
            embedding_service=embedding_service,
            milvus_service=milvus_service,
            redis_cache_service=redis_cache_service,
            domain=domain or "ecommerce",
            top_k=top_k,
        )
    if node_type == "rag_generate":
        # 编译期不强制 llm_service（图可构建）；运行期 run 再 fail-closed 校验
        output_filter = getattr(cfg, "rag_output_filter", True)
        if isinstance(cfg, dict):
            output_filter = cfg.get("rag_output_filter", True)
        return RagGenerateHandler(
            llm_service=llm_service,
            embedding_service=embedding_service,
            milvus_service=milvus_service,
            redis_cache_service=redis_cache_service,
            domain=domain or "ecommerce",
            output_filter=output_filter,
        )
    if node_type == "dispute":
        if llm_service is None or tool_service is None:
            raise ValueError(
                "dispute 节点需要 llm_service 与 tool_service，请在 FlowCompiler 注入"
            )
        return DisputeHandler(
            llm_service=llm_service,
            tool_service=tool_service,
            redis_cache_service=redis_cache_service,
            skill=skill,
            skill_registry=skill_registry,
        )
    if node_type == "human_approval":
        return HumanApprovalHandler(
            approval_type=approval_type or "confirm",
            tool_service=tool_service,
            approval_store=approval_store,
            execution_store=execution_store,
        )
    if node_type in ("input_filter", "output_filter", "lock", "persist", "observe"):
        return GuardHandler(node_type)

    logger.warning(f"未知节点类型 '{node_type}'，回落 GuardHandler 占位")
    return GuardHandler(node_type)
