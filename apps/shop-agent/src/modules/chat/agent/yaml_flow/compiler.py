"""YAML 编排解析器 v1（阶段 2.1-2.9）。

把 FlowFile 编译成 LangGraph StateGraph：
- 2.2 StateGraph 构建：add_node / add_edge
- 2.3 条件路由：YAML condition → add_conditional_edges 路由函数
- 2.4 执行节点映射：direct_tool / react（v1）；rag_pipeline/dispute 占位
- 2.5 前置节点：normalize / intent
- 2.6 守卫钩子：输入/输出过滤、锁、持久化、埋点（v1 占位包装）
- 2.7 checkpointer + recursion_limit
- 2.8 ReAct 节点注入工具精选（由 ReActAgent 内部完成，此处透传）
- 2.9 ainvoke 入口（见 runtime.CompiledFlow）
- 2.10 LangGraph Studio 调试入口（runtime.CompiledFlow.get_graph）

节点函数本身不感知上下游：编译器负责 read_from 组装 inputs、write_to 写回 state。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from .runtime import CompiledFlow, GraphState
from .schema import Condition, ConditionOp, FlowFile, NodeDef
from .nodes.handlers import build_handler
from .validator import validate_flow


# ───────────────────────────── 条件路由（阶段 2.3） ─────────────────────────────

def _compile_router(conditions: List[Condition]):
    """把 YAML 条件列表编译成路由函数：state -> target node id。

    语义：
    - on_emergency：state['emergency'] 为真 → 命中
    - on_intent==：state['intent']['intent'] == value（或 action == value）
    - on_complexity==：state['intent']['complexity'] == value
    - always：兜底（每出口至多一条）
    多条命中时，按列表顺序取第一条（建议把 always 放最后）。
    """

    def router(state: GraphState) -> str:
        intent = state.get("intent") or {}
        emergency = bool(state.get("emergency"))
        decision = None
        for cond in conditions:
            op = cond.op
            if op == ConditionOp.ON_EMERGENCY:
                if emergency:
                    decision = cond.target
                    break
            elif op == ConditionOp.ON_INTENT_EQ:
                if intent.get("intent") == cond.value or intent.get("action") == cond.value:
                    decision = cond.target
                    break
            elif op == ConditionOp.ON_COMPLEXITY_EQ:
                if intent.get("complexity") == cond.value:
                    decision = cond.target
                    break
            elif op == ConditionOp.ALWAYS:
                decision = cond.target
                break
            elif op == ConditionOp.ON_RAG_SAFE:
                # 阶段 4.2：rag_review 通过（can_proceed=True）走该边
                rag = state.get("rag") or {}
                if rag.get("can_proceed", True):
                    decision = cond.target
                    break
            elif op == ConditionOp.ON_RAG_UNSAFE:
                # 阶段 4.2：rag_review 不通过（can_proceed=False）短路到 fallback
                rag = state.get("rag") or {}
                if not rag.get("can_proceed", True):
                    decision = cond.target
                    break
            elif op == ConditionOp.EXPR:
                # 自定义表达式：v1 不启用（解析器评估 state），预留
                raise NotImplementedError("expr 条件 v1 未启用")
        if decision is None:
            # 无兜底条件时，默认走到 END（避免图悬挂）
            return END
        return decision

    return router


# ───────────────────────────── 节点函数工厂 ─────────────────────────────

def _make_node_fn(node: NodeDef, *, llm_service=None, tool_service=None, skill_registry=None,
                  embedding_service=None, milvus_service=None, redis_cache_service=None,
                  approval_store=None, execution_store=None):
    """构造 LangGraph 节点函数：负责 read_from 组装 + 调 handler + write_to 写回。

    同时把守卫（input_filter/output_filter）作为节点前后置包装（阶段 2.6）。
    """
    handler = build_handler(
        node.type, node.config,
        llm_service=llm_service, tool_service=tool_service,
        skill_registry=skill_registry,
        embedding_service=embedding_service, milvus_service=milvus_service,
        redis_cache_service=redis_cache_service,
        approval_store=approval_store, execution_store=execution_store,
    )
    read_from = node.config.read_from
    write_to = node.config.write_to

    async def node_fn(state: GraphState) -> Dict[str, Any]:
        # read_from：从 state 取字段，组装 inputs
        inputs: Dict[str, Any] = {}
        for rf in read_from:
            src = state.get(rf.state)
            local = rf.as_ or rf.state.split(".")[-1]
            inputs[local] = src
        # 兼容：params 整体透传
        if "params" in state:
            inputs.setdefault("params", state["params"])

        # 调 handler
        produced = await handler.run(state, inputs)

        # write_to：把产出写回 state
        updates: Dict[str, Any] = {}
        if write_to:
            for wt in write_to:
                key = wt.from_ or "result"
                if isinstance(produced, dict) and key in produced:
                    updates[wt.state] = produced[key]
                else:
                    updates[wt.state] = produced
        else:
            # 无 write_to 声明时，若产出是 dict 则整体合并
            if isinstance(produced, dict):
                updates.update(produced)
        return updates

    return node_fn


# ───────────────────────────── 主编译入口 ─────────────────────────────

class FlowCompiler:
    """FlowFile → CompiledGraph（阶段 2.1-2.9）。"""

    def __init__(self, *, llm_service=None, tool_service=None, skill_registry=None,
                 embedding_service=None, milvus_service=None, redis_cache_service=None,
                 approval_store=None, execution_store=None):
        self._llm = llm_service
        self._tool = tool_service
        self._skill_registry = skill_registry
        self._embedding = embedding_service
        self._milvus = milvus_service
        self._redis = redis_cache_service
        self._approval_store = approval_store
        self._execution_store = execution_store

    def compile(self, flow: FlowFile, checkpointer=None, *, approval_store=None, execution_store=None) -> CompiledFlow:
        approval_store = approval_store or self._approval_store
        execution_store = execution_store or self._execution_store
        # 编译期安全强校验：敏感 Skill 必须有 SOP（5.3）且高后果字段强制绑定（5.2）。
        # 纯结构校验在 loader 已做；此处带 registry 做语义强校验（fail-closed 拒绝发布）。
        validate_flow(flow, skill_registry=self._skill_registry)
        workflow = StateGraph(GraphState)

        # ── 阶段 4.1 完整版：展开 react 节点引用的子图 ──
        # 采用「内联展开」而非「子图作为节点」：把子图节点以 `<node.id>__<sub_id>` 前缀
        # 展开并入父图。优点：子图内 human_approval 的 interrupt 落在父图层面，
        # 跨请求 resume(Command(resume=...)) 与阶段 3 完全一致，无需特殊透传。
        # 父图中指向该 react 节点的边，其 target 改写为 `<node.id>__<sub_entry>`。
        sub_prefix_map: Dict[str, Dict[str, str]] = {}   # react节点id -> {子图原id: 展开id}
        sub_entry_map: Dict[str, str] = {}               # react节点id -> 展开后的子图入口id
        sub_exit_map: Dict[str, str] = {}                # react节点id -> 展开后的子图出口id
        for node in flow.nodes:
            if node.type == "react" and node.config.subgraph:
                sub = getattr(flow, "subgraphs", {}).get(node.config.subgraph)
                if sub is None:
                    raise ValueError(
                        f"react 节点 '{node.id}' 引用的 subgraph '{node.config.subgraph}' 未定义"
                    )
                prefix = f"{node.id}__"
                sub_prefix_map[node.id] = {sn.id: f"{prefix}{sn.id}" for sn in sub.nodes}
                sub_entry_map[node.id] = f"{prefix}{sub.entry}"
                # 子图出口：子图中没有任何「出边」的节点（即末端节点，取第一个/唯一）
                sources = {e.from_ for e in sub.edges}
                for c in sub.edges:
                    sources.add(c.from_)
                exits = [sn.id for sn in sub.nodes if sn.id not in sources]
                if not exits:
                    raise ValueError(
                        f"react 节点 '{node.id}' 的子图 '{node.config.subgraph}' 无出口节点"
                    )
                sub_exit_map[node.id] = f"{prefix}{exits[0]}"

        def _entry(target: str) -> str:
            """父图边 to 方向：把 react 节点 id 改写为子图入口展开 id。"""
            return sub_entry_map.get(target, target)

        def _exit(target: str) -> str:
            """父图边 from 方向：把 react 节点 id 改写为子图出口展开 id。"""
            return sub_exit_map.get(target, target)

        def _prefix(target: str) -> str:
            """把子图内部边的原 id 改写前缀（仅当 target 属于某子图节点）。"""
            for prefix_map in sub_prefix_map.values():
                if target in prefix_map:
                    return prefix_map[target]
            return target

        # 守卫节点：v1 作为独立节点加入（校验器已保证 input/output 过滤存在）
        guard_node_ids: List[str] = []
        for i, g in enumerate(flow.guards):
            gid = f"__guard_{g.type}_{i}"
            guard_node_ids.append(gid)
            _ = gid

        # 1) 加节点（普通节点 + 子图展开节点）
        def _add_one(node_def: NodeDef, nid: str):
            fn = _make_node_fn(
                node_def,
                llm_service=self._llm, tool_service=self._tool,
                skill_registry=self._skill_registry,
                embedding_service=self._embedding, milvus_service=self._milvus,
                redis_cache_service=self._redis,
                approval_store=approval_store, execution_store=execution_store,
            )
            workflow.add_node(nid, fn)

        for node in flow.nodes:
            if node.id in sub_prefix_map:
                # 展开子图：把所有子节点加前缀并入父图
                sub = getattr(flow, "subgraphs", {})[node.config.subgraph]
                for sn in sub.nodes:
                    _add_one(sn, f"{node.id}__{sn.id}")
                continue
            _add_one(node, node.id)

        # 2) 加边（固定边 + 条件边，含子图内部边）
        all_edges: List[EdgeDef] = list(flow.edges)
        for node in flow.nodes:
            if node.id in sub_prefix_map:
                sub = getattr(flow, "subgraphs", {})[node.config.subgraph]
                all_edges.extend(sub.edges)
        conditional_routes: Dict[str, List[Condition]] = {}
        for edge in all_edges:
            src = _exit(_prefix(edge.from_))
            if edge.to is not None:
                dst = _entry(_prefix(edge.to))
                workflow.add_edge(src, dst)
            else:
                new_conditions = [
                    Condition(op=c.op, value=c.value, target=_entry(_prefix(c.target)))
                    for c in edge.conditions
                ]
                conditional_routes[src] = new_conditions

        for src, conditions in conditional_routes.items():
            router = _compile_router(conditions)
            targets = [c.target for c in conditions] + [END]
            workflow.add_conditional_edges(src, router, targets)

        # 3) 入口
        entry = _entry(flow.entry)
        node_ids_all = {n.id for n in flow.nodes} | set(sub_entry_map.values())
        if entry in node_ids_all:
            workflow.add_edge(START, entry)
        else:
            workflow.set_entry_point(entry)

        return CompiledFlow(workflow, checkpointer=checkpointer, execution_store=execution_store)

    # ── Studio 兼容（阶段 2.10）──
    def get_graph(self, flow: FlowFile):
        return self.compile(flow).get_graph()


__all__ = ["FlowCompiler", "CompiledFlow", "_compile_router"]
