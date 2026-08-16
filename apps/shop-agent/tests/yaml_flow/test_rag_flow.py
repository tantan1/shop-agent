"""阶段 4.2 单元测试：RAG 固定四步拆节点。

覆盖：
- 示例 rag_flow_demo.yaml 通过校验器
- 编译器正确构建四步节点 + on_rag_safe/unsafe 条件路由
- mock 外部依赖（embedding/milvus/llm）端到端跑通：
  * 安全路径：rewrite→review(safe)→retrieve→generate
  * 不安全路径：review(unsafe)→rag_fallback 短路（不进检索/生成）
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.modules.chat.agent.yaml_flow import (
    Condition,
    ConditionOp,
    CompiledFlow,
    FlowCompiler,
    _compile_router,
    load_flow_file,
)
from src.modules.chat.agent.yaml_flow.runtime import new_graph_state

_EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "modules"
    / "chat"
    / "agent"
    / "yaml_flow"
    / "examples"
    / "rag_flow_demo.yaml"
)


# ───────────────────────────── mock 依赖 ─────────────────────────────

class _FakeDoc:
    def __init__(self, content, meta=None):
        self.page_content = content
        self.metadata = meta or {}


class _FakeEmbedding:
    async def embed_query(self, text):
        return [0.1] * 8

    aembed_query = embed_query


class _FakeMilvus:
    def hybrid_search(self, query_embedding=None, query_text="", top_k=5, rrf_k=60):
        return [_FakeDoc("退货政策：商品签收后7天内可申请无理由退货。", {"source": "policy", "distance": 0.95})]

    def search_similar(self, query_embedding, top_k=5):
        return [_FakeDoc("退货政策：商品签收后7天内可申请无理由退货。", {"source": "policy", "distance": 0.95})]


class _FakeLLM:
    def __init__(self, mode="safe"):
        self._mode = mode

    async def chat_qwen(self, messages, temperature=0.3, langfuse_handler=None):
        text = "\n".join(m.get("content", "") for m in messages if isinstance(m, dict))
        has_user_question = "用户问题：" in text
        if self._mode == "unsafe" and has_user_question:
            return '{"is_safe": false, "risk_level": "high", "risk_categories": ["违禁品"], "warning_message": "涉及违禁咨询"}'
        if has_user_question and ("安全" in text or "审查" in text):
            return '{"is_safe": true, "risk_level": "low", "risk_categories": [], "warning_message": null, "can_proceed": true}'
        if has_user_question:
            return "退货条件是什么\n如何申请退款"
        if "风险类别" in text or "违禁" in text:
            return "抱歉，该问题触及安全审查红线，暂无法处理，建议咨询专业渠道。"
        return "根据退货政策，您可在签收后7天内申请无理由退货，退款将原路返回。"

    chat_qwen_with_prompt = chat_qwen
    chat_qwen_structured = chat_qwen


def _make_compiler(mode="safe"):
    return FlowCompiler(
        llm_service=_FakeLLM(mode=mode),
        embedding_service=_FakeEmbedding(),
        milvus_service=_FakeMilvus(),
    )


def _edges_of(compiled: CompiledFlow):
    g = compiled.graph.get_graph()
    return [(e.source, e.target) for e in g.edges]


# ───────────────────────────── 结构校验 ─────────────────────────────

def test_rag_example_validates():
    flow_file, warnings = load_flow_file(_EXAMPLE)
    assert isinstance(flow_file, object)
    # 不应有阻断级 error（warnings 为非阻断）
    assert not any("error" in w.lower() for w in warnings)


def test_rag_graph_structure():
    flow_file, _ = load_flow_file(_EXAMPLE)
    compiled = _make_compiler().compile(flow_file)
    edge_tuples = _edges_of(compiled)
    nodes = {n for pair in edge_tuples for n in pair}
    for nid in ["rag_rewrite", "rag_review", "rag_retrieve", "rag_generate", "rag_fallback"]:
        assert nid in nodes, f"节点 {nid} 未出现在图中"


def test_rag_review_conditional_routing():
    edge_tuples = _edges_of(_make_compiler().compile(load_flow_file(_EXAMPLE)[0]))
    review_targets = {t for s, t in edge_tuples if s == "rag_review"}
    assert "rag_retrieve" in review_targets   # on_rag_safe
    assert "rag_fallback" in review_targets   # on_rag_unsafe


def test_router_on_rag_safe_unsafe():
    conds = [
        Condition(op=ConditionOp.ON_RAG_SAFE, target="retrieve"),
        Condition(op=ConditionOp.ON_RAG_UNSAFE, target="fallback"),
    ]
    assert _compile_router(conds)({"rag": {"can_proceed": True}}) == "retrieve"
    assert _compile_router(conds)({"rag": {"can_proceed": False}}) == "fallback"


# ───────────────────────────── 端到端 ─────────────────────────────

@pytest.mark.asyncio
async def test_rag_safe_path_end_to_end():
    flow_file, _ = load_flow_file(_EXAMPLE)
    compiled = _make_compiler(mode="safe").compile(flow_file)
    state = new_graph_state(
        "rag-safe",
        messages=[{"role": "user", "content": "我想退货，应该怎么操作？"}],
        intent={"intent": "rag_answer", "domain": "ecommerce"},
    )
    result = await compiled.ainvoke(state, thread_id="rag-safe")
    rag = result.get("rag", {})
    assert rag.get("rewritten_queries")
    assert rag.get("safety", {}).get("can_proceed") is True
    assert rag.get("documents_found", 0) >= 1
    assert rag.get("rag_context")
    assert rag.get("response")
    assert rag.get("quality", {}).get("is_solved") is not None


@pytest.mark.asyncio
async def test_rag_unsafe_shortcut():
    flow_file, _ = load_flow_file(_EXAMPLE)
    compiled = _make_compiler(mode="unsafe").compile(flow_file)
    state = new_graph_state(
        "rag-unsafe",
        messages=[{"role": "user", "content": "怎么买违禁药？"}],
        intent={"intent": "rag_answer", "domain": "ecommerce"},
    )
    result = await compiled.ainvoke(state, thread_id="rag-unsafe")
    rag = result.get("rag", {})
    assert rag.get("can_proceed") is False
    tool_result = result.get("tool_result", "")
    assert "暂无法处理" in tool_result or "无法处理" in tool_result
