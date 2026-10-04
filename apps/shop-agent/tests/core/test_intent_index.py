"""意图索引一致性测试（关注点分离重构 · 第 1 步）。

核心断言：意图索引的示例**唯一数据源是 SkillRegistry**（即各 SKILL.md 的
``examples``），而不是代码里硬编码的常量。改 SKILL.md 即生效，
杜绝「改了 SKILL.md 但意图识别不跟随」的双份事实源问题。
"""

import asyncio

import pytest

from src.modules.chat.core.intent.index_builder import (
    build_index,
    ensure_intent_index_sync,
    examples_from_registry,
    flatten_examples,
    reset_intent_index_cache,
)


# ── 测试替身 ──


class _FakeSkill:
    def __init__(self, name: str, examples):
        self.name = name
        self.examples = examples


class _FakeRegistry:
    def __init__(self, skills):
        self.skills = skills


class _FakeEmbeddings:
    """按文本长度产出确定性 2 维向量，避免测试依赖真实 BGE 模型。"""

    def embed_documents(self, texts):
        return [[float(len(t)), 1.0] for t in texts]


class _FakeEmbeddingService:
    def get_embeddings(self):
        return _FakeEmbeddings()


@pytest.fixture(autouse=True)
def _clean_index_cache():
    reset_intent_index_cache()
    yield
    reset_intent_index_cache()


# ── 示例抽取 ──


def test_examples_from_registry_reads_skill_examples():
    reg = _FakeRegistry(
        [
            _FakeSkill("query-order", ["查订单", "我的订单呢"]),
            _FakeSkill("check-balance", ["余额多少"]),
            _FakeSkill("no-example", []),  # 无示例的 skill 不应进入索引
        ]
    )
    assert examples_from_registry(reg) == {
        "query-order": ["查订单", "我的订单呢"],
        "check-balance": ["余额多少"],
    }


def test_examples_from_registry_tolerates_empty_registry():
    assert examples_from_registry(None) == {}
    assert examples_from_registry(_FakeRegistry([])) == {}


# ── 索引构建 ──


def test_flatten_examples_aligns_actions_with_rows():
    actions, phrases = flatten_examples(
        {"query-order": ["a", "b"], "check-balance": ["c"]}
    )
    assert actions == ["query-order", "query-order", "check-balance"]
    assert phrases == ["a", "b", "c"]


def test_build_index_maps_row_to_action_and_example():
    examples = {"query-order": ["a", "b"], "check-balance": ["c"]}
    actions, phrases = flatten_examples(examples)
    vecs = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]

    idx = build_index(vecs, actions, phrases)

    assert idx is not None
    assert idx.dim == 2
    assert idx.actions == actions
    assert idx.examples == phrases

    top = idx.search([1.0, 0.0], k=1)
    assert len(top) == 1
    assert top[0].action == "query-order"
    assert top[0].matched == "a"


def test_build_index_rejects_mismatched_rows():
    assert build_index([[1.0, 0.0]], ["a", "b"], ["x", "y"]) is None


# ── 一致性：索引跟随 SkillRegistry（本重构的核心保证）──


def test_index_follows_registry_examples():
    """修改 SKILL.md 的 examples（此处用替身模拟）后，重建的索引必须跟随。"""
    reg = _FakeRegistry([_FakeSkill("query-order", ["查订单"])])
    idx = ensure_intent_index_sync(
        _FakeEmbeddingService(), lambda: examples_from_registry(reg)
    )
    assert idx is not None
    assert set(idx.actions) == {"query-order"}

    # 模拟「给某个 SKILL.md 新增示例 / 新增一个 skill」
    reset_intent_index_cache()
    reg2 = _FakeRegistry(
        [
            _FakeSkill("query-order", ["查订单"]),
            _FakeSkill("coupon-inquiry", ["我的券"]),
        ]
    )
    idx2 = ensure_intent_index_sync(
        _FakeEmbeddingService(), lambda: examples_from_registry(reg2)
    )
    assert idx2 is not None
    assert set(idx2.actions) == {"query-order", "coupon-inquiry"}


def test_index_is_reused_until_cache_reset():
    provider = lambda: examples_from_registry(  # noqa: E731
        _FakeRegistry([_FakeSkill("query-order", ["查订单"])])
    )
    first = ensure_intent_index_sync(_FakeEmbeddingService(), provider)
    second = ensure_intent_index_sync(_FakeEmbeddingService(), provider)
    assert first is second  # 命中缓存，不重复构建


# ── 回归护栏：禁止再把示例硬编码回代码 ──


def test_recognizer_no_longer_hardcodes_intent_examples():
    import src.modules.chat.core.intent_recognizer as ir

    assert not hasattr(ir, "INTENT_EXAMPLES"), (
        "意图示例必须来自 SkillRegistry（SKILL.md 的 examples），"
        "不得在 intent_recognizer 里重新硬编码一份"
    )


# ── 端到端冒烟：recognize() 经新分类器仍能正确产出 IntentResult ──


class _AsyncFakeEmbeddings:
    async def aembed_documents(self, texts):
        return [[float(len(t)), 1.0] for t in texts]

    async def aembed_query(self, text):
        return [float(len(text)), 1.0]


class _AsyncFakeEmbeddingService:
    def get_embeddings(self):
        return _AsyncFakeEmbeddings()


def test_recognize_routes_via_registry_examples():
    from src.modules.chat.core.intent_recognizer import IntentRecognizer

    reg = _FakeRegistry(
        [
            _FakeSkill("query-order", ["我的订单到哪了", "查一下订单"]),
            _FakeSkill("request-return", ["我要退货"]),
        ]
    )
    recognizer = IntentRecognizer(
        embedding_service=_AsyncFakeEmbeddingService(), skill_registry=reg
    )

    # 否定词（政策咨询）→ 走 RAG
    neg_res = asyncio.run(recognizer.recognize("退货政策是什么"))
    assert neg_res.mode == "rag_pipeline"

    # 命中 registry 示例 → 走工具，且 action 来自 SKILL.md 的示例所属意图
    hit_res = asyncio.run(recognizer.recognize("查一下订单"))
    assert hit_res.mode == "direct_tool"
    assert hit_res.action == "query-order"
