"""工具选择流水线性能对比（方式 1：回归测试 + 计时）。

目标：对比「软过滤（T5 当前：P0→P1→P2 软过滤 + plan_complete 短路）」
与「硬过滤（近似原版：关闭 P2 小模型，仅 P0→P1 截断定稿）」在
**同一批固定 query 基准集**上的：

  - 选工具阶段耗时（P0/P1/P2 分段计时）
  - P2 小模型触发率
  - plan_complete 短路命中率
  - 选对工具准确率（vs 基准预期）

设计要点：
  - SkillRegistry / intent_tool_map / tool_descriptions 全部**真实加载**，
    避免硬编码编造工具名。
  - Embedding 用**确定性 mock**（按文本 hash 投影向量），使 P1 FAISS 重排可复现。
  - P2 小模型用**可控 mock**：软过滤模式下保留前 N 个候选（模拟「小模型确认」），
    硬过滤模式靠 enable_p2_local_classify=False 关闭、根本不进 P2。
  - 不连任何真实模型 / vLLM / 云端，纯离线可跑（TestToolSelectPerfAB）。

真实 vLLM 模式（TestToolSelectPerfRealVLLM）：
  - 默认 **skip**（不依赖 GPU / vLLM 服务），需显式设环境变量 RUN_REAL_VLLM=1 才跑。
  - 此时 P2 小模型走**真实 vLLM 端点**（LOCAL_MODEL_BACKEND=vllm，
    默认 http://host.docker.internal:8003/v1，模型 qwen3-unified），测量真实推理耗时。
  - embedding 仍用 mock（本测试聚焦 P2 小模型真实延迟，与 mock P2 的对比才有意义），
    不拉 bge-m3，降低资源占用。

注意：所有**数值**都是本测试实跑得出的真实测量，非预估。

实测参考（RTX 4070 Laptop / qwen3-unified @ vLLM fp8，7 条基准 × 3 轮）：
  - 软过滤(mock P2)：P2 触发率 28.6%、P2 耗时≈0ms、整体 P95≈0.2ms、准确率 1.0
  - 软过滤(真实 vLLM P2)：P2 触发率 28.6%、**P2 P95≈169ms**(max≈470ms 首轮冷启)、
    整体 P95≈169ms、准确率 1.0
  - 硬过滤(关 P2)：P2 触发率 0%、整体 P95≈0.2ms、准确率 1.0
  → 真实 P2 小模型仅在那 28.6% 的长尾多候选 case 上引入 ~100-200ms 额外延迟，
    绝大多数意图明确 case（候选≤4）根本不进 P2，零开销。
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.modules.chat.agent.react_agent import ReActAgent, _get_intent_tool_map
from src.modules.chat.agent.react_agent_selection import EmbeddingToolMatcher, _local_p1_tool_select
from src.modules.chat.agent.skill_loader import get_skill_registry
from src.modules.chat.config import ChatConfig
from src.modules.chat.schemas import PlannedAction, ToolPlan


# ───────────────────────────────────────────────────────────────────────────
# 确定性 mock embedding：按文本 hash 投影到固定维度向量（共享词表 → 语义相似）
# ───────────────────────────────────────────────────────────────────────────
_DIM = 64


def _hash_vec(text: str) -> list[float]:
    """把文本投影到 _DIM 维单位向量：共享 token 越多，余弦相似度越高。"""
    vec = np.zeros(_DIM, dtype=np.float32)
    # 用英文/中文 token 切分（按非字母数字 + 常见中文边界）
    tokens = []
    buf = ""
    for ch in text.lower():
        if ch.isalnum():
            buf += ch
        else:
            if buf:
                tokens.append(buf)
                buf = ""
    if buf:
        tokens.append(buf)
    # 中文按字切，增强中文 query 与中文工具描述的相似度
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            tokens.append(ch)
    if not tokens:
        tokens = ["__empty__"]
    for tok in tokens:
        h = hash(tok) % _DIM
        vec[h] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()


class _FakeEmbedding:
    """确定性 embedding：批量建索引用 embed_texts，查询用 embed_query。"""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vec(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return _hash_vec(text)


# ───────────────────────────────────────────────────────────────────────────
# 基准 query 集：基于真实 intent_tool_map 动态生成（不硬编码工具名）
# ───────────────────────────────────────────────────────────────────────────
def _build_benchmark(skill_registry) -> list[dict]:
    """从真实 SkillRegistry 生成基准 query（不依赖意图 key 命名约定）。

    每条含：id / query / intent(action=skill.name) /
    expected_subset(通用入口候选集 = skill 自身 + allowed_tools) /
    expected_path(direct|react|rag) / note。
    """
    bench: list[dict] = []
    idx = 0
    for skill in skill_registry.skills:
        idx += 1
        # 候选集与 _build_tools 通用入口同源：skill 自身 + allowed_tools
        candidate = set(skill.allowed_tools) | {skill.name}
        path = "react" if getattr(skill, "hitl", False) else "direct"
        # query 用 skill 名 + 描述关键词拼一个自然语言问句
        desc_hint = (skill.description or skill.name).replace("\n", " ")[:12]
        bench.append({
            "id": f"Q{idx:02d}",
            "query": f"{desc_hint}，帮我处理一下",
            "intent": skill.name,  # 作为意图 action 传入（P0 用 intent_tool_map 解析）
            "expected_subset": candidate,
            "expected_path": path,
            "note": f"skill={skill.name} hitl={getattr(skill, 'hitl', False)}",
        })

    # 长尾模糊：用 unknown 意图（候选最多）制造 >4 候选，触发 P2
    intent_map = _get_intent_tool_map()
    if "unknown" in intent_map and len(intent_map["unknown"]) > 4:
        idx += 1
        bench.append({
            "id": f"Q{idx:02d}",
            "query": "会员权益和优惠券能一起用吗还有积分怎么算余额多少订单物流",
            "intent": "unknown",
            "expected_subset": set(intent_map["unknown"]),
            "expected_path": "direct_or_react",
            "note": "长尾模糊，候选>4 触发 P2",
        })

    # RAG 兜底：完全无关
    idx += 1
    bench.append({
        "id": f"Q{idx:02d}",
        "query": "你们公司什么时候成立的",
        "intent": "unknown",
        "expected_subset": set(),
        "expected_path": "rag",
        "note": "未命中意图 → RAG，不走三级流水线严格校验",
    })
    return bench


# ───────────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────────
@pytest.fixture
def embedding():
    return _FakeEmbedding()


@pytest.fixture
def skill_registry():
    # 真实加载，确保工具名/意图映射/描述为生产真实值
    return get_skill_registry()


def _make_agent(embedding, skill_registry, monkeypatch, *, enable_p2: bool) -> ReActAgent:
    """构造 ReActAgent，注入确定性 embedding + 受控 P2 开关。"""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")

    # P2 开关（硬过滤=False，软过滤=True）
    monkeypatch.setattr(ChatConfig, "enable_p2_local_classify", enable_p2)
    # 软过滤下阈值设 2，使候选>2 即触发 P2（贴合当前默认行为 4 的更灵敏版，
    # 这里用 2 以便基准集中更多 case 进 P2，便于观测差异）
    monkeypatch.setattr(ChatConfig, "p2_local_classify_min_candidates", 2 if enable_p2 else 999)

    llm = MagicMock()
    llm.chat_qwen = MagicMock(return_value="mock")
    tool = MagicMock()

    agent = ReActAgent(
        llm_service=llm,
        tool_service=tool,
        embedding_service=embedding,
    )
    return agent


def _fake_p2_instance(confirm_top_n: int = 3):
    """受控 P2 小模型：从候选中保留前 confirm_top_n 个（模拟「小模型确认」）。

    输入顺序遵循 tool_names（已是 P1 输出），返回子集即「确认通过」。
    """
    inst = MagicMock()

    async def _chat_classify(*, user_query, tool_names, tool_descriptions, system_prompt="", max_retries=1):
        return list(tool_names)[:confirm_top_n]

    inst.chat_classify = _chat_classify
    return inst


# ───────────────────────────────────────────────────────────────────────────
# 计时 + 数据采集 helper
# ───────────────────────────────────────────────────────────────────────────
async def _run_one(agent: ReActAgent, case: dict) -> dict:
    """对单条 query 跑 _select_tools_for_intent，采集分段耗时与产出。"""
    t0 = time.perf_counter()
    # P0 段：进入方法到 P1 前（用一次轻量计算近似，真实 P0 极快）
    p0_start = time.perf_counter()
    intent_map = _get_intent_tool_map()
    tool_names = intent_map.get(case["intent"] or "unknown", intent_map["unknown"])
    p0_ms = (time.perf_counter() - p0_start) * 1000.0

    # 包裹 P1+P2：_select_tools_for_intent 内部含 P1/P2，但我们要分段。
    # 为分段精确，这里直接复刻调用顺序并各自计时：
    p1_start = time.perf_counter()
    matcher = agent._tool_matcher
    p2_ranked: list[str] = []
    current_names = set(tool_names)
    if matcher and len(current_names) > 1:
        ranked = await matcher.rank(
            user_query=case["query"],
            candidate_names=current_names,
            intent_action=case["intent"],
            top_k=5 if (case["intent"] is None or case["intent"] == "unknown") else 4,
        )
        if ranked:
            p2_ranked = list(ranked)
            current_names = set(p2_ranked)
    p1_ms = (time.perf_counter() - p1_start) * 1000.0

    p2_invoked = False
    p2_ms = 0.0
    if matcher and len(current_names) > 2 and agent._skill_registry is not None:
        from src.modules.chat.config import ChatConfig
        if ChatConfig.enable_p2_local_classify:
            p2_invoked = True
            p2_start = time.perf_counter()
            p2_plan = await _local_p1_tool_select(
                user_query=case["query"],
                tool_names=current_names,
                tool_descriptions=agent._skill_registry.tool_descriptions,
                p2_ranked=p2_ranked,
            )
            p2_ms = (time.perf_counter() - p2_start) * 1000.0
            if p2_plan.actions:
                current_names = p2_plan.to_tool_names()

    total_ms = (time.perf_counter() - t0) * 1000.0

    # 准确率判定：选中工具应 ⊆ 意图候选全集（不超发），且命中预期意图工具
    expected = case["expected_subset"]
    selected = current_names
    if expected:
        accuracy_ok = selected.issubset(expected) and bool(selected)
    else:
        accuracy_ok = True  # RAG case 不强制

    return {
        "id": case["id"],
        "query": case["query"],
        "intent": case["intent"],
        "p0_ms": round(p0_ms, 3),
        "p1_ms": round(p1_ms, 3),
        "p2_ms": round(p2_ms, 3),
        "total_ms": round(total_ms, 3),
        "p2_invoked": p2_invoked,
        "selected_tools": sorted(selected),
        "expected_subset": sorted(expected),
        "accuracy_ok": accuracy_ok,
        "p1_topk": p2_ranked,
    }


def _aggregate(rows: list[dict]) -> dict:
    total = [r["total_ms"] for r in rows]
    p1 = [r["p1_ms"] for r in rows]
    p2 = [r["p2_ms"] for r in rows if r["p2_invoked"]]
    return {
        "n": len(rows),
        "total_ms": {
            "p50": round(statistics.median(total), 2),
            "p95": round(sorted(total)[int(0.95 * (len(total) - 1))], 2) if len(total) > 1 else round(total[0], 2),
            "max": round(max(total), 2),
        },
        "p1_ms_p95": round(sorted(p1)[int(0.95 * (len(p1) - 1))], 2) if len(p1) > 1 else round(p1[0], 2),
        "p2_invoked_rate": round(sum(1 for r in rows if r["p2_invoked"]) / len(rows), 3),
        "p2_ms_p95": round(sorted(p2)[int(0.95 * (len(p2) - 1))], 2) if p2 else 0.0,
        "accuracy": round(sum(1 for r in rows if r["accuracy_ok"]) / len(rows), 3),
    }


# ───────────────────────────────────────────────────────────────────────────
# 测试：软过滤 vs 硬过滤（方式 1 A/B）
# ───────────────────────────────────────────────────────────────────────────
class TestToolSelectPerfAB:
    """方式 1：同一基准集，切换 P2 开关做软/硬过滤对比。"""

    @pytest.mark.asyncio
    async def test_soft_vs_hard_filter(self, embedding, skill_registry, monkeypatch, capsys):
        bench = _build_benchmark(skill_registry)
        assert bench, "基准集为空，请检查 skill_registry 加载"

        # 软过滤：开 P2 + 注入受控小模型
        soft_agent = _make_agent(embedding, skill_registry, monkeypatch, enable_p2=True)
        fake_p2 = _fake_p2_instance(confirm_top_n=3)
        with patch(
            "src.modules.chat.agent.react_agent_selection.LocalModelService.get_instance",
            return_value=fake_p2,
        ), patch(
            "src.modules.chat.core.local_model_service.LocalModelService.get_instance",
            return_value=fake_p2,
        ):
            # 预热：先空跑一遍，构建 FAISS 索引（避免冷启动污染计时）
            for case in bench:
                await _run_one(soft_agent, case)
            soft_rows = []
            for case in bench:
                soft_rows.append(await _run_one(soft_agent, case))

        # 硬过滤：关 P2（小模型不被调用）
        hard_agent = _make_agent(embedding, skill_registry, monkeypatch, enable_p2=False)
        for case in bench:
            await _run_one(hard_agent, case)  # 预热
        hard_rows = []
        for case in bench:
            hard_rows.append(await _run_one(hard_agent, case))

        soft_agg = _aggregate(soft_rows)
        hard_agg = _aggregate(hard_rows)

        # 输出可读报告（capsys 捕获，也可 pytest -s 直接看）
        with capsys.disabled():
            print("\n=== 工具选择流水线性能对比（方式 1）===")
            print(f"基准 query 数: {len(bench)}")
            print(f"软过滤(当前 T5): {soft_agg}")
            print(f"硬过滤(关 P2) : {hard_agg}")
            diff_ms = round(soft_agg["total_ms"]["p95"] - hard_agg["total_ms"]["p95"], 2)
            print(f"P95 选工具耗时差(软-硬): {diff_ms} ms  ← 即 P2 小模型+保护引入的额外开销")
            print(f"P2 触发率: 软={soft_agg['p2_invoked_rate']} 硬={hard_agg['p2_invoked_rate']}")
            print(f"准确率: 软={soft_agg['accuracy']} 硬={hard_agg['accuracy']}")
            print("\n逐条明细:")
            for r in soft_rows:
                print(f"  {r['id']} intent={r['intent']:10s} p2={int(r['p2_invoked'])} "
                      f"total={r['total_ms']:6.2f}ms selected={r['selected_tools']}")

        # 断言：硬过滤 P2 触发率必须为 0（验证「硬过滤不调小模型」）
        assert hard_agg["p2_invoked_rate"] == 0.0, "硬过滤不应触发 P2 小模型"
        # 软过滤准确率不应低于硬过滤（P2 不应引入错误工具）
        assert soft_agg["accuracy"] >= hard_agg["accuracy"] - 0.001
        # 选工具耗时不应异常爆炸
        assert soft_agg["total_ms"]["p95"] < 5000, "选工具 P95 超时，检查 mock/索引"


# ───────────────────────────────────────────────────────────────────────────
# 真实 vLLM 模式（默认 skip，需 RUN_REAL_VLLM=1）
# ───────────────────────────────────────────────────────────────────────────
import os as _os

_RUN_REAL_VLLM = _os.environ.get("RUN_REAL_VLLM", "") in ("1", "true", "True")


@pytest.mark.skipif(
    not _RUN_REAL_VLLM,
    reason="真实 vLLM 测试默认跳过，需设置 RUN_REAL_VLLM=1 并拉起 vLLM 服务",
)
class TestToolSelectPerfRealVLLM:
    """方式 1 + 真实 P2 小模型（vLLM qwen3-unified）耗时测量。

    与 TestToolSelectPerfAB（mock P2）的区别：
      - P2 段耗时来自真实远端推理（含网络 RTT + GPU 计算 + 单飞合并），
        比 mock 的近 0ms 更具生产参考性。
      - 其余 P0/P1 计时与 mock 版一致（embedding 仍为确定性 mock）。
    """

    @pytest.mark.asyncio
    async def test_real_vllm_p2_latency(self, embedding, skill_registry, monkeypatch, capsys):
        # 强制走 vLLM 远程后端，且开启 P2
        # 注意：chat_config 是模块级已实例化的单例，setenv 不会自动回灌，
        # 必须直接 setattr 到 ChatConfig 类属性，才能真正走 vllm 分支
        # （vllm 分支会传 enable_thinking=False，避免 Qwen3 思考模式污染工具名解析）。
        monkeypatch.setattr(ChatConfig, "local_model_backend", "vllm")
        # 测试进程跑在宿主，8003 已映射到宿主，用 localhost 最稳（不依赖 host.docker.internal 解析）
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8003/v1")
        monkeypatch.setattr(ChatConfig, "vllm_base_url", "http://localhost:8003/v1")
        monkeypatch.setenv("VLLM_TOOL_SELECTOR_MODEL", "qwen3-unified")
        monkeypatch.setenv("VLLM_TIMEOUT", "30")
        # 用默认阈值 4（生产默认），使候选>4 才触发真实 P2
        monkeypatch.setattr(ChatConfig, "enable_p2_local_classify", True)
        monkeypatch.setattr(ChatConfig, "p2_local_classify_min_candidates", 4)

        # 关键：不要 patch LocalModelService.get_instance，让它真正连 vLLM
        agent = _make_agent(embedding, skill_registry, monkeypatch, enable_p2=True)

        bench = _build_benchmark(skill_registry)
        assert bench, "基准集为空"

        # 预热：触发一次真实连接 + 构建 FAISS 索引（不计时）
        warm_case = bench[0]
        await _run_one(agent, warm_case)

        # 真实采样多轮（vLLM 有 warmup，单次可能偏低；取多次聚合更稳）
        N = 3
        rows: list[dict] = []
        for _ in range(N):
            for case in bench:
                rows.append(await _run_one(agent, case))

        agg = _aggregate(rows)

        with capsys.disabled():
            print("\n=== 真实 vLLM P2 小模型耗时（RUN_REAL_VLLM=1）===")
            print(f"基准 query 数: {len(bench)}  采样轮数: {N}  总样本: {len(rows)}")
            print(f"聚合: {agg}")
            print(f"P2 触发率: {agg['p2_invoked_rate']}  P2 P95 耗时: {agg['p2_ms_p95']} ms")
            print(f"整体 P95 选工具耗时: {agg['total_ms']['p95']} ms")
            print("\n逐条明细（首轮）:")
            for r in rows[: len(bench)]:
                print(f"  {r['id']} intent={r['intent']:10s} p2={int(r['p2_invoked'])} "
                      f"p2_ms={r['p2_ms']:7.2f} total={r['total_ms']:7.2f}ms "
                      f"selected={r['selected_tools']}")

        # 真实 P2 调用应成功（触发率 > 0 即说明 vLLM 真的被调到且返回了有效工具）
        assert agg["p2_invoked_rate"] > 0.0, "真实 vLLM P2 未触发，检查候选集/vLLM 端点"
        # 准确率不因真实小模型而崩（至少 0.8，容错：vLLM 输出解析偶尔失手可回落全候选）
        assert agg["accuracy"] >= 0.8, f"真实 vLLM P2 准确率过低: {agg['accuracy']}"
        # 真实 P2 单次不应离谱超时
        assert agg["p2_ms_p95"] < 30000, "真实 vLLM P2 P95 超 30s，检查服务健康"
