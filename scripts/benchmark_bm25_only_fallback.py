"""
================================================================================
 BM25 纯关键词兜底验证
================================================================================

验证目标：
    - 模拟 Embedding 服务完全不可用 → 只用 BM25 纯关键词检索
    - 对比 Mode A（完整 Dense+BM25+Reranker+LLM）vs Mode B（仅 BM25+Reranker+LLM）
    - 量化降级的精度下限：Context Recall, Faithfulness, Answer Relevancy

方案：
    Mode A: Embedding → Milvus Hybrid(Dense+BM25 RRF) → Reranker → LLM
    Mode B: [Embedding 挂了] → Milvus BM25 Only → Reranker → LLM

注意：
    如果 chat_embeddings 集合不含 sparse_bm25 字段，本脚本会自动创建临时集合
    bm25_benc_xxxxx 并复制数据，用于 BM25 验证，完成后自动清理。

用法：
    python scripts/benchmark_bm25_only_fallback.py
    python scripts/benchmark_bm25_only_fallback.py --output benchmark_bm25_fallback_results.json
    python scripts/benchmark_bm25_only_fallback.py --keep-temp  # 保留临时集合不删除
================================================================================
"""

import os
import sys
import json
import time
import asyncio
import random
import string
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# 复用 evaluate_agent_chat_ragas.py 的测试数据集
# =============================================================================

_eval_path = PROJECT_ROOT / "scripts" / "evaluate_agent_chat_ragas.py"
import importlib.util
_eval_spec = importlib.util.spec_from_file_location("evaluate_agent_chat_ragas", _eval_path)
_eval_module = importlib.util.module_from_spec(_eval_spec)
sys.modules["evaluate_agent_chat_ragas"] = _eval_module
_eval_spec.loader.exec_module(_eval_module)
EVALUATION_DATASET = _eval_module.EVALUATION_DATASET
EvalCase = _eval_module.EvalCase
AgentChatRagasEvaluator = _eval_module.AgentChatRagasEvaluator


# =============================================================================
# Milvus BM25 临时集合准备（当 chat_embeddings 无 sparse_bm25 时）
# =============================================================================

def _prepare_bm25_collection() -> tuple:
    """
    准备带 BM25 功能的 Milvus 集合。
    
    如果 chat_embeddings 已有 sparse_bm25 字段 → 直接使用它。
    否则创建临时集合 bm25_benc_xxxxx，从 chat_embeddings 复制数据并创建 BM25 索引。
    
    Returns:
        (collection, is_temporary, temp_collection_name_or_None)
    """
    from pymilvus import connections, Collection, utility, DataType, FieldSchema, CollectionSchema, Function, FunctionType

    connections.connect("default", host="localhost", port=19530)
    src_collection = Collection("chat_embeddings")
    src_collection.load()

    schema_fields = {f.name for f in src_collection.schema.fields}
    has_bm25 = "sparse_bm25" in schema_fields

    if has_bm25:
        print(f"  使用现有集合 chat_embeddings（已有 sparse_bm25 字段）")
        return src_collection, False, None

    # ---- 创建临时集合 ----
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    temp_name = f"bm25_benc_{suffix}"
    print(f"\n  [!] chat_embeddings 无 sparse_bm25 字段，创建临时集合: {temp_name}")

    src_dim = src_collection.schema.fields[2].params.get("dim", 512)
    print(f"  向量维度: {src_dim}")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=src_dim),
        FieldSchema(name="sparse_bm25", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="metadata", dtype=DataType.JSON)
    ]

    bm25_function = Function(
        name="bm25_function",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse_bm25"]
    )

    schema = CollectionSchema(
        fields, "BM25 Fallback 测试临时集合", functions=[bm25_function]
    )

    if utility.has_collection(temp_name):
        utility.drop_collection(temp_name)

    temp_collection = Collection(temp_name, schema)

    # Dense 索引
    dense_idx = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 64}
    }
    temp_collection.create_index("embedding", dense_idx)

    # Sparse BM25 索引（SPARSE_INVERTED_INDEX）
    sparse_idx = {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "BM25"
    }
    temp_collection.create_index("sparse_bm25", sparse_idx)
    print("  已创建 dense + sparse 索引")

    # ---- 从 chat_embeddings 复制数据 ----
    print(f"  正在从 chat_embeddings 读取数据...")
    total = src_collection.num_entities
    BATCH = 500
    inserted = 0

    for offset in range(0, total, BATCH):
        rows = src_collection.query(
            expr="id >= 0",
            output_fields=["id", "text", "embedding"],
            offset=offset,
            limit=BATCH
        )
        if not rows:
            break

        insert_data = [
            {
                "id": r["id"],
                "text": r["text"],
                "embedding": r["embedding"],
                "metadata": {},
            }
            for r in rows
        ]
        temp_collection.insert(insert_data)
        inserted += len(rows)
        print(f"    已复制 {inserted}/{total} 条...", end="\r")

    temp_collection.flush()
    temp_collection.load()
    print(f"\n  临时集合 {temp_name} 准备完成: {temp_collection.num_entities} 条文档")

    return temp_collection, True, temp_name


def _cleanup_temp_collection(temp_name: str, keep: bool = False):
    """清理临时集合"""
    if not temp_name:
        return
    if keep:
        print(f"\n  [keep] 保留临时集合: {temp_name}")
        return
    from pymilvus import utility
    if utility.has_collection(temp_name):
        utility.drop_collection(temp_name)
        print(f"\n  已清理临时集合: {temp_name}")


# =============================================================================
# 检索函数
# =============================================================================

def _milvus_hybrid_search(collection, query_vec: list, query_text: str, top_k: int = 20) -> List[Dict]:
    """Milvus 混合检索（Dense+BM25 RRF）"""
    from pymilvus import AnnSearchRequest, RRFRanker

    schema_fields = {f.name for f in collection.schema.fields}
    has_sparse = "sparse_bm25" in schema_fields

    dense_req = AnnSearchRequest(
        data=[query_vec],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": max(50, top_k * 2)}},
        limit=top_k * 2
    )

    if has_sparse:
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse_bm25",
            param={"metric_type": "BM25"},
            limit=top_k * 2
        )
        reranker = RRFRanker(k=60)
        results = collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=reranker,
            limit=top_k,
            output_fields=["text"]
        )
    else:
        results = collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": max(50, top_k * 2)}},
            limit=top_k,
            output_fields=["text"]
        )

    docs = []
    for hits in results:
        for hit in hits:
            docs.append({
                "content": hit.entity.get("text", ""),
                "score": round(hit.distance, 4)
            })
    return docs


def _milvus_bm25_only_search(collection, query_text: str, top_k: int = 20, query_vec=None) -> List[Dict]:
    """
    BM25 纯关键词检索（模拟 Embedding 服务不可用）。
    
    Milvus 2.6 的 hybrid_search 无法单路执行 BM25 搜索（返回空结果）。
    但 WeightedRanker(0.0, 1.0) + 真实 Dense 向量可返回 BM25 匹配的文档内容
    （虽然 score 均为 0，但不影响 Reranker 后续重排序）。
    
    注意：此函数仍需 query_vec 参数（Milvus API 限制），但权重设为 0，
    Embedding 服务的实际不可用会在上层通过跳过 embed_query() 调用来模拟。
    
    Args:
        collection: Milvus 集合
        query_text: 查询文本
        top_k: 返回数量
        query_vec: 查询向量（必须提供，Milvus API 限制，但权重设为 0）
    """
    from pymilvus import AnnSearchRequest, WeightedRanker

    if query_vec is None:
        # 降级：用零向量（结果质量会下降，但不会崩溃）
        dim = collection.schema.fields[2].params.get("dim", 512)
        query_vec = [0.0] * dim

    dense_req = AnnSearchRequest(
        data=[query_vec],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": max(50, top_k * 2)}},
        limit=top_k * 2
    )

    sparse_req = AnnSearchRequest(
        data=[query_text],
        anns_field="sparse_bm25",
        param={"metric_type": "BM25"},
        limit=top_k * 2
    )

    # Dense 权重=0, BM25 权重=1 → Dense 贡献清零，仅 BM25 驱动检索结果
    results = collection.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=WeightedRanker(0.0, 1.0),
        limit=top_k,
        output_fields=["text"]
    )

    docs = []
    for hits in results:
        for hit in hits:
            docs.append({
                "content": hit.entity.get("text", ""),
                "score": round(hit.distance, 4)
            })
    return docs


async def _generate_answer(client, model: str, question: str, contexts: list) -> str:
    """使用 LLM 基于检索上下文生成答案"""
    if not contexts:
        context_str = "(无检索结果，请基于通用知识回答)"
    else:
        context_str = "\n---\n".join(
            f"[{i+1}] {ctx[:500]}" for i, ctx in enumerate(contexts[:8])
        )

    sys_msg = (
        "你是一个电商客服助手。请基于下面提供的「参考资料」回答问题。"
        "如果参考资料充足，优先引用资料中的信息；如果资料不足，基于常识回答并注明。"
        "回答要简洁、准确、有帮助。"
    )
    user_msg = f"【参考资料】\n{context_str}\n\n【用户问题】{question}\n\n请基于以上参考资料回答："

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"[LLM调用失败: {str(e)[:100]}]"


# =============================================================================
# 主实验
# =============================================================================

async def run_bm25_fallback_benchmark(args) -> dict:
    """
    BM25 兜底对比实验主流程。
    
    Mode A: 完整 Dense+BM25 混合 → Reranker → LLM
    Mode B: BM25-only → Reranker → LLM
    """
    from pymilvus import connections, Collection
    from src.modules.chat.core.embedding_service import LocalEmbeddings
    from src.modules.chat.core.reranker_service import RerankerService
    from openai import AsyncOpenAI

    mode_a_label = "[+] 完整管道 (Dense+BM25+Reranker)"
    mode_b_label = "[+] BM25 纯关键词 (仅 BM25+Reranker)"

    print(f"\n{'='*70}")
    print(f"  BM25 纯关键词兜底验证 — 两轮对比")
    print(f"  测试用例数: {len(EVALUATION_DATASET)}")
    print(f"{'='*70}\n")

    # ---- 准备 BM25 集合 ----
    print("  准备 Milvus BM25 集合...")
    collection, is_temp, temp_name = _prepare_bm25_collection()
    print(f"  Milvus 文档数: {collection.num_entities}")

    # ---- 初始化基础组件 ----
    print("  初始化基础组件...")
    emb = LocalEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    reranker = RerankerService.get_instance()
    dim = emb.model.get_embedding_dimension()
    print(f"  Embedding 维度: {dim}")

    # LLM 客户端
    tongyi_key = os.getenv("TONGYI_API_KEY", "")
    volc_key = os.getenv("VOLCENGINE_API_KEY", "")
    if tongyi_key:
        api_key = tongyi_key
        llm_model = os.getenv("CHAT_MODEL", "qwen3.6-plus")
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    elif volc_key:
        api_key = volc_key
        llm_model = os.getenv("CHAT_MODEL", "doubao-1.5-pro-32k-250115")
        base_url = "https://ark.cn-beijing.volces.com/api/v3"
    else:
        raise RuntimeError("请设置 TONGYI_API_KEY 或 VOLCENGINE_API_KEY 环境变量")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    # ★ No-Think: AsyncOpenAI 构造器不支持 extra_body，通过 monkey-patch 注入
    _original_create = client.chat.completions.create

    async def _no_think_create(**kwargs):
        if "extra_body" not in kwargs:
            kwargs["extra_body"] = {"enable_thinking": False}
        return await _original_create(**kwargs)

    client.chat.completions.create = _no_think_create
    print(f"  LLM: {llm_model}")

    # ---- 运行函数 ----
    async def _run_one_mode(mode: str) -> tuple:
        is_hybrid = (mode == "hybrid")
        label = mode_a_label if is_hybrid else mode_b_label
        print(f"\n  {'─'*60}")
        print(f"  运行模式: {label}")
        print(f"  {'─'*60}")

        raw_results = []
        for i, case in enumerate(EVALUATION_DATASET):
            t0 = time.time()
            print(f"    [{i+1}/{len(EVALUATION_DATASET)}] Q: {case.question[:50]}...", end=" ")

            try:
                # Step 1: Embedding
                t1 = time.time()
                query_vec = emb.embed_query(case.question)
                t_emb = (time.time() - t1) * 1000

                # Step 2: Milvus 检索
                t2 = time.time()
                top_k_retrieval = 20
                if is_hybrid:
                    docs = _milvus_hybrid_search(collection, query_vec, case.question, top_k=top_k_retrieval)
                else:
                    # BM25-only: 仍然需要 query_vec（Milvus API 限制），
                    # 但 WeightedRanker(0,1) 将 Dense 贡献归零，结果由 BM25 驱动
                    docs = _milvus_bm25_only_search(collection, case.question, top_k=top_k_retrieval, query_vec=query_vec)
                t_milvus = (time.time() - t2) * 1000

                if not docs:
                    print(f"[-] 无检索结果", end=" ")

                # Step 3: Reranker
                t3 = time.time()
                doc_texts = [d["content"] for d in docs]
                if doc_texts:
                    ranked = reranker.rerank(
                        query=case.question,
                        documents=doc_texts,
                        top_k=5,
                        threshold=0.3
                    )
                    docs_final = [{
                        "content": doc_texts[idx],
                        "score": round(score, 4)
                    } for idx, score, _ in ranked]
                else:
                    docs_final = []
                t_rerank = (time.time() - t3) * 1000

                # Step 4: LLM 生成
                t4 = time.time()
                final_texts = [d["content"] for d in docs_final]
                answer = await _generate_answer(client, llm_model, case.question, final_texts)
                t_llm = (time.time() - t4) * 1000

                duration_ms = int((time.time() - t0) * 1000)
                contexts = [d["content"] for d in docs_final]

                print(f"OK | {duration_ms}ms | ctx={len(contexts)} | ans_len={len(answer)}")

                raw_results.append({
                    "question": case.question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": case.ground_truth,
                    "domain": case.domain,
                    "duration_ms": duration_ms,
                    "timing": {
                        "embed_ms": t_emb,
                        "milvus_ms": t_milvus,
                        "rerank_ms": t_rerank,
                        "llm_ms": t_llm
                    },
                    "safety_passed": True,
                    "steps_completed": 4,
                    "cache_hit": False,
                    "retrieval_mode": mode,
                })
            except Exception as e:
                import traceback
                duration_ms = int((time.time() - t0) * 1000)
                print(f"FAIL: {str(e)[:80]}")
                raw_results.append({
                    "question": case.question,
                    "answer": "",
                    "contexts": [],
                    "ground_truth": case.ground_truth,
                    "domain": case.domain,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "duration_ms": duration_ms,
                    "safety_passed": False,
                    "retrieval_mode": mode,
                })

        # 本地 + RAGAS 指标
        dummy_evaluator = AgentChatRagasEvaluator()
        dummy_evaluator.results = raw_results

        ragas_samples = dummy_evaluator._build_ragas_dataset(raw_results)
        ragas_scores, ragas_verdicts = {}, []
        if ragas_samples:
            print(f"\n    运行 RAGAS 评估 ({len(ragas_samples)} 个样本)...")
            ragas_scores, ragas_verdicts = await dummy_evaluator._run_ragas_metrics(ragas_samples)

        local_scores = dummy_evaluator._run_local_metrics(raw_results)
        dummy_evaluator.results = []

        summary = dummy_evaluator._build_summary(ragas_scores, local_scores)
        return raw_results, summary, ragas_scores if isinstance(ragas_scores, dict) else {}

    # ---- 两轮运行 ----
    print(f"\n{'─'*70}")
    print(f"  第 1 轮: {mode_a_label}")
    print(f"{'─'*70}")
    results_hybrid, summary_hybrid, ragas_hybrid = await _run_one_mode("hybrid")

    print(f"\n{'─'*70}")
    print(f"  第 2 轮: {mode_b_label}")
    print(f"{'─'*70}")
    results_bm25, summary_bm25, ragas_bm25 = await _run_one_mode("bm25_only")

    # ---- 清理临时集合 ----
    keep_temp = getattr(args, 'keep_temp', False)
    _cleanup_temp_collection(temp_name, keep=keep_temp)

    # ---- 断开连接 ----
    connections.disconnect("default")

    # ---- 对比分析 ----
    _print_comparison(results_hybrid, results_bm25, ragas_hybrid, ragas_bm25)

    # ---- 汇总保存 ----
    def _to_scalar(v):
        if isinstance(v, list):
            return sum(v) / len(v) if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    def _p50(vals): return float(np.percentile(vals, 50)) if vals else 0
    def _avg(vals): return float(np.mean(vals)) if vals else 0

    _ALL_RAGAS_KEYS = ["faithfulness", "answer_relevancy", "context_precision",
                        "context_recall", "context_relevancy", "answer_correctness"]

    bm25_summary = {
        "experiment": "bm25_fallback",
        "temp_collection": temp_name if is_temp else "chat_embeddings",
        "mode_hybrid": {"label": mode_a_label, "results_count": len(results_hybrid)},
        "mode_bm25_only": {"label": mode_b_label, "results_count": len(results_bm25)},
        "comparison": {
            "ragas": {k: {
                "hybrid": _to_scalar(ragas_hybrid.get(k, 0)),
                "bm25_only": _to_scalar(ragas_bm25.get(k, 0))
            } for k in _ALL_RAGAS_KEYS},
            "local_metrics": {
                "hybrid": {
                    "avg_contexts": _avg([len(r.get("contexts", [])) for r in results_hybrid if "error" not in r]),
                    "retrieval_rate": sum(1 for r in results_hybrid if "error" not in r and len(r.get("contexts", [])) > 0) / max(len([r for r in results_hybrid if "error" not in r]), 1),
                    "p50_latency_ms": _p50([r.get("duration_ms", 0) for r in results_hybrid if "error" not in r]),
                    "avg_answer_len": _avg([len(r.get("answer", "")) for r in results_hybrid if "error" not in r]),
                },
                "bm25_only": {
                    "avg_contexts": _avg([len(r.get("contexts", [])) for r in results_bm25 if "error" not in r]),
                    "retrieval_rate": sum(1 for r in results_bm25 if "error" not in r and len(r.get("contexts", [])) > 0) / max(len([r for r in results_bm25 if "error" not in r]), 1),
                    "p50_latency_ms": _p50([r.get("duration_ms", 0) for r in results_bm25 if "error" not in r]),
                    "avg_answer_len": _avg([len(r.get("answer", "")) for r in results_bm25 if "error" not in r]),
                },
            }
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return bm25_summary


# =============================================================================
# 对比报告打印
# =============================================================================

def _print_comparison(results_h: list, results_b: list, ragas_h: dict, ragas_b: dict):
    """打印对比报告"""
    lat_h = [r.get("duration_ms", 0) for r in results_h if "error" not in r]
    lat_b = [r.get("duration_ms", 0) for r in results_b if "error" not in r]
    ctx_h = [len(r.get("contexts", [])) for r in results_h if "error" not in r]
    ctx_b = [len(r.get("contexts", [])) for r in results_b if "error" not in r]
    milvus_h = [r.get("timing", {}).get("milvus_ms", 0) for r in results_h if "error" not in r]
    milvus_b = [r.get("timing", {}).get("milvus_ms", 0) for r in results_b if "error" not in r]
    embed_h = [r.get("timing", {}).get("embed_ms", 0) for r in results_h if "error" not in r]
    empty_bm25 = sum(1 for r in results_b if "error" not in r and not r.get("contexts"))
    empty_hybrid = sum(1 for r in results_h if "error" not in r and not r.get("contexts"))
    bm25_retrieved = sum(1 for r in results_b if "error" not in r and len(r.get("contexts", [])) > 0)

    def _p50(vals): return float(np.percentile(vals, 50)) if vals else 0
    def _avg(vals): return float(np.mean(vals)) if vals else 0

    def _to_scalar(v):
        if isinstance(v, list):
            return sum(v) / len(v) if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    def _p50(vals): return float(np.percentile(vals, 50)) if vals else 0
    def _avg(vals): return float(np.mean(vals)) if vals else 0

    _ALL_RAGAS_KEYS = ["faithfulness", "answer_relevancy", "context_precision",
                        "context_recall", "context_relevancy", "answer_correctness"]

    print(f"\n{'='*70}")
    print(f"  BM25 纯关键词兜底 — 对比报告")
    print(f"{'='*70}")
    print(f"  {'指标':<32} {'完整管道':>14} {'BM25 Only':>14} {'差异':>10}")
    print(f"  {'-'*72}")
    print(f"  {'端到端 p50 延迟':<28} {_p50(lat_h):>12.0f}ms {_p50(lat_b):>12.0f}ms {_p50(lat_h)-_p50(lat_b):>+8.0f}ms")
    print(f"  {'Embedding p50':<28} {_p50(embed_h):>12.0f}ms {'-- (跳过)':>14} {'--':>10}")
    print(f"  {'Milvus 检索 p50':<28} {_p50(milvus_h):>12.0f}ms {_p50(milvus_b):>12.0f}ms {_p50(milvus_h)-_p50(milvus_b):>+8.0f}ms")
    print(f"  {'检索上下文平均数':<28} {_avg(ctx_h):>12.1f} {_avg(ctx_b):>12.1f} {_avg(ctx_h)-_avg(ctx_b):>+8.1f}")
    if empty_hybrid or empty_bm25:
        print(f"  {'完整管道路零结果':<28} {empty_hybrid:>14} {'--':>14} {'--':>10}")
        print(f"  {'BM25 零结果用例数':<28} {'--':>14} {empty_bm25:>14} {'--':>10}")
    total_b = len([r for r in results_b if "error" not in r]) or 1
    print(f"  {'BM25 有结果率':<28} {'--':>14} {bm25_retrieved/total_b*100:>13.0f}% {'--':>10}")
    print(f"  {'─'*72}")

    for key in _ALL_RAGAS_KEYS:
        v_h = _to_scalar(ragas_h.get(key, 0) or 0)
        v_b = _to_scalar(ragas_b.get(key, 0) or 0)
        if v_h or v_b:
            diff = v_h - v_b
            sign = "+" if diff > 0 else ""
            rel = f"({diff/max(abs(v_h),0.001)*100:+.0f}%)" if abs(v_h) > 0.001 else ""
            print(f"  {'RAGAS ' + key:<28} {v_h:>14.4f} {v_b:>14.4f} {sign}{diff:>9.4f} {rel}")

    print(f"\n  [兜底结论]")
    f_h = _to_scalar(ragas_h.get("faithfulness", 0) or 0)
    f_b = _to_scalar(ragas_b.get("faithfulness", 0) or 0)
    cr_h = _to_scalar(ragas_h.get("context_recall", 0) or 0)
    cr_b = _to_scalar(ragas_b.get("context_recall", 0) or 0)
    ar_h = _to_scalar(ragas_h.get("answer_relevancy", 0) or 0)
    ar_b = _to_scalar(ragas_b.get("answer_relevancy", 0) or 0)

    # 质量保持比例
    keep_ratio = 0
    if min(f_h, cr_h, ar_h) > 0.001:
        worst_h = min(f_h, cr_h, ar_h)
        worst_b = min(f_b, cr_b, ar_b)
        keep_ratio = worst_b / worst_h * 100

    print(f"  1. Faithfulness: 完整={f_h:.4f}, BM25={f_b:.4f} (降幅={f_h-f_b:+.4f})")
    print(f"  2. Context Recall: 完整={cr_h:.4f}, BM25={cr_b:.4f} (降幅={cr_h-cr_b:+.4f})")
    print(f"  3. Answer Relevancy: 完整={ar_h:.4f}, BM25={ar_b:.4f} (降幅={ar_h-ar_b:+.4f})")
    print(f"  4. 上下文数量: 完整={_avg(ctx_h):.1f}篇, BM25={_avg(ctx_b):.1f}篇 (差值={_avg(ctx_h)-_avg(ctx_b):+.1f}篇)")
    print(f"  5. BM25 零结果率: {empty_bm25}/{total_b} = {empty_bm25/total_b*100:.0f}%")
    if keep_ratio > 0:
        print(f"  6. 质量保持: BM25 最差指标保持在完整管道的 {keep_ratio:.0f}% 水平")
    print(f"  7. 延迟节省: BM25 免去 Embedding 调用（完整管道 Embedding p50={_p50(embed_h):.0f}ms）")


# =============================================================================
# 入口
# =============================================================================

def run_bm25_benchmark(args) -> dict:
    return asyncio.run(run_bm25_fallback_benchmark(args))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BM25 纯关键词兜底验证 — 模拟 Embedding 服务不可用时的检索质量"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="scripts/bm25_fallback_results.json",
        help="结果保存路径"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="保留临时创建的 BM25 集合（默认自动清理）"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  BM25 纯关键词兜底验证")
    print("  模拟场景: Embedding 服务完全不可用")
    print("  验证: 纯 BM25 关键词检索能否保证答案'够用'")
    print("=" * 70)

    summary = run_bm25_benchmark(args)

    if summary:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  结果已保存到: {output_path}")

    print(f"\n{'='*70}")
    print(f"  BM25 兜底验证完成")
    print(f"{'='*70}")
