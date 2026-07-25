"""
不同数据量级延迟基准测试 — FAISS vs Milvus HNSW
================================================================================

测量 FAISS HNSW（进程序内工具匹配）和 Milvus HNSW（知识检索）
在不同数据量级下的 p50/p99 搜索延迟，验证 HNSW 在目标规模下
是否满足 < 150ms 的延迟目标。

测试维度：
    - 数据量级：[1K, 5K, 10K, 50K, 100K]
    - FAISS HNSW: M=16, efConstruction=64, efSearch=32
    - Milvus HNSW: M=16, efConstruction=200, ef=max(50, top_k*2)
    - 50 条随机 query
    - top_k ∈ [3, 5, 10]
    - p50 / p99 / mean 延迟 + Milvus 内存估算

前置条件：
    1. pip install pymilvus numpy faiss-cpu
    2. Milvus Standalone 运行中 (docker-compose up standalone etcd minio)

用法：
    python scripts/benchmark_latency_by_scale.py
    python scripts/benchmark_latency_by_scale.py --scales 1k,5k,10k,50k,100k
    python scripts/benchmark_latency_by_scale.py --output results.json --dim 512
    python scripts/benchmark_latency_by_scale.py --faiss-only   # 只测 FAISS
    python scripts/benchmark_latency_by_scale.py --milvus-only  # 只测 Milvus
================================================================================
"""

import os
import sys
import json
import time
import math
import argparse
from typing import List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import numpy as np

# =============================================================================
# 常量（从项目源码提取，保持一致）
# =============================================================================

# FAISS HNSW 参数（对应 tool_registry.py 的 EmbeddingToolMatcher）
FAISS_M = 16
FAISS_EF_CONSTRUCTION = 64
FAISS_EF_SEARCH = 32

# Milvus HNSW 参数（对应 milvus_service.py）
MILVUS_M = 16
MILVUS_EF_CONSTRUCTION = 200
MILVUS_SEARCH_EF_MIN = 50  # max(50, top_k*2)

# Milvus 连接
MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))

# 向量维度（默认 bge-small-zh-v1.5）
DEFAULT_DIM = 512

# 数据量级
DEFAULT_SCALES = [1_000, 5_000, 10_000, 50_000, 100_000]

# Top-K 取值
DEFAULT_TOPK_VALUES = [3, 5, 10]
DEFAULT_SEARCH_REPEATS = 50  # 每个量级跑多少条 query

# 每 chunk 估算大小（text 字段平均 500 字节 + metadata 400 字节）
ESTIMATED_CHUNK_TEXT_BYTES = 500
ESTIMATED_METADATA_BYTES = 400


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class ScaleResult:
    """单个量级的测试结果"""
    scale: int  # 向量数
    vector_dim: int
    # FAISS
    faiss_latency: Dict[int, Dict[str, float]] = field(default_factory=dict)  # {top_k: {p50,p99,mean}}
    faiss_build_ms: float = 0
    faiss_memory_mb: float = 0
    # Milvus
    milvus_latency: Dict[int, Dict[str, float]] = field(default_factory=dict)
    milvus_build_ms: float = 0
    milvus_insert_ms: float = 0
    milvus_memory_mb: float = 0
    milvus_estimated_index_mb: float = 0


# =============================================================================
# FAISS HNSW 基准测试
# =============================================================================

def _faiss_benchmark_one_scale(scale: int, dim: int,
                                 topk_values: List[int],
                                 search_repeats: int) -> Dict[str, Any]:
    """对单个量级做 FAISS HNSW 基准测试"""
    print(f"  [FAISS] 生成 {scale} 条 {dim}维随机向量...", end=" ", flush=True)

    # 生成数据（均匀分布，模拟 L2 归一化后的向量）
    rng = np.random.RandomState(42)
    data = rng.randn(scale, dim).astype(np.float32)
    # L2 归一化（余弦相似度）
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms

    # 生成 50 条随机 query（每次基准用不同 seed 取 query）
    query_rng = np.random.RandomState(999)
    query_vecs = query_rng.randn(search_repeats, dim).astype(np.float32)
    q_norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
    query_vecs = query_vecs / q_norms

    # 构建 FAISS HNSW 索引
    import faiss
    t0 = time.time()
    index = faiss.IndexHNSWFlat(dim, FAISS_M)
    index.hnsw.efConstruction = FAISS_EF_CONSTRUCTION
    index.hnsw.efSearch = FAISS_EF_SEARCH
    index.add(data)
    build_ms = (time.time() - t0) * 1000

    # 估算内存：向量数据 + HNSW 图结构
    vec_bytes = data.nbytes  # scale * dim * 4
    graph_bytes = scale * FAISS_M * 2 * 4  # 每个节点 M*2 条边(上下双向)，每条边 4 字节
    faiss_memory_mb = (vec_bytes + graph_bytes) / (1024 * 1024)
    print(f"done ({build_ms:.0f}ms)")

    # 预热
    for _ in range(5):
        _ = index.search(query_vecs[0:1], max(topk_values))

    # 基准测试
    max_k = max(topk_values)
    results: Dict[int, Dict[str, float]] = {}
    for top_k in topk_values:
        latencies = []
        for i in range(search_repeats):
            q = query_vecs[i:i+1]
            t0 = time.time()
            _ = index.search(q, min(max_k, scale))
            latencies.append((time.time() - t0) * 1000)

        results[top_k] = {
            "p50": float(np.percentile(latencies, 50)),
            "p99": float(np.percentile(latencies, 99)),
            "mean": float(np.mean(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
        }
        print(f"    FAISS top_k={top_k:<3} p50={results[top_k]['p50']:6.2f}ms "
              f"p99={results[top_k]['p99']:6.2f}ms mean={results[top_k]['mean']:6.2f}ms")

    return {
        "latency": results,
        "build_ms": build_ms,
        "memory_mb": faiss_memory_mb,
    }


# =============================================================================
# Milvus HNSW 基准测试
# =============================================================================

def _milvus_benchmark_one_scale(scale: int, dim: int,
                                  topk_values: List[int],
                                  search_repeats: int) -> Dict[str, Any]:
    """对单个量级做 Milvus HNSW 基准测试"""
    from pymilvus import (
        connections, Collection, FieldSchema, CollectionSchema, DataType,
        utility, connections as mc
    )

    COLL_NAME = f"benchmark_scale_{scale}"
    print(f"  [Milvus] 创建临时集合 {COLL_NAME} ({scale} 条 {dim}维)...", end=" ", flush=True)

    # ---- 创建集合 ----
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description=f"latency benchmark @ {scale}")
    collection = Collection(COLL_NAME, schema)

    # ---- 生成数据 ----
    rng = np.random.RandomState(42)
    data = rng.randn(scale, dim).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms

    query_rng = np.random.RandomState(999)
    query_vecs = query_rng.randn(search_repeats, dim).astype(np.float32)
    q_norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
    query_vecs = query_vecs / q_norms

    # ---- 插入数据 ----
    t0 = time.time()
    batch_size = min(1000, scale)
    for start in range(0, scale, batch_size):
        end = min(start + batch_size, scale)
        collection.insert([data[start:end].tolist()])
    insert_ms = (time.time() - t0) * 1000

    collection.flush()
    print(f"inserted ({insert_ms:.0f}ms), ", end="", flush=True)

    # ---- 创建 HNSW 索引 ----
    t0 = time.time()
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": MILVUS_M, "efConstruction": MILVUS_EF_CONSTRUCTION}
    }
    collection.create_index("embedding", index_params)
    import time as _t
    while True:
        progress = utility.index_building_progress(COLL_NAME)
        if progress.get("total_rows", 0) > 0 and \
           progress.get("pending_index_rows", 0) == 0:
            break
        _t.sleep(0.5)
    build_ms = (time.time() - t0) * 1000

    collection.load()
    print(f"indexed ({build_ms:.0f}ms)", flush=True)

    # ---- 估算内存 ----
    # Milvus 内部: 向量数据 + HNSW 图 + overhead
    vec_mb = scale * dim * 4 / (1024 * 1024)
    graph_mb = scale * MILVUS_M * 2 * 4 / (1024 * 1024)
    estimated_index_mb = vec_mb + graph_mb

    # ---- 预热 ----
    search_params = {"metric_type": "COSINE", "params": {"ef": max(MILVUS_SEARCH_EF_MIN, max(topk_values) * 2)}}
    for _ in range(5):
        collection.search([query_vecs[0].tolist()], "embedding", search_params,
                          limit=max(topk_values))

    # ---- 基准测试 ----
    results: Dict[int, Dict[str, float]] = {}
    for top_k in topk_values:
        search_params = {"metric_type": "COSINE", "params": {"ef": max(MILVUS_SEARCH_EF_MIN, top_k * 2)}}
        latencies = []
        for i in range(search_repeats):
            t0 = time.time()
            _ = collection.search([query_vecs[i].tolist()], "embedding", search_params,
                                  limit=min(top_k, scale))
            latencies.append((time.time() - t0) * 1000)

        results[top_k] = {
            "p50": float(np.percentile(latencies, 50)),
            "p99": float(np.percentile(latencies, 99)),
            "mean": float(np.mean(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
        }
        print(f"    Milvus top_k={top_k:<3} p50={results[top_k]['p50']:6.2f}ms "
              f"p99={results[top_k]['p99']:6.2f}ms mean={results[top_k]['mean']:6.2f}ms")

    # ---- 清理 ----
    collection.release()
    utility.drop_collection(COLL_NAME)
    print(f"  [Milvus] 释放集合 {COLL_NAME}")

    return {
        "latency": results,
        "build_ms": build_ms,
        "insert_ms": insert_ms,
        "vector_mb": vec_mb,
        "estimated_index_mb": estimated_index_mb,
    }


# =============================================================================
# 主流程
# =============================================================================

def run_benchmark(scales: List[int], dim: int,
                  topk_values: List[int], search_repeats: int,
                  run_faiss: bool = True, run_milvus: bool = True) -> List[ScaleResult]:
    """运行全量级基准测试"""

    all_results: List[ScaleResult] = []

    print(f"\n{'='*72}")
    print(f"  数据量级延迟基准测试")
    print(f"{'='*72}")
    print(f"  量级: {[f'{s//1000}K' for s in scales]}")
    print(f"  维度: {dim}")
    print(f"  Top-K: {topk_values}")
    print(f"  每量级 query 数: {search_repeats}")
    print(f"  FAISS: {run_faiss}, Milvus: {run_milvus}")
    print(f"{'='*72}")

    for scale in scales:
        label = f"{scale} ({scale//1000}K)" if scale >= 1000 else str(scale)
        print(f"\n{'─'*72}")
        print(f"  [量级] {label} 条向量")
        print(f"{'─'*72}")

        result = ScaleResult(scale=scale, vector_dim=dim)

        if run_faiss:
            faiss_data = _faiss_benchmark_one_scale(scale, dim, topk_values, search_repeats)
            result.faiss_latency = faiss_data["latency"]
            result.faiss_build_ms = faiss_data["build_ms"]
            result.faiss_memory_mb = faiss_data["memory_mb"]

        if run_milvus:
            milvus_data = _milvus_benchmark_one_scale(scale, dim, topk_values, search_repeats)
            result.milvus_latency = milvus_data["latency"]
            result.milvus_build_ms = milvus_data["build_ms"]
            result.milvus_insert_ms = milvus_data["insert_ms"]
            result.milvus_memory_mb = milvus_data["vector_mb"]
            result.milvus_estimated_index_mb = milvus_data["estimated_index_mb"]

        all_results.append(result)

    return all_results


def print_summary(results: List[ScaleResult], topk_values: List[int]):
    """打印汇总对比表"""

    print(f"\n\n{'='*90}")
    print(f"  [Summary] 延迟 vs 数据量级 — p50/p99 (ms)")
    print(f"{'='*90}")

    # ---- FAISS 表 ----
    has_faiss = any(r.faiss_latency for r in results)
    if has_faiss:
        for top_k in topk_values:
            print(f"\n  --- FAISS HNSW (M={FAISS_M}, ef={FAISS_EF_SEARCH}), top_k={top_k} ---")
            print(f"  {'量级':>10} {'p50(ms)':>9} {'p99(ms)':>9} {'mean(ms)':>9} {'构建(ms)':>9} {'内存(MB)':>9}")
            print(f"  {'─'*60}")
            for r in results:
                if r.faiss_latency and top_k in r.faiss_latency:
                    pt = r.faiss_latency[top_k]
                    print(f"  {r.scale:>8,} {pt['p50']:>9.2f} {pt['p99']:>9.2f} "
                          f"{pt['mean']:>9.2f} {r.faiss_build_ms:>9.0f} {r.faiss_memory_mb:>9.1f}")

    # ---- Milvus 表 ----
    has_milvus = any(r.milvus_latency for r in results)
    if has_milvus:
        for top_k in topk_values:
            print(f"\n  --- Milvus HNSW (M={MILVUS_M}, ef=max(50, top_k*2)), top_k={top_k} ---")
            print(f"  {'量级':>10} {'p50(ms)':>9} {'p99(ms)':>9} {'mean(ms)':>9} "
                  f"{'插入(ms)':>9} {'索引(ms)':>9} {'索引(MB)':>9}")
            print(f"  {'─'*75}")
            for r in results:
                if r.milvus_latency and top_k in r.milvus_latency:
                    pt = r.milvus_latency[top_k]
                    print(f"  {r.scale:>8,} {pt['p50']:>9.2f} {pt['p99']:>9.2f} "
                          f"{pt['mean']:>9.2f} {r.milvus_insert_ms:>9.0f} "
                          f"{r.milvus_build_ms:>9.0f} {r.milvus_estimated_index_mb:>9.1f}")

    # ---- 结论 ----
    print(f"\n\n  [结论] 延迟随数据量级变化趋势")
    print(f"  {'─'*60}")
    _print_trend(results, "FAISS", "faiss_latency", has_faiss)
    _print_trend(results, "Milvus", "milvus_latency", has_milvus)

    _print_150ms_check(results)

    # ---- 内存 ----
    print(f"\n\n  [内存] 索引及向量数据估算")
    print(f"  {'─'*60}")
    print(f"  {'量级':>10} {'FAISS(MB)':>12} {'Milvus向量(MB)':>15} {'Milvus索引(MB)':>15}")
    print(f"  {'─'*60}")
    for r in results:
        fm = r.faiss_memory_mb if has_faiss else 0
        mm = r.milvus_memory_mb if has_milvus else 0
        mi = r.milvus_estimated_index_mb if has_milvus else 0
        print(f"  {r.scale:>8,} {fm:>12.1f} {mm:>15.1f} {mi:>15.1f}")

    # ---- 汇总 ----
    _print_total_ram(results)


def _print_trend(results: List[ScaleResult], label: str,
                  attr: str, has_data: bool):
    """打印单组趋势"""
    if not has_data:
        print(f"  {label}: (未测试)")
        return

    # 取 top_k=5 的 p50 看趋势
    for top_k in DEFAULT_TOPK_VALUES:
        vals = []
        for r in results:
            d = getattr(r, attr, {})
            if top_k in d:
                vals.append((r.scale, d[top_k]["p50"]))
        if vals:
            scales_str = " → ".join(f"{s//1000}K" if s >= 1000 else str(s) for s, _ in vals)
            lat_str = " → ".join(f"{v:.1f}ms" for _, v in vals)
            print(f"  {label} top_k={top_k} p50: {lat_str}")
            print(f"    (量级: {scales_str})")


def _print_150ms_check(results: List[ScaleResult]):
    """检查 150ms 目标是否满足"""
    print(f"\n  [目标验证] 150ms 延迟上限检查 (top_k=5):")
    # FAISS
    faiss_ok = True
    for r in results:
        if r.faiss_latency:
            p50 = r.faiss_latency.get(5, {}).get("p50", 999)
            p99 = r.faiss_latency.get(5, {}).get("p99", 999)
            if p50 > 150 or p99 > 150:
                print(f"    FAISS {r.scale//1000}K: p50={p50:.1f}ms, p99={p99:.1f}ms [!] 超出 150ms")
                faiss_ok = False
            else:
                print(f"    FAISS {r.scale//1000}K: p50={p50:.1f}ms, p99={p99:.1f}ms [+] 达标")
    if faiss_ok:
        print(f"    FAISS: 所有量级 p99 均在 150ms 以内 [+]")

    # Milvus
    milvus_ok = True
    for r in results:
        if r.milvus_latency:
            p50 = r.milvus_latency.get(5, {}).get("p50", 999)
            p99 = r.milvus_latency.get(5, {}).get("p99", 999)
            if p50 > 150 or p99 > 150:
                print(f"    Milvus {r.scale//1000}K: p50={p50:.1f}ms, p99={p99:.1f}ms [!] 超出 150ms")
                milvus_ok = False
            else:
                print(f"    Milvus {r.scale//1000}K: p50={p50:.1f}ms, p99={p99:.1f}ms [+] 达标")
    if milvus_ok:
        print(f"    Milvus: 所有量级 p99 均在 150ms 以内 [+]")

    # 综合判断
    if faiss_ok and milvus_ok:
        print(f"\n  [+] 结论: 1K-100K 量级下，FAISS + Milvus HNSW 均满足 < 150ms 延迟目标")
    else:
        print(f"\n  [!] 发现超限情况，见上述标记")


def _print_total_ram(results: List[ScaleResult]):
    """打印总内存估算（向量+索引）"""
    print(f"\n  [总体 RAM 估算] 100K 量级:")
    for r in results:
        if r.scale == 100_000:
            total_faiss = r.faiss_memory_mb
            total_milvus = r.milvus_estimated_index_mb
            total = total_faiss + total_milvus
            print(f"    FAISS 进程序内: {total_faiss:.1f} MB")
            print(f"    Milvus 独立进程: {total_milvus:.1f} MB (不含服务本身开销)")
            print(f"    总和: {total:.1f} MB ≈ {total/1024:.1f} GB")
            break


# =============================================================================
# 数据序列化
# =============================================================================

def _convert_for_json(obj):
    """递归转换 numpy 类型为 Python 原生类型"""
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def results_to_json(results: List[ScaleResult]) -> list:
    """将 ScaleResult 列表转为 JSON 可序列化格式"""
    out = []
    for r in results:
        out.append({
            "scale": r.scale,
            "vector_dim": r.vector_dim,
            "faiss": {
                "latency": _convert_for_json(r.faiss_latency),
                "build_ms": r.faiss_build_ms,
                "memory_mb": r.faiss_memory_mb,
            },
            "milvus": {
                "latency": _convert_for_json(r.milvus_latency),
                "build_ms": r.milvus_build_ms,
                "insert_ms": r.milvus_insert_ms,
                "vector_mb": r.milvus_memory_mb,
                "estimated_index_mb": r.milvus_estimated_index_mb,
            },
        })
    return out


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="不同数据量级延迟基准测试 — FAISS vs Milvus HNSW"
    )
    parser.add_argument("--scales", type=str,
                        default="1k,5k,10k,50k,100k",
                        help="测试量级，逗号分隔 (默认: 1k,5k,10k,50k,100k)")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM,
                        help=f"向量维度 (默认: {DEFAULT_DIM})")
    parser.add_argument("--topk", type=str, default="3,5,10",
                        help="Top-K 取值，逗号分隔 (默认: 3,5,10)")
    parser.add_argument("--repeats", type=int, default=DEFAULT_SEARCH_REPEATS,
                        help=f"每个量级 query 数 (默认: {DEFAULT_SEARCH_REPEATS})")
    parser.add_argument("--faiss-only", action="store_true",
                        help="只测试 FAISS HNSW")
    parser.add_argument("--milvus-only", action="store_true",
                        help="只测试 Milvus HNSW")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSON 文件路径")
    args = parser.parse_args()

    # 解析参数
    scales = []
    for s in args.scales.split(","):
        s = s.strip().lower()
        if s.endswith("k"):
            scales.append(int(s[:-1]) * 1000)
        else:
            scales.append(int(s))
    topk_values = [int(k.strip()) for k in args.topk.split(",")]
    run_faiss = not args.milvus_only
    run_milvus = not args.faiss_only

    # 连接 Milvus（仅当需要时）
    if run_milvus:
        from pymilvus import connections
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        print(f"[OK] Milvus 已连接: {MILVUS_HOST}:{MILVUS_PORT}")

    try:
        results = run_benchmark(scales, args.dim, topk_values,
                                args.repeats, run_faiss, run_milvus)
        print_summary(results, topk_values)

        # 保存结果
        if args.output:
            json_data = {
                "experiment": "latency_by_scale",
                "config": {
                    "dim": args.dim,
                    "scales": scales,
                    "topk_values": topk_values,
                    "search_repeats": args.repeats,
                    "faiss": {"M": FAISS_M, "efConstruction": FAISS_EF_CONSTRUCTION,
                              "efSearch": FAISS_EF_SEARCH},
                    "milvus": {"M": MILVUS_M, "efConstruction": MILVUS_EF_CONSTRUCTION,
                               "ef": f"max({MILVUS_SEARCH_EF_MIN}, top_k*2)"},
                },
                "results": results_to_json(results),
            }
            output_path = Path(args.output).absolute()
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(_convert_for_json(json_data), f, ensure_ascii=False, indent=2)
            print(f"\n[SAVED] 详细结果已保存到: {output_path}")
    finally:
        if run_milvus:
            from pymilvus import connections
            connections.disconnect("default")


if __name__ == "__main__":
    main()
