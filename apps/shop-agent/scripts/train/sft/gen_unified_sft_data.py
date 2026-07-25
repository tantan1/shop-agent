"""
生成 unified 数据集：参数抽取 + 工具选择 合并为单模型 SFT 训练集。

背景（多任务合并第一步）：
  参数抽取（PARAM_EXTRACTION_PROMPTS）与工具选择（build_system 工具列表）
  是两种 system 结构差异很大的任务。把它们合并进一个 1.7B 模型（unified），
  替代生产环境两份独立 Qwen3-1.7B，省一份内存。safety 任务已砍（合规主判
  归云端大模型，见设计决策），本数据集不含 safety 数据。

设计决策（模板一致性，呼应微调坑④）：
  - **不改 prompt 结构，纯合并**：param 样本保留 PARAM_EXTRACTION_PROMPTS
    作 system，tool_select 样本保留工具列表 system。两个任务的 system 语义
    差异足够大，模型可自然区分任务，无需显式 task 前缀。
  - task 字段仅作**顶层元信息**（LLaMA-Factory 不使用），供统计/后续升级
    （若评测发现任务混淆，再升级为前缀注入 + 同步改评测端）。
  - 因此现有评测脚本（eval_tool_select_sft.py / eval_field_level.py /
    eval_sft_before_after.py）零改动即可直接评测 unified 模型（训练=评测同构）。

数据规模（默认）：
  - shop_param_v1_aug.json        (1729 条，param_extract)
  - shop_tool_select_S.json       ( 267 条，5 工具)
  - shop_tool_select_M8.json      ( 960 条，8 候选 Top-K 对齐生产预过滤)
  合计约 2956 条，param:tool_select ≈ 1.4:1。

用法：
  python scripts/gen_unified_sft_data.py              # 默认 S + M8
  python scripts/gen_unified_sft_data.py --register   # 注册 dataset_info.json
  # 可选把 M（40 工具全集）也并入，覆盖更宽泛候选分布：
  python scripts/gen_unified_sft_data.py --include-full-m
"""

import sys
import os
import json
import argparse
import random
from typing import Dict, List, Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 仓库根 = apps/shop-agent/scripts -> apps/shop-agent -> apps -> 仓库根
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, ROOT)

DATASET_INFO_TAGS = {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "system_tag": "system",
    "messages": "conversations",
}


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mark_param(rec: Dict, task: str = "param_extract") -> Dict:
    """参数抽取样本：保留 conversations 原样，加 task/intent 元信息。"""
    return {
        "task": task,
        "intent": rec.get("intent", ""),
        "conversations": rec.get("conversations", []),
    }


def mark_tool(rec: Dict, task: str = "tool_select") -> Dict:
    """工具选择样本：保留 conversations 原样，加 task/scale/level 元信息。"""
    return {
        "task": task,
        "scale": rec.get("scale", ""),
        "level": rec.get("level", ""),
        "domain_signal": rec.get("domain_signal", ""),
        "correct_tool": rec.get("correct_tool", ""),
        "conversations": rec.get("conversations", []),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data/llamafactory/shop_unified_v1.json"))
    ap.add_argument("--param-data", default=os.path.join(ROOT, "data/llamafactory/shop_param_v1_aug.json"))
    ap.add_argument("--tool-select-s", default=os.path.join(ROOT, "data/llamafactory/shop_tool_select_S.json"))
    ap.add_argument("--tool-select-m8", default=os.path.join(ROOT, "data/llamafactory/shop_tool_select_M8.json"))
    ap.add_argument("--include-full-m", action="store_true",
                    help="额外并入 40 工具全集 shop_tool_select_M.json（覆盖更宽候选分布）")
    ap.add_argument("--tool-select-m", default=os.path.join(ROOT, "data/llamafactory/shop_tool_select_M.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--register", action="store_true", help="注册进 dataset_info.json")
    args = ap.parse_args()

    samples: List[Dict] = []
    stats = {}

    # 1) 参数抽取
    param = load_json(args.param_data)
    samples.extend(mark_param(r) for r in param)
    stats["param_extract"] = len(param)

    # 2) 工具选择（S 5 工具）
    ts_s = load_json(args.tool_select_s)
    samples.extend(mark_tool(r) for r in ts_s)
    stats["tool_select_S"] = len(ts_s)

    # 3) 工具选择（M8 8 候选，对齐生产 Top-K 预过滤）
    ts_m8 = load_json(args.tool_select_m8)
    samples.extend(mark_tool(r) for r in ts_m8)
    stats["tool_select_M8"] = len(ts_m8)

    # 4) 可选：40 工具全集
    if args.include_full_m:
        ts_m = load_json(args.tool_select_m)
        samples.extend(mark_tool(r) for r in ts_m)
        stats["tool_select_M_full"] = len(ts_m)

    # 校验：至少 3 轮对话、assistant 是 JSON
    before = len(samples)
    rng = random.Random(args.seed)
    valid = []
    for s in samples:
        convs = s.get("conversations", [])
        if len(convs) < 3:
            continue
        if not any(c.get("role") == "system" for c in convs):
            continue
        valid.append(s)
    rng.shuffle(valid)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(valid, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 写出 {len(valid)} 条（过滤 {before - len(valid)} 条异常）-> {args.out}")
    print(f"[STATS] {stats}")

    ds_name = os.path.splitext(os.path.basename(args.out))[0]
    info = {ds_name: {
        "file_name": os.path.basename(args.out),
        "formatting": "sharegpt",
        "tags": DATASET_INFO_TAGS,
    }}
    print("\n# dataset_info.json 片段：")
    print(json.dumps(info, ensure_ascii=False, indent=2))

    if args.register:
        di_path = os.path.join(os.path.dirname(args.out), "dataset_info.json")
        existing: Dict = {}
        if os.path.exists(di_path):
            with open(di_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing.update(info)
        with open(di_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"[REGISTER] 已写入 {di_path}")


if __name__ == "__main__":
    main()
