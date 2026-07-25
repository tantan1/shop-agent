"""
生成「工具选择」SFT 微调数据（ShareGPT 格式），复用参数抽取微调范式。

为什么不直接用现成的 Qwen2.5-1.5B-Instruct-sft：
  那是「参数抽取」适配器（给定意图抽字段），与本计划的「工具选择」适配器
  （给定 query 从多工具中选对工具）是不同任务、不同输出 schema。套到工具选择
  评测上会因任务/模板错位污染 H2/H3 结论。本脚本按参数抽取「同一套微调范式」
  生成工具选择训练集，供 Phase 4 训练专用适配器。

范式对齐（关键，呼应微调坑：对话模板一致性 / 数据分布对齐）：
  - 与现有 param-sft 完全一致：assistant 直接输出 JSON 文本，工具列表以文本
    嵌入 system 提示，不用原生 tools= schema（与 shop_param_v1.json 同构）。
  - 训练/评测共享同一 system 模板与 JSON 契约 {"name": "<工具名>"}，约束解码
    时把 grammar 约束到该契约即可（H1/H3 的「约束解码」落点）。

数据来源（分布对齐）：直接复用 gen_synthetic_lscale.py 产出的
  data/lscale_S.json（5 工具）、data/lscale_M.json（40 工具）。
  它们的 queries 已覆盖 exact/synonym/implied/ambiguous/mixed 五级难度，
  ambiguous/mixed 即近义判别命门，天然覆盖。

输出：
  - data/llamafactory/shop_tool_select_v1.json   (ShareGPT 训练集)
  - 终端打印 dataset_info.json 片段（粘进 LLaMA-Factory 的 data/dataset_info.json）

用法：
  # 离线合成（默认，零依赖，立即可跑）
  python scripts/gen_tool_selection_sft_data.py

  # 指定规模 + 近义增广（强化 H2/H3 的 ambiguous 子集）
  python scripts/gen_tool_selection_sft_data.py --scales S M --augment --per-aug 2

  # 注册进 dataset_info.json
  python scripts/gen_tool_selection_sft_data.py --register
"""

import sys
import os
import json
import argparse
import random
from typing import Dict, List, Any, Optional

# ── 允许从项目根运行脚本 ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 与现有 param-sft 相同的 ShareGPT 标签结构（data/llamafactory/dataset_info.json）
# _to_sharegpt 产出 conversations 列表，每条含 role/content，故 tags 必须含 role_tag/content_tag/messages
DATASET_INFO_TAGS = {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "system_tag": "system",
    "messages": "conversations",
}

# =============================================================================
# 0. 工具选择 system 模板 + JSON 契约（训练/评测必须共用，避免模板错位）
# =============================================================================
TOOL_SELECT_SYSTEM_TMPL = """你是一个电商客服工具选择器。根据用户消息，从下面的候选工具中选出【最相关的一个】（单选，只输出一个工具名）。
若消息同时涉及多个操作，选最核心的那一个。
候选工具：
{tool_list}

输出格式（严格 JSON，不要任何解释或多余文字）：
{{"name": "<工具名>"}}"""

# 口语前缀池：镜像测试集真实口语分布，强化「看意图吃饭」以外的表层鲁棒性
PREFIX_POOL = ["帮我看下，", "那个，", "在吗，", "我想问下，", "哎你好，", "麻烦问下，", "呃，", "就是，"]
SUFFIX_POOL = ["，帮我弄下", "，怎么搞", "，该点哪个", "，谢谢", "，急", "，帮忙看看"]


def render_tool_list(tools: List[Dict]) -> str:
    """把工具列表渲染成文本，嵌入 system 提示（与 param-sft 同构：文本而非 native tools=）。"""
    lines = []
    for t in tools:
        kw = "、".join(t.get("trigger_keywords", [])[:6])
        disp = t.get("display_name", "") or t.get("name", "")
        lines.append(f"- {t['name']}（{disp}）：{t.get('description', '')}［触发词：{kw}］")
    return "\n".join(lines)


def build_system(tools: List[Dict]) -> str:
    return TOOL_SELECT_SYSTEM_TMPL.format(tool_list=render_tool_list(tools))


def _to_sharegpt(system: str, query: str, correct_tool: str, meta: Dict) -> Dict:
    return {
        "scale": meta.get("scale"),
        "level": meta.get("level"),
        "domain_signal": meta.get("domain_signal"),
        "correct_tool": correct_tool,  # 元信息，LLaMA-Factory 不使用
        "conversations": [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
            {"role": "assistant", "content": json.dumps({"name": correct_tool}, ensure_ascii=False)},
        ],
    }


# =============================================================================
# 1. 主数据：复用 lscale 数据集的 query + tool 列表（分布对齐）
# =============================================================================
def build_from_lscale(data_dir: str, scales: List[str], M: int = 0,
                      rng: random.Random = None) -> List[Dict]:
    """生成训练样本。

    M==0（默认）：沿用原行为，每条样本用该 scale 的完整工具列表作候选（S=5 / M=40）。
    M>0 ：候选集大小钉为 M —— 对每条 query 从工具池随机抽 M-1 个干扰项 + gold，
          仅把这组 M 个工具渲染进 system。这对应线上软预过滤把候选压到 Top-K=M
          的分布（训练/推理同分布），且提示最短、训练最快、规模外推失效被预过滤吸收。
    """
    samples = []
    for sc in scales:
        path = os.path.join(data_dir, f"lscale_{sc}.json")
        if not os.path.exists(path):
            print(f"[WARN] 找不到 {path}，跳过 scale={sc}（先跑 gen_synthetic_lscale.py 生成）")
            continue
        with open(path, "r", encoding="utf-8") as f:
            ds = json.load(f)
        tools = ds.get("tools", [])
        name_to_tool = {t["name"]: t for t in tools}
        tool_names = {t["name"] for t in tools}
        n_used = 0
        for q in ds.get("queries", []):
            correct = q.get("correct_tool")
            if correct not in tool_names:
                continue  # 工具不在本规模列表里则跳过，避免错误标签
            if M and M > 0 and len(tools) >= M:
                others = [t for t in tools if t["name"] != correct]
                distract = rng.sample(others, M - 1) if rng is not None else others[:M - 1]
                cand = [name_to_tool[correct]] + distract
                system = build_system(cand)
            else:
                if M and M > 0:
                    print(f"[WARN] scale={sc} 工具池仅 {len(tools)} < M={M}，退化为全集候选")
                system = build_system(tools)
            samples.append(_to_sharegpt(system, q["message"], correct,
                                        {"scale": sc, "level": q.get("level"),
                                         "domain_signal": q.get("domain_signal")}))
            n_used += 1
        print(f"[INFO] scale={sc}: 工具池 {len(tools)} | 候选 M={M or len(tools)} | 写入 {n_used} 条")
    return samples


# =============================================================================
# 2. 近义增广：对 ambiguous/mixed 难例做口语前缀/后缀改写（同标签，强化 H2/H3）
# =============================================================================
def augment_hard(samples: List[Dict], per_aug: int, rng: random.Random) -> List[Dict]:
    aug = []
    for s in samples:
        if s.get("level") not in ("ambiguous", "mixed"):
            continue
        user_turn = s["conversations"][1]
        base_msg = user_turn["content"]
        system = s["conversations"][0]["content"]
        correct = s["correct_tool"]
        meta = {"scale": s.get("scale"), "level": s.get("level"),
                "domain_signal": s.get("domain_signal")}
        for _ in range(per_aug):
            # 随机选前缀或后缀改写，制造表层扰动但不改意图
            if rng.random() < 0.6:
                msg = rng.choice(PREFIX_POOL) + base_msg
            else:
                msg = base_msg + rng.choice(SUFFIX_POOL)
            aug.append(_to_sharegpt(system, msg, correct, meta))
    print(f"[INFO] 近义增广（ambiguous/mixed 口语改写）：+{len(aug)} 条")
    return aug


# =============================================================================
# 3. 主流程
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/llamafactory/shop_tool_select_v1.json"))
    ap.add_argument("--scales", nargs="+", default=["S", "M"],
                    help="复用哪些 lscale 数据集的规模（默认 S M，对齐评测 S/M 规模）")
    ap.add_argument("--M", type=int, default=0,
                    help="候选集大小：每条样本从工具池随机抽 M-1 个干扰项+gold 作候选；"
                         "0=用完整工具列表（默认）。训练建议 8（对齐软预过滤 Top-K）。")
    ap.add_argument("--augment", action="store_true", help="对 ambiguous/mixed 做口语改写增广")
    ap.add_argument("--per-aug", type=int, default=2, help="每条难例增广条数")
    ap.add_argument("--register", action="store_true", help="把 dataset_info 片段写进 data/llamafactory/dataset_info.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # M>0 时强制用大工具池（需 >= M），自动把 scales 收敛为 M（40 工具）以免池不够抽
    scales = args.scales
    if args.M and args.M > 0 and "M" not in scales:
        scales = ["M"]
        print(f"[INFO] --M={args.M}>0，候选从 40 工具池抽样，scales 收敛为 ['M']")
    samples = build_from_lscale(args.data_dir, scales, M=args.M, rng=rng)
    if args.augment:
        samples.extend(augment_hard(samples, args.per_aug, rng))
    rng.shuffle(samples)

    # 写文件
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"[DONE] 写出 {len(samples)} 条 -> {args.out}")

    # dataset_info.json 片段
    ds_name = os.path.splitext(os.path.basename(args.out))[0]
    info = {ds_name: {
        "file_name": os.path.basename(args.out),
        "formatting": "sharegpt",
        "tags": DATASET_INFO_TAGS,
    }}
    print("\n# 把下面这段加进 LLaMA-Factory 的 data/dataset_info.json：")
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
