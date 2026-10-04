"""工具选择训练数据质量校验器。

目的：检测工具选择 SFT 数据集（ShareGPT 格式）中的「脏数据 / 错误标签」，
使训练数据出错时有能力被发现，产出报告供人工复核；标注回流走 Langfuse（设计 3）。

检测项：
  1. LABEL_MISSING      correct_tool 缺失或为空
  2. LABEL_NOT_IN_CAND  correct_tool 不在该样本 system 提示列出的候选工具内（典型错误标签）
  3. ASSISTANT_MISMATCH assistant 的 JSON 输出 {\"name\": ...} 与 correct_tool 不一致
  4. EMPTY_QUERY        用户 query 为空
  5. CONFLICTING_LABEL  同一 query（归一化后）出现多个不同 correct_tool（标签冲突）

用法：
  # 仅检测并产出报告
  python scripts/train/sft/validate_tool_select_data.py \
      --in data/llamafactory/shop_tool_select_v1.json

  # 校验报告 JSON 可用于人工复核；标注完成后再由 Langfuse 导出回流 SFT
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# system 提示里候选工具行的格式："- query-order（查询订单）：..." 或 "- query_order: ..."
TOOL_LINE_RE = re.compile(r"^\s*-\s*([A-Za-z0-9_\-]+)\s*[（(：:]")
ASSISTANT_RE = re.compile(r'\{\s*"name"\s*:\s*"([^"]+)"\s*\}')


def _norm_query(q: str) -> str:
    """归一化 query：去标点/空白/大小写，用于冲突检测。"""
    return re.sub(r"[\s\W]+", "", (q or "").lower())


def _parse_conversations(sample: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """返回 (system, user_query, assistant_text)。

    兼容两种格式：
      - ShareGPT：conversations=[{system},{user},{assistant}]
      - 简化prod：conversations=[{user}] + 顶层 correct_tool
    """
    convs = sample.get("conversations", [])
    system = user = assistant = None
    for turn in convs:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "system":
            system = content
        elif role == "user":
            user = content
        elif role == "assistant":
            assistant = content
    return system, user, assistant


def _parse_candidate_tools(system: Optional[str]) -> List[str]:
    if not system:
        return []
    return TOOL_LINE_RE.findall(system)


def _parse_assistant_name(assistant: Optional[str]) -> Optional[str]:
    if not assistant:
        return None
    m = ASSISTANT_RE.search(assistant)
    return m.group(1) if m else None


def validate(samples: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    """返回 (issues, stats)。每个 issue: {index, sample_id, issues:[...], sample}。"""
    issues: List[Dict] = []
    stats = defaultdict(int)

    # 第一遍：单样本检查 + 收集 query->correct_tool 映射
    query_map: Dict[str, set] = defaultdict(set)
    for i, s in enumerate(samples):
        sample_issues: List[str] = []
        correct = s.get("correct_tool")
        system, user, assistant = _parse_conversations(s)
        cands = _parse_candidate_tools(system)
        asst_name = _parse_assistant_name(assistant)

        if not correct:
            sample_issues.append("LABEL_MISSING")
        elif cands and correct not in cands:
            sample_issues.append("LABEL_NOT_IN_CAND")
        if asst_name is not None and correct is not None and asst_name != correct:
            sample_issues.append("ASSISTANT_MISMATCH")
        if not user or not user.strip():
            sample_issues.append("EMPTY_QUERY")
        if user:
            query_map[_norm_query(user)].add(correct)

        for code in sample_issues:
            stats[code] += 1
        if sample_issues:
            issues.append({
                "index": i,
                "correct_tool": correct,
                "candidate_tools": cands,
                "issues": sample_issues,
                "sample": s,
            })

    # 第二遍：冲突标签（同一 query 多个不同 correct_tool）
    conflicted = {q: cset for q, cset in query_map.items() if len(cset) > 1 and None not in cset}
    if conflicted:
        norm_to_samples: Dict[str, List[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            _, user, _ = _parse_conversations(s)
            if user and _norm_query(user) in conflicted:
                norm_to_samples[_norm_query(user)].append(i)
        seen = set()
        for q, idxs in norm_to_samples.items():
            key = tuple(sorted(idxs))
            if key in seen:
                continue
            seen.add(key)
            stats["CONFLICTING_LABEL"] += len(idxs)
            for i in idxs:
                issues.append({
                    "index": i,
                    "correct_tool": samples[i].get("correct_tool"),
                    "candidate_tools": _parse_candidate_tools(_parse_conversations(samples[i])[0]),
                    "issues": ["CONFLICTING_LABEL"],
                    "sample": samples[i],
                })

    return issues, dict(stats)





def main():
    ap = argparse.ArgumentParser(description="工具选择训练数据质量校验器")
    ap.add_argument("--in", dest="in_path", default="data/llamafactory/shop_tool_select_v1.json")
    ap.add_argument("--report", default=None, help="校验报告输出路径（默认 <in>.validation.json）")
    args = ap.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"[INFO] 载入 {len(samples)} 条样本：{args.in_path}")

    issues, stats = validate(samples)
    print("[INFO] 校验统计：")
    for code, n in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  - {code}: {n}")
    print(f"[INFO] 共 {len(issues)} 条样本存在问题")

    report_path = args.report or (args.in_path + ".validation.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"total": len(samples), "stats": stats, "issues": issues}, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 校验报告已写出：{report_path}")

    # 有错误时以非 0 退出码，便于 CI / 数据门禁
    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
