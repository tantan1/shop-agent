"""
工具选择 SFT 适配器评测：组 5 (SFT+自由生成) 与 组 6 (SFT+约束解码)。

验证计划 H2(语义层)/ H3(约束解码最优)：
  - 组 5：自由生成，解析 {"name": "<工具名>"}，验证 SFT 语义准确率
  - 组 6：约束解码，输出强制落在 5 工具名契约内，验证格式合规 100% 且语义不劣

与训练端共用同一 system 模板与 JSON 契约（避免模板错位，呼应微调坑④）。
评测集：data/lscale_S.json（S 规模 5 工具，五级难度，ambiguous/mixed 为近义判别命门）。

用法：
  .\\venv_cuda\\Scripts\\python scripts/eval_tool_select_sft.py --device cuda
  # 可选 --runs 3 --model ./models/Qwen2.5-1.5B-Instruct-tool-select --out outputs/eval_tool_select_group5_6.json
"""
import sys, os, json, argparse, time, statistics, random
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# scripts 目录无 __init__.py，直接用 importlib 加载模块（避免包结构问题）
import importlib.util
_gen_spec = importlib.util.spec_from_file_location(
    "gen_tool_selection_sft_data",
    os.path.join(os.path.dirname(ROOT), "train", "sft", "gen_tool_selection_sft_data.py"),
)
_gen_mod = importlib.util.module_from_spec(_gen_spec)
_gen_spec.loader.exec_module(_gen_mod)
build_system = _gen_mod.build_system
render_tool_list = _gen_mod.render_tool_list

CONTRACT_TMPL = '{{"name": "{tool}"}}'  # 与训练 assistant 输出一致


def load_model(model_path: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else "cpu", trust_remote_code=True)
    model.eval()
    tok._supports_thinking_ctrl = "enable_thinking" in (tok.chat_template or "")
    return model, tok


def build_prompt(tok, system: str, message: str) -> List[int]:
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": message}]
    kwargs = {"enable_thinking": False} if getattr(tok, "_supports_thinking_ctrl", False) else {}
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs)
    return tok(text, return_tensors="pt")["input_ids"][0].tolist()


def parse_free(gen: str, tool_set) -> (Optional[str], bool):
    """解析自由生成文本，返回 (name, 合规)。合规=可解析为单一 {'name': 合法工具}。"""
    g = gen.strip()
    # 取第一个 {...} 块
    import re
    m = re.search(r"\{[^{}]*\}", g, re.DOTALL)
    if not m:
        return None, False
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None, False
    if not isinstance(obj, dict) or "name" not in obj:
        return None, False
    name = obj["name"]
    if name in tool_set:
        return name, True
    # 容错：去连字符归一
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    for t in tool_set:
        if re.sub(r"[^a-z0-9]", "", t.lower()) == key:
            return t, True
    return name, False  # 输出结构合法但工具名超 schema


def gen_free(model, tok, input_ids, max_new_tokens=48):
    with torch.no_grad():
        out = model.generate(input_ids=torch.tensor([input_ids]).to(model.device),
                             max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][len(input_ids):], skip_special_tokens=True)
    return gen


def make_constrained_logits_processor(tok, full_strings, eos_id):
    """grammar 约束：每一步只允许生成 full_strings 中某一串的前缀 token；
    命中完整串后只允许 EOS。保证输出必为 {"name": "<合法工具>"}。
    （旧实现，保留以备对照；主路径改用下方 build_constrained_prefix_fn + generate 的 KV cache）"""
    def processor(generated_text: str):
        if generated_text in full_strings:
            return {eos_id}
        cands = [s for s in full_strings if s.startswith(generated_text)]
        if not cands:
            return {eos_id}
        allowed = set()
        for s in cands:
            suf = s[len(generated_text):]
            ids = tok.encode(suf, add_special_tokens=False)
            if ids:
                allowed.add(ids[0])
        return allowed
    return processor


def build_constrained_prefix_fn(tok, full_strings, eos_id, prompt_len):
    """transformers generate 用的 prefix_allowed_tokens_fn。

    语义与原 make_constrained_logits_processor 完全一致（基于解码文本的前缀约束），
    但借助 generate 内部的 KV cache，每个新 token 只做一次 O(1) 前向，
    而不是像原 gen_constrained 那样每次重算整段 prompt（O(L*T) 二次方）。
    M=8 这类长 prompt 下加速明显（约束解码从二次方降回线性）。
    """
    def prefix_allowed_tokens(batch_id, input_ids):
        seq = input_ids[batch_id] if input_ids.dim() > 1 else input_ids
        gen_ids = seq[prompt_len:].tolist()
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        if gen_text in full_strings:
            return [eos_id]
        cands = [s for s in full_strings if s.startswith(gen_text)]
        if not cands:
            return [eos_id]
        allowed = set()
        for s in cands:
            suf = s[len(gen_text):]
            ids = tok.encode(suf, add_special_tokens=False)
            if ids:
                allowed.add(ids[0])
        return list(allowed)
    return prefix_allowed_tokens


@torch.no_grad()
def gen_constrained(model, tok, input_ids, full_strings, max_new_tokens=48):
    eos_id = tok.eos_token_id
    prefix_fn = build_constrained_prefix_fn(tok, full_strings, eos_id, len(input_ids))
    out = model.generate(
        input_ids=torch.tensor([input_ids]).to(model.device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=eos_id,
        prefix_allowed_tokens_fn=prefix_fn,
    )
    gen_ids = out[0][len(input_ids):].tolist()
    return tok.decode(gen_ids, skip_special_tokens=True)


def majority(votes):
    from collections import Counter
    c = Counter(v for v in votes if v is not None)
    if not c:
        return None
    return c.most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/Qwen3-1.7B-unified")
    ap.add_argument("--data", default=os.path.join(ROOT, "data/lscale_S.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--out", default="outputs/eval_tool_select_group5_6.json")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 条（分块/冒烟用，0=全部）")
    ap.add_argument("--M", type=int, default=0,
                    help="评测候选集大小：每条 query 从工具池随机抽 M-1 个干扰项+gold 作候选（对齐 M8 训练/线上 Top-K）；"
                         "0=用完整工具列表（默认 S 规模 5 工具）。M>=8 时 --data 默认切到 lscale_M.json。")
    ap.add_argument("--distract-seed", type=int, default=20260728,
                    help="M>0 时干扰项抽样种子（故意不同于训练 seed=42，避免与训练干扰项完全相同）")
    args = ap.parse_args()

    # M>0 且未显式指定 --data 时，默认用 40 工具池评测集（与规模扫描同分布）
    if args.M and args.M > 0 and args.data == os.path.join(ROOT, "data/lscale_S.json"):
        args.data = os.path.join(ROOT, "data/lscale_M.json")
        print(f"[INFO] M={args.M}>0，评测数据默认切到 {args.data}")

    ds = json.load(open(args.data, encoding="utf-8"))
    pool = ds["tools"]
    pool_names = [t["name"] for t in pool]
    pool_by_name = {t["name"]: t for t in pool}
    rng = random.Random(args.distract_seed) if (args.M and args.M > 0) else None
    queries = ds["queries"]
    # 默认（M=0）用全集工具；M>0 时每条 query 单独构造候选集
    if not (args.M and args.M > 0) or args.M >= len(pool):
        tool_names = pool_names
        tool_set = set(pool_names)
        full_strings = [CONTRACT_TMPL.format(tool=t) for t in pool_names]
        use_full = True
    else:
        use_full = False
        # tool_names/full_strings 在循环内按候选集构造

    if use_full:
        print(f"[INFO] 模型 {args.model} | 全集工具 {len(tool_names)} 个 | 评测 {len(queries)} 条 | runs={args.runs} | device={args.device}")
    else:
        print(f"[INFO] 模型 {args.model} | M={args.M} 候选（gold+{args.M-1}干扰，从 {len(pool)} 池抽样）| 评测 {len(queries)} 条 | runs={args.runs} | device={args.device}")
    model, tok = load_model(args.model, args.device)

    rows = []
    lat_free, lat_con = [], []
    for i, q in enumerate(queries):
        msg, gold = q["message"], q["correct_tool"]
        level = q.get("level", "exact")
        # 构造该 query 的候选集与 system
        if use_full:
            system = build_system(pool)
            tool_set = set(tool_names)
            full_strings = [CONTRACT_TMPL.format(tool=t) for t in tool_names]
        else:
            others = [t for t in pool if t["name"] != gold]
            distract = rng.sample(others, args.M - 1)
            cands = [pool_by_name[gold]] + distract
            system = build_system(cands)
            tool_set = {t["name"] for t in cands}
            full_strings = [CONTRACT_TMPL.format(tool=t["name"]) for t in cands]
        input_ids = build_prompt(tok, system, msg)

        # ---- 组 5：自由生成 ----
        free_votes, free_compliant = [], 0
        t0 = time.monotonic()
        for _ in range(args.runs):
            g = gen_free(model, tok, input_ids, args.max_new)
            name, ok = parse_free(g, tool_set)
            if ok:
                free_compliant += 1
            free_votes.append(name)
        lat_free.append((time.monotonic() - t0) * 1000 / args.runs)
        free_sel = majority(free_votes)
        free_hit = (free_sel == gold)

        # ---- 组 6：约束解码 ----
        con_votes = []
        t1 = time.monotonic()
        for _ in range(args.runs):
            g = gen_constrained(model, tok, input_ids, full_strings, args.max_new)
            name, ok = parse_free(g, tool_set)
            con_votes.append(name if ok else None)
        lat_con.append((time.monotonic() - t1) * 1000 / args.runs)
        con_sel = majority(con_votes)
        con_hit = (con_sel == gold)
        con_compliant = all(v is not None for v in con_votes)

        rows.append({"idx": i, "message": msg, "level": level,
                     "correct_tool": gold, "free_sel": free_sel, "free_hit": free_hit,
                     "con_sel": con_sel, "con_hit": con_hit,
                     "free_compliant": free_compliant == args.runs,
                     "con_compliant": con_compliant})

        # 每条都打印（紧凑），避免长任务前台 idle 超时
        print(f"  [{i+1}/{len(queries)}] lv={level:9s} gold={gold:16s} "
              f"free={free_sel}/{('Y' if free_hit else 'N')} con={con_sel}/{('Y' if con_hit else 'N')} "
              f"acc={sum(r['free_hit'] for r in rows)/(i+1):.1%}|{sum(r['con_hit'] for r in rows)/(i+1):.1%}",
              flush=True)

    def acc(subset):
        n = len(subset)
        return round(sum(r["free_hit"] for r in subset) / n, 4), \
               round(sum(r["con_hit"] for r in subset) / n, 4) if n else (0, 0)

    free_overall, con_overall = acc(rows)
    levels = sorted({r["level"] for r in rows})
    per_level = {}
    for lv in levels:
        sub = [r for r in rows if r["level"] == lv]
        f, c = acc(sub)
        per_level[lv] = {"n": len(sub), "free_acc": f, "con_acc": c}
    amb = [r for r in rows if r["level"] == "ambiguous"]
    amb_free, amb_con = acc(amb)
    free_fmt = round(sum(r["free_compliant"] for r in rows) / len(rows), 4)
    con_fmt = round(sum(r["con_compliant"] for r in rows) / len(rows), 4)

    summary = {
        "model": args.model, "n": len(rows), "runs": args.runs,
        "eval_M": args.M if (args.M and args.M > 0) else len(pool_names),
        "pool_size": len(pool_names),
        "tools": pool_names,
        "group5_free_gen": {"overall_acc": free_overall, "format_compliance": free_fmt,
                            "latency_ms_p50": round(statistics.median(lat_free), 1),
                            "latency_ms_p95": round(sorted(lat_free)[int(0.95 * (len(lat_free) - 1))], 1)},
        "group6_constrained": {"overall_acc": con_overall, "format_compliance": con_fmt,
                               "latency_ms_p50": round(statistics.median(lat_con), 1),
                               "latency_ms_p95": round(sorted(lat_con)[int(0.95 * (len(lat_con) - 1))], 1)},
        "ambiguous_subset": {"n": len(amb), "free_acc": amb_free, "con_acc": amb_con},
        "per_level": per_level,
        "h2_pass": free_overall > 0.95,  # 简化判据：SFT 整体准确率≥95%
        "h3_pass": (con_overall >= free_overall) and (amb_con >= amb_free),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("\n========== 结果 ==========")
    print(f"组5 自由生成 : 整体准确率={free_overall:.1%}  格式合规={free_fmt:.1%}  p95延迟={summary['group5_free_gen']['latency_ms_p95']}ms")
    print(f"组6 约束解码 : 整体准确率={con_overall:.1%}  格式合规={con_fmt:.1%}  p95延迟={summary['group6_constrained']['latency_ms_p95']}ms")
    print(f"ambiguous子集: 自由={amb_free:.1%}  约束={amb_con:.1%}")
    print("逐难度:")
    for lv, d in per_level.items():
        print(f"  {lv:10s} n={d['n']:3d}  free={d['free_acc']:.1%}  con={d['con_acc']:.1%}")
    print(f"H2 PASS={summary['h2_pass']}  H3 PASS={summary['h3_pass']}")
    print(f"[DONE] 明细 -> {args.out}")


if __name__ == "__main__":
    main()
