"""
工具选择规模外推扫描：用 base 模型测候选集大小 M∈{5,10,15,20,30,40} 的准确率，
定位规模外推失效拐点（为"减小 M 以省训练时间但仍保明显结果"提供定量依据）。

实验设计（关键，保证规模变量被干净隔离）：
  - 同一批 query 固定不变，仅"候选集大小 M"变化；
  - 对每条 query，候选集 = {gold} ∪ (工具池除 gold 外的前 M-1 个)，
    因此 gold 必在候选内，且小 M 候选是更大 M 候选的子集（嵌套可比）；
  - 难度只来自"干扰项数量 = M-1"，纯粹反映规模/注意力稀释效应。
评测：自由生成(组5) + 约束解码(组6)；base 模型 do_sample=False 确定性，runs=1。

用法：
  .\venv_cuda\Scripts\python scripts/eval_tool_select_scale_sweep.py --device cuda
  # 自定义档位: --ms 5,10,15,20,30,40
"""
import sys, os, json, argparse, time, statistics
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# 复用训练/评测同款 system 模板（避免模板错位，呼应微调坑④）
from scripts.gen_tool_selection_sft_data import build_system

CONTRACT_TMPL = '{{"name": "{tool}"}}'  # 与训练 assistant 输出一致


def load_model(model_path: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else "cpu", trust_remote_code=False)
    model.eval()
    return model, tok


def build_prompt(tok, system: str, message: str) -> List[int]:
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": message}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt")["input_ids"][0].tolist()


def parse_free(gen: str, tool_set) -> (Optional[str], bool):
    """解析自由生成文本，返回 (name, 合规)。合规=可解析为单一 {'name': 合法工具}。"""
    import re
    g = gen.strip()
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
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    for t in tool_set:
        if re.sub(r"[^a-z0-9]", "", t.lower()) == key:
            return t, True
    return name, False


def gen_free(model, tok, input_ids, max_new_tokens=24):
    with torch.no_grad():
        out = model.generate(input_ids=torch.tensor([input_ids]).to(model.device),
                             max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][len(input_ids):], skip_special_tokens=True)
    return gen


def make_constrained_logits_processor(tok, full_strings, eos_id):
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


class ConstraintLogitsProcessor(LogitsProcessor):
    """将逐 token 前缀匹配约束搬入 transformers 的 LogitsProcessor，复用 generate 的 KV cache。

    原 gen_constrained 用手写循环、每步重算整段输入的 attention（O(n^2) 浪费）。
    改为 generate + 本 processor 后，KV cache 自动生效：仅首步做全序列 prefill、
    其后每步只解码单 token，约束语义与原实现完全等价。
    """

    def __init__(self, tok, full_strings, eos_id, prompt_len):
        self.tok = tok
        self.full_strings = full_strings
        self.eos_id = eos_id
        self.prompt_len = prompt_len

    def _allowed(self, gen_text: str):
        if gen_text in self.full_strings:
            return {self.eos_id}
        cands = [s for s in self.full_strings if s.startswith(gen_text)]
        if not cands:
            return {self.eos_id}
        allowed = set()
        for s in cands:
            suf = s[len(gen_text):]
            ids = self.tok.encode(suf, add_special_tokens=False)
            if ids:
                allowed.add(ids[0])
        return allowed

    def __call__(self, input_ids, scores):
        gen_ids = input_ids[0, self.prompt_len:]
        gen_text = self.tok.decode(gen_ids, skip_special_tokens=True)
        allowed = self._allowed(gen_text)
        mask = torch.full_like(scores, float("-inf"))
        for a in allowed:
            mask[0, a] = 0.0
        return scores + mask


@torch.no_grad()
def gen_constrained(model, tok, input_ids, full_strings, max_new_tokens=24):
    eos_id = tok.eos_token_id
    proc = ConstraintLogitsProcessor(tok, full_strings, eos_id, len(input_ids))
    out = model.generate(input_ids=torch.tensor([input_ids]).to(model.device),
                         max_new_tokens=max_new_tokens, do_sample=False,
                         logits_processor=[proc],
                         pad_token_id=eos_id, eos_token_id=eos_id)
    gen = tok.decode(out[0][len(input_ids):], skip_special_tokens=True)
    return gen


def majority(votes):
    from collections import Counter
    c = Counter(v for v in votes if v is not None)
    if not c:
        return None
    return c.most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models/Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data/lscale_M.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ms", default="5,10,15,20,30,40",
                    help="候选集大小档位，逗号分隔（含 S=5 与 M=40 作对照）")
    ap.add_argument("--runs", type=int, default=1, help="base 确定性，默认 1")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/eval_scale_sweep_base.json"))
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 条 query（0=全部）")
    args = ap.parse_args()

    ds = json.load(open(args.data, encoding="utf-8"))
    pool = ds["tools"]
    pool_names = [t["name"] for t in pool]
    name_to_tool = {t["name"]: t for t in pool}
    queries = ds["queries"]
    if args.limit:
        queries = queries[:args.limit]
    ms = [int(x) for x in args.ms.split(",")]
    for M in ms:
        assert M <= len(pool), f"M={M} 超过工具池大小 {len(pool)}"

    print(f"[INFO] 模型 {args.model} | 工具池 {len(pool)} | query {len(queries)} | "
          f"档位 M={ms} | runs={args.runs} | device={args.device}")
    model, tok = load_model(args.model, args.device)

    results = {}
    for M in ms:
        rows = []
        lat_free, lat_con = [], []
        for i, q in enumerate(queries):
            gold = q["correct_tool"]
            if gold not in name_to_tool:
                continue
            # 候选：gold + 池顺序除 gold 外前 M-1（嵌套可比，gold 必在）
            others = [t for t in pool if t["name"] != gold]
            cand = [name_to_tool[gold]] + others[:M - 1]
            cand_names = [t["name"] for t in cand]
            tool_set = set(cand_names)
            system = build_system(cand)
            full_strings = [CONTRACT_TMPL.format(tool=t) for t in cand_names]
            msg = q["message"]
            input_ids = build_prompt(tok, system, msg)

            # ---- 自由生成 ----
            free_votes = []
            t0 = time.monotonic()
            for _ in range(args.runs):
                g = gen_free(model, tok, input_ids)
                name, ok = parse_free(g, tool_set)
                free_votes.append(name)
            lat_free.append((time.monotonic() - t0) * 1000 / args.runs)
            free_sel = majority(free_votes)
            free_hit = (free_sel == gold)

            # ---- 约束解码 ----
            con_votes = []
            t1 = time.monotonic()
            for _ in range(args.runs):
                g = gen_constrained(model, tok, input_ids, full_strings)
                name, ok = parse_free(g, tool_set)
                con_votes.append(name if ok else None)
            lat_con.append((time.monotonic() - t1) * 1000 / args.runs)
            con_sel = majority(con_votes)
            con_hit = (con_sel == gold)
            con_compliant = all(v is not None for v in con_votes)

            rows.append({"idx": i, "level": q.get("level", "exact"), "correct_tool": gold,
                         "free_sel": free_sel, "free_hit": free_hit,
                         "con_sel": con_sel, "con_hit": con_hit, "con_compliant": con_compliant})
            print(f"  [M={M}] [{i+1}/{len(queries)}] lv={q.get('level','exact'):9s} "
                  f"gold={gold:16s} free={free_sel}/{('Y' if free_hit else 'N')} "
                  f"con={con_sel}/{('Y' if con_hit else 'N')}", flush=True)

        n = len(rows)
        free_acc = round(sum(r["free_hit"] for r in rows) / n, 4) if n else 0
        con_acc = round(sum(r["con_hit"] for r in rows) / n, 4) if n else 0
        con_fmt = round(sum(r["con_compliant"] for r in rows) / n, 4) if n else 0
        levels = sorted({r["level"] for r in rows})
        per_level = {}
        for lv in levels:
            sub = [r for r in rows if r["level"] == lv]
            per_level[lv] = {"n": len(sub),
                             "free_acc": round(sum(r["free_hit"] for r in sub) / len(sub), 4),
                             "con_acc": round(sum(r["con_hit"] for r in sub) / len(sub), 4)}
        amb = [r for r in rows if r["level"] == "ambiguous"]
        amb_free = round(sum(r["free_hit"] for r in amb) / len(amb), 4) if amb else 0
        amb_con = round(sum(r["con_hit"] for r in amb) / len(amb), 4) if amb else 0
        results[M] = {
            "n": n, "free_acc": free_acc, "con_acc": con_acc, "con_fmt": con_fmt,
            "ambiguous": {"n": len(amb), "free_acc": amb_free, "con_acc": amb_con},
            "per_level": per_level,
            "latency_ms_p50_free": round(statistics.median(lat_free), 1) if lat_free else 0,
            "latency_ms_p95_free": round(sorted(lat_free)[int(0.95 * (len(lat_free) - 1))], 1) if lat_free else 0,
            "latency_ms_p50_con": round(statistics.median(lat_con), 1) if lat_con else 0,
            "latency_ms_p95_con": round(sorted(lat_con)[int(0.95 * (len(lat_con) - 1))], 1) if lat_con else 0,
        }
        print(f"[M={M}] free_acc={free_acc:.1%} con_acc={con_acc:.1%} con_fmt={con_fmt:.1%} "
              f"amb_free={amb_free:.1%} amb_con={amb_con:.1%} "
              f"p95free={results[M]['latency_ms_p95_free']}ms")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"model": args.model, "ms": ms, "results": results},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n==== 规模外推扫描汇总 ====")
    print(f"{'M':>4} {'free':>7} {'con':>7} {'con_fmt':>8} {'amb_free':>9} {'amb_con':>9} {'p95free(ms)':>11}")
    for M in ms:
        r = results[M]; a = r["ambiguous"]
        print(f"{M:>4} {r['free_acc']:>7.1%} {r['con_acc']:>7.1%} {r['con_fmt']:>8.1%} "
              f"{a['free_acc']:>9.1%} {a['con_acc']:>9.1%} {r['latency_ms_p95_free']:>11}")
    print(f"[DONE] -> {args.out}")


if __name__ == "__main__":
    main()
