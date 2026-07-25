"""
eval_lscale_routing.py — L 规模分层路由测试 harness 扩展
========================================================

依据：docs/L规模分层路由测试计划.md (Phase 1-7)
复用：benchmark_tool_selection_pipeline 的 embedding 软预过滤 + FC 思路，
      但作为独立脚本实现一层「域路由」(domain_route) 与路由开关，
      不改动共享 harness，避免与 doc 23 计划并行冲突。

域路由在 run_soft_filter 之前：把「全集」替换为「命中域并集」再进既有软预过滤。
支持：
  --route-mode  none(基线 R0) / hard(硬域路由 R1) / soft(软域路由 R2, 默认)
  --classifier  keyword(规则) / embedding(语义, 主线) / functional(功能聚类, H3 对照)
  --domain-top-m 软路由保留候选域数(默认 2)
  --scale      S/M/L/LL（载入 data/lscale_<scale>.json）
  --top-k       recall@K 与 FC-on-TopK 的 K
  --fc on/off  是否在域内候选上跑 Function Calling 做端到端准确率（默认 off，快；embedding 已够验 H1/H2/H3）
  --fc-model / --fc-device  FC 模型（固定 ./models/Qwen2.5-1.5B-Instruct + cuda）
  --runs        FC 每条 runs 次多数票消抖
  --sweep       一次性跑 none/hard/soft（embedding 分类器）并打印对比表
  --output      JSON 输出路径

指标：recall@K(路由后候选含正确工具) / 端到端FC准确率 / 上下文token比 /
      硬路由不可逆丢条率 / 软路由救回率 / 域分类准确率(strong/weak) /
      功能聚类对照(H3) / 延迟。

用法示例：
  # 快速 embedding 扫描（不含 FC）
  python scripts/eval_lscale_routing.py --scale L --route-mode soft --classifier embedding --sweep
  # 端到端（含 FC，L 规模 R2）
  python scripts/eval_lscale_routing.py --scale L --route-mode soft --classifier embedding --fc on --output lscale_L_R2.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------- embedding 模型（复用 harness 同款） ----------------
EMB_MODEL = "./models/BAAI/bge-m3"
FC_MODEL = "./models/Qwen2.5-1.5B-Instruct"

FUNCTIONAL_REPR = {
    "query": "查询查看类功能：罗列列表、查看详情、跟踪进度、查询状态等只读操作，不修改数据。",
    "operate": "操作提交类功能：申请、提交、取消、修改等会真实写入或变更数据的动作。",
    "feedback": "反馈投诉类功能：投诉、建议、举报、差评处理等面向诉求的表达。",
}


def _load_embed_model(path: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(path)


def _encode(model, texts: List[str]) -> List[List[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()


# ================================================================
# 数据加载与索引
# ================================================================
def load_dataset(scale: str, data_dir: str = "data") -> Dict[str, Any]:
    path = os.path.join(data_dir, f"lscale_{scale}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(data: Dict[str, Any]):
    tools = data["tools"]
    domains = data["domains"]
    tool_by_name = {t["name"]: t for t in tools}
    domain_to_tools: Dict[str, Set[str]] = defaultdict(set)
    func_to_tools: Dict[str, Set[str]] = defaultdict(set)
    for t in tools:
        domain_to_tools[t["domain"]].add(t["name"])
        for cd in t.get("cross_domains", []):
            domain_to_tools[cd].add(t["name"])
        func_to_tools[t["functional"]].add(t["name"])
    dom_repr = {k: v["repr"] for k, v in domains.items()}
    return tool_by_name, domain_to_tools, func_to_tools, dom_repr


# ================================================================
# 路由
# ================================================================
def route_keyword(msg: str, domain_to_tools: Dict[str, Set[str]], tool_by_name: Dict[str, Any],
                  domains: Dict[str, Any]) -> List[str]:
    matched = []
    for dk, dm in domains.items():
        kws = dm.get("trigger_keywords", [])
        if any(kw in msg for kw in kws):
            matched.append(dk)
    return matched  # 可能多个（跨域关键词）；空=无命中


def route_embedding(msg_emb: List[float], dom_repr_emb: Dict[str, List[float]],
                    top_m: int) -> List[Tuple[str, float]]:
    sims = [(dk, float(sum(a * b for a, b in zip(msg_emb, eb)))) for dk, eb in dom_repr_emb.items()]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_m]


def route_functional(msg_emb: List[float], func_repr_emb: Dict[str, List[float]],
                     top_m: int) -> List[Tuple[str, float]]:
    sims = [(fk, float(sum(a * b for a, b in zip(msg_emb, eb)))) for fk, eb in func_repr_emb.items()]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_m]


# ================================================================
# FC 推理（域内候选集上）
# ================================================================
def _build_prefix_fn(tok, full_strings, eos_id, prompt_len):
    """约束解码（同 eval_tool_select_sft.py 的 KV cache 线性路径）：
    每一步只允许生成 full_strings（候选工具名）中某一串的前缀 token，
    命中完整串后只允许 EOS —— 输出必为合法工具名，格式噪声清零。"""
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


def _fc_infer(model, tokenizer, device: str, cand_names: List[str],
              tool_by_name: Dict[str, Any], msg: str,
              decode: str = "constrained") -> str:
    import re
    _norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    norm_map = {_norm(n): n for n in cand_names}

    def resolve(raw: str):
        if not raw:
            return None
        if raw in cand_names:
            return raw
        key = _norm(raw)
        if key in norm_map:
            return norm_map[key]
        return None

    desc_lines = "\n".join(f"- {n}: {tool_by_name[n]['description']}" for n in cand_names)
    system = ("你是一个电商客服工具路由器。根据用户消息，从候选工具列表中选择最相关的一个工具名。"
              "每个工具都有功能描述，请根据语义匹配。若用户同时涉及多个操作，选最相关的一个。"
              "只输出工具名，不要解释。")
    user = f"候选工具:\n{desc_lines}\n\n用户消息: {msg}\n\n请输出最相关的工具名:"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    import torch
    t1 = time.monotonic()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    gen_kwargs = dict(max_new_tokens=32, do_sample=False,
                      pad_token_id=tokenizer.eos_token_id)
    if decode == "constrained":
        prompt_len = inputs["input_ids"].shape[1]
        gen_kwargs["prefix_allowed_tokens_fn"] = _build_prefix_fn(
            tokenizer, set(cand_names), tokenizer.eos_token_id, prompt_len)
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()
    elapsed = (time.monotonic() - t1) * 1000
    if decode == "constrained":
        return (generated if generated in cand_names else resolve(generated) or generated), elapsed
    selected = None
    for line in generated.strip().splitlines():
        name = line.strip().lstrip("-* 0123456789.、，").strip().strip('\'"`,，:')
        r = resolve(name)
        if r:
            selected = r
            break
    if selected is None:
        fb = "".join(generated.split())
        selected = resolve(fb) or fb
    return selected, elapsed


# ================================================================
# 单组评测
# ================================================================
def eval_group(data: Dict[str, Any], emb_model, route_mode: str, classifier: str,
               domain_top_m: int, top_k: int, use_fc: bool, fc_model_path: str,
               fc_device: str, runs: int, scale: str,
               fc_decode: str = "constrained") -> Dict[str, Any]:
    tool_by_name, domain_to_tools, func_to_tools, dom_repr = build_index(data)
    domains = data["domains"]
    all_tools = list(tool_by_name.keys())

    # 预编码
    t0 = time.monotonic()
    tool_texts = [f"工具名称：{n}；功能描述：{tool_by_name[n]['description']}" for n in all_tools]
    tool_emb = dict(zip(all_tools, _encode(emb_model, tool_texts)))
    dom_repr_emb = {k: _encode(emb_model, [v])[0] for k, v in dom_repr.items()}
    func_repr_emb = {k: _encode(emb_model, [v])[0] for k, v in FUNCTIONAL_REPR.items()}
    encode_time = time.monotonic() - t0

    fc_bundle = None
    if use_fc:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(fc_model_path, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            fc_model_path, torch_dtype=torch.float16,
            device_map="auto" if fc_device == "cuda" else "cpu", trust_remote_code=True)
        fc_bundle = (mdl, tok)

    per_case = []
    disp_sum = 0.0
    disp_n = 0
    # 统计容器
    recall_hit = 0
    end2end_hit = 0
    fc_latencies: List[float] = []
    ctx_chars_total = 0
    ctx_toolcount_total = 0
    # 域分类准确率
    dom_acc_strong = dom_acc_weak = 0
    dom_n_strong = dom_n_weak = 0
    # 硬路由丢条
    hard_lost = 0           # R1 中正确工具不在选定域、且域内 top-k 也救不回
    hard_total = 0
    # 软路由救回（相对 R1）：同 query 在 soft 下的候选是否含正确
    # 这里单独再跑一遍 none/hard 作对照在 sweep 中处理；单组只记自身
    for q in data["queries"]:
        msg = q["message"]
        correct = q["correct_tool"]
        correct_dom = q["correct_domain"]
        signal = q["domain_signal"]
        msg_emb = _encode(emb_model, [msg])[0]

        # ---- 路由 ----
        if route_mode == "none":
            selected_doms: List[str] = list(domain_to_tools.keys())  # 全集
            routed_tools: Set[str] = set(all_tools)
            dom_sim = None
        else:
            if classifier == "keyword":
                matched = route_keyword(msg, domain_to_tools, tool_by_name, domains)
                if route_mode == "hard":
                    selected_doms = [matched[0]] if matched else list(domain_to_tools.keys())
                else:  # soft
                    selected_doms = matched if matched else list(domain_to_tools.keys())
                dom_sim = None
            elif classifier == "embedding":
                ranked = route_embedding(msg_emb, dom_repr_emb, domain_top_m)
                dom_sim = ranked
                if route_mode == "hard":
                    selected_doms = [ranked[0][0]]
                else:
                    selected_doms = [d for d, _ in ranked]
            elif classifier == "functional":
                ranked = route_functional(msg_emb, func_repr_emb, domain_top_m)
                dom_sim = ranked
                if route_mode == "hard":
                    selected_groups = [ranked[0][0]]
                else:
                    selected_groups = [d for d, _ in ranked]
                routed_tools = set()
                for g in selected_groups:
                    routed_tools |= func_to_tools.get(g, set())
                selected_doms = selected_groups  # 功能组
            # 由 selected_doms 求候选工具集
            if classifier != "functional":
                routed_tools = set()
                for d in selected_doms:
                    routed_tools |= domain_to_tools.get(d, set())

        # 域（或功能组）分类准确率
        if classifier == "functional":
            correct_in = tool_by_name[correct]["functional"] in set(selected_doms)
        else:
            correct_in = correct_dom in set(selected_doms)
        # 置信度分散度（top1-top2 相似度差；越小越「犹豫」，用于 H3）
        if dom_sim is not None and len(dom_sim) >= 2:
            disp_sum += (dom_sim[0][1] - dom_sim[1][1])
            disp_n += 1
        if signal == "strong":
            dom_n_strong += 1
            dom_acc_strong += 1 if correct_in else 0
        else:
            dom_n_weak += 1
            dom_acc_weak += 1 if correct_in else 0

        # ---- 域内软预过滤（embedding 在 routed_tools 上取 top-k）----
        cand = list(routed_tools)
        if len(cand) == 0:
            cand = all_tools
        sims = [(n, float(sum(a * b for a, b in zip(msg_emb, tool_emb[n])))) for n in cand]
        sims.sort(key=lambda x: x[1], reverse=True)
        topk = [n for n, _ in sims[:top_k]]
        recall_ok = correct in topk
        recall_hit += 1 if recall_ok else 0

        # 上下文 token 近似（用字符数做比例代理；中文 ~1.5 tok/char）
        ctx_chars = sum(len(tool_by_name[n]["description"]) for n in cand)
        ctx_chars_total += ctx_chars
        ctx_toolcount_total += len(cand)

        # ---- 端到端 FC（可选）----
        e2e = None
        if use_fc and fc_bundle is not None:
            mdl, tok = fc_bundle
            votes, lats = [], []
            fc_cands = topk if topk else cand[:top_k]
            for _ in range(max(1, runs)):
                sel, ms = _fc_infer(mdl, tok, fc_device, fc_cands, tool_by_name, msg,
                                    decode=fc_decode)
                votes.append(sel)
                lats.append(ms)
            e2e = Counter(votes).most_common(1)[0][0]
            fc_latencies.append(statistics.median(lats))
            end2end_hit += 1 if e2e == correct else 0

        # 硬路由丢条（仅 hard 模式有意义）
        if route_mode == "hard" and classifier != "functional":
            hard_total += 1
            if correct_dom not in set(selected_doms):
                # 正确工具域不在选定域；看域内 top-k 能否救回（不能→不可逆丢）
                if not recall_ok:
                    hard_lost += 1

        per_case.append({
            "id": q["id"], "message": msg, "correct_tool": correct,
            "correct_domain": correct_dom, "level": q["level"], "domain_signal": signal,
            "selected_doms": selected_doms, "routed_count": len(cand),
            "correct_in_set": correct in cand,
            "recall_topk_hit": recall_ok, "fc_selected": e2e,
        })

    # ---- 汇总 ----
    n = len(data["queries"])
    result = {
        "scale": scale, "route_mode": route_mode, "classifier": classifier,
        "domain_top_m": domain_top_m, "top_k": top_k, "use_fc": use_fc,
        "n_queries": n, "n_tools": len(all_tools), "n_domains": len(domain_to_tools),
        "recall_at_k": round(recall_hit / n, 4),
        "recall_hits": recall_hit,
        "ctx_avg_chars": round(ctx_chars_total / n, 1),
        "ctx_avg_toolcount": round(ctx_toolcount_total / n, 2),
        "domain_acc_strong": round(dom_acc_strong / dom_n_strong, 4) if dom_n_strong else None,
        "domain_acc_weak": round(dom_acc_weak / dom_n_weak, 4) if dom_n_weak else None,
        "domain_n_strong": dom_n_strong, "domain_n_weak": dom_n_weak,
        "encode_time_s": round(encode_time, 2),
        "mean_dom_top1_top2_gap": round(disp_sum / disp_n, 4) if disp_n else None,
        "_per_case": per_case,
    }
    if use_fc:
        result["fc_decode"] = fc_decode
        result["e2e_acc"] = round(end2end_hit / n, 4)
        result["e2e_hits"] = end2end_hit
        if fc_latencies:
            result["fc_p50_ms"] = round(statistics.median(fc_latencies), 1)
            result["fc_p95_ms"] = round(sorted(fc_latencies)[max(0, int(len(fc_latencies) * 0.95) - 1)], 1)
    if route_mode == "hard" and classifier != "functional":
        result["hard_irreversible_loss_rate"] = round(hard_lost / hard_total, 4) if hard_total else 0.0
        result["hard_lost"] = hard_lost
        result["hard_total"] = hard_total
    return result


def _print_result(r: Dict[str, Any]):
    print(f"  scale={r['scale']} mode={r['route_mode']} cls={r['classifier']} "
          f"top-m={r['domain_top_m']} K={r['top_k']} fc={r['use_fc']}")
    print(f"    recall@{r['top_k']}       = {r['recall_at_k']*100:.1f}%  ({r['recall_hits']}/{r['n_queries']})")
    if r["use_fc"]:
        print(f"    end2end FC acc     = {r['e2e_acc']*100:.1f}%  "
              f"p95={r.get('fc_p95_ms')}ms")
    print(f"    ctx avg chars       = {r['ctx_avg_chars']}  (tool count {r['ctx_avg_toolcount']})")
    if r["domain_acc_strong"] is not None:
        print(f"    domain acc strong    = {r['domain_acc_strong']*100:.1f}%  "
              f"weak = {r['domain_acc_weak']*100:.1f}%")
    if "hard_irreversible_loss_rate" in r:
        print(f"    hard irreversible loss = {r['hard_irreversible_loss_rate']*100:.1f}%  "
              f"({r['hard_lost']}/{r['hard_total']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["S", "M", "L", "LL"], required=True)
    ap.add_argument("--route-mode", choices=["none", "hard", "soft"], default="soft")
    ap.add_argument("--classifier", choices=["keyword", "embedding", "functional"], default="embedding")
    ap.add_argument("--domain-top-m", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--fc", choices=["on", "off"], default="off")
    ap.add_argument("--fc-decode", choices=["free", "constrained"], default="constrained",
                    help="域内 FC 解码方式：constrained=约束解码(默认,格式噪声清零) / free=自由生成+启发式解析(旧)")
    ap.add_argument("--fc-model", default=FC_MODEL)
    ap.add_argument("--fc-device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output", default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="对 embedding 分类器跑 none/hard/soft 并打印对比表")
    args = ap.parse_args()

    data = load_dataset(args.scale, args.data_dir)
    print(f"载入 {args.scale} 数据集: tools={data['n_tools']} domains={data['n_domains']} "
          f"queries={len(data['queries'])}")
    emb_model = _load_embed_model(EMB_MODEL)
    print(f"embedding 模型: {EMB_MODEL}")

    use_fc = args.fc == "on"

    def run_one(mode, cls) -> Dict[str, Any]:
        return eval_group(data, emb_model, mode, cls, args.domain_top_m,
                          args.top_k, use_fc, args.fc_model, args.fc_device,
                          args.runs, args.scale, fc_decode=args.fc_decode)

    if args.sweep:
        modes = ["none", "hard", "soft"]
        results = []
        for m in modes:
            r = run_one(m, "embedding")
            results.append(r)
            _print_result(r)
        by_mode = {r["route_mode"]: r for r in results}
        # 软路由救回率（H2）：R1 丢失的条中 R2 救回多少
        pc_by_id = {}
        for r in results:
            pc_by_id[r["route_mode"]] = {c["id"]: c for c in r.get("_per_case", [])}
        rescue_rate = None
        if "hard" in pc_by_id and "soft" in pc_by_id:
            hard_pc = pc_by_id["hard"]
            soft_pc = pc_by_id["soft"]
            lost_ids = [i for i, c in hard_pc.items() if not c["correct_in_set"]]
            rescued = sum(1 for i in lost_ids if soft_pc[i]["correct_in_set"])
            rescue_rate = (rescued / len(lost_ids)) if lost_ids else None
            print(f"\n  [H2] R1(hard) 丢失 {len(lost_ids)} 条；R2(soft) 救回 {rescued} 条 "
                  f"→ 软路由救回率 = {rescue_rate*100:.1f}%")
        # 对比表
        print(f"\n{'='*60}\n对比表 (scale={args.scale}, embedding 分类器, K={args.top_k})\n{'='*60}")
        print(f"{'mode':<6} {'recall@K':>9} {'ctx_chars':>10} {'domAccS':>9} {'domAccW':>9} {'gap':>7}")
        for r in results:
            print(f"{r['route_mode']:<6} {r['recall_at_k']*100:>8.1f}% "
                  f"{r['ctx_avg_chars']:>10} "
                  f"{(r['domain_acc_strong'] or 0)*100:>8.1f}% "
                  f"{(r['domain_acc_weak'] or 0)*100:>8.1f}% "
                  f"{r.get('mean_dom_top1_top2_gap')}")
        r1 = by_mode["hard"]
        if "hard_irreversible_loss_rate" in r1:
            print(f"\n  R1(hard) 硬路由不可逆丢条率 = {r1['hard_irreversible_loss_rate']*100:.1f}% "
                  f"({r1['hard_lost']}/{r1['hard_total']})")
            print(f"  R2(soft) 端到端 recall 代理 = {by_mode['soft']['recall_at_k']*100:.1f}% "
                  f"vs R1 = {r1['recall_at_k']*100:.1f}% "
                  f"→ R2 {'≥' if by_mode['soft']['recall_at_k'] >= r1['recall_at_k'] else '<'} R1")
        if args.output:
            out = {"scale": args.scale, "sweep": results,
                   "rescue_rate": rescue_rate}
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"\nJSON -> {args.output}")
    else:
        r = run_one(args.route_mode, args.classifier)
        _print_result(r)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=2)
            print(f"\nJSON -> {args.output}")


if __name__ == "__main__":
    main()
