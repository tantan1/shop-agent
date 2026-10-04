"""
对比微调前(base)与微调后(sft)的参数抽取准确性。

关键设计：
  - 评测 prompt 严格使用项目 PARAM_EXTRACTION_PROMPTS（与训练数据一致），
    避免 benchmark_local_model_comparison.py 内联 prompt 与训练不一致导致的"坑④"。
  - 测试集默认直接复用训练集 shop_param_v1.json（其 assistant 字段即 gold label，
    同分布最公平）；也可用 --data 指向独立 holdout 集。
  - base / sft 均为独立模型目录（sft 为 merge 后的权重目录）。

用法：
  # 1) 不加载模型，先校验测试集分布 + prompt 一致性（无需 GPU）
  python scripts/eval_sft_before_after.py --dry-run

  # 2) 跑对比（需 GPU + 模型权重）
  python scripts/eval_sft_before_after.py \
      --base ./models/Qwen2.5-1.5B-Instruct \
      --sft  ./models/Qwen2.5-1.5B-Instruct-sft \
      --data data/llamafactory/shop_param_v1.json \
      --device cuda --max-samples 200

输出：
  - 控制台对比表（field_f1 / value_exact_match_rate 的 Δ 提升 + 逐意图 Δ）
  - eval_sft_before_after.json（总体 + 逐意图 + 回归 case 列表）
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

# 进度条：优先用 tqdm，缺失则回退到纯文本百分比（不引入硬依赖）。
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# 评测 prompt 与训练严格一致（对应博客坑④：template 命门）。
# 内联自 src/modules/chat/schemas.py 的 PARAM_EXTRACTION_PROMPTS，
# 避免 import 项目模块带来的 pydantic 依赖（venv_cuda 推理环境无 pydantic）。
# dry-run 会校验测试集样本的 system 字段与此处是否一致，捕获漂移。
PARAM_EXTRACTION_PROMPTS: Dict[str, str] = {
    "query-order": (
        "从用户消息中提取查询订单的参数。\n"
        "- order_id: 订单号通常是数字组合(如 202405270001)\n"
        "- phone: 手机号后四位\n"
        "- status_filter: 用户想看的订单状态(待付款/已发货/派送中/已签收)\n"
        "- 如果没有提到某个参数，留空即可"
    ),
    "check-shipping": (
        "从用户消息中提取查询物流的参数。\n"
        "- tracking_number: 快递单号(如 SF1234567890、YT123456、JD001234567)\n"
        "- order_id: 订单号\n"
        "- 如果没有提到某个参数，留空即可"
    ),
    "request-return": (
        "从用户消息中提取退货退款的参数。\n"
        "- order_id: 要退货的订单号\n"
        "- reason: 退货原因(质量问题/不想要/发错货/与描述不符/其他)\n"
        "- 如果没有提到某个参数，留空即可"
    ),
    "check-balance": (
        "用户查询账户余额或积分，当前无需额外参数。"
    ),
    "coupon-inquiry": (
        "从用户消息中提取查询优惠券的参数。\n"
        "- coupon_type: 券类型(满减券/折扣券/运费券/通用)\n"
        "- 如果没有提到某个参数，留空即可"
    ),
}

# ── JSON 提取（镜像 local_model_service / benchmark 逻辑） ──────────────
_STRIP_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think_tags(text: str) -> str:
    return _STRIP_THINK_RE.sub("", text).strip()


def extract_json_from_text(text: str) -> Optional[dict]:
    """从模型输出中提取 JSON 对象（兼容 ```json 代码块 / 多余文字）。"""
    text = strip_think_tags(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*\n?(\{.*?\})\s*\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return None


def eval_extraction(predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> dict:
    """字段级精确率/召回率/F1 + 值完全匹配率（与 benchmark 一致）。"""
    predict_keys = set(predicted.keys())
    truth_keys = set(ground_truth.keys())
    tp = len(predict_keys & truth_keys)
    fp = len(predict_keys - truth_keys)
    fn = len(truth_keys - predict_keys)
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not truth_keys else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if not truth_keys else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # 严格匹配：键集必须完全一致。否则 gold 为空(如 check-balance / 缺参难例)
    # 时 all() 对空集恒为 True，模型乱吐字段也会被记满分（见对话核对）。
    value_match = (set(predicted.keys()) == set(ground_truth.keys())) and all(
        predicted.get(k) == v for k, v in ground_truth.items()
    )
    # 失败样本的错误类型细分（仅当 value_match=False 时有意义）：
    #   over_extract  : 预测多出 gold 没有的字段（fp>0）——主要嫌疑：信息不足时硬抽
    #   under_extract : gold 有字段被漏抽（fn>0 且 fp==0）
    #   value_wrong   : 键集一致但值不等（fp==0 且 fn==0）
    #   mixed         : 既多抽又漏抽（fp>0 且 fn>0）
    if value_match:
        error_type = "correct"
    elif fp > 0 and fn > 0:
        error_type = "mixed"
    elif fp > 0:
        error_type = "over_extract"
    elif fn > 0:
        error_type = "under_extract"
    else:
        error_type = "value_wrong"
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "value_exact_match": value_match,
        "error_type": error_type,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def build_messages(intent: str, query: str) -> list:
    """评测 prompt 严格使用 PARAM_EXTRACTION_PROMPTS（与训练一致）。"""
    system = PARAM_EXTRACTION_PROMPTS[intent]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]


def load_model(model_path: str, device: str):
    import json
    import os

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  [load] tokenizer from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 批量推理关键①：decoder-only 模型必须左填充，否则右侧 padding 会顶掉
    # 生成起点，导致批内除最长样本外全部输出错乱。
    tokenizer.padding_side = "left"
    # 批量推理关键②：确保有 pad_token（Qwen 一般有 <|endoftext|>，兜底用 eos）。
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if device == "cuda":
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32

    # 支持未 merge 的 PEFT(LoRA) 适配器目录：先加载基座，再挂载适配器。
    # 这样 smoke 训练（不 merge）产出的 adapter 也能被评测直接加载。
    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        print(f"  [load] 检测到 PEFT 适配器，按 LoRA 加载: {model_path}", flush=True)
        base_name = json.load(open(adapter_cfg, encoding="utf-8"))["base_model_name_or_path"]
        base = AutoModelForCausalLM.from_pretrained(base_name, **kwargs)
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, model_path)
    else:
        print(f"  [load] model from {model_path} (dtype={kwargs.get('dtype')})", flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages: list, max_new_tokens: int, device: str,
             enable_thinking: bool = True) -> tuple:
    import torch

    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if not enable_thinking:
        chat_kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    target = "cuda" if device == "cuda" else "cpu"
    inputs = {k: v.to(target) for k, v in inputs.items()}
    t0 = time.monotonic()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    elapsed_ms = (time.monotonic() - t0) * 1000
    return text, elapsed_ms


def generate_batch(model, tokenizer, batch_messages: List[list], max_new_tokens: int,
                   device: str, enable_thinking: bool = True) -> tuple:
    """批量生成：一次把 N 条塞进 model.generate，显著提升 GPU 利用率。

    返回 (texts, per_latency_ms)：单条延迟无法精确拆分，用「批总耗时/批大小」
    作为每条的近似值（对比 base/sft 快慢仍公平，因两者同批大小同口径）。
    """
    import torch

    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if not enable_thinking:
        chat_kwargs["enable_thinking"] = False
    prompts = [tokenizer.apply_chat_template(m, **chat_kwargs) for m in batch_messages]
    # 左填充对齐（padding_side 已在 load_model 设为 left）
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=2048)
    target = "cuda" if device == "cuda" else "cpu"
    inputs = {k: v.to(target) for k, v in inputs.items()}
    t0 = time.monotonic()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # 左填充下所有行 prompt 段等长，input_ids.shape[1] 即生成起点，逐行切分即可
    input_len = inputs["input_ids"].shape[1]
    gen = outputs[:, input_len:]
    texts = [tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]
    elapsed_ms = (time.monotonic() - t0) * 1000
    per_latency = elapsed_ms / max(1, len(batch_messages))
    return texts, per_latency


def load_testset(path: str, max_samples: Optional[int]) -> List[dict]:
    """读取 ShareGPT 测试集，解析出 (intent, query, gold)。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: List[dict] = []
    for rec in raw:
        intent = rec.get("intent")
        convs = rec.get("conversations", [])
        query = None
        gold_raw = None
        for c in convs:
            if c["role"] == "user":
                query = c["content"]
            elif c["role"] == "assistant":
                gold_raw = c["content"]
        if query is None or gold_raw is None or intent not in PARAM_EXTRACTION_PROMPTS:
            continue
        try:
            gold = json.loads(gold_raw)
        except Exception:
            continue
        if not isinstance(gold, dict):
            continue
        out.append({"intent": intent, "query": query, "gold": gold})
        if max_samples and len(out) >= max_samples:
            break
    return out


def run_eval(model_path: str, device: str, testset: List[dict],
             max_new_tokens: int = 128, batch_size: int = 8) -> tuple:
    """对单个模型跑完整评测，返回 (summary, per_case)。

    batch_size>1 时走批量推理（大幅提升 GPU 利用率、缩短总时长）；
    batch_size=1 退化为逐条，行为与旧版一致。
    """
    model, tokenizer = load_model(model_path, device)
    enable_thinking = "Qwen3" not in model_path

    per_case: List[dict] = []
    latencies: List[float] = []
    total_tp = total_fp = total_fn = 0
    value_matches = 0
    intent_stat: Dict[str, List[int]] = {}  # intent -> [matches, total]
    err_counter: "Counter" = Counter()              # 全局错误类型计数
    intent_err: Dict[str, "Counter"] = {}          # intent -> 错误类型计数

    total = len(testset)
    tag = Path(model_path).name
    print(f"  [eval] {tag}: {total} 条样本 (batch_size={batch_size})...", flush=True)

    def _accumulate(i: int, case: dict, raw: str, elapsed: float):
        """把单条结果并入统计（批量/逐条共用）。"""
        nonlocal total_tp, total_fp, total_fn, value_matches
        latencies.append(elapsed)
        data = extract_json_from_text(raw) or {}
        if not isinstance(data, dict):
            data = {}
        predicted = {k: v for k, v in data.items() if v is not None}
        gold = case["gold"]
        m = eval_extraction(predicted, gold)
        total_tp += m["tp"]
        total_fp += m["fp"]
        total_fn += m["fn"]
        if m["value_exact_match"]:
            value_matches += 1
        istat = intent_stat.setdefault(case["intent"], [0, 0])
        istat[1] += 1
        if m["value_exact_match"]:
            istat[0] += 1
        et = m["error_type"]
        err_counter[et] += 1
        intent_err.setdefault(case["intent"], Counter())[et] += 1
        per_case.append({
            "idx": i + 1,
            "intent": case["intent"],
            "query": case["query"],
            "predicted": predicted,
            "gold": gold,
            "latency_ms": round(elapsed, 1),
            **m,
        })

    # 按批切分（batch_size=1 即逐条）
    batch_starts = range(0, total, max(1, batch_size))
    if _HAS_TQDM:
        pbar = _tqdm(total=total, desc=f"  {tag}", unit="case",
                     dynamic_ncols=True, leave=True)
    else:
        pbar = None

    for start in batch_starts:
        chunk = testset[start:start + max(1, batch_size)]
        batch_messages = [build_messages(c["intent"], c["query"]) for c in chunk]
        texts, per_lat = generate_batch(
            model, tokenizer, batch_messages, max_new_tokens, device, enable_thinking
        )
        for j, (case, raw) in enumerate(zip(chunk, texts)):
            _accumulate(start + j, case, raw, per_lat)

        done = start + len(chunk)
        hit_rate = value_matches / done if done else 0.0
        if pbar is not None:
            pbar.update(len(chunk))
            pbar.set_postfix(hit=f"{hit_rate:.2%}", ms=f"{per_lat:.0f}")
        elif done % 20 < len(chunk) or done == total:
            print(f"    [{tag}] {done}/{total} ({done/total:.0%})  "
                  f"hit={hit_rate:.2%}  ms/条≈{per_lat:.0f}", flush=True)
    if pbar is not None:
        pbar.close()

    # 清理显存/内存
    del model, tokenizer
    gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()

    macro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    macro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    intent_rate = {
        k: round(v[0] / v[1], 4) if v[1] else 0.0 for k, v in intent_stat.items()
    }
    summary = {
        "model": model_path,
        "total_samples": total,
        "field_precision": round(macro_p, 4),
        "field_recall": round(macro_r, 4),
        "field_f1": round(macro_f1, 4),
        "value_exact_match_rate": round(value_matches / total, 4) if total else 0.0,
        "value_matches": value_matches,
        "avg_latency_ms": round(avg_lat, 1),
        "per_intent_value_match_rate": intent_rate,
        "error_breakdown": dict(err_counter),
        "per_intent_error_breakdown": {
            k: dict(v) for k, v in intent_err.items()
        },
    }
    return summary, per_case


def dry_run(data_path: str):
    """不加载模型，校验测试集分布 + 评测/训练 prompt 一致性（坑④自检）。"""
    raw = json.loads(Path(data_path).read_text(encoding="utf-8"))
    cnt = Counter()
    mismatch = 0
    usable = 0
    for rec in raw:
        intent = rec.get("intent")
        if intent not in PARAM_EXTRACTION_PROMPTS:
            continue
        sys_prompt = next(
            (c["content"] for c in rec.get("conversations", []) if c["role"] == "system"),
            None,
        )
        # 坑④自检：训练时 system 必须与评测 prompt 逐字一致
        if sys_prompt != PARAM_EXTRACTION_PROMPTS[intent]:
            mismatch += 1
        cnt[intent] += 1
        usable += 1
    print(f"[DRY-RUN] 测试集可用样本: {usable} 条")
    print(f"[DRY-RUN] 意图分布: {dict(cnt)}")
    print(f"[DRY-RUN] system prompt 与 PARAM_EXTRACTION_PROMPTS 不一致数: {mismatch}")
    if mismatch:
        print("[DRY-RUN][WARN] 存在不一致！训练/评测 prompt 不同会导致对比不公平（博客坑④）")
    else:
        print("[DRY-RUN][OK] 评测 prompt 与训练完全一致")
    return


# =============================================================================
# 指标中文说明（写入结果 json 的「指标说明」字段，便于人读）
# =============================================================================
METRIC_DOC = {
    "total_samples": "评测样本总数",
    "field_precision": "字段级精确率：预测抽出的字段中，键名确实属于 gold 的比例（乱抽字段会被惩罚）",
    "field_recall": "字段级召回率：gold 字段中被正确抽出的比例（漏抽字段会被惩罚）",
    "field_f1": "字段级 F1：精确率与召回率的调和平均",
    "value_exact_match_rate": "整条达标率（最严格）：该条所有 gold 字段的键和值都完全相等才算对，是衡量「任务是否达标」的核心指标",
    "value_matches": "value_exact_match 达标的样本数",
    "avg_latency_ms": "单条生成平均耗时（毫秒），越低越快",
    "per_intent_value_match_rate": "逐意图的整条达标率，定位哪个意图最弱",
    "error_breakdown": "失败样本的错误类型分布：over_extract=多抽字段(该空不空)；under_extract=漏抽字段；value_wrong=键对但值错；mixed=既多抽又漏抽；correct=达标",
    "per_intent_error_breakdown": "逐意图的错误类型分布，定位某意图主要错在哪类",
    "regressions": "回归 case：BASE 答对但 SFT 答错的样本（应尽量避免）",
    "diff": "SFT 相对 BASE 的差值（SFT - BASE），正值表示 SFT 更优",
    "per_intent_diff": "逐意图 value_exact_match_rate 的差值",
    "config": "本次评测的配置（模型路径、测试集、设备等）",
}


def main():
    ap = argparse.ArgumentParser(description="对比微调前后参数抽取准确性")
    ap.add_argument("--base", default=None, help="微调前模型目录（独立目录）")
    ap.add_argument("--sft", default=None, help="微调后模型目录（独立目录，已 merge）")
    ap.add_argument("--data", default="data/llamafactory/shop_param_v1.json",
                    help="测试集 ShareGPT json（默认复用训练集）")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="批量推理大小（>1 提升 GPU 利用率，显存不足时调小；=1 为逐条）")
    ap.add_argument("--max-samples", type=int, default=None, help="限制测试样本数（调试用）")
    ap.add_argument("--out", default="eval_sft_before_after.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="只校验测试集分布与 prompt 一致性，不加载模型")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args.data)
        return

    if not args.base or not args.sft:
        ap.error("--base 和 --sft 都必填（除非 --dry-run）")

    testset = load_testset(args.data, args.max_samples)
    print(f"[INFO] 测试集: {len(testset)} 条 (from {args.data})")

    print(f"\n[1/2] 评测 BASE: {args.base}")
    base_sum, base_cases = run_eval(args.base, args.device, testset,
                                    batch_size=args.batch_size)
    print(f"  field_f1={base_sum['field_f1']:.4f} "
          f"value_match={base_sum['value_exact_match_rate']:.4f} "
          f"avg_lat={base_sum['avg_latency_ms']}ms")

    print(f"\n[2/2] 评测 SFT: {args.sft}")
    sft_sum, sft_cases = run_eval(args.sft, args.device, testset,
                                  batch_size=args.batch_size)
    print(f"  field_f1={sft_sum['field_f1']:.4f} "
          f"value_match={sft_sum['value_exact_match_rate']:.4f} "
          f"avg_lat={sft_sum['avg_latency_ms']}ms")

    # ── 对比 ──
    def delta(a: float, b: float) -> float:
        return round(b - a, 4)

    diff = {
        "field_precision": delta(base_sum["field_precision"], sft_sum["field_precision"]),
        "field_recall": delta(base_sum["field_recall"], sft_sum["field_recall"]),
        "field_f1": delta(base_sum["field_f1"], sft_sum["field_f1"]),
        "value_exact_match_rate": delta(
            base_sum["value_exact_match_rate"], sft_sum["value_exact_match_rate"]
        ),
        "avg_latency_ms": delta(base_sum["avg_latency_ms"], sft_sum["avg_latency_ms"]),
    }
    intents = set(base_sum["per_intent_value_match_rate"]) | set(
        sft_sum["per_intent_value_match_rate"]
    )
    intent_diff = {
        k: delta(
            base_sum["per_intent_value_match_rate"].get(k, 0.0),
            sft_sum["per_intent_value_match_rate"].get(k, 0.0),
        )
        for k in intents
    }
    # 回归 case：base 对、sft 错
    regressions = []
    for b, s in zip(base_cases, sft_cases):
        if b["value_exact_match"] and not s["value_exact_match"]:
            regressions.append({
                "intent": b["intent"],
                "query": b["query"],
                "gold": b["gold"],
                "base_pred": b["predicted"],
                "sft_pred": s["predicted"],
            })

    # ── 打印对比表 ──
    print("\n" + "=" * 70)
    print("对比结果 (SFT - BASE)")
    print("=" * 70)
    print(f"{'指标':<26}{'BASE':<12}{'SFT':<12}{'Δ':<12}")
    print("-" * 62)
    for k in ("field_precision", "field_recall", "field_f1", "value_exact_match_rate"):
        print(f"{k:<26}{base_sum[k]:<12.4f}{sft_sum[k]:<12.4f}{diff[k]:<+12.4f}")
    print(f"{'avg_latency_ms':<26}{base_sum['avg_latency_ms']:<12}"
          f"{sft_sum['avg_latency_ms']:<12}{diff['avg_latency_ms']:<+12.1f}")

    print("\n逐意图 value_exact_match_rate Δ:")
    for k in sorted(intent_diff):
        print(f"  {k:<18} base={base_sum['per_intent_value_match_rate'].get(k, 0):.3f} "
              f"sft={sft_sum['per_intent_value_match_rate'].get(k, 0):.3f} "
              f"Δ={intent_diff[k]:+.3f}")
    print(f"\n回归 case (base 对 / sft 错): {len(regressions)} 条")

    # ── 中文指标说明 ──
    print("\n" + "=" * 70)
    print("指标说明（详见结果 json 的「指标说明」字段）")
    print("=" * 70)
    for k, v in METRIC_DOC.items():
        print(f"  {k}: {v}")
    result = {
        "指标说明": METRIC_DOC,
        "base": base_sum,
        "sft": sft_sum,
        "diff": diff,
        "per_intent_diff": intent_diff,
        "regressions": regressions,
        "config": {
            "base": args.base,
            "sft": args.sft,
            "data": args.data,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
        },
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[DONE] 详细结果 -> {args.out}")


if __name__ == "__main__":
    main()
