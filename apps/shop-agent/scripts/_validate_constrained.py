"""校验新约束解码 (prefix_allowed_tokens_fn) 与旧实现 (逐token循环) 的允许集合逐token一致。
仅用 tokenizer，不加载 GPU 模型。两者允许集合一致 => 贪心结果一致 => 输出等价。"""
import sys, os, json, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from transformers import AutoTokenizer
from scripts.eval_tool_select_sft import build_constrained_prefix_fn, make_constrained_logits_processor

MODEL = "models/Qwen2.5-1.5B-Instruct-tool-select-M8"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
eos = tok.eos_token_id
full = ['{"name": "get_order"}', '{"name": "query_logistics"}', '{"name": "refund_apply"}']
prompt_len = 10

old_proc = make_constrained_logits_processor(tok, full, eos)
new_fn = build_constrained_prefix_fn(tok, full, eos, prompt_len)

# 对每条完整串作为"真值目标"，模拟贪心推进，比较新旧允许集合
mismatch = 0
for target in full:
    gen_ids, gen_text = [], ""
    for step in range(48):
        old_allowed = old_proc(gen_text)
        seq = [0] * prompt_len + gen_ids
        new_allowed = set(new_fn(0, torch.tensor(seq)))
        if old_allowed != new_allowed:
            mismatch += 1
            print(f"  MISMATCH target={target} step={step} old={old_allowed} new={new_allowed}")
            break
        # 选下一个 token：优先让 gen_text 朝 target 走（取 target 在该步期望的 token）
        suf = target[len(gen_text):]
        want_ids = tok.encode(suf, add_special_tokens=False)
        nxt = want_ids[0] if (want_ids and want_ids[0] in old_allowed) else min(old_allowed)
        gen_ids.append(nxt)
        gen_text += tok.decode([nxt])
        if gen_text in full:
            break
    ok = gen_text == target
    print(f"  target={target!r} -> gen={gen_text!r} {'OK' if ok else 'FAIL'}")
    if not ok:
        mismatch += 1

print("RESULT:", "ALL VALID" if mismatch == 0 else f"{mismatch} MISMATCH(ES)")
