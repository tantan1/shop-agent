import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "./models/Qwen2.5-1.5B-Instruct-tool-select"
DATA = "data/llamafactory/shop_tool_select_S.json"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="auto", trust_remote_code=False)
model.eval()

data = json.load(open(DATA, encoding="utf-8"))
# 每个 level 取 2 条做冒烟
seen = {}
picks = []
for x in data:
    lv = x["level"]
    if seen.get(lv, 0) < 2:
        picks.append(x); seen[lv] = seen.get(lv, 0) + 1
    if sum(seen.values()) >= 10:
        break

ok = 0
for x in picks:
    sys_c = x["conversations"][0]["content"]
    usr_c = x["conversations"][1]["content"]
    msgs = [{"role": "system", "content": sys_c}, {"role": "user", "content": usr_c}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=48, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    # 解析 {"name": ...}
    name = None
    try:
        name = json.loads(gen.replace("'", '"')).get("name")
    except Exception:
        name = None
    hit = (name == x["correct_tool"])
    ok += int(hit)
    print(f"[{x['level']}] pred={name!r} gold={x['correct_tool']!r} {'OK' if hit else 'X'}  raw={gen!r}")

print(f"\nSMOKE acc = {ok}/{len(picks)}")
