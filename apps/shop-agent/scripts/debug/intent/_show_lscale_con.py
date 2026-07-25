"""对比 lscale L 档 FC 自由生成(旧) vs 约束解码(新)，并分解瓶颈。"""
import json
from collections import Counter

PAIRS = [
    ("R0", "lscale_L_R0_fc.json", "lscale_L_R0_fc_con.json"),
    ("R2", "lscale_L_R2_fc.json", "lscale_L_R2_fc_con.json"),
]

for tag, old, new in PAIRS:
    o = json.load(open(old, encoding="utf-8"))
    nw = json.load(open(new, encoding="utf-8"))
    print(f"== {tag}: e2e free={o['e2e_acc']:.1%} -> con={nw['e2e_acc']:.1%}  "
          f"recall@K={nw['recall_at_k']:.1%}")
    pc = nw["_per_case"]
    hit = [c for c in pc if c["recall_topk_hit"]]
    cond = sum(1 for c in hit if c["fc_selected"] == c["correct_tool"]) / len(hit)
    print(f"   召回命中 n={len(hit)}  条件FC准确率(命中前提下)={cond:.1%}")
    lv, lvh = Counter(), Counter()
    for c in hit:
        lv[c["level"]] += 1
        if c["fc_selected"] == c["correct_tool"]:
            lvh[c["level"]] += 1
    for k in sorted(lv):
        print(f"     {k:10s} {lvh[k]:3d}/{lv[k]:3d} = {lvh[k]/lv[k]:.1%}")
    print()
