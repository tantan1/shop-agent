"""
gen_synthetic_lscale.py — 生成 L 规模分层路由测试用的合成数据集
=============================================================

计划依据：docs/L规模分层路由测试计划.md (Phase 0)
设计要点：
  - 4 档规模：S(5)/M(40)/L(200)/LL(500)，每档带「业务域(business)」标签 + 「功能聚类(functional)」标签
  - 域内注入近义陷阱（描述互相点名区分），约每域 1-3 对
  - 5-10% 工具挂 2 个域（跨域工具，验 H2 多域候选自动并入）
  - 每条 query 带 correct_tool / correct_domain 标签 + 5 级难度 + 域信号强度(strong/weak)
  - 域代表描述用手写/独立来源（避免与 query 同模板导致 embedding 域分类器虚高，计划 §7 风险）
  - 固定 random seed，可复现

产出：data/lscale_<scale>.json，结构同时兼容 benchmark_tool_selection_pipeline 的字段约定
      （tools: name/display_name/domain/cross_domains/description/trigger_keywords；
       queries: message/correct_tool/correct_domain/level/domain_signal）

用法：
  python scripts/gen_synthetic_lscale.py --scale all --out-dir data
  python scripts/gen_synthetic_lscale.py --scale L --out-dir data
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

# ================================================================
# 业务域定义（10 个，LL/L 用全量；M 取前 8；S 取前 5 对齐现有 5 工具集）
# repr 为「独立来源」的域描述，不与 query 模板同构，降低 leakage
# ================================================================
DOMAINS: List[Dict[str, Any]] = [
    {"key": "order", "display": "订单管理", "trigger_keywords": ["订单", "下单", "我的订单", "待付款", "已购"],
     "repr": "订单管理域：围绕用户已生成的交易订单展开，包括订单列表、订单详情、下单、订单状态变更、订单查询与订单相关的退款进度查询。"},
    {"key": "logistics", "display": "物流配送", "trigger_keywords": ["物流", "快递", "配送", "到货", "包裹", "运单"],
     "repr": "物流配送域：围绕商品发货后的实体流转，包括快递跟踪、预计送达、修改收货地址、签收、配送异常处理。"},
    {"key": "aftersale", "display": "售后退货", "trigger_keywords": ["退货", "退款", "售后", "退换", "维修"],
     "repr": "售后退货域：围绕交易完成前后产生的逆向流程，包括退货申请、退款办理、换货、售后咨询与维修。"},
    {"key": "account", "display": "账户余额积分", "trigger_keywords": ["余额", "积分", "钱包", "账户", "资金"],
     "repr": "账户余额积分域：围绕用户账户内的资产，包括余额查询、积分查询、钱包明细、资金变动。"},
    {"key": "coupon", "display": "优惠营销", "trigger_keywords": ["优惠券", "满减", "活动", "折扣", "福利", "券"],
     "repr": "优惠营销域：围绕促销与优惠权益，包括优惠券查询、满减活动、折扣、新人福利、领券。"},
    {"key": "product", "display": "商品咨询", "trigger_keywords": ["商品", "规格", "参数", "材质", "详情", "库存"],
     "repr": "商品咨询域：围绕在售商品本身的属性，包括规格参数、材质、库存、商品详情、比价。"},
    {"key": "membership", "display": "会员权益", "trigger_keywords": ["会员", "等级", "成长值", "权益", "VIP"],
     "repr": "会员权益域：围绕会员体系，包括会员等级、成长值、会员专属权益、积分兑换会员。"},
    {"key": "complaint", "display": "投诉建议", "trigger_keywords": ["投诉", "差评", "举报", "建议", "反馈"],
     "repr": "投诉建议域：围绕用户的负面反馈与诉求，包括投诉、差评处理、举报、建议收集。"},
    {"key": "preorder", "display": "预售预订", "trigger_keywords": ["预售", "预订", "定金", "尾款", "预约"],
     "repr": "预售预订域：围绕未正式发货的预购行为，包括预售活动、预订、定金、尾款支付、预约。"},
    {"key": "service", "display": "客服通用", "trigger_keywords": ["客服", "转人工", "帮助", "咨询", "找不到"],
     "repr": "客服通用域：围绕通用客服入口，包括转人工、帮助中心、通用咨询、找不到入口时的引导。"},
]

# weak 域信号：topical 但不含 trigger_keywords 的短语（验 H1/H3 的 weak 子集）。
# 关键：保留「域主题」但避开字面触发词，避免 §7 的 leakage，也避免 weak 完全不可路由。
WEAK_TOPIC: Dict[str, List[str]] = {
    "order":     ["我买的东西", "之前下的那单", "刚拍的宝贝", "付完钱的那笔", "购物车里那件"],
    "logistics": ["我那个件", "发出的货", "要收的包", "派送的东西", "在路上那单"],
    "aftersale": ["不想要的那笔", "要退的那桩", "出问题的工作", "收到的坏件", "要处理的退换"],
    "account":   ["卡里的钱", "攒的那些分", "户头里的", "我的资产", "里面的余额"],
    "coupon":    ["能领的福利", "可以减的", "活动给的", "能省的", "平台送的券"],
    "product":   ["看中的款", "想买的那个", "货架上的", "相中的宝贝", "要下单的物"],
    "membership":["等级相关的", "会员那块", "权益方面", "成长的那点", "VIP 相关"],
    "complaint": ["要吐槽的", "不爽的那桩", "想反映的", "闹心的经历", "要说的毛病"],
    "preorder":  ["预定的那个", "付了定金的", "尾款那笔", "预约的物", "还没发货的购"],
    "service":   ["找不到入口", "搞不定的事", "想找人聊聊", "卡住的流程", "弄不明白的"],
}

# ================================================================
# 动作原型（跨域复用的 verb；qualifier 用于在同一域内扩出更多工具）
# functional 分组用于 H3 功能聚类对照
# ================================================================
ACTIONS: List[Dict[str, Any]] = [
    {"verb": "query", "display": "查询", "functional": "query",
     "func": "查询/查看该类{domain}的概览信息（列表或状态）。区分：只查不改，提交类请走 apply。",
     "kw": ["查询", "查看", "怎么查"],
     "syn": ["我想看看相关的情况", "帮我了解下这块的内容", "这块我想查一下"],
     "impl": ["之前提到的那个，再确认一下", "那个事我想再核实"],
     "amb": ["帮我找出来相关的那些", "把相关的挑出来给我看"]},
    {"verb": "detail", "display": "详情", "functional": "query",
     "func": "查看单条{domain}的明细内容。区分：与 query 不同——query 是列表概览，detail 是单条明细。",
     "kw": ["详情", "明细", "具体信息"],
     "syn": ["我想看具体那一条的内容", "把那条的具体情况告诉我", "那条里面具体的部分看一下"],
     "impl": ["上一条的具体内容是啥", "刚才那个的明细给我"],
     "amb": ["那条里面具体的部分看一下", "把单条的细节点出来"]},
    {"verb": "apply", "display": "申请", "functional": "operate",
     "func": "提交{domain}相关申请/办理（真实写操作）。区分：本工具会真正提交，仅查看请用 query。",
     "kw": ["申请", "办理", "提交"],
     "syn": ["我想办一下这个", "帮我弄个这个", "走一下这个流程"],
     "impl": ["这个该怎么去办", "我想发起这个"],
     "amb": ["走一下这个流程", "帮我提交这个申请"]},
    {"verb": "cancel", "display": "取消", "functional": "operate",
     "func": "取消/撤销已提交的{domain}。区分：仅取消用本工具，查询状态用 query。",
     "kw": ["取消", "撤销", "退掉"],
     "syn": ["我想停了那个", "帮我中止这个", "那个不要了"],
     "impl": ["之前那个不要了", "不想要了帮我撤"],
     "amb": ["把那个撤回来", "帮我中止并取消"]},
    {"verb": "modify", "display": "修改", "functional": "operate",
     "func": "修改{domain}的配置/信息。区分：修改用本工具，查看用 query。",
     "kw": ["修改", "更改", "调整"],
     "syn": ["我想改一下这个", "帮我调一下", "那个地方换个设置"],
     "impl": ["那个地方不对，改掉", "想把信息更新下"],
     "amb": ["把设置换一下", "帮我调整这个配置"]},
    {"verb": "list", "display": "列表", "functional": "query",
     "func": "列出{domain}列表。区分：list=列表罗列，detail=单条详情，二者近义易混。",
     "kw": ["列表", "都有哪些", "罗列"],
     "syn": ["都给我列出来", "把相关的都摆出来", "有哪些全部列一下"],
     "impl": ["之前那些都在哪", "相关的都汇总给我"],
     "amb": ["把相关的都翻出来", "列一下所有相关的"]},
    {"verb": "track", "display": "跟踪", "functional": "query",
     "func": "跟踪{domain}的进度/状态变化。区分：track=动态进度，detail=静态明细，二者近义易混。",
     "kw": ["跟踪", "进度", "到哪了"],
     "syn": ["我想跟着看进展", "帮我盯一下进度", "现在进行得怎样"],
     "impl": ["现在进行得怎样", "走到哪一步了没"],
     "amb": ["看看走到哪一步了", "帮我盯一下进度到哪"]},
    {"verb": "complain", "display": "投诉", "functional": "feedback",
     "func": "对{domain}发起投诉/反馈。区分：投诉用本工具，普通咨询用 query。",
     "kw": ["投诉", "反馈", "举报"],
     "syn": ["我要反映个问题", "帮我提个意见", "这事得说道说道"],
     "impl": ["这事得说道说道", "有点不满想说下"],
     "amb": ["对这个有点不满想说下", "帮我提个投诉"]},
]

# 每域内工具数 → 动作 × qualifier 扩出；qualifier 用于区分同 verb 的不同工具
QUALIFIERS = ["", "今日", "历史", "本月", "全部", "待处理", "已完成", "异常"]

# 规模配置：tools_per_domain 与每工具 query 数
SCALE_CONFIG = {
    "S":  {"n_domains": 5,  "tools_per_domain": 1,  "q_per_tool": 27},
    "M":  {"n_domains": 8,  "tools_per_domain": 5,  "q_per_tool": 12},
    "L":  {"n_domains": 10, "tools_per_domain": 20, "q_per_tool": 7},
    "LL": {"n_domains": 10, "tools_per_domain": 50, "q_per_tool": 5},
}

# 难度分布权重（ambiguous 拉到 ~30% 保证统计意义，计划 §7）
LEVEL_WEIGHTS = {"exact": 0.18, "synonym": 0.18, "implied": 0.14, "ambiguous": 0.32, "mixed": 0.18}


def _weighted_levels(budget: int, rng: random.Random) -> List[str]:
    levels: List[str] = []
    # 按比例分配，最后用 rng 补齐余数
    alloc = {}
    remaining = budget
    for lv, w in LEVEL_WEIGHTS.items():
        n = int(round(budget * w))
        alloc[lv] = n
        remaining -= n
    # 余数随机补到某档
    for _ in range(abs(remaining)):
        lv = rng.choice(list(LEVEL_WEIGHTS.keys()))
        alloc[lv] += 1 if remaining > 0 else -1
    for lv, n in alloc.items():
        levels.extend([lv] * max(0, n))
    rng.shuffle(levels)
    return levels[:budget]


def build_tools(scale: str, rng: random.Random) -> List[Dict[str, Any]]:
    cfg = SCALE_CONFIG[scale]
    n_dom = cfg["n_domains"]
    tpd = cfg["tools_per_domain"]
    domains = DOMAINS[:n_dom]
    tools: List[Dict[str, Any]] = []
    # 跨域工具比例 5-10%
    cross_ratio = 0.08

    for di, dom in enumerate(domains):
        # 该域工具：动作 × qualifier 扩出 tpd 个
        combos = []
        idx = 0
        while len(combos) < tpd:
            act = ACTIONS[idx % len(ACTIONS)]
            qual = QUALIFIERS[(idx // len(ACTIONS)) % len(QUALIFIERS)]
            combos.append((act, qual))
            idx += 1
        # 近义对：域内前 1-3 对（相邻 combo）标记为 near
        near_pairs_plan: List[List[int]] = []
        if tpd >= 2:
            n_pairs = min(3, max(1, tpd // 7))
            for p in range(n_pairs):
                a = p * 2
                b = p * 2 + 1
                if b < tpd:
                    near_pairs_plan.append([a, b])

        for ti, (act, qual) in enumerate(combos):
            name = f"{dom['key']}-{act['verb']}{('-' + qual) if qual else ''}"
            disp = f"{dom['display']}{act['display']}{qual}"
            # 描述：函数行 + 区分行（引用近义对）
            desc = act["func"].format(domain=dom["display"])
            # 近义区分
            pairmates = [j for pr in near_pairs_plan if ti in pr for j in pr if j != ti]
            if pairmates:
                mates_disp = "、".join(
                    f"{domains[di]['display']}{ACTIONS[combos[j][0]['verb'] == combos[j][0]['verb'] and 0]['display']}"
                    if False else f"{domains[di]['display']}{combos[j][1] and ''}{ACTIONS[combos[j][0]['verb'] == combos[j][0]['verb'] and 0]['display']}"
                    for j in pairmates
                )
                # 简化：直接列近义工具显示名
                mates_disp = "、".join(
                    f"{domains[di]['display']}{combos[j][1] and ''}{combos[j][0]['display']}"
                    for j in pairmates
                )
                desc += f" 近义区分：本工具易与 {mates_disp} 混淆——" \
                        f"前者侧重{dom['display']}的{combos[pairmates[0]][0]['display']}语义，" \
                        f"本工具侧重{dom['display']}的{act['display']}语义，请按用户实际意图区分。"
            trigger = list(dom["trigger_keywords"]) + list(act["kw"])
            # functional 标签
            functional = act["functional"]
            tools.append({
                "name": name,
                "display_name": disp,
                "domain": dom["key"],
                "cross_domains": [],
                "description": desc,
                "trigger_keywords": trigger,
                "functional": functional,
                "near_pairs": [f"{dom['key']}-{combos[j][0]['verb']}{('-' + combos[j][1]) if combos[j][1] else ''}" for j in pairmates],
            })

    # 跨域工具：随机挑 ~8% 挂第二个域
    n_cross = max(0, int(len(tools) * cross_ratio))
    other_domains = [d["key"] for d in domains]
    for _ in range(n_cross):
        t = rng.choice(tools)
        cand = [d for d in other_domains if d != t["domain"] and d not in t["cross_domains"]]
        if cand:
            second = rng.choice(cand)
            t["cross_domains"].append(second)
    return tools


def build_queries(tools: List[Dict[str, Any]], scale: str, rng: random.Random) -> List[Dict[str, Any]]:
    cfg = SCALE_CONFIG[scale]
    qpt = cfg["q_per_tool"]
    # 动作原型短语查找：按 verb
    act_by_verb = {a["verb"]: a for a in ACTIONS}
    queries: List[Dict[str, Any]] = []
    qid = 0
    for t in tools:
        dom_key = t["domain"]
        dom_disp = next(d["display"] for d in DOMAINS if d["key"] == dom_key)
        dom_kw = next(d["trigger_keywords"] for d in DOMAINS if d["key"] == dom_key)
        verb = t["name"].split("-")[1].split("-")[0]
        act = act_by_verb.get(verb, ACTIONS[0])
        # 找近义兄弟工具（用于 mixed / ambiguous 混淆源）
        near_tool = None
        if t["near_pairs"]:
            near_tool = next((x for x in tools if x["name"] == t["near_pairs"][0]), None)

        levels = _weighted_levels(qpt, rng)
        weak_topics = WEAK_TOPIC.get(dom_key, ["相关的"])
        for lv in levels:
            msg = ""
            signal = "strong"
            if lv == "exact":
                kw = rng.choice(act["kw"])
                msg = f"{dom_kw}{kw}怎么弄" if rng.random() < 0.7 else f"帮我{dom_kw}{kw}"
                signal = "strong"
            elif lv == "synonym":
                # weak：域主题短语 + 动作语义（不含触发词）
                msg = f"{rng.choice(weak_topics)}，{rng.choice(act['syn'])}"
                signal = "weak"
            elif lv == "implied":
                msg = f"{rng.choice(weak_topics)}，{rng.choice(act['impl'])}"
                signal = "weak"
            elif lv == "ambiguous":
                if near_tool is not None and rng.random() < 0.6:
                    # 用近义兄弟的短语，制造域内近义混淆（弱域信号）
                    nverb = near_tool["name"].split("-")[1].split("-")[0]
                    nact = act_by_verb.get(nverb, ACTIONS[0])
                    msg = f"{rng.choice(weak_topics)}，{rng.choice(nact['amb'] + nact['syn'])}"
                else:
                    msg = f"{rng.choice(weak_topics)}，{rng.choice(act['amb'])}"
                signal = "weak"
            elif lv == "mixed":
                if near_tool is not None and rng.random() < 0.6:
                    nverb = near_tool["name"].split("-")[1].split("-")[0]
                    nact = act_by_verb.get(nverb, ACTIONS[0])
                    msg = f"{rng.choice(weak_topics)}，{rng.choice(act['syn'])}，另外{rng.choice(nact['syn'])}"
                else:
                    msg = f"{rng.choice(weak_topics)}，{rng.choice(act['syn'])}，再{rng.choice(act['impl'])}"
                signal = "weak"

            queries.append({
                "id": qid,
                "message": msg,
                "correct_tool": t["name"],
                "correct_domain": t["domain"],
                "level": lv,
                "domain_signal": signal,
            })
            qid += 1
    return queries


def generate_scale(scale: str, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed + hash(scale) % 1000)
    cfg = SCALE_CONFIG[scale]
    domains_meta = {d["key"]: {"display": d["display"],
                               "trigger_keywords": d["trigger_keywords"],
                               "repr": d["repr"]}
                    for d in DOMAINS[:cfg["n_domains"]]}
    tools = build_tools(scale, rng)
    queries = build_queries(tools, scale, rng)
    # 自检统计
    from collections import Counter
    dom_counter = Counter(t["domain"] for t in tools)
    cross = sum(1 for t in tools if t["cross_domains"])
    lvl_counter = Counter(q["level"] for q in queries)
    sig_counter = Counter(q["domain_signal"] for q in queries)
    amb_ratio = lvl_counter.get("ambiguous", 0) / max(1, len(queries))
    return {
        "scale": scale,
        "n_tools": len(tools),
        "n_domains": cfg["n_domains"],
        "domains": domains_meta,
        "tools": tools,
        "queries": queries,
        "_selfcheck": {
            "tools_per_domain": dict(dom_counter),
            "cross_domain_tools": cross,
            "cross_ratio": round(cross / max(1, len(tools)), 3),
            "level_counts": dict(lvl_counter),
            "ambiguous_ratio": round(amb_ratio, 3),
            "signal_counts": dict(sig_counter),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["all", "S", "M", "L", "LL"], default="all")
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    scales = ["S", "M", "L", "LL"] if args.scale == "all" else [args.scale]
    for sc in scales:
        data = generate_scale(sc, args.seed)
        path = os.path.join(args.out_dir, f"lscale_{sc}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        sck = data["_selfcheck"]
        print(f"[gen] {sc}: tools={data['n_tools']} domains={data['n_domains']} "
              f"queries={len(data['queries'])} cross={sck['cross_domain_tools']} "
              f"({sck['cross_ratio']}) amb={sck['ambiguous_ratio']} -> {path}")


if __name__ == "__main__":
    main()
