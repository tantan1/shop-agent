import json


def show(tag, f):
    d = json.load(open(f, encoding="utf-8"))["summary"]
    g5 = d["group5_free_gen"]
    g6 = d["group6_constrained"]
    amb = d["ambiguous_subset"]
    print("=== %s (holdout, unseen queries) ===" % tag)
    print("  free_gen  overall=%.1f%%  fmt=%.1f%%  p50=%sms p95=%sms" % (
        g5["overall_acc"] * 100, g5["format_compliance"] * 100,
        g5["latency_ms_p50"], g5["latency_ms_p95"]))
    print("  constr    overall=%.1f%%  fmt=%.1f%%  p50=%sms p95=%sms" % (
        g6["overall_acc"] * 100, g6["format_compliance"] * 100,
        g6["latency_ms_p50"], g6["latency_ms_p95"]))
    print("  ambiguous free=%.1f%% con=%.1f%% (n=%s)" % (
        amb["free_acc"] * 100, amb["con_acc"] * 100, amb["n"]))
    print("  逐level:")
    for lv, v in d["per_level"].items():
        print("    %-10s n=%3d free=%.1f%% con=%.1f%%" % (
            lv, v["n"], v["free_acc"] * 100, v["con_acc"] * 100))
    print()


show("BASE", "outputs/eval_M8_holdout_base.json")
show("SFT-M8", "outputs/eval_M8_holdout_sft.json")
