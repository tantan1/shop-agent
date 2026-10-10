"""GrowthBook Flag 生命周期审计脚本（scope §9.3 陈旧检测）。

扫描 GB 实例中的 flag，识别以下「陈旧/需治理」情形并告警：
  1) 超期未清理：feature 带 tag `expected_end_date:<YYYY-MM-DD>`，且日期 < 今天  → HIGH
  2) 长期 100% 单一变体（实验已赢但从未清理/固化）：
       - exp_ 类：rollout==100 且仅 1 个有效变体（treatment 流量=0）            → MEDIUM
       - canary_ 类：rollout==100 长期（金丝雀应推进或回滚）                   → LOW
  3) 孤儿 flag：GB 中存在，但代码里没有任何 `eval_variant("<key>")` 引用         → MEDIUM

用法：
  python scripts/audit_stale_flags.py \
      --api-host http://localhost:3100 \
      --api-key <GB_SERVER_API_KEY> \
      [--code-root apps/shop-agent/src] \
      [--strict]          # 出现 MEDIUM/HIGH 即以 exit=1 退出（CI 门禁）

输出人类可读报告；任何 HIGH 或（--strict 下）MEDIUM 都会令 exit code 非 0，
便于 CI 兜底（scope §9.4）。

不依赖 shop-agent 运行时，纯标准库（urllib + json + re + datetime）。
"""
import argparse
import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.request

# 命名前缀（与 growthbook_client._FLAG_PREFIXES 保持一致）
_FLAG_PREFIXES = ("exp_", "canary_", "switch_", "perm_")
_FLAG_PREFIXES_WITH_END_DATE = ("exp_", "canary_")
_EVAL_REF_RE = re.compile(r'eval_variant\(\s*[\'"](?P<key>[^\'"]+)[\'"]')


def _call(method, url, api_key, timeout=15):
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        return e.code, {"_error": e.read().decode("utf-8", "replace")}
    except Exception as e:  # noqa: BLE001
        return -1, {"_error": str(e)}


def _iter_features(resp):
    """兼容 GB 返回 list 与 dict 两种形态。"""
    features = resp.get("features", resp)
    if isinstance(features, dict):
        for key, feat in features.items():
            if isinstance(feat, dict):
                feat.setdefault("key", key)
                yield feat
    elif isinstance(features, list):
        for feat in features:
            if isinstance(feat, dict):
                yield feat


def _tag_value(tags, prefix):
    for t in tags or []:
        if isinstance(t, str) and t.startswith(prefix):
            return t.split(":", 1)[1]
    return ""


def _scan_code_refs(code_root):
    """返回代码中被 eval_variant("<key>") 引用的 key 集合。"""
    refs = set()
    if not code_root or not os.path.isdir(code_root):
        print(f"[warn] 代码根目录不存在，跳过孤儿检测: {code_root}")
        return refs
    for root, _dirs, files in os.walk(code_root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        m = _EVAL_REF_RE.search(line)
                        if m:
                            refs.add(m.group("key"))
            except OSError:
                continue
    return refs


def audit(host, api_key, code_root, strict):
    host = host.rstrip("/")
    st, resp = _call("GET", f"{host}/api/features", api_key)
    if st < 200 or st >= 300:
        print(f"[error] GET /api/features 失败: status={st} resp={resp}")
        return 2
    if isinstance(resp, dict) and not resp.get("features"):
        # 某些版本返回 {"features": {}，这里兜底打印顶层 keys 帮助排查
        print(f"[warn] 响应中无 features 字段，顶层 keys={list(resp.keys())}；如字段名不同请对齐。")
        return 0

    refs = _scan_code_refs(code_root)
    today = datetime.date.today()
    findings = []  # (severity, key, reason)

    for feat in _iter_features(resp):
        key = feat.get("key") or feat.get("id") or "<unknown>"
        tags = feat.get("tags", []) or []
        prod = (feat.get("environments", {}) or {}).get("production", {}) or {}
        rollout = prod.get("rolloutPercentage", 0)
        variations = prod.get("variations", []) or feat.get("variations", []) or []

        # 1) 超期
        end_date = _tag_value(tags, "expected_end_date:")
        if end_date:
            try:
                end = datetime.date.fromisoformat(end_date[:10])
                if end < today:
                    days = (today - end).days
                    findings.append(
                        ("HIGH", key, f"已超 expected_end_date({end_date}) {days} 天，未清理/固化")
                    )
            except ValueError:
                findings.append(("LOW", key, f"expected_end_date 格式非法: {end_date!r}"))

        # 2) 长期 100% 单一有效变体
        if rollout == 100 and isinstance(variations, list):
            effective = [v for v in variations if (v.get("weight", 1) or 0) > 0]
            if len(variations) <= 1 or len(effective) <= 1:
                if key.startswith("exp_"):
                    findings.append(
                        ("MEDIUM", key, "实验长期 100% 单变体（实验已赢但从未固化/清理）")
                    )
                elif key.startswith("canary_"):
                    findings.append(("LOW", key, "金丝雀长期 100% 开量（应推进固化或回滚）"))

        # 3) 孤儿
        if key not in refs:
            findings.append(("MEDIUM", key, "GB 中存在但代码无 eval_variant 引用（疑似孤儿 flag）"))

    # 报告
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    findings.sort(key=lambda f: order.get(f[0], 9))
    print("=" * 72)
    print(f"GrowthBook Flag 生命周期审计 — {today.isoformat()}")
    print(f"实例: {host}   代码根: {code_root or '(未指定)'}")
    print("=" * 72)
    if not findings:
        print("✅ 未发现陈旧/需治理的 flag。")
        return 0
    for sev, key, reason in findings:
        print(f"[{sev:6}] {key:40} {reason}")
    print("-" * 72)
    counts = {}
    for sev, _, _ in findings:
        counts[sev] = counts.get(sev, 0) + 1
    print("汇总: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    high = counts.get("HIGH", 0)
    med = counts.get("MEDIUM", 0)
    if high > 0 or (strict and med > 0):
        print(f"\n❌ 存在需治理项（HIGH={high}, MEDIUM={med}）-> exit 1")
        return 1
    print("\n⚠️ 仅 LOW 级别提示，exit 0。")
    return 0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_code_root = os.path.normpath(os.path.join(here, "..", "apps", "shop-agent", "src"))
    ap = argparse.ArgumentParser(description="GrowthBook Flag 生命周期审计（scope §9）")
    ap.add_argument("--api-host", required=True, help="GrowthBook API host，如 http://localhost:3100")
    ap.add_argument("--api-key", required=True, help="GrowthBook Server API Key")
    ap.add_argument("--code-root", default=default_code_root, help="待扫描代码根目录（检测孤儿 flag）")
    ap.add_argument("--strict", action="store_true", help="出现 MEDIUM/HIGH 即 exit 1（CI 门禁）")
    args = ap.parse_args()
    sys.exit(audit(args.api_host, args.api_key, args.code_root, args.strict))


if __name__ == "__main__":
    main()
