"""GrowthBook 接入联调校准脚本（供真实 GB 实例验证 client.py 的请求/响应映射）。

用法：
  python scripts/calibrate_growthbook.py \
      --api-host http://localhost:3100 \
      --api-key <GB_SERVER_API_KEY> \
      --feature-key cal_test_<random>

脚本会：
  1) 用与 growthbook_client.create_experiment 完全一致的请求体 POST /api/features（json 类型，value 为 JSON 字符串）
  2) GET /api/features/:key 回读，打印 GB 实际存储的 value 形态（字符串 or 已解析）
  3) POST /api/experiments 建 experiment，打印返回
  4) PUT /api/features/:key 把 rollout 置 0（模拟暂停）
  5) DELETE /api/features/:key 清理
每步打印「请求体 / 响应状态码 / 响应 JSON」，方便核对 client.py 的字段名是否与你的 GB 版本一致。

不依赖 shop-agent 运行时，纯标准库（urllib + json）。
"""
import argparse
import json
import sys
import urllib.request

API = type(sys).modules["__main__"]
EXP_VALUE = {"rerank_threshold": 0.1, "llm_model": "qwen3.6-plus"}


def _call(method, url, api_key, payload=None, timeout=10):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        return e.code, {"_error": body}
    except Exception as e:  # noqa
        return -1, {"_error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-host", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--feature-key", default="cal_test_demo")
    args = ap.parse_args()

    host = args.api_host.rstrip("/")
    key = args.feature_key
    variations = [
        {"name": "Control", "value": json.dumps({})},
        {"name": "Treatment", "value": json.dumps(EXP_VALUE)},
    ]

    print("=" * 70)
    print("STEP 1: POST /api/features (json feature, value=JSON字符串)")
    print("=" * 70)
    feature_body = {
        "key": key,
        "name": "calibration test",
        "type": "json",
        "defaultValue": "{}",
        "project": "default",
        "environments": {
            "production": {"enabled": True, "rolloutPercentage": 100.0, "variations": variations}
        },
    }
    print("REQUEST:", json.dumps({"feature": feature_body}, ensure_ascii=False)[:2000])
    st, resp = _call("POST", f"{host}/api/features", args.api_key, {"feature": feature_body})
    print("STATUS:", st)
    print("RESPONSE:", json.dumps(resp, ensure_ascii=False)[:2000])

    print("\n" + "=" * 70)
    print("STEP 2: GET /api/features/:key (回读实际存储形态)")
    print("=" * 70)
    st, resp = _call("GET", f"{host}/api/features/{key}", args.api_key)
    print("STATUS:", st)
    feat = resp.get("feature", resp)
    prod = feat.get("environments", {}).get("production", {})
    print("回读 variations:", json.dumps(prod.get("variations"), ensure_ascii=False))
    print("→ 若 value 是字符串，说明 GB 接受 JSON 字符串（与 client.py 一致）；若被解析为对象，则 client.py 需改为传 dict")

    print("\n" + "=" * 70)
    print("STEP 3: POST /api/experiments (featureId 关联)")
    print("=" * 70)
    exp_body = {
        "name": "cal exp",
        "featureId": key,
        "type": "code",
        "status": "running",
        "variations": variations,
        "coverage": 1.0,
        "hashAttribute": "id",
        "phases": [{"dateStarted": "2026-01-01T00:00:00.000Z", "variationWeights": [0.5, 0.5]}],
    }
    print("REQUEST:", json.dumps({"experiment": exp_body}, ensure_ascii=False)[:2000])
    st, resp = _call("POST", f"{host}/api/experiments", args.api_key, {"experiment": exp_body})
    print("STATUS:", st)
    print("RESPONSE:", json.dumps(resp, ensure_ascii=False)[:2000])
    exp_id = resp.get("experiment", {}).get("id") if isinstance(resp.get("experiment"), dict) else None

    print("\n" + "=" * 70)
    print("STEP 4: PUT /api/features/:key (rollout=0 模拟暂停)")
    print("=" * 70)
    st, resp = _call(
        "PUT",
        f"{host}/api/features/{key}",
        args.api_key,
        {"feature": {"environments": {"production": {"enabled": True, "rolloutPercentage": 0.0}}}},
    )
    print("STATUS:", st, "RESPONSE:", json.dumps(resp, ensure_ascii=False)[:500])

    print("\n" + "=" * 70)
    print("STEP 5: DELETE /api/features/:key (清理)")
    print("=" * 70)
    st, resp = _call("DELETE", f"{host}/api/features/{key}", args.api_key)
    print("STATUS:", st, "RESPONSE:", json.dumps(resp, ensure_ascii=False)[:500])
    if exp_id:
        st, resp = _call("DELETE", f"{host}/api/experiments/{exp_id}", args.api_key)
        print("DELETE experiment STATUS:", st)

    print("\n校准脚本结束。请核对上面各步的 STATUS==2xx 与字段形态是否与 client.py 一致。")


if __name__ == "__main__":
    main()
