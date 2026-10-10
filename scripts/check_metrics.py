import urllib.request, json, urllib.parse

BASE = "http://localhost:19090/api/v1/query"
queries = [
    "shop_agent_agent_chat_total",
    "shop_agent_l2_save_triggered_total",
    "shop_agent_l2_save_success_total",
    "shop_agent_l2_save_failure_total",
    "shop_agent_embedding_requests_total",
    "gateway_tokens_total",
    "gateway_requests_total",
    "shop_agent_memory_blocks_total",
    "shop_agent_l2_summary_tokens_sum",
    "shop_agent_exceptions_total",
    "shop_agent_redis_available",
    "shop_agent_forgetting_job_duration_seconds_count",
]

for q in queries:
    url = BASE + "?" + urllib.parse.urlencode({"query": q})
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            d = json.load(r)
        res = d["data"]["result"]
        if not res:
            print(f"{q:50s} => EMPTY")
        else:
            vals = ", ".join(f"{m['metric'].get('status') or m['metric'].get('trigger') or m['metric'].get('provider') or m['metric'].get('block_type') or ''}={m['value'][1]}" for m in res)
            print(f"{q:50s} => {vals}")
    except Exception as e:
        print(f"{q:50s} => ERROR {e}")
