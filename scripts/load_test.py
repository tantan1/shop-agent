"""Representative load generator for the shop-agent full chain (mock mode).

Drives many users x many turns against the chat endpoint to populate
Prometheus with typical throughput / latency / L2 / embedding / token data
so Grafana dashboards show meaningful curves for parameter tuning.
"""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8000/agent/api/v1/chatagent/agent/chat"
KEY = "ak_bigdata_internal_2024"
USERS = 15
TURNS = 8          # >=5 so every conversation triggers an L2 save
CONCURRENCY = 5    # parallel users

QUESTIONS = [
    "你好，我的订单到哪了",
    "我想查一下昨天下单的连衣裙发货了没",
    "帮我看看订单 A20240912003 的物流状态",
    "这款手机的保修期是多久",
    "我的积分还有多少，能换什么",
    "再帮我查下订单 A20240912003 的退款进度",
    "我想退掉上周买的耳机",
    "推荐几款适合夏天的连衣裙",
    "我的优惠券为什么用不了",
    "店铺的营业时间是几点到几点",
]

def chat(uid, conv, msg):
    body = json.dumps({
        "user_id": uid, "conversation_id": conv, "message": msg
    }).encode("utf-8")
    req = urllib.request.Request(URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("X-API-Key", KEY)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            code = r.getcode()
            r.read()
        dt = time.time() - t0
        return code, dt
    except Exception as e:
        return getattr(e, "code", "ERR"), time.time() - t0

def run_user(i):
    uid = f"u_load_{i}"
    conv = f"conv_load_{i}"
    rows = []
    for t in range(TURNS):
        msg = QUESTIONS[(i + t) % len(QUESTIONS)]
        code, dt = chat(uid, conv, msg)
        rows.append((t + 1, code, round(dt, 3)))
    return rows

if __name__ == "__main__":
    t_start = time.time()
    ok = err = 0
    lat = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for rows in ex.map(run_user, range(USERS)):
            for (turn, code, dt) in rows:
                if code == 200:
                    ok += 1
                    lat.append(dt)
                else:
                    err += 1
    elapsed = time.time() - t_start
    print(f"DONE users={USERS} turns={TURNS} ok={ok} err={err} "
          f"wall={elapsed:.1f}s throughput={ok/elapsed:.2f} req/s")
    if lat:
        lat.sort()
        p50 = lat[len(lat)//2]
        p95 = lat[int(len(lat)*0.95)]
        print(f"client latency p50={p50:.3f}s p95={p95:.3f}s max={max(lat):.3f}s")
