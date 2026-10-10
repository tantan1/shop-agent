"""四层工具选择 Pipeline 负载生成器。

只发「react 模式」查询（含 >=2 个推理触发词），强制进入
ToolSelectPipeline（P0->P1->P2->P3），从而让 shop_agent_tool_select_*
指标在 Prometheus/Grafana 上产生数据。

用法: python scripts/load_test_toolselect.py --count 30 --concurrency 4
"""
import argparse
import json
import random
import threading
import time
import urllib.request

URL = "http://localhost:8000/agent/api/v1/chatagent/agent/chat"
KEY = "ak_bigdata_internal_2024"
# 含多个触发词(同时/另外/然后/对比/趋势...)确保 react 模式
QUERIES = [
    "查一下我的积分同时帮我看看会员等级，另外再对比下上个月的消费趋势和物流进度",
    "帮我看看订单状态同时分析下我的消费习惯，另外再推荐几个商品",
    "查一下优惠券然后对比上个月的优惠力度，另外再看看我的会员权益",
    "分析我的购物偏好同时对比不同品类的花费，另外再给个省钱建议",
]


def send(n: int):
    q = random.choice(QUERIES)
    body = json.dumps({
        "user_id": f"u_ts{n}",
        "conversation_id": f"conv_ts{n}_{int(time.time())}",
        "message": q,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json",
        "X-API-Key": KEY,
    }, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            r.read()
    except Exception as e:
        print(f"  req {n} err: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    sem = threading.BoundedSemaphore(args.concurrency)
    def worker(i):
        with sem:
            send(i)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.count)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print(f"done {args.count} react queries in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
