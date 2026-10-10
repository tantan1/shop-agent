"""诊断：直接 POST 一条 GatewayDown alert 到 monitoring /ingest/alert，
验证 monitoring 内部链路（ingest -> rca -> websocket broadcast）是否打通。
剥离 Prometheus/Alertmanager，专测 monitoring 自身。
"""
import asyncio
import json
import urllib.request

import websockets

MON_WS = "ws://localhost/monitor/ws"
MON_INGEST = "http://localhost/monitor/ingest/alert"
TOKEN = "local-webhook-token"


async def main():
    print("==> 连接 WS")
    received = []
    ws = await websockets.connect(MON_WS, open_timeout=10, ping_interval=20)

    async def listener():
        async for msg in ws:
            m = json.loads(msg)
            print(f"[WS] type={m.get('type')} sev={m.get('severity')} cause={m.get('root_cause')}")
            received.append(m)

    task = asyncio.create_task(listener())
    await asyncio.sleep(1)

    print("==> POST GatewayDown alert 到 /ingest/alert")
    payload = {
        "version": "4",
        "groupKey": "{}:{alertname=\"GatewayDown\"}",
        "status": "firing",
        "receiver": "monitoring-agent",
        "alerts": [{
            "status": "firing",
            "labels": {
                "alertname": "GatewayDown",
                "severity": "critical",
                "service_name": "gateway",
                "job": "gateway",
            },
            "annotations": {
                "summary": "网关 Gateway 不可用（up==0）",
                "description": "Prometheus 连续 1 分钟无法抓取 gateway:80/metrics",
            },
        }],
    }
    req = urllib.request.Request(
        MON_INGEST,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        print(f"[ingest] HTTP {resp.status}: {resp.read().decode()[:200]}")
    except Exception as e:
        print(f"[ingest FAIL] {e}")

    # 等 WS 推送
    for _ in range(30):
        await asyncio.sleep(1)
        if any(m.get("type") == "alert" for m in received):
            print("[PASS] monitoring 收到 alert 并经 WS 推送成功")
            await ws.close()
            task.cancel()
            return True

    print("[FAIL] 未收到 WS alert 推送")
    await ws.close()
    task.cancel()
    return False


if __name__ == "__main__":
    import sys
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
