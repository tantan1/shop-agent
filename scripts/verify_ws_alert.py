"""真实故障端到端验证：gateway 缩容 → Prometheus 告警 → Alertmanager →
monitoring /ingest/alert → RCA → WebSocket 推送 demo。

用法：python verify_ws_alert.py
前置：kubectl 在当前 context，gateway 正常 Running。
"""
import asyncio
import json
import subprocess
import time
import sys

import websockets

MON_WS = "ws://localhost/monitor/ws"
NS = "shop-agent"


def kubectl(args):
    r = subprocess.run(["kubectl"] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[kubectl ERROR] {' '.join(args)}\n{r.stderr}")
    return r


async def main():
    print("==> 1. 连接 WebSocket ws://localhost/monitor/ws")
    received = []
    try:
        ws = await websockets.connect(MON_WS, open_timeout=10, ping_interval=20)
    except Exception as e:
        print(f"[FAIL] WebSocket 连接失败: {e}")
        return False

    async def listener():
        try:
            async for msg in ws:
                m = json.loads(msg)
                if m.get("type") == "alert":
                    print(f"[WS ALERT] sev={m.get('severity')} cause={m.get('root_cause')}")
                    print(f"           affected={m.get('affected')} recs={m.get('recommendations')}")
                received.append(m)
        except Exception as e:
            print(f"[WS listener 结束] {e}")

    task = asyncio.create_task(listener())
    await asyncio.sleep(1)

    print("==> 2. 制造真实故障：scale down gateway")
    kubectl(["scale", "deployment/gateway", "-n", NS, "--replicas=0"])
    print("    等待 Prometheus(15s eval + 1m for) + Alertmanager(group_wait 30s) ...")

    deadline = time.time() + 200
    while time.time() < deadline:
        await asyncio.sleep(10)
        alerts = [m for m in received if m.get("type") == "alert"]
        if alerts:
            a = alerts[-1]
            print(f"\n[PASS] 收到真实故障告警推送（severity={a.get('severity')}）")
            print(f"   root_cause={a.get('root_cause')}")
            print(f"   recommendations={a.get('recommendations')}")
            await ws.close()
            task.cancel()
            return True
        # 进度提示：查询 Prometheus 是否已 firing
        if int(time.time()) % 30 < 10:
            try:
                r = subprocess.run(
                    ["kubectl", "exec", "-n", NS, "prometheus-f47c99ddc-ck9fb",
                     "--", "wget", "-qO-", "http://localhost:9090/api/v1/alerts"],
                    capture_output=True, text=True, timeout=10)
                import re
                names = re.findall(r'"alertname":"([^"]+)"', r.stdout)
                if names:
                    print(f"    [Prometheus firing] {set(names)}")
            except Exception:
                pass

    print("\n[FAIL] 200 秒内未收到告警推送")
    await ws.close()
    task.cancel()
    return False


if __name__ == "__main__":
    ok = asyncio.run(main())
    print("\n==> 3. 恢复 gateway")
    kubectl(["scale", "deployment/gateway", "-n", NS, "--replicas=1"])
    kubectl(["rollout", "status", "deployment/gateway", "-n", NS, "--timeout=120s"])
    sys.exit(0 if ok else 1)
