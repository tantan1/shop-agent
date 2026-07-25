"""
A2A 协议端到端测试脚本
=======================
测试流程：
  1. Agent 发现 (GET /.well-known/agent-card.json)
  2. A2A 健康检查 (GET /a2a/health)
  3. 注册 Webhook 订阅 (POST /a2a/webhooks)
  4. 提交异步任务——查询订单 (POST /a2a/tasks/send, 带 callback_url)
  5. 轮询任务状态 (GET /a2a/tasks/{task_id})
  6. 验证 Webhook 回调是否收到
  7. 查询任务列表 (GET /a2a/tasks)
  8. 查询对话历史 (GET /a2a/conversations)
  9. 取消 Webhook 订阅 (DELETE /a2a/webhooks/{subscription_id})

用法:
  1. 确保 uvicorn 已启动:  cd e:\workspace\shop-agent && python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
  2. 运行测试:          cd e:\workspace\shop-agent && python scripts/test_a2a_protocol.py
  3. 自定义消息:        python scripts/test_a2a_protocol.py --message "查物流运单 SHIP-888"
  4. 仅发现测试:        python scripts/test_a2a_protocol.py --discovery-only
  5. 纯轮询模式(无webhook): python scripts/test_a2a_protocol.py --no-webhook
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import sys
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import Optional

# ── 配置 ────────────────────────────────────────────────────────────
BASE_URL = "http://localhost:8000"
API_KEY = "ak_bigdata_internal_2024"
AUTH_HEADER = {"Authorization": f"Bearer {API_KEY}"}
DEFAULT_MESSAGE = "帮我查一下订单 ORD-2024-001 的状态和物流信息"
WEBHOOK_PORT = 18765  # 本地 webhook 接收端口

# ── 全局状态：webhook 接收器 ──
webhook_received: list[dict] = []
webhook_server: Optional[HTTPServer] = None


# =============================================================================
# 内置 Webhook 接收服务器（线程安全）
# =============================================================================

class WebhookHandler(BaseHTTPRequestHandler):
    """接收 A2A webhook 回调的 HTTP handler。"""

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # 验证 HMAC 签名（如果提供）
        secret = b"test-webhook-secret"
        expected_sig = self.headers.get("X-A2A-Signature", "")
        if expected_sig.startswith("sha256="):
            computed = f"sha256={hmac.new(secret, body, hashlib.sha256).hexdigest()}"
            verified = hmac.compare_digest(computed, expected_sig)
        else:
            verified = True  # 无签名 header 时跳过验证

        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {"raw": body.decode("utf-8", errors="replace")}

        event = self.headers.get("X-A2A-Event", "unknown")

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "signature_verified": verified,
            "payload": payload,
            "headers": dict(self.headers),
        }
        webhook_received.append(record)

        print(f"\n  [Webhook] event={event} verified={verified}")
        print(f"     task_id={payload.get('task_id', '?')} status={payload.get('status', '?')}")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"received": true}')

    def log_message(self, format, *args) -> None:
        pass  # 静默，避免干扰测试输出


def start_webhook_receiver(port: int = WEBHOOK_PORT) -> HTTPServer:
    """在后台线程启动 webhook 接收服务器。"""
    global webhook_server
    server = HTTPServer(("127.0.0.1", port), WebhookHandler)
    webhook_server = server
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"  [OK] Webhook 接收器已启动: http://127.0.0.1:{port}")
    return server


def stop_webhook_receiver() -> None:
    """关闭 webhook 接收服务器。"""
    global webhook_server
    if webhook_server:
        webhook_server.shutdown()
        webhook_server = None


# =============================================================================
# HTTP 辅助函数
# =============================================================================

async def http_get(path: str, auth: bool = True) -> tuple[int, dict]:
    """GET 请求。"""
    import aiohttp
    headers = {**AUTH_HEADER} if auth else {}
    url = f"{BASE_URL}{path}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            try:
                body = await resp.json()
            except Exception:
                body = {"_raw": await resp.text()}
            return resp.status, body


async def http_post(path: str, data: dict, auth: bool = True) -> tuple[int, dict]:
    """POST 请求。"""
    import aiohttp
    headers = {**AUTH_HEADER, "Content-Type": "application/json"} if auth else {"Content-Type": "application/json"}
    url = f"{BASE_URL}{path}"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            try:
                body = await resp.json()
            except Exception:
                body = {"_raw": await resp.text()}
            return resp.status, body


async def http_delete(path: str) -> tuple[int, dict]:
    """DELETE 请求。"""
    import aiohttp
    url = f"{BASE_URL}{path}"
    async with aiohttp.ClientSession() as session:
        async with session.delete(url, headers=AUTH_HEADER, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            try:
                body = await resp.json()
            except Exception:
                body = {"_raw": await resp.text()}
            return resp.status, body


# =============================================================================
# 测试用例
# =============================================================================

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "(skip)"

_results: list[tuple[str, str, float]] = []  # (name, status, elapsed)


def record(name: str, ok: bool, elapsed: float, detail: str = "") -> None:
    status = PASS if ok else FAIL
    msg = f"  {status} {name} ({elapsed*1000:.0f}ms)"
    if detail:
        msg += f" — {detail}"
    print(msg)
    _results.append((name, status, elapsed))


def _unwrap(body: dict) -> dict:
    """从 success_response 包装中提取 data 字段。"""
    if "data" in body and isinstance(body.get("data"), dict):
        return body["data"]
    return body


async def test_01_discovery() -> bool:
    """测试 Agent Card 发现（无需认证）。"""
    t0 = time.perf_counter()
    code, body = await http_get("/.well-known/agent-card.json", auth=False)
    elapsed = time.perf_counter() - t0

    ok = (
        code == 200
        and body.get("name")
        and len(body.get("skills", [])) > 0
        and len(body.get("endpoints", [])) > 0
        and body.get("authentication") is not None
        and body.get("capabilities", {}).get("asyncTasks") is True
    )
    record("Agent Card 发现", ok, elapsed, f"skills={len(body.get('skills',[]))} endpoints={len(body.get('endpoints',[]))} name={body.get('name','?')}")
    if not ok:
        print(f"     body keys={list(body.keys())} auth={body.get('authentication')}")
    return ok


def _data(body: dict) -> dict:
    """从 success_response 包装中提取 data 字段（兼容直接返回的 JSON）。"""
    if "data" in body and isinstance(body.get("data"), dict):
        return body["data"]
    return body


async def test_02_health() -> bool:
    """测试 A2A 健康检查（无需认证）。"""
    t0 = time.perf_counter()
    code, body = await http_get("/a2a/health", auth=False)
    elapsed = time.perf_counter() - t0

    d = _unwrap(body)
    ok = code == 200 and d.get("status") is not None
    deps = d.get("dependencies", {})
    detail = f"status={d.get('status')} deps={json.dumps(deps, ensure_ascii=False)}"
    record("A2A 健康检查", ok, elapsed, detail)
    return ok


async def test_03_subscribe_webhook() -> Optional[str]:
    """测试 Webhook 订阅注册。"""
    t0 = time.perf_counter()
    code, body = await http_post("/a2a/webhooks", {
        "url": f"http://127.0.0.1:{WEBHOOK_PORT}/webhook",
        "events": ["task.completed", "task.failed"],
        "secret": "test-webhook-secret",
        "ttl_seconds": 3600,
    })
    elapsed = time.perf_counter() - t0

    ok = code == 200 and _unwrap(body).get("subscription_id", "").startswith("wh_")
    sub_id = _unwrap(body).get("subscription_id") if ok else None
    detail = f"sub_id={sub_id}" if sub_id else f"code={code} body={json.dumps(body, ensure_ascii=False)[:100]}"
    record("Webhook 订阅注册", ok, elapsed, detail)
    return sub_id


async def test_04_send_task_with_callback(callback_url: str) -> dict | None:
    """测试提交异步任务（带 Webhook 回调）。"""
    t0 = time.perf_counter()
    code, body = await http_post("/a2a/tasks/send", {
        "message": DEFAULT_MESSAGE,
        "domain": "ecommerce",
        "callback_url": callback_url,
    })
    elapsed = time.perf_counter() - t0

    ok = code == 200 and _unwrap(body).get("task_id") and _unwrap(body).get("status") == "pending"
    if not ok:
        record("提交异步任务", False, elapsed, f"code={code} body={json.dumps(body, ensure_ascii=False)[:100]}")
        return None

    task_id = _unwrap(body).get("task_id")
    record("提交异步任务", True, elapsed, f"task_id={task_id}")
    return {"task_id": task_id, "conversation_id": _unwrap(body).get("conversation_id"), "domain": _unwrap(body).get("domain")}


async def test_05_poll_task(task_id: str, max_wait: float = 120.0) -> bool:
    """轮询任务直到完成。"""
    t0 = time.perf_counter()
    interval = 2.0
    last_status = "?"
    error_msg = ""

    while (time.perf_counter() - t0) < max_wait:
        code, body = await http_get(f"/a2a/tasks/{task_id}")
        if code != 200:
            error_msg = f"HTTP {code}"
            break

        d = _unwrap(body)
        status = d.get("status", "?")
        if status != last_status:
            elapsed_sofar = time.perf_counter() - t0
            print(f"     [{elapsed_sofar:.0f}s] status={status}")
            last_status = status

        if status in ("completed", "failed", "cancelled"):
            elapsed = time.perf_counter() - t0
            ok = status == "completed"
            result_preview = (d.get("result") or d.get("error") or "")[:120]
            record(
                "任务执行结果",
                ok,
                elapsed,
                f"status={status} {result_preview}",
            )
            if ok:
                print(f"\n{'─'*60}")
                print(f"  [Reply] Agent 回复:\n{d.get('result', '(无)')}")
                print(f"{'─'*60}")
            return ok

        await asyncio.sleep(interval)

    record("任务执行结果", False, max_wait, f"timeout after {max_wait}s, last_status={last_status} {error_msg}")
    return False


async def test_06_verify_webhook(expect_task_id: str) -> bool:
    """验证 Webhook 是否收到回调。"""
    t0 = time.perf_counter()

    # 等待 webhook 送达（最多 10s）
    for _ in range(10):
        if webhook_received:
            break
        await asyncio.sleep(1.0)

    elapsed = time.perf_counter() - t0

    if not webhook_received:
        record("Webhook 回调验证", False, elapsed, "未收到任何 webhook 回调")
        return False

    match = None
    for rec in webhook_received:
        payload = rec.get("payload", {})
        if payload.get("task_id") == expect_task_id:
            match = rec
            break

    if match is None:
        # 也接受第一条
        match = webhook_received[0]
        ok = match.get("payload", {}).get("task_id") == expect_task_id
    else:
        ok = True

    detail = f"收到 {len(webhook_received)} 条回调 event={match.get('event')} verified={match.get('signature_verified')}"
    record("Webhook 回调验证", ok, elapsed, detail)
    return ok


async def test_07_list_tasks() -> bool:
    """测试任务列表。"""
    t0 = time.perf_counter()
    code, body = await http_get("/a2a/tasks?limit=10")
    elapsed = time.perf_counter() - t0

    d = _unwrap(body)
    tasks = d.get("tasks", [])
    ok = code == 200 and isinstance(tasks, list)
    detail = f"total={d.get('total')} returned={len(tasks)}"
    record("任务列表查询", ok, elapsed, detail)
    return ok


async def test_08_conversations() -> bool:
    """测试对话列表。"""
    t0 = time.perf_counter()
    code, body = await http_get("/a2a/conversations?limit=10")
    elapsed = time.perf_counter() - t0

    d = _unwrap(body)
    convs = d.get("conversations", [])
    ok = code == 200 and isinstance(convs, list)
    detail = f"total={d.get('total')} returned={len(convs)}"

    # 如果有对话，测试消息历史
    if ok and convs:
        conv_id = convs[0].get("conversation_id")
        code2, body2 = await http_get(f"/a2a/conversations/{conv_id}/messages")
        msgs = _unwrap(body2).get("messages", [])
        ok = ok and code2 == 200
        detail += f" msgs={len(msgs)}"

    record("对话列表 & 消息历史", ok, elapsed, detail)
    return ok


async def test_09_cancel_task() -> bool:
    """测试取消任务（创建一个新任务然后取消）。"""
    # 先创建一个任务
    code, body = await http_post("/a2a/tasks/send", {
        "message": "这个任务会被立即取消",
        "domain": "ecommerce",
    })
    if code != 200:
        record("取消任务", False, 0, "创建测试任务失败")
        return False

    task_id = _unwrap(body).get("task_id")

    t0 = time.perf_counter()
    code, body = await http_post(f"/a2a/tasks/{task_id}/cancel", {})
    elapsed = time.perf_counter() - t0

    # 409 表示任务已终态（太快了），也算通过
    if code == 409:
        record("取消任务", True, elapsed, f"task_id={task_id} (任务已终态，无法取消)")
        return True

    ok = code == 200 and _unwrap(body).get("status") == "cancelled"
    detail = f"task_id={task_id} status={_unwrap(body).get('status')}"
    record("取消任务", ok, elapsed, detail)
    return ok


async def test_10_unsubscribe_webhook(subscription_id: str) -> bool:
    """测试取消 Webhook 订阅。"""
    t0 = time.perf_counter()
    code, body = await http_delete(f"/a2a/webhooks/{subscription_id}")
    elapsed = time.perf_counter() - t0

    ok = code == 200 and _unwrap(body).get("deleted") is True
    detail = f"sub_id={subscription_id} deleted={_unwrap(body).get('deleted')}"
    record("取消 Webhook 订阅", ok, elapsed, detail)
    return ok


# =============================================================================
# 主流程
# =============================================================================

async def run_all_tests(no_webhook: bool = False, discovery_only: bool = False) -> bool:
    """执行完整 A2A 协议测试套件。"""
    print("=" * 60)
    print("  [TEST] A2A 协议端到端测试")
    print(f"    目标服务: {BASE_URL}")
    print(f"    测试消息: {DEFAULT_MESSAGE[:60]}...")
    print(f"    Webhook:  {'禁用' if no_webhook else f'http://127.0.0.1:{WEBHOOK_PORT}'}")
    print("=" * 60)
    print()

    all_pass = True
    webhook_sub_id: Optional[str] = None

    try:
        # ── Step 0: 启动 Webhook 接收器 ──
        if not no_webhook:
            start_webhook_receiver(WEBHOOK_PORT)
            await asyncio.sleep(0.5)

        # ── Step 1: 发现 ──
        if not await test_01_discovery():
            all_pass = False

        if discovery_only:
            print("\n  (--discovery-only 模式，跳过后续测试)")
            return all_pass

        # ── Step 2: 健康检查 ──
        if not await test_02_health():
            all_pass = False

        # ── Step 3: Webhook 订阅 ──
        if not no_webhook:
            webhook_sub_id = await test_03_subscribe_webhook()
            if not webhook_sub_id:
                all_pass = False

        # ── Step 4: 提交任务 ──
        callback_url = f"http://127.0.0.1:{WEBHOOK_PORT}/webhook" if not no_webhook else None
        task_info = await test_04_send_task_with_callback(callback_url) if callback_url else None

        if not callback_url:
            # 无 webhook 模式：不带 callback_url 提交任务
            task_info = await test_04_send_task_with_callback(None) if not no_webhook else None
            if task_info is None:
                # fallback: POST /a2a/tasks/send without callback
                t0 = time.perf_counter()
                code, body = await http_post("/a2a/tasks/send", {
                    "message": DEFAULT_MESSAGE,
                    "domain": "ecommerce",
                })
                elapsed = time.perf_counter() - t0
                if code == 200:
                    task_info = {"task_id": _unwrap(body).get("task_id"), "conversation_id": _unwrap(body).get("conversation_id"), "domain": _unwrap(body).get("domain")}
                    record("提交异步任务(无webhook)", True, elapsed, f"task_id={task_info['task_id']}")

        if task_info is None:
            print("  [FAIL] 提交任务失败，跳过后续测试")
            return False

        task_id = task_info["task_id"]

        # ── Step 5: 轮询任务 ──
        if not await test_05_poll_task(task_id, max_wait=180):
            all_pass = False

        # ── Step 6: 验证 Webhook ──
        if not no_webhook:
            await asyncio.sleep(1.0)  # 给 webhook 送达留缓冲
            if not await test_06_verify_webhook(task_id):
                all_pass = False

        # ── Step 7: 任务列表 ──
        if not await test_07_list_tasks():
            all_pass = False

        # ── Step 8: 对话历史 ──
        if not await test_08_conversations():
            all_pass = False

        # ── Step 9: 取消任务 ──
        if not await test_09_cancel_task():
            all_pass = False

        # ── Step 10: 取消 Webhook 订阅 ──
        if webhook_sub_id:
            if not await test_10_unsubscribe_webhook(webhook_sub_id):
                all_pass = False

    finally:
        stop_webhook_receiver()

    # ── 汇总 ──
    print(f"\n{'='*60}")
    passed = sum(1 for _, s, _ in _results if s == PASS)
    total = len(_results)
    print(f"  [RESULT] {passed}/{total} 通过  {'[PASS] 全部通过！' if passed == total else '[FAIL] 存在失败项'}")
    print(f"{'='*60}")

    return all_pass


async def run_discovery_only() -> bool:
    """仅执行发现相关测试。"""
    print("=" * 60)
    print("  [TEST] A2A Agent Card 发现测试")
    print(f"    目标服务: {BASE_URL}")
    print("=" * 60)
    all_pass = await test_01_discovery()
    await test_02_health()
    return all_pass


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    global DEFAULT_MESSAGE, BASE_URL

    parser = argparse.ArgumentParser(
        description="A2A 协议端到端测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/test_a2a_protocol.py                           # 完整测试
  python scripts/test_a2a_protocol.py --message "查运单 SHIP-001"  # 自定义消息
  python scripts/test_a2a_protocol.py --no-webhook              # 不带 webhook 测试
  python scripts/test_a2a_protocol.py --discovery-only          # 仅发现
        """,
    )
    parser.add_argument(
        "--message", type=str, default=DEFAULT_MESSAGE,
        help="测试消息内容",
    )
    parser.add_argument(
        "--no-webhook", action="store_true",
        help="不使用 webhook 回调（纯轮询模式）",
    )
    parser.add_argument(
        "--discovery-only", action="store_true",
        help="仅执行发现/健康检查测试，不提交任务",
    )
    parser.add_argument(
        "--base-url", type=str, default=BASE_URL,
        help="服务地址",
    )

    args = parser.parse_args()

    DEFAULT_MESSAGE = args.message
    BASE_URL = args.base_url

    try:
        import aiohttp  # noqa: F401
    except ImportError:
        print("[FAIL] 缺少 aiohttp，请安装: pip install aiohttp")
        sys.exit(1)

    ok = asyncio.run(run_all_tests(
        no_webhook=args.no_webhook,
        discovery_only=args.discovery_only,
    ))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
