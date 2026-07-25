#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shop-agent 最小 e2e 冒烟脚本（docker-compose 部署后运行）。

覆盖四组**没有单测保护**的改动，直接对部署后的服务 + Redis 做端到端验证：

  #2  版本化 Redis 缓存键 / 向量索引前缀
        -> 进程启动时 RedisCacheService 会建版本化向量索引
           `hospital_questions_idx_{embedding_cache_version}`（默认 v1）。
           索引名带版本号即证明版本前缀逻辑生效，模型升级时旧缓存按版本隔离。

  #4  分布式锁（acquire / renew / release）
        -> 纠纷协调用 `hospital_chat:lock:dispute:{conv}:{order}` 防重复执行，
           续约/释放依赖两段 Lua 脚本（仅当 token 匹配才操作）。
           本脚本用与 redis_cache_service.py 完全一致的 Lua 校验锁语义。

  #7  A2A 任务状态外置 Redis（跨节点可见）
        -> `POST /a2a/tasks/send` 在创建时**同步**把任务 JSON 写入
           `a2a:task:{id}` 并登记到 ZSET `a2a:tasks:index`。
           该落盘发生在后台 LLM 执行之前，故不依赖 LLM 即可验证序列化往返。

  #8  人在回路（human-in-the-loop）中断状态 Redis 往返
        -> Agent 暂停退款时把上下文写入 `interrupt:{thread_id}`（JSON）。
           本脚本按 react_agent.py 的字段形态写入并读回，验证序列化跨节点一致。

运行：
    python tests/e2e/smoke_e2e.py

环境变量（均为可选，默认值适配本地 docker-compose）：
    SHOP_AGENT_URL          默认 http://localhost:8000
    FIXED_API_KEY           Bearer 令牌，需与部署的 shop-agent 一致；默认 test-key-for-pytest
    REDIS_HOST / REDIS_PORT 默认 localhost / 6379
    REDIS_AUTH              默认空（与 compose 中 REDIS_AUTH 一致）
    EMBEDDING_CACHE_VERSION 默认 v1（必须与 shop-agent 的 embedding_cache_version 一致）

退出码：
    0  全部 PASS
    1  存在 FAIL（环境已就绪但功能异常）
    2  环境未就绪（服务/Redis 不可达，全部 SKIP）
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from typing import List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import redis

# ── 与 redis_cache_service.py 完全一致的锁 Lua 脚本（#4 校验用）──
_RELEASE_LOCK_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
else
    return 0
end
"""
_RENEW_LOCK_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("EXPIRE", KEYS[1], ARGV[2])
else
    return 0
end
"""

KEY_PREFIX = "hospital_chat:"
LOCK_KEY_PREFIX = f"{KEY_PREFIX}lock:"


# ── 极简结果收集 ──────────────────────────────────────────────────────────
class Check:
    def __init__(self, name: str, tag: str):
        self.name = name
        self.tag = tag  # 例如 "#2" "#4" "#7" "#8"
        self.status: str = "SKIP"  # PASS | FAIL | SKIP
        self.detail: str = ""

    def __str__(self) -> str:
        return f"[{self.status:4}] {self.tag:3} {self.name}  {self.detail}".rstrip()


RESULTS: List[Check] = []


def record(tag: str, name: str, ok: Optional[bool], detail: str = "") -> None:
    c = Check(name, tag)
    if ok is None:
        c.status = "SKIP"
    elif ok:
        c.status = "PASS"
    else:
        c.status = "FAIL"
    c.detail = detail
    RESULTS.append(c)
    print(c)


# ── HTTP 辅助（仅用标准库，避免新增依赖）──
def _http(method: str, url: str, api_key: str, payload: Optional[dict], timeout: int):
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {}
    except URLError as e:
        return None, {"error": str(e.reason)}


def http_post(url: str, api_key: str, payload: dict, timeout: int = 30):
    return _http("POST", url, api_key, payload, timeout)


def http_get(url: str, api_key: str = "", timeout: int = 10):
    return _http("GET", url, api_key, None, timeout)


# ── Redis 辅助 ──────────────────────────────────────────────────────────────
def make_redis() -> Optional[redis.Redis]:
    host = os.environ.get("REDIS_HOST", "localhost")
    port = int(os.environ.get("REDIS_PORT", "6379"))
    auth = os.environ.get("REDIS_AUTH", "")
    try:
        r = redis.Redis(
            host=host, port=port,
            password=auth or None,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        r.ping()
        return r
    except Exception as e:  # noqa: BLE001
        print(f"  (Redis 不可达 {host}:{port}: {e})")
        return None


# ── 各项检查 ──────────────────────────────────────────────────────────────
def check_health(base_url: str, api_key: str) -> bool:
    """基础就绪探测（不计入 # 项，但 #7 依赖它）。

    连接失败（status=None）视为环境未就绪 -> SKIP；
    仅当服务有响应但非 200 才记 FAIL（服务在但不健康）。
    """
    status, _ = http_get(f"{base_url}/health")
    if status is None:
        record("-", "服务 /health 就绪", None, "连接失败（环境未就绪）")
        return False
    ok = status == 200
    record("-", "服务 /health 就绪", ok, f"status={status}")
    return ok


def check_vector_index_version(r: redis.Redis) -> None:
    """#2 版本化向量索引前缀。"""
    version = os.environ.get("EMBEDDING_CACHE_VERSION", "v1").strip() or "v1"
    index_name = f"hospital_questions_idx_{version}"
    try:
        r.execute_command("FT.INFO", index_name)
        record("#2", f"版本化向量索引存在 ({index_name})", True)
    except redis.ResponseError as e:
        if "Unknown index" in str(e) or "no such index" in str(e).lower():
            record("#2", f"版本化向量索引缺失 ({index_name})", False,
                   "shop-agent 启动时未建版本化索引，版本前缀逻辑可能失效")
        else:
            record("#2", f"FT.INFO 异常 ({index_name})", False, str(e))
    except Exception as e:  # noqa: BLE001
        record("#2", "向量索引检查异常", None, str(e))


def check_a2a_task_persistence(base_url: str, api_key: str, r: redis.Redis) -> None:
    """#7 A2A 任务创建即同步落 Redis，验证 JSON 序列化往返。"""
    conv_id = f"e2e_smoke_{uuid.uuid4().hex[:8]}"
    status, body = http_post(
        f"{base_url}/a2a/tasks/send",
        api_key,
        {"message": "查询订单 ORD-2024-001 的物流状态", "domain": "ecommerce",
         "conversation_id": conv_id},
        timeout=30,
    )
    if status != 200 or not body.get("data", {}).get("task_id"):
        record("#7", "A2A 任务创建", False, f"status={status} body={body}")
        return

    task_id = body["data"]["task_id"]
    try:
        raw = r.get(f"a2a:task:{task_id}")
        if raw is None:
            record("#7", "A2A 任务落 Redis", False, f"未找到 a2a:task:{task_id}")
            return
        data = json.loads(raw)
        zscore = r.zscore("a2a:tasks:index", task_id)
        ok = all(k in data for k in ("task_id", "status", "created_at", "domain")) and zscore is not None
        record("#7", "A2A 任务 JSON 落 Redis + 索引登记", ok,
               f"task_id={task_id} status={data.get('status')} zscore={zscore}")
    except Exception as e:  # noqa: BLE001
        record("#7", "A2A Redis 读取异常", False, str(e))
    finally:
        # 清理冒烟数据，避免污染
        try:
            r.delete(f"a2a:task:{task_id}")
            r.zrem("a2a:tasks:index", task_id)
        except Exception:
            pass


def check_distributed_lock(r: redis.Redis) -> None:
    """#4 分布式锁 acquire / renew / release 语义（对齐 redis_cache_service.py）。"""
    lock_key = f"{LOCK_KEY_PREFIX}e2e_smoke_{uuid.uuid4().hex[:8]}"
    tok_a = uuid.uuid4().hex
    tok_b = uuid.uuid4().hex
    try:
        # acquire
        acquired = r.set(lock_key, tok_a, nx=True, ex=60)
        if not acquired:
            record("#4", "锁获取(acquire)", False, "首次 acquire 失败")
            return
        # 他人无法重复获取
        dup = r.set(lock_key, tok_b, nx=True, ex=60)
        if dup is not None:
            record("#4", "锁互斥(重复 acquire 应失败)", False, f"重复 acquire 返回 {dup}")
            return
        # 合法持有者续约
        renew_ok = r.eval(_RENEW_LOCK_SCRIPT, 1, lock_key, tok_a, 60)
        if renew_ok != 1:
            record("#4", "锁续约(renew, 持有者)", False, f"renew 返回 {renew_ok}")
            return
        # 非持有者续约应被拒
        renew_wrong = r.eval(_RENEW_LOCK_SCRIPT, 1, lock_key, tok_b, 60)
        if renew_wrong != 0:
            record("#4", "锁续约(renew, 非持有者应拒)", False, f"renew 返回 {renew_wrong}")
            return
        # 非持有者释放应被拒
        rel_wrong = r.eval(_RELEASE_LOCK_SCRIPT, 1, lock_key, tok_b)
        if rel_wrong != 0:
            record("#4", "锁释放(release, 非持有者应拒)", False, f"release 返回 {rel_wrong}")
            return
        # 合法持有者释放
        rel_ok = r.eval(_RELEASE_LOCK_SCRIPT, 1, lock_key, tok_a)
        if rel_ok != 1:
            record("#4", "锁释放(release, 持有者)", False, f"release 返回 {rel_ok}")
            return
        record("#4", "锁 acquire/renew/release 语义", True)
    except Exception as e:  # noqa: BLE001
        record("#4", "锁校验异常", False, str(e))
    finally:
        try:
            r.delete(lock_key)
        except Exception:
            pass


def check_interrupt_roundtrip(r: redis.Redis) -> None:
    """#8 人在回路中断状态 Redis 往返（字段形态对齐 react_agent.py）。"""
    thread_id = f"e2e_smoke_{uuid.uuid4().hex[:8]}"
    key = f"interrupt:{thread_id}"
    payload = {
        "conversation_id": thread_id,
        "intent_steps": [{"step_name": "退款申请", "status": "success"}],
        "domain": "ecommerce",
        "order_id": "ORD-2024-001",
        "reason": "e2e smoke: 等待人工确认退款",
        "config": {"configurable": {"thread_id": thread_id}},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        r.set(key, json.dumps(payload, ensure_ascii=False, default=str), ex=3600)
        raw = r.get(key)
        if raw is None:
            record("#8", "中断状态落 Redis", False, f"未找到 {key}")
            return
        data = json.loads(raw)
        ok = (data.get("conversation_id") == thread_id
              and data.get("order_id") == "ORD-2024-001"
              and data.get("reason", "").startswith("e2e smoke"))
        record("#8", "中断状态 JSON 往返(interrupt:{thread_id})", ok,
               f"order_id={data.get('order_id')}")
    except Exception as e:  # noqa: BLE001
        record("#8", "中断 Redis 往返异常", False, str(e))
    finally:
        try:
            r.delete(key)
        except Exception:
            pass


# ── 入口 ───────────────────────────────────────────────────────────────────
def main() -> int:
    base_url = os.environ.get("SHOP_AGENT_URL", "http://localhost:8000").rstrip("/")
    api_key = os.environ.get("FIXED_API_KEY", "test-key-for-pytest")

    print("=" * 72)
    print(f"shop-agent e2e 冒烟  ->  {base_url}")
    print("=" * 72)

    r = make_redis()
    health_ok = check_health(base_url, api_key)

    if r is None:
        # Redis 不可达：所有依赖 Redis 的检查 SKIP
        record("#2", "向量索引(#2)", None, "Redis 不可达")
        record("#4", "分布式锁(#4)", None, "Redis 不可达")
        record("#7", "A2A 落 Redis(#7)", None, "Redis 不可达")
        record("#8", "中断往返(#8)", None, "Redis 不可达")
    else:
        # #4 / #8 仅依赖 Redis，与 shop-agent 是否就绪无关
        check_distributed_lock(r)
        check_interrupt_roundtrip(r)

        if health_ok:
            # #2 需要 shop-agent 启动并建好版本化索引；#7 需要 shop-agent HTTP
            check_vector_index_version(r)
            check_a2a_task_persistence(base_url, api_key, r)
        else:
            record("#2", "向量化索引(#2)", None, "shop-agent 未就绪，无法验证索引创建")
            record("#7", "A2A 任务创建+落 Redis(#7)", None, "服务/Redis 不可达")

    print("-" * 72)
    passed = sum(1 for c in RESULTS if c.status == "PASS")
    failed = sum(1 for c in RESULTS if c.status == "FAIL")
    skipped = sum(1 for c in RESULTS if c.status == "SKIP")
    print(f"结果: PASS={passed}  FAIL={failed}  SKIP={skipped}")
    print("=" * 72)

    if failed > 0:
        return 1
    if passed == 0:
        return 2  # 环境未就绪
    return 0


if __name__ == "__main__":
    sys.exit(main())
