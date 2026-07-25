"""系统级健康监控 Agent（FastAPI）。

职责分层：
- 被动巡检：周期/按需探测各组件健康，聚合为 /status 拓扑健康矩阵（04 篇 §3）。
- 主动分析：被告警 webhook 叫醒后做 RCA 根因分析（04 篇 §5/§6）。
  - POST /ingest/alert   ← Alertmanager（指标+日志双通道同源）
  - POST /ingest/event   ← Langfuse 应急口（不经网关 /v1，不污染计量）
  - POST /rca            ← 手动触发 RCA（给定事件 payload + 当前拓扑）
  - GET  /rca/{id}       ← 取最近一次 RCA 结果（仅派生产物，无原始流）
- 探活旁路业务计量：探 /health 不计 token（04 §4）；用模型分析须走网关 /v1 带 monitoring 标识。

被探测：shop-agent、gateway、redis、postgres、milvus(standalone)、minio、langfuse 等。
（见 docs/oracle-cloud-deploy.md §5 三服务之一）
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .alerts import IngestEvent, parse_alertmanager, parse_langfuse, redact
from . import prom_client as prom
from . import skywalking_client as sky
from . import ws as ws_hub
from .logging_json import setup_logging, trace_binding_middleware
from .rca import RcaResult, analyze

setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("monitoring_agent")

app = FastAPI(title="Monitoring Agent", version="1.2.0")

# 允许 demo 前端跨域访问（前端跑在 http://localhost，API 在 :9091/:30091）。
# 固定白名单：本地 demo（80）、port-forward（9091）、LoadBalancer（30091）。
_CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:80",
    "http://localhost:9091",
    "http://localhost:30091",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(trace_binding_middleware)

SHOP_AGENT_URL = os.getenv("SHOP_AGENT_URL", "http://shop-agent:8000/health")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8001/health")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "local-redis-password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "local-postgres-password")

# 告警摄入端点鉴权（B-6）：告警 webhook 是写入口，必须带 Bearer token。
# 不配置 MONITORING_WEBHOOK_TOKEN 时仅允许本地回环（default-deny 在生产必须配置）。
_WEBHOOK_TOKEN = os.getenv("MONITORING_WEBHOOK_TOKEN", "").strip()

# HTTP 探针目标（方案 A：动态服务发现 + 静态补充）
#
# 动态来源：Prometheus /api/v1/targets（正在被抓取的服务 = 真实运行服务清单），
# 这样新增服务只要接入 Prometheus 抓取，monitoring-agent 自动纳入巡检，
# 无需改代码/重启。健康路径由 _HEALTH_PATHS 映射（job -> health path）。
#
# 静态补充（k8s 外 / 未接入 Prometheus 的服务）：STATIC_TARGETS env，
# 格式 "name=url,name2=url2"，用于覆盖动态发现的 url（如本机 Redis/PG 走
# 独立探针、集群外服务无 /metrics 端点时手工登记）。静态优先级高于动态。
HTTP_TARGETS = {
    "shop-agent": SHOP_AGENT_URL,
    "gateway": GATEWAY_URL,
    "milvus": "http://milvus:9091/healthz",
    "minio": "http://minio:9000/minio/health/live",
    "langfuse": "http://langfuse-web:3000/health",
    "prometheus": "http://prometheus:9090/-/healthy",
}

# job -> 健康检查路径（动态发现时拼接；Prometheus metrics_path 不等于 health path）
_HEALTH_PATHS = {
    "shop-agent": "/health",
    "gateway": "/health",
    "monitoring-agent": "/health",
    "milvus": "/healthz",
    "minio": "/minio/health/live",
    "langfuse": "/health",
    "prometheus": "/-/healthy",
}

_DISCOVER_ENABLED = os.getenv("DISCOVER_ENABLED", "1") == "1"
_DISCOVER_CACHE_TTL_S = float(os.getenv("DISCOVER_CACHE_TTL_S", "60"))


def _parse_static_targets(raw: str) -> dict[str, str]:
    """解析 STATIC_TARGETS：逗号分隔的 name=url，空条目跳过。"""
    out: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            logger.warning("STATIC_TARGETS 条目缺少 '='：%r，已跳过", item)
            continue
        name, _, url = item.partition("=")
        name, url = name.strip(), url.strip()
        if name and url:
            out[name] = url
    return out


STATIC_TARGETS = _parse_static_targets(os.getenv("STATIC_TARGETS", ""))


def _discover_from_prometheus() -> dict[str, str]:
    """从 Prometheus activeTargets 动态发现服务清单：{job: health_url}。

    只取 active 且未被禁用的 target；instance 形如 host:port（或带 scheme/path），
    取 host:port 后拼接 _HEALTH_PATHS[job] 的健康路径。Prometheus 不可达时
    返回空 dict（调用方回退到静态清单），不抛异常——发现是增强而非依赖。
    """
    discovered: dict[str, str] = {}
    try:
        targets = prom.scrape_targets()
    except prom.PrometheusUnavailable:
        logger.warning("Prometheus 不可达，服务发现降级为静态清单")
        return discovered

    for t in targets:
        if t.get("health") != "up":
            continue
        labels = t.get("labels", {})
        job = labels.get("job")
        instance = labels.get("instance")
        if not job or not instance:
            continue
        # instance 可能是 host:port / scheme://host:port / host:port/path
        host_port = instance.split("://")[-1].split("/")[0]
        if ":" not in host_port:
            continue
        path = _HEALTH_PATHS.get(job, "/health")
        discovered[job] = f"http://{host_port}{path}"
    return discovered


_discover_cache: dict[str, str] = {}
_discover_cached_at = 0.0


def build_targets() -> dict[str, str]:
    """合并动态发现 + 静态补充，返回最终探测目标清单。

    优先级：STATIC_TARGETS（显式配置，k8s 外服务）> 动态发现 > HTTP_TARGETS 兜底。
    带 TTL 缓存，避免每次 /status 都打 Prometheus（巡检是高频路径）。
    """
    global _discover_cache, _discover_cached_at
    now = time.monotonic()
    if _DISCOVER_ENABLED and (now - _discover_cached_at > _DISCOVER_CACHE_TTL_S):
        _discover_cache = _discover_from_prometheus()
        _discover_cached_at = now

    targets = dict(HTTP_TARGETS)
    targets.update(_discover_cache)   # 动态发现的 url 覆盖默认（host 更贴近实际）
    targets.update(STATIC_TARGETS)    # 显式静态配置最高优先（k8s 外/独立探针）
    return targets


# ── 方案 C：SkyWalking 真实调用依赖边（动态）+ 静态依赖兜底 ────────────────
# 拓扑依赖矩阵用于 RCA 级联归因：A 依赖 B（A 调 B），B 故障则 A 是受影响者，
# 由依赖边方向推断根因传播（替代纯拓扑健康矩阵的「平铺」假设）。
# 动态来源：OAP getGlobalTopology（只覆盖接 agent 上报的服务）；
# k8s 外 / 未接 agent 的服务用 STATIC_DEPENDENCIES 静态声明依赖边。

# 静态依赖：逗号分隔 "依赖方=被依赖方"（即 依赖方 调用 被依赖方）。
# 例：STATIC_DEPENDENCIES="shop-agent=gateway,gateway=redis" 表示
#     shop-agent 依赖 gateway，gateway 依赖 redis。
def _parse_dependencies(raw: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        dep, _, target = item.partition("=")
        dep, target = dep.strip(), target.strip()
        if dep and target and dep != target:
            out.append({"source": dep, "target": target})
    return out


STATIC_DEPENDENCIES = _parse_dependencies(os.getenv("STATIC_DEPENDENCIES", ""))

_SKYWALKING_ENABLED = os.getenv("SKYWALKING_ENABLED", "1") == "1"
_SKY_CACHE_TTL_S = float(os.getenv("SKYWALKING_CACHE_TTL_S", "120"))
_sky_deps_cache: list[dict[str, str]] = []
_sky_deps_cached_at = 0.0


def _discover_dependencies_from_skywalking() -> list[dict[str, str]]:
    """从 OAP 拉真实调用依赖边（方案 C 核心）。OAP 不可达返回空列表，降级静态。"""
    try:
        return sky.dependency_edges()
    except sky.SkyWalkingUnavailable:
        logger.warning("SkyWalking 不可达，依赖拓扑降级为静态清单")
        return []


def build_dependencies() -> list[dict[str, str]]:
    """合并动态（OAP 真实调用边）+ 静态（STATIC_DEPENDENCIES，k8s 外兜底）。

    返回 [{source, target}]，source 依赖 target（source 调用 target）。
    带 TTL 缓存；SKYWALKING_ENABLED=0 时只走静态。
    """
    global _sky_deps_cache, _sky_deps_cached_at
    now = time.monotonic()
    if _SKYWALKING_ENABLED and (now - _sky_deps_cached_at > _SKY_CACHE_TTL_S):
        _sky_deps_cache = _discover_dependencies_from_skywalking()
        _sky_deps_cached_at = now

    merged = list(_sky_deps_cache)
    seen = {(e["source"], e["target"]) for e in merged}
    for e in STATIC_DEPENDENCIES:
        key = (e["source"], e["target"])
        if key not in seen:
            merged.append(e)
            seen.add(key)
    return merged

_status: dict[str, bool] = {}
# 最近一次 RCA 结果缓存（派生产物，非原始流；单实例足够，不持久化）
_last_rca: RcaResult | None = None

# ── RCA 可观测指标（Prometheus 抓取，事件驱动型保留最近值） ──
_rca_total: dict[str, int] = {}          # {severity: 累计次数}，counter 语义
_rca_last: dict[str, str | float] | None = None  # 最近一次 RCA 派生产物（info gauge）


def _escape_label(v: str) -> str:
    """Prometheus label 值转义（\\ 与双引号）。"""
    return v.replace("\\", "\\\\").replace('"', '\\"')


async def _probe(name: str, url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(url)
        return r.status_code < 500
    except Exception:
        return False


async def _probe_redis() -> bool:
    try:
        import redis  # 可选依赖

        r = redis.Redis(
            host=REDIS_HOST, port=6379, password=REDIS_PASSWORD, socket_timeout=3
        )
        return bool(r.ping())
    except Exception:
        return False


async def _probe_postgres() -> bool:
    try:
        import psycopg2  # 可选依赖

        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=5432,
            dbname="postgres",
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            connect_timeout=3,
        )
        conn.close()
        return True
    except Exception:
        return False


async def _build_topology() -> dict:
    """聚合拓扑健康矩阵（04 篇 §3）：{component: {status, latency_ms, last_check, detail}}。"""
    results: dict = {}
    for name, url in build_targets().items():
        t0 = time.perf_counter()
        up = await _probe(name, url)
        results[name] = {
            "status": up,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "last_check": int(time.time()),
            "detail": "ok" if up else "unreachable",
        }
    results["redis"] = {
        "status": await _probe_redis(),
        "latency_ms": None,
        "last_check": int(time.time()),
        "detail": "ok" if _status.get("redis", True) else "unreachable",
    }
    results["postgres"] = {
        "status": await _probe_postgres(),
        "latency_ms": None,
        "last_check": int(time.time()),
        "detail": "ok" if _status.get("postgres", True) else "unreachable",
    }
    return results


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── WebSocket 实时告警推送（demo 经 Ingress /monitor/ws → /ws）─────────────
# 连接池由 ws_hub 维护；RCA 完成后主动 broadcast 告警 + 处置建议。
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    await ws_hub.register(websocket)
    try:
        # 连接即推一次当前拓扑快照（demo 实时显示健康矩阵）
        await ws_hub.send_topology(websocket, await _build_topology())
        # 保持连接：接收前端心跳/控制消息，断开即退出
        while True:
            try:
                await websocket.receive_text()
            except Exception:
                break
    finally:
        await ws_hub.unregister(websocket)


# 演示页面静态托管（兼容 /demo 和 /demo/，避免 307 重定向）
from starlette.staticfiles import StaticFiles as StarletteStaticFiles

_demo_dir = Path("monitoring_agent/static/demo")
_demo_index = (_demo_dir / "index.html").read_text(encoding="utf-8")


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
@app.get("/demo/", response_class=HTMLResponse, include_in_schema=False)
async def demo_root():
    return _demo_index


# 静态资源（js/css 等）挂载在 /demo/static/ 下
app.mount("/demo/static", StarletteStaticFiles(directory=_demo_dir), name="demo-static")


@app.get("/status")
async def status():
    results = await _build_topology()
    _status.update({k: v["status"] for k, v in results.items()})
    return {
        "overall": all(v["status"] for v in results.values()),
        "components": results,
        "dependencies": build_dependencies(),
        "ts": int(time.time()),
    }


@app.get("/metrics")
async def metrics():
    lines = ["# TYPE component_up gauge"]
    for name, up in _status.items():
        lines.append(f'component_up{{name="{name}"}} {1 if up else 0}')

    # RCA 事件计数（counter 语义：按 severity 累计）
    lines.append("# TYPE rca_total counter")
    for sev, n in _rca_total.items():
        lines.append(f'rca_total{{severity="{sev}"}} {n}')

    # 最近一次 RCA（info gauge：保留最近值供 Grafana 面板展示；事件型指标不滚动）
    if _rca_last:
        lines.append("# TYPE rca_last_info gauge")
        rc = _escape_label(str(_rca_last.get("root_cause", "")))
        affected = _escape_label(str(_rca_last.get("affected", "")))
        recs = _escape_label(str(_rca_last.get("recommendations", "")))
        sev = _escape_label(str(_rca_last.get("severity", "")))
        ts = float(_rca_last.get("ts", 0) or 0)
        lines.append(
            f'rca_last_info{{severity="{sev}",'
            f'used_llm="{_rca_last.get("used_llm", "")}",'
            f'root_cause="{rc}",affected="{affected}",recommendations="{recs}"}} 1'
        )
        lines.append(f"rca_last_timestamp_seconds{{severity=\"{sev}\"}} {ts}")
    return Response("\n".join(lines) + "\n", media_type="text/plain")


# ── 告警摄入与 RCA（被叫醒才工作）────────────────────────────────────────


def _json_safe(obj):
    """递归把 NaN/Inf（不可 JSON 编码）替换为 None，避免 RCA 接口 500。

    Prometheus/Loki 等数据源可能返回非法浮点（如 ``NaN`` 字符串被 float() 解析），
    直接 json.dumps 会抛 ``ValueError: Out of range float values``。在响应序列化前
    兜底清洗，确保 RCA 结果始终可被 JSON 编码（B 防御层）。
    """
    if isinstance(obj, float):
        return None if (obj != obj or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _auth_ok(request: Request) -> bool:
    """校验告警摄入端点 Bearer token（B-6）。

    未配置 MONITORING_WEBHOOK_TOKEN 时拒绝一切外部调用（default-deny）；
    配置后须 ``Authorization: Bearer <token>``。本地回环（127.0.0.1）免鉴权，
    便于 compose 内健康联动。Langfuse 应急口与手动触发同样适用。
    """
    if not _WEBHOOK_TOKEN:
        client = (request.client.host if request.client else "") or ""
        return client in ("127.0.0.1", "::1")
    auth = request.headers.get("Authorization", "")
    return auth == f"Bearer {_WEBHOOK_TOKEN}"


def _deny() -> JSONResponse:
    return JSONResponse(status_code=401, content={"error": "unauthorized"})


@app.post("/ingest/alert")
async def ingest_alert(request: Request):
    """Alertmanager webhook 入口（指标+日志双通道同源）。

    入站即脱敏，PII 不进分析/落库路径（06/07 跨平面）。
    """
    if not _auth_ok(request):
        return _deny()
    try:
        payload = await request.json()
    except Exception as exc:
        raw = await request.body()
        logging.warning("ingest/alert payload 非 JSON: %s | raw=%r", exc, raw[:500])
        return JSONResponse(status_code=400, content={"error": "invalid json body", "detail": str(exc)})
    event = parse_alertmanager(payload)
    # 现场聚合真实拓扑（含 redis 等必备依赖的实时探测），让 RCA 能据真实
    # 健康矩阵收敛根因（如 redis 不可达 → 产出结构化修复动作）。
    # 若告警 payload 自带 topology 快照则优先采用（兼容外部传入）。
    topology = payload.get("topology") or await _build_topology()
    return await _run_rca(event, topology)


@app.post("/ingest/event")
async def ingest_event(request: Request):
    """Langfuse 应急口（不经 Alertmanager、不经网关 /v1，不污染业务计量）。"""
    if not _auth_ok(request):
        return _deny()
    try:
        payload = await request.json()
    except Exception as exc:
        raw = await request.body()
        logging.warning("ingest/event payload 非 JSON: %s | raw=%r", exc, raw[:500])
        return JSONResponse(status_code=400, content={"error": "invalid json body", "detail": str(exc)})
    event = parse_langfuse(payload)
    return await _run_rca(event)


# ── Demo 自愈闭环：故障注入 / 确认恢复（纯人工，不自动恢复）─────────────────
# /demo/inject-fault：显式把 redis 副本置 0，制造真实故障（chat 经 readiness
#   摘除流量 + 缓存/会话降级）。不自动恢复，须下方 /demo/remediate 确认。
# /demo/remediate：用户在 demo 点「确认执行修复」后调用，把 redis 副本置 1 恢复。
# 两者共享 _auth_ok（与 /ingest/* 同源 token），且目标受 k8s_client 白名单约束。
from . import k8s_client as k8s

_DEMO_TARGET = os.getenv("DEMO_FAULT_TARGET", "redis")


@app.post("/demo/inject-fault")
async def demo_inject_fault(request: Request):
    """Demo：注入真实故障（redis scale -> 0）。需 Bearer token 鉴权。"""
    if not _auth_ok(request):
        return _deny()
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    target = (payload.get("target") or _DEMO_TARGET) if isinstance(payload, dict) else _DEMO_TARGET
    try:
        result = k8s.inject_fault(target)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": "bad_target", "detail": str(exc)})
    except k8s.K8sUnavailable as exc:
        return JSONResponse(status_code=503, content={"error": "k8s_unavailable", "detail": str(exc)})
    logger.info("Demo 故障注入：target=%s", target)
    # 注入后主动推一条 topology 快照，前端可即时看到 redis 不可用
    try:
        await ws_hub.broadcast({"type": "fault_injected", "ts": int(time.time()),
                                "target": target, "replicas": 0})
    except Exception:
        pass
    return {"injected": True, **result}


@app.post("/demo/remediate")
async def demo_remediate(request: Request):
    """Demo：确认执行修复（redis scale -> 1）。需 Bearer token 鉴权。"""
    if not _auth_ok(request):
        return _deny()
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    target = (payload.get("target") or _DEMO_TARGET) if isinstance(payload, dict) else _DEMO_TARGET
    try:
        result = k8s.remediate(target)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": "bad_target", "detail": str(exc)})
    except k8s.K8sUnavailable as exc:
        return JSONResponse(status_code=503, content={"error": "k8s_unavailable", "detail": str(exc)})
    logger.info("Demo 修复执行：target=%s", target)
    try:
        await ws_hub.broadcast({"type": "resolved", "ts": int(time.time()),
                                "target": target, "replicas": 1})
    except Exception:
        pass
    return {"remediated": True, **result}


@app.post("/rca")
async def run_rca_manual(request: Request):
    """手动触发 RCA：给定事件 payload（含 source 标识）+ 可选拓扑快照。"""
    if not _auth_ok(request):
        return _deny()
    try:
        payload = await request.json()
    except Exception as exc:  # JSON 解析失败（外部代理可能改写/截断 body）
        raw = await request.body()
        logging.warning("RCA payload 非 JSON: %s | raw=%r", exc, raw[:500])
        return JSONResponse(status_code=400, content={"error": "invalid json body", "detail": str(exc)})
    source = (payload.get("source") or "alertmanager").lower()
    event: IngestEvent = (
        parse_langfuse(payload) if source == "langfuse" else parse_alertmanager(payload)
    )
    topology = payload.get("topology") or {}
    # 未给拓扑则现场聚合一份
    if not topology:
        topology = await _build_topology()
    return await _run_rca(event, topology)


@app.get("/rca/last")
async def get_last_rca(request: Request):
    """取最近一次 RCA 结果（派生产物）。"""
    if not _auth_ok(request):
        return _deny()
    if _last_rca is None:
        return {"found": False}
    return _json_safe({"found": True, **_last_rca.to_dict()})


async def _run_rca(event: IngestEvent, topology: dict | None = None) -> dict:
    """异步执行 RCA，分析层内的同步 IO（Prometheus/Loki/LLM 查询）经 to_thread 卸载，
    避免阻塞事件循环（B-1）。
    """
    # topology 为空时构造最小矩阵，保证分析不崩
    if not topology:
        topology = {name: {"status": True} for name in build_targets()}
    # 注入依赖拓扑（方案 C），供级联归因：A 依赖 B，B 故障则 A 是受影响者
    topology.setdefault("_dependencies", build_dependencies())
    result = await asyncio.to_thread(analyze, event, topology)
    global _last_rca
    _last_rca = result
    _record_rca(result)
    logger.info("RCA 完成 sev=%s cause=%s used_llm=%s",
                result.severity, result.root_cause, result.used_llm)
    # 实时推送给所有已连接 demo（异步 fire-and-forget，不影响本次响应）
    try:
        await ws_hub.broadcast(ws_hub.build_alert_message(result))
    except Exception as exc:
        logger.warning("WebSocket 广播失败（不影响 RCA 响应）: %s", exc)
    return _json_safe({
        "accepted": True,
        "severity": result.severity,
        **result.to_dict(),
    })


def _record_rca(result: RcaResult) -> None:
    """把 RCA 结果登记为 Prometheus 指标（事件计数 + 最近值 info gauge）。"""
    global _rca_last
    _rca_total[result.severity] = _rca_total.get(result.severity, 0) + 1
    _rca_last = {
        "severity": result.severity,
        "used_llm": str(result.used_llm).lower(),
        "root_cause": redact(result.root_cause),
        "affected": ",".join(result.affected) if result.affected else "none",
        "recommendations": " | ".join(redact(r) for r in result.recommendations),
        "ts": time.time(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9091)
