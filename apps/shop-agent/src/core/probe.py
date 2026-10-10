"""独立探针服务（与业务端口/事件循环/线程池解耦）。

- liveness:  GET /healthz  —— 仅判断进程是否存活，不做任何业务/依赖检查，永不因高并发失败
- readiness: GET /readyz   —— 检查依赖（Redis）连通性；高并发资源饱和导致依赖超时即返回 503，
                            让 k8s Service 摘除该 Pod（只摘流量，不杀进程）

独立监听 PROBE_PORT（默认 8001），与业务端口 8000 隔离。通过 SO_REUSEPORT 允许
多个 uvicorn worker 各起探针线程共同监听同一端口，任一 worker 存活即可响应。
"""

from __future__ import annotations

import json
import socket
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.shared.logger import APILogger

logger = APILogger("probe")

# readiness 依赖检查复用全局连接，避免每次探测重建连接
_redis_client = None
_redis_lock = threading.Lock()


def _redis_target() -> tuple[str, int, str | None] | None:
    """解析 Redis 连接目标，返回 (host, port, password)。

    依赖分级（04 §8 / 大厂实践）：
    - Redis 在 shop-agent 中作为**可选缓存/记忆**依赖（对话历史、问题去重、
      向量检索兜底）。它不可用时业务侧已做优雅降级（记忆静默丢失、chat 继续），
      故探针默认不把它当作阻塞就绪的必备依赖。
    - 如需把 Redis 提升为必备依赖（fail-closed），显式设置环境变量
      ``REDIS_CRITICAL=true`` 即可；此时 Redis 不可达会让 readiness 返回 503，
      k8s 摘除 Pod（用于强一致会话等场景）。
    """
    import os

    from src.modules.chat.config import chat_config

    cfg = chat_config
    host = getattr(cfg, "redis_host", None) or "localhost"
    port = int(getattr(cfg, "redis_port", None) or 6379)
    password = getattr(cfg, "redis_password", None) or None
    env_host = os.environ.get("REDIS_HOST")
    if env_host:
        host = env_host
    # 密码优先取环境变量 REDIS_AUTH（聊天/限流侧用 secret 注入，chat_config 不为它赋值）
    env_auth = os.environ.get("REDIS_AUTH")
    if env_auth:
        password = env_auth
    # 未配置 Redis（默认 localhost 且未显式覆盖）视为完全不用 Redis，不检查。
    if host in ("localhost", "127.0.0.1") and not env_host:
        return None
    # 显式配置了 Redis：默认视为可选依赖（fail-soft），仅当 REDIS_CRITICAL=true
    # 时才当作必备依赖参与 readiness 判定。
    if os.environ.get("REDIS_CRITICAL", "").lower() != "true":
        return None
    return host, port, password


def _get_redis():
    """懒加载 Redis 客户端（readiness 用，超时 1s）。"""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    target = _redis_target()
    if target is None:
        return None
    host, port, password = target
    try:
        import redis as redis_lib

        _redis_client = redis_lib.Redis(
            host=host,
            port=port,
            password=password,
            db=0,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
    except Exception:
        return None
    return _redis_client


def _check_ready() -> tuple[bool, str]:
    """readiness：进程存活 + 必备外部依赖（显式配置的 Redis）可用。

    未显式配置 Redis（默认 localhost）时视为可选依赖，不阻塞就绪——
    与业务一致（Redis 不可用时应用走内存降级，仍能接新流量）。

    GrowthBook 为可选实验后端（GROWTHBOOK_ENABLED 默认 false），其 degraded 状态
    不阻塞就绪（设计红线：GB 不可用时进程仍能起来，仅以安全默认/缓存快照上岗）。
    """
    target = _redis_target()
    if target is None:
        return True, "no mandatory external dependency"
    try:
        client = _get_redis()
        if client is None:
            return False, "redis client unavailable"
        return bool(client.ping()), "redis ok"
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:120]


def _gb_health_snapshot() -> dict | None:
    """GrowthBook 健康快照（非阻塞；GB 禁用/未安装时返回 None）。"""
    try:
        from src.core.growthbook_client import GrowthBookClient

        return GrowthBookClient.get_instance().health()
    except Exception:  # noqa: BLE001
        return None


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 抑制探针请求日志刷屏
        return

    def _reply(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except Exception:  # noqa: BLE001
            pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/healthz":
            # liveness：仅判断进程存活，永不因业务/依赖失败告警
            self._reply(200, {"status": "ok", "reason": "process alive"})
            return
        if path == "/readyz":
            ok, detail = _check_ready()
            payload = {"status": "ok" if ok else "unhealthy", "detail": detail}
            # GrowthBook 健康（非阻塞：degraded 不影响就绪，仅作观测）
            gb = _gb_health_snapshot()
            if gb is not None:
                payload["growthbook"] = gb
            self._reply(200 if ok else 503, payload)
            return
        self._reply(404, {"status": "not found"})


class _ReusePortHTTPServer(ThreadingHTTPServer):
    """SO_REUSEPORT 允许多 worker 探针线程共同监听同一端口。"""

    allow_reuse_address = True

    def server_bind(self):
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass  # 平台不支持 SO_REUSEPORT 时退回普通绑定
        socketserver.TCPServer.server_bind(self)


class ProbeServer:
    """以独立守护线程维护的探针 HTTP 服务。"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        self.host = host
        self.port = port
        self._server = None
        self._thread = None

    def start(self) -> None:
        if self._server is not None:
            return
        try:
            self._server = _ReusePortHTTPServer((self.host, self.port), _Handler)
        except OSError as e:
            logger.warning(f"探针服务启动失败（可能已被其它 worker 占用）: {e}")
            self._server = None
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="probe-server",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"独立探针服务已启动 http://{self.host}:{self.port} (/healthz /readyz)")

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:  # noqa: BLE001
                pass
            self._server = None
            self._thread = None


_probe_server = None


def start_probe_server(host: str = "0.0.0.0", port: int = 8001):
    """启动探针服务（幂等），返回实例。"""
    global _probe_server
    if _probe_server is None:
        _probe_server = ProbeServer(host, port)
        _probe_server.start()
    return _probe_server


def stop_probe_server() -> None:
    global _probe_server
    if _probe_server is not None:
        try:
            _probe_server.stop()
        except Exception:  # noqa: BLE001
            pass
        _probe_server = None
