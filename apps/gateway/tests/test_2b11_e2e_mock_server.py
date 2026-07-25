"""批次 2b-11：Gateway 真实 HTTP 端到端集成测试（mock 上游，无真实 API）。

与 01~10 单元级验收（stub AsyncClient）不同，本文件起一个**真实 HTTP mock 上游**
（标准库 http.server 后台线程），并把网关路由的后端地址指向它，走真实
httpx.AsyncClient 全链路转发：入向注入闸 → 治理面 → 路由 → 出向 fail 开关 →
转发 → 语义缓存 → PII 脱敏 → 计量。验证网关在真实网络栈下的端到端行为。

覆盖：
  - 基础 chat 透传（真实转发 + 上游响应原样返回）
  - PII 脱敏在出向响应生效（上游返回含手机号，网关脱敏后返回）
  - 语义缓存集成（相同请求第二次命中缓存，不二次打到上游）
  - 流式 SSE 透传（mock 以 stream 模式返回，网关按 content-type 透传并脱敏）
  - 上游 5xx 故障转移（主/备均 500 → 统一网关 503 fail-closed）
"""
from __future__ import annotations

import importlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # apps/gateway

CHAT_OK = {
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "已为您处理，客服电话 13812348000。"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13},
}
# 不含 PII 的响应（用于缓存命中用例：含 PII 的 answer 会被缓存 PII 门禁拦截，无法写入）
CHAT_OK_NOPII = {
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "已为您处理完成。"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13},
}
STREAM_TEXT = "已为您处理，电话 13900001234。"


class _MockHandler(BaseHTTPRequestHandler):
    stats: dict = {"calls": 0, "last_request_body": None}
    mode: str = "ok"

    def log_message(self, *args):  # 静默
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            _MockHandler.stats["last_request_body"] = json.loads(raw)
        except Exception:
            pass
        _MockHandler.stats["calls"] += 1

        # 网关按 OpenAI 兼容约定拼接：base_url + "/chat/completions"（base_url 不带 /v1）
        if self.path.rstrip("/") != "/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        if _MockHandler.mode == "fail":
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "boom"}).encode())
            return

        if _MockHandler.mode == "ok_nopii":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(CHAT_OK_NOPII).encode())
            return

        if _MockHandler.mode == "stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            # 单块内含完整手机号（流式逐块脱敏不跨块，故手机号须在单块内才可被检测）
            frame = "data: " + json.dumps({"choices": [{"delta": {"content": STREAM_TEXT}}]}) + "\n\n"
            self.wfile.write(frame.encode())
            self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(CHAT_OK).encode())


@pytest.fixture
def mock_upstream(request):
    """起一个真实 HTTP mock 上游，返回 (base_url, stats)。mode 由 request.param 决定。"""
    mode = getattr(request, "param", "ok")
    _MockHandler.stats = {"calls": 0, "last_request_body": None}
    _MockHandler.mode = mode

    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    base_url = f"http://127.0.0.1:{port}"

    yield base_url, _MockHandler.stats

    server.shutdown()
    server.server_close()


def _point_gateway_to(monkeypatch, base_url):
    """把 gpt-4o 的两个 deployment（azure 主 + openai 备）都指向本地 mock 上游，并重建 Router。

    生产级：通过临时 litellm_config_path YAML 驱动 Router（与运行时 YAML 同源），
    不依赖 legacy 字段，确保 mock server（裸 OpenAI 兼容）被 Router 真实调用。
    """
    import gateway.config as config
    import gateway.litellm_router as lr

    # E2E 关注转发/脱敏/缓存/故障转移，不受限流干扰：放宽令牌桶与 refill 速率。
    monkeypatch.setenv("RATE_LIMIT_BURST", "100")
    monkeypatch.setenv("RATE_LIMIT_TENANT_RPS", "100")
    monkeypatch.setenv("RATE_LIMIT_GLOBAL_RPS", "100")

    # 写临时 YAML：gpt-4o 两个 deployment 均指向 mock server（openai/ 前缀，裸兼容）
    yaml_text = f"""
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: gpt-4o
      custom_llm_provider: openai
      api_base: {base_url}
      api_key: sk-noop
  - model_name: gpt-4o
    litellm_params:
      model: gpt-4o
      custom_llm_provider: openai
      api_base: {base_url}
      api_key: sk-noop
router_settings:
  routing_strategy: simple-shuffle
  num_retries: 1
  timeout: 10
"""
    import tempfile

    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    tmp.write(yaml_text)
    tmp.close()
    monkeypatch.setenv("LITELLM_CONFIG_PATH", tmp.name)

    config.settings = config.Settings()  # 重建单例，读取新 env
    importlib.reload(config)
    # 关键：多个模块用 `from .config import settings` 绑定了旧对象引用，需 reload 使
    # 令牌桶/循环防护/缓存/治理钩子按新 settings 重建；reload 同时清空模块级单例状态。
    importlib.reload(importlib.import_module("gateway.cache.store"))
    importlib.reload(importlib.import_module("gateway.cache.policy"))
    importlib.reload(importlib.import_module("gateway.hooks.governance"))
    importlib.reload(importlib.import_module("gateway.limiter"))
    importlib.reload(importlib.import_module("gateway.loopguard"))
    importlib.reload(importlib.import_module("gateway.router"))
    # litellm_router 也必须 reload，使其 `from .config import settings` 绑定到新 settings
    # （含 LITELLM_CONFIG_PATH），否则 build() 仍走 legacy 字段，指向不存在的 vllm 地址。
    importlib.reload(lr)
    import gateway.litellm_router as lr  # 重新绑定 reload 后的模块
    # 重建 LiteLLM Router（读新 YAML），使 mock server 地址生效
    lr.llm_router.build()
    # 进程内语义缓存是模块级单例，跨测试持久会污染；每次重建后清空，保证测试隔离。
    import gateway.cache.store as cache_store_mod

    cache_store_mod.cache.clear()
    # reload main 使 app 重新 include_router 注册到新 proxy（绑定新 llm_router 单例），
    # 否则 main.app 仍持有旧 proxy 函数引用（旧单例，指向不存在地址），流式/路由判定失效。
    importlib.reload(importlib.import_module("gateway.main"))
    return importlib.reload(importlib.import_module("gateway.controllers.proxy"))


def _client():
    from fastapi.testclient import TestClient

    import gateway.main as main

    return TestClient(main.app)


@pytest.mark.parametrize("mock_upstream", ["ok"], indirect=True)
def test_e2e_basic_chat_passthrough(mock_upstream, monkeypatch):
    """真实转发到 mock 上游，网关返回上游 chat completion（请求真实打到上游）。

    注：网关默认开启出向 PII 脱敏，故响应中上游原文手机号会被掩码——透传语义只
    验证「请求真实到达上游一次 + 上游业务内容被原样返回」，脱敏由
    test_e2e_pii_redaction_on_egress 单独覆盖。
    """
    base_url, stats = mock_upstream
    _point_gateway_to(monkeypatch, base_url)
    client = _client()
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, r.text
    assert stats["calls"] == 1, "应真实打到上游一次"
    # 上游业务文案原样透传（脱敏仅掩码 PII 字段，不影响普通文本）
    assert "已为您处理" in r.json()["choices"][0]["message"]["content"]


@pytest.mark.parametrize("mock_upstream", ["ok"], indirect=True)
def test_e2e_pii_redaction_on_egress(mock_upstream, monkeypatch):
    """出向 PII 脱敏在真实链路生效：上游返回手机号，网关脱敏后返回。"""
    base_url, _ = mock_upstream
    _point_gateway_to(monkeypatch, base_url)
    client = _client()
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, r.text
    content = r.json()["choices"][0]["message"]["content"]
    assert "13812348000" not in content, "出向 PII 应被脱敏"
    assert "*" in content, "脱敏结果应含掩码"


@pytest.mark.parametrize("mock_upstream", ["ok_nopii"], indirect=True)
def test_e2e_semantic_cache_hit(mock_upstream, monkeypatch):
    """相同请求第二次命中语义缓存，不再打到上游（上游只被调用一次）。

    用不含 PII 的响应（ok_nopii）：含 PII 的 answer 会被缓存 PII 门禁硬拦截、无法写入，
    故缓存命中用例必须用可写入的非敏感回答，才走真实「查找未命中→进模型→egress 写入→
    二次查找命中短路」链路。
    """
    base_url, stats = mock_upstream
    _point_gateway_to(monkeypatch, base_url)
    client = _client()
    payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "cache me"}]}
    r1 = client.post("/v1/chat/completions", json=payload)
    r2 = client.post("/v1/chat/completions", json=payload)
    assert r1.status_code == 200 and r2.status_code == 200
    assert stats["calls"] == 1, f"缓存命中时应只打上游一次，实际 {stats['calls']}"


@pytest.mark.parametrize("mock_upstream", ["stream"], indirect=True)
def test_e2e_stream_passthrough(mock_upstream, monkeypatch):
    """流式 SSE 真实透传，且出向 PII 在流中脱敏。"""
    base_url, _ = mock_upstream
    _point_gateway_to(monkeypatch, base_url)
    client = _client()
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert r.status_code == 200, r.text
    body = r.text
    assert "data:" in body, "应透传 SSE 帧"
    assert "[DONE]" in body, "应透传结束标记"
    assert "13900001234" not in body, "流式出向 PII 应被脱敏"
    assert "*" in body, "流式脱敏结果应含掩码"


@pytest.mark.parametrize("mock_upstream", ["fail"], indirect=True)
def test_e2e_upstream_5xx_fail_closed(mock_upstream, monkeypatch):
    """主/备上游均 5xx → 网关统一返回 503 fail-closed，不伪装成功。"""
    base_url, _ = mock_upstream
    _point_gateway_to(monkeypatch, base_url)
    client = _client()
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503, f"全部上游 5xx 应网关 503，得到 {r.status_code}: {r.text}"
    assert r.json().get("error") == "all backends unavailable"
