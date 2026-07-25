"""Gateway 正式测试的共享固件。

把 apps/gateway 加入 sys.path，使 `import gateway.*` 可用；并提供公共
reload / TestClient 辅助，供各批次验收测试复用（迁移自 _verify_2bXX.py）。
"""

import importlib
import sys
from pathlib import Path

GW_DIR = str(Path(__file__).resolve().parent.parent)  # apps/gateway
if GW_DIR not in sys.path:
    sys.path.insert(0, GW_DIR)

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)  # 仓库根


def reload_gateway(*names):
    """重载 gateway 下若干子模块，隔离 prometheus Counter 等全局状态。

    重载顺序约束：
      - config 恒先：settings 单例在 config 重载时重建，router/limiter/proxy 等用
        `from .config import settings` 绑定，config 后于它们会导致持有旧 settings。
      - litellm_router 须在依赖它的 main / proxy 之前：否则末尾 reload litellm_router
        会新建单例，使 proxy/decide 持有旧（空）单例，路由注册判定失效。
    其余模块按原顺序。
    """
    ordered = list(names)
    # config 提到最前
    if "config" in ordered:
        ordered.remove("config")
        ordered.insert(0, "config")
    # litellm_router 提到 config 之后（依赖 config，须在 main/proxy 之前）
    if "litellm_router" in ordered:
        ordered.remove("litellm_router")
        insert_at = 1 if "config" in ordered else 0
        ordered.insert(insert_at, "litellm_router")
    mods = {}
    for n in ordered:
        import importlib

        m = importlib.import_module(f"gateway.{n}")
        importlib.reload(m)
        mods[n] = m
    return mods


def app_paths(app):
    """收集 FastAPI app 注册的全部路由路径。

    新 starlette 将 include_router 挂载的路由包装为 _IncludedRouter（无 .path），
    需经 original_router 展开其内部 APIRouter 才能拿到真实 path（如 /v1/*）。
    """
    paths: set[str] = set()
    for r in app.routes:
        p = getattr(r, "path", None)
        if p:
            paths.add(p)
            continue
        orig = getattr(r, "original_router", None)
        if orig is not None and hasattr(orig, "routes"):
            for sub in orig.routes:
                sp = getattr(sub, "path", None)
                if sp:
                    paths.add(sp)
            continue
        for sub in getattr(r, "routes", []) or []:
            sp = getattr(sub, "path", None)
            if sp:
                paths.add(sp)
    return paths


class FakeRouter:
    """端到端测试用假 LiteLLM Router：实现 proxy 依赖的 acompletion / astream 协议。

    - acompletion(model, messages, **kwargs) -> (dict, actual_model, used_fallback)
    - astream(model, messages, **kwargs) -> async generator of (chunk_dict, actual_model)
    通过 `seq` 控制返回状态（200 / 500），`behavior` 控制是否标记 fallback。
    """

    def __init__(self, seq=None, used_fallback=False, content="ok"):
        self.seq = list(seq or [200])
        self.i = 0
        self.used_fallback = used_fallback
        self.content = content
        self.last_model = None
        self.last_messages = None

    async def acompletion(self, model, messages, **kwargs):
        self.last_model = model
        self.last_messages = messages
        code = self.seq[self.i] if self.i < len(self.seq) else 500
        self.i += 1
        if code >= 400:
            # 模拟上游失败：抛 RateLimitError 让 proxy 走 fail-closed 分流
            from litellm.exceptions import RateLimitError

            raise RateLimitError(
                message="stub upstream failure",
                model=model,
                llm_provider="fake",
            )
        return (
            {
                "id": "stub",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": self.content}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            model,
            self.used_fallback,
        )

    async def astream(self, model, messages, **kwargs):
        self.last_model = model
        self.last_messages = messages
        code = self.seq[self.i] if self.i < len(self.seq) else 500
        self.i += 1
        if code >= 400:
            from litellm.exceptions import RateLimitError

            raise RateLimitError(message="stub upstream failure", model=model, llm_provider="fake")
        yield (
            {
                "id": "stub",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {"content": self.content}, "finish_reason": None}],
            },
            model,
        )
        yield (
            {
                "id": "stub",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
            model,
        )


def stub_proxy(monkeypatch, fake_router=None, reload_modules=("controllers.proxy", "router", "config", "main", "litellm_router")):
    """pytest 友好：注入 FakeRouter 到 llm_router，替代原 httpx stub。

    返回 (TestClient, reloaded_proxy_module)。monkeypatch 自动恢复。

    注意：proxy 语义缓存(cache_store)为进程内全局，跨用例共享会短路端到端请求、
    绕过 stub。这里一并禁用 cache_lookup，使端到端用例真实打到 FakeRouter；缓存行为
    由批次 08 的单元测试单独覆盖。
    """
    from fastapi.testclient import TestClient

    import gateway.hooks.governance as governance
    import gateway.litellm_router as lr

    async def _no_cache(stage, payload):
        return None

    mods = reload_gateway(*reload_modules)
    proxy = mods["controllers.proxy"]
    main = mods["main"]
    fake = fake_router or FakeRouter()
    # FakeRouter 实现 acompletion/astream，直接替换 proxy 实际引用的 llm_router 单例的 router 属性。
    # 用 proxy 模块绑定的引用设 fake，确保 proxy 调用的是被替换的对象（而非 reload 产生的新单例）。
    llm_router_ref = proxy.llm_router
    monkeypatch.setattr(llm_router_ref, "_router", fake)
    monkeypatch.setattr(llm_router_ref, "_model_list", [{"model_name": "*"}])  # 任意 model 均视为已注册
    monkeypatch.setattr(governance.hooks, "cache_lookup", _no_cache)
    return TestClient(main.app), proxy
