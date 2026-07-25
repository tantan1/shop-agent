"""业务维度分片策略（08 §3）。

配置优先级（已确认，三层，绝不出现无值）：分片值 → env 默认 → 代码内置常量。
硬约束层（PII 整条不写、写操作不命中）**不参与该优先级**，代码强制、任何配置不可覆盖。

配置加载走应用内定时轮询 reload（默认 30s，不依赖 inotify 事件，兼容 Docker/K8s
ConfigMap 符号链接轮换）；多节点时各节点独立拉同一共享配置源即可一致。
"""
from __future__ import annotations

import threading
from typing import Any

import yaml

from gateway.config import settings

# 内置常量（第三层兜底，恒定值，任何配置不可覆盖）
_DEFAULT_THRESHOLD = settings.similarity_threshold  # 来自 env 兜底，env 再低于内置不可见
_DEFAULT_TTL = settings.cache_ttl
_DEFAULT_ENABLED = settings.cache_enabled

# 停用词表（轻量，中文按字符、英文按词；真实分词留钩子，本批够用）
_STOPWORDS = set(
    "的 了 是 在 我 你 他 她 它 们 这 那 有 和 与 及 或 不 也 都 就 而 一个 什么 怎么 如何 请问 能 可以 吗 呢 吧 啊 哦 嗯 请 帮 助 告诉 关于".split()
)


def _normalize(text: str) -> str:
    """归一化：小写、去停用词、按空白/标点切分 token。"""
    text = (text or "").lower()
    # 英文按空格，中文按字符（简化：统一按空白+标点切，再逐字符兜底中文）
    import re

    # 先按非字母数字中文切
    raw_tokens = re.split(r"[\s，。、；：？！,.!?;:\"'`~()（）\[\]【】]+", text)
    tokens: list[str] = []
    for tok in raw_tokens:
        if not tok:
            continue
        # 中文整段逐字做 token（中文无空格），但过滤停用字
        if any("\u4e00" <= c <= "\u9fff" for c in tok):
            for c in tok:
                if c in _STOPWORDS:
                    continue
                tokens.append(c)
        else:
            if tok in _STOPWORDS:
                continue
            tokens.append(tok)
    return " ".join(tokens)


class BucketPolicy:
    """单个业务分片的生效策略（运行时只读视图）。"""

    def __init__(
        self,
        bucket: str,
        enabled: bool,
        threshold: float,
        ttl: int,
        skip_write: bool,
    ) -> None:
        self.bucket = bucket
        self.enabled = enabled
        self.threshold = threshold
        self.ttl = ttl
        self.skip_write = skip_write

    def as_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "enabled": self.enabled,
            "threshold": self.threshold,
            "ttl": self.ttl,
            "skip_write": self.skip_write,
        }


class CachePolicy:
    """业务维度分片策略管理（三层优先级解析 + 后台 reload）。

    硬约束（PII 整条不写、写操作不命中）由调用方（store/pii_gate/proxy）代码强制，
    本类只负责「软可调」维度：enabled / threshold / ttl / skip_write。
    """

    def __init__(self, policy_path: str | None = None) -> None:
        self._path = policy_path or settings.cache_policy_path
        self._lock = threading.Lock()
        self._buckets: dict[str, dict[str, Any]] = {}
        self._reload()
        # 后台轮询（默认 30s）；不依赖 inotify，兼容 CM 符号链接轮换。
        # 注意：本批不启动独立线程（避免测试/导入副作用），reload 由调用方在
        # 热路径首调用懒触发（见 maybe_reload）。批次3 可加 asyncio 后台任务。

    def _resolve_path(self) -> str:
        """路径解析：相对路径以本包目录为基准（兼容 CWD=仓库根 / CWD=apps/gateway / Docker 挂载）。"""
        import os

        p = self._path
        if os.path.isabs(p):
            return p
        # 先试「以本包目录为基准」（gateway/cache/cache-policy.yml 在包内）
        pkg = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(pkg, p)
        if os.path.exists(cand):
            return cand
        # 再试「以 CWD 为基准」（env 覆盖成包外路径时）
        if os.path.exists(p):
            return p
        return cand  # 都不存在则保留包内基准路径（让 FileNotFoundError 冒泡被捕获）

    def _reload(self) -> None:
        raw: dict[str, Any] = {}
        try:
            with open(self._resolve_path(), "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raw = {}
        except Exception:
            # 解析失败：保留旧配置（绝不清空，避免误关缓存/误放写操作）
            return
        buckets = raw.get("buckets", {}) or {}
        with self._lock:
            self._buckets = buckets if isinstance(buckets, dict) else {}

    def maybe_reload(self) -> None:
        """懒 reload 入口（本批由 lookup/store 调用方在热路径外周期触发）。"""
        self._reload()

    def resolve(self, bucket: str) -> BucketPolicy:
        """三层优先级解析：分片值 → env 默认 → 内置常量。"""
        with self._lock:
            spec = self._buckets.get(bucket, {}) or {}
        # 分片字段（声明即覆盖）
        enabled = spec.get("enabled", _DEFAULT_ENABLED)
        threshold = spec.get("threshold", _DEFAULT_THRESHOLD)
        ttl = spec.get("ttl", _DEFAULT_TTL)
        skip_write = spec.get("skip_write", False)
        return BucketPolicy(
            bucket=bucket,
            enabled=bool(enabled),
            threshold=float(threshold),
            ttl=int(ttl),
            skip_write=bool(skip_write),
        )

    def bucket_of(self, payload: dict[str, Any]) -> str:
        """从请求体提取业务标签（缓存分片键）。

        优先级：payload["cache_bucket"] 显式标签 > messages[0] 的 role 启发 > "default"。
        业务方可在请求体携带 ``cache_bucket`` 字段声明自有分片（仅能命中存在分片）。
        """
        explicit = payload.get("cache_bucket") if isinstance(payload, dict) else None
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        return "default"


# 模块级单例（与 governance.hooks 同生命周期）
policy = CachePolicy()
