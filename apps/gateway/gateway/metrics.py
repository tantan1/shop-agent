"""指标注册表（01 §3 可观测 / 03 §3 计量）。

D2 重构：由「手写字符串拼接的固定四指标」升级为**可注册 registry**——各模块
（06 guardrails / 08 cache / 10 injection）在导入时 `register_counter(...)` 声明
自己的计数器，`render()` 统一遍历渲染，**不再需要改动 render() 本体**。

批次3 落地（scope-04）：内部实现从自研 MetricSeries 切换为 **prometheus_client.Counter**
标准封装，对外调用签名（register_counter / inc / get / render / reset）完全不变，
调用方零改动。`render()` 现在输出标准 Prometheus exposition 格式
（`prometheus_client.generate_latest`）。

设计约束：
- 标签任意：计数器按 label 元组分桶，`inc(labels={...})`。
- 幂等注册：同名重复 register 返回同一实例（模块被多次导入也安全）。
- 线程安全：prometheus_client.Counter 内部已线程安全；注册表加 Lock 防重复注册竞态。
- prometheus_client 是**客户端库**（指标在进程内存，/metrics 为 pull 端点），
  不引入运行时外部依赖 SPOF。
"""
from __future__ import annotations

import threading
from collections.abc import Mapping

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

# 独立 registry，避免与进程内其它 prometheus_client 默认 registry 串扰。
_REGISTRY = CollectorRegistry()
_LOCK = threading.Lock()


class MetricSeries:
    """prometheus_client.Counter 的薄封装，保持与原自研实现一致的调用签名。

    调用方使用 `series.inc(amount, labels=...)` / `series.get(labels=...)`，
    与原 MetricSeries 完全兼容；内部委托给 prometheus_client.Counter.labels(...)。
    """

    name: str
    help_text: str
    labelnames: tuple[str, ...]

    def __init__(
        self, name: str, help_text: str, labelnames: tuple[str, ...] = ()
    ) -> None:
        self.name = name
        self.help_text = help_text
        self.labelnames = tuple(labelnames)
        self._pc = Counter(
            name, help_text, list(self.labelnames), registry=_REGISTRY
        )

    def _labels(self, labels: Mapping[str, str] | None) -> dict[str, str]:
        labels = dict(labels or {})
        unknown = set(labels) - set(self.labelnames)
        if unknown:
            raise ValueError(f"{self.name}: unknown labels {sorted(unknown)}")
        # 补齐缺失 label 为空串（prometheus_client 要求 label 值必须提供）
        return {n: str(labels.get(n, "")) for n in self.labelnames}

    def inc(
        self, amount: int = 1, labels: Mapping[str, str] | None = None
    ) -> None:
        if self.labelnames:
            self._pc.labels(**self._labels(labels)).inc(amount)
        else:
            self._pc.inc(amount)

    def get(self, labels: Mapping[str, str] | None = None) -> int:
        if self.labelnames:
            sample = self._pc.labels(**self._labels(labels))._value.get()
            return int(sample)
        return int(self._pc._value.get())

    def samples(self):  # pragma: no cover - 兼容旧接口，本批未使用
        raise NotImplementedError("use render() for exposition")

    def clear(self) -> None:  # pragma: no cover - 仅供测试；prometheus 无清零语义
        # prometheus_client 的 Counter 不支持清零；测试改用 _REGISTRY 重建。
        raise NotImplementedError("prometheus Counter 不可清零，测试用 reset() 重建 registry")

    def render_lines(self) -> list[str]:  # pragma: no cover - 由 generate_latest 替代
        raise NotImplementedError("use generate_latest()")


class HistogramSeries:
    """prometheus_client.Histogram 的薄封装（09 业务级延迟 golden signal）。

    提供 `observe(value, labels=...)`，内部委托给 prometheus_client.Histogram.labels(...)。
    注册表同为幂等：同名重复 register 返回同一实例。
    """

    name: str
    help_text: str
    labelnames: tuple[str, ...]
    buckets: tuple[float, ...]

    def __init__(
        self,
        name: str,
        help_text: str,
        labelnames: tuple[str, ...] = (),
        buckets: tuple[float, ...] = (),
    ) -> None:
        self.name = name
        self.help_text = help_text
        self.labelnames = tuple(labelnames)
        self.buckets = tuple(buckets)
        kwargs: dict = {"registry": _REGISTRY}
        if self.buckets:
            kwargs["buckets"] = list(self.buckets)
        self._ph = Histogram(name, help_text, list(self.labelnames), **kwargs)

    def _labels(self, labels: Mapping[str, str] | None) -> dict[str, str]:
        labels = dict(labels or {})
        unknown = set(labels) - set(self.labelnames)
        if unknown:
            raise ValueError(f"{self.name}: unknown labels {sorted(unknown)}")
        return {n: str(labels.get(n, "")) for n in self.labelnames}

    def observe(
        self, value: float, labels: Mapping[str, str] | None = None
    ) -> None:
        if self.labelnames:
            self._ph.labels(**self._labels(labels)).observe(value)
        else:
            self._ph.observe(value)

    def get_count(self, labels: Mapping[str, str] | None = None) -> int:
        if self.labelnames:
            return int(self._ph.labels(**self._labels(labels))._created.value)
        return int(self._ph._created.value)


# 注册表：name -> MetricSeries（保持与原 _REGISTRY 字典语义）
_SERIES: dict[str, MetricSeries] = {}

# 注册表：name -> HistogramSeries（09 业务级延迟）
_HISTOGRAMS: dict[str, HistogramSeries] = {}


def register_counter(
    name: str, help_text: str, labelnames: tuple[str, ...] = ()
) -> MetricSeries:
    """注册（或取回同名）指标序列。各模块自行调用，无需改动 render()。"""
    labelnames = tuple(labelnames)
    with _LOCK:
        existing = _SERIES.get(name)
        if existing is not None:
            if existing.labelnames != labelnames:
                raise ValueError(
                    f"metric {name} already registered with labels {existing.labelnames}"
                )
            return existing
        series = MetricSeries(name, help_text, labelnames)
        _SERIES[name] = series
        return series


def get_metric(name: str) -> MetricSeries | None:
    with _LOCK:
        return _SERIES.get(name)


def register_histogram(
    name: str,
    help_text: str,
    labelnames: tuple[str, ...] = (),
    buckets: tuple[float, ...] = (),
) -> HistogramSeries:
    """注册（或取回同名）直方图指标（09 业务级延迟用）。"""
    labelnames = tuple(labelnames)
    with _LOCK:
        existing = _HISTOGRAMS.get(name)
        if existing is not None:
            if existing.labelnames != labelnames:
                raise ValueError(
                    f"histogram {name} already registered with labels {existing.labelnames}"
                )
            return existing
        series = HistogramSeries(name, help_text, labelnames, buckets)
        _HISTOGRAMS[name] = series
        return series


# 09 业务级延迟 golden signal：按 tenant / model / mode / status 分桶。
# 桶边界覆盖 50ms~30s，兼顾实时交互与长文生成场景。
# 在 register_histogram 定义之后注册，保证调用顺序正确。
request_latency_seconds = register_histogram(
    "gateway_request_latency_seconds",
    "LLM gateway request latency in seconds",
    ("tenant", "model", "mode", "status"),
    buckets=(
        0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0,
    ),
)


# --- 批次0/1 既有指标（迁移到 registry，对外函数签名不变） ---

requests_total = register_counter(
    "gateway_requests_total", "total LLM requests", ("tenant",)
)
tokens_total = register_counter(
    "gateway_tokens_total", "estimated tokens", ("tenant",)
)
rate_limited_total = register_counter(
    "gateway_rate_limited_total", "requests rejected by rate limit", ("tenant",)
)
budget_exceeded_total = register_counter(
    "gateway_budget_exceeded_total", "requests rejected by budget", ("tenant",)
)


def record(tenant: str, est_tokens: int) -> None:
    requests_total.inc(labels={"tenant": tenant})
    tokens_total.inc(est_tokens, labels={"tenant": tenant})


def tenant_tokens(tenant: str) -> int:
    """返回该 tenant 累计 est_tokens（供预算熔断判断）。"""
    return tokens_total.get(labels={"tenant": tenant})


def mark_rate_limited(tenant: str) -> None:
    rate_limited_total.inc(labels={"tenant": tenant})


def mark_budget_exceeded(tenant: str) -> None:
    budget_exceeded_total.inc(labels={"tenant": tenant})


def estimate_tokens(raw: bytes) -> int:
    """骨架估算（按字符数近似），批次3 替换为真实计数。"""
    try:
        import json

        parsed: object = json.loads(raw or b"{}")
        if not isinstance(parsed, dict):
            return 0
        msgs: object = parsed.get("messages", [])
        if not isinstance(msgs, list):
            return 0
        text = "".join(
            str(m.get("content", "")) for m in msgs if isinstance(m, dict)
        )
        return max(len(text) // 2, 0)
    except Exception:
        return 0


def render() -> str:
    """输出标准 Prometheus exposition（prometheus_client.generate_latest）。

    新增指标无需改动本函数——registry 自动包含全部已注册 Counter。
    """
    return generate_latest(_REGISTRY).decode("utf-8")


def reset() -> None:
    """清空所有计数（仅供测试使用，生产不调用）。

    prometheus_client 的 Counter 不可原地清零，故重建 registry 与序列表。
    已 import 的模块级引用（requests_total 等）仍指向旧实例——测试通过
    re-import 或统一经 get_metric() 访问；本批验收脚本改用 module 级 fixture 重建。
    """
    global _REGISTRY, _SERIES, _HISTOGRAMS
    with _LOCK:
        _REGISTRY = CollectorRegistry()
        # 重建所有已注册序列（保留 name/help/labelnames）
        for name, series in list(_SERIES.items()):
            _SERIES[name] = MetricSeries(
                series.name, series.help_text, series.labelnames
            )
        # 重建所有已注册直方图（保留 name/help/labelnames/buckets）
        for name, hist in list(_HISTOGRAMS.items()):
            _HISTOGRAMS[name] = HistogramSeries(
                hist.name, hist.help_text, hist.labelnames, hist.buckets
            )
