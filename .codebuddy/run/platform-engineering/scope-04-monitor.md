# scope-04 可观测：prometheus_client 标准化 + Langfuse 钩子

> 关联论证文：`04-监控子弧.md`、`09-可观测.md`
> 关联代码：`apps/gateway/gateway/metrics.py`、`controllers/metrics_endpoint.py`
> 关联决策：用户确认「用 prometheus_client（选项2）」，05 做最小失控循环熔断版。

## 1. 目标
把网关指标从「自研文本格式」升级为**标准 Prometheus exposition**，并预留 Langfuse trace
钩子（默认关闭、不引入推送式 SPOF）。

## 2. 非目标（本批边界）
- Langfuse trace 不默认启用：仅留 env 开关 + 失败降级钩子，不强制推送外部服务。
- 多实例分布式聚合：Prometheus server 侧 pull 聚合，网关侧不引入 redis（沿用 limiter 注释约定）。
- 真实 token 计数：仍用 `estimate_tokens` 字符近似，真实计数留后续（09 §3）。

## 3. 改动点
### 3.1 metrics.py 内部实现替换
- 保留对外调用签名：`register_counter(name, help_text, labelnames)`、`series.inc(amount, labels)`、
  `series.get(labels)`、`get_metric(name)`、`record/tenant_tokens/mark_*/estimate_tokens/render/reset`。
- 内部用 `prometheus_client.Counter` 封装，**调用方零改动**。
- `render()` 改为返回 `prometheus_client.generate_latest()` 的标准 exposition。
- 多标签 Counter 用 `Counter.labels(**kwargs)` 写、`labels` 须与注册 labelnames 一致（校验防御）。
- 删除自研 `MetricSeries` / `_escape` / 手写 render（被库取代）。

### 3.2 metrics_endpoint.py
- 返回 `render()`（已是标准 exposition），`media_type="text/plain; version=0.0.4"`。
- 不变更多逻辑。

### 3.3 Langfuse 钩子（预留）
- 新增 `gateway/tracing.py`：`init_tracing()` 读 `LANGFUSE_ENABLED` env，False 时 no-op；
  `trace_llm(tenant, model, tokens)` / `trace_error(...)` 在 enabled 时推送，异常静默降级。
- proxy 在「模型调用成功/失败」处留调用点（本批仅接线、默认关闭）。

## 4. 验收项（scope-04 §6）
1. `/metrics` 输出标准 Prometheus 格式（`# TYPE ... counter` + 各 label 样本）。
2. 已注册指标（requests/tokens/rate_limited/budget_exceeded/cache_hits/cache_writes/
   cache_pii_blocked/injection_denied/governance_error/guardrails_review）均可 scrape。
3. 重复 `register_counter` 同名同 labels 幂等返回（不抛异常）。
4. 标签缺失/多余时 `inc` 抛清晰错误（与原校验语义一致）。
5. `LANGFUSE_ENABLED` 未设/False 时网关启动与 `/metrics` 不受影响（无外部依赖）。
6. 进程内计数正确累加，`reset()` 测试可用。

## 5. 不做
- 不改任何调用方（`cache/store.py`、`hooks/injection.py`、`hooks/governance.py`、`proxy.py`）。
