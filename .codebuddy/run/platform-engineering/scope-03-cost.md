# 子 Scope · 批次1-03：成本治理与限流

> 上游：`docs/platform-engineering/03-成本治理与限流.md`（目录大纲）+ 总 `scope.md` §3/§5
> 性质：落地 03 篇「计量 → 限流/配额 → 预算熔断」三段中的**批次1 可落地部分**；真实 token 计数留批次3
> 下游消费者：`python-coder`（批次1 实现）、`code-reviewer`、`test-generator`、用户验收

---

## 1. 目标（一句话可验证）

在批次0 `metrics.py` 估算占位基础上，落地 03 篇三层治理的**集中卡口**：按租户/API Key 解析 + 进程内令牌桶租户隔离限流 + 预算熔断截断；计量仍用估算（真实计数批次3 接 Prometheus+Langfuse，本批仅把接口留好）。

## 2. 设计依据（必读）

- `docs/platform-engineering/03-成本治理与限流.md` §2（三层治理）、§3（计量精度是地基）、§4（令牌桶原理引分布式第4篇）
- 现有代码：`apps/gateway/gateway/metrics.py`（`record/estimate_tokens/render`，进程内 `_stats`）、`controllers/proxy.py`（`_tenant_of` 默认 default，待扩展）、`config.py`、`types.py`（`TenantUsage` 已定义）

## 3. 子 scope 边界

**做**：
- **租户解析**：`controllers/proxy.py` 的 `_tenant_of` 从 `X-Tenant` / `Authorization`(API Key) 解析 tenant；默认 `default`。API Key → tenant 映射走 env 配置（`TENANT_API_KEYS`，逗号分隔 `tenant:key`）。
- **限流（令牌桶，租户隔离）**：新增 `limiter.py`，实现进程内令牌桶（token bucket）按 tenant 隔离；全局桶 + 单租户桶两层阈值（引 03 §4）。超限返回 429 + `Retry-After`。配置项：`RATE_LIMIT_GLOBAL_RPS`、`RATE_LIMIT_TENANT_RPS`、`RATE_LIMIT_BURST`。
- **预算熔断**：`limiter.py` 或 `metrics.py` 增预算检查；按 tenant 累计 est_tokens 超 `BUDGET_TENANT_TOKENS`（env，默认 0=不熔断）即截断 429 + `{"error":"budget exceeded"}`。熔断为**软熔断**（估算口径，引 03 §3「估算偏差」），真实精度批次3 接。
- **计量升级**：`metrics.py` 保持估算但结构清晰化，`render()` 已按 tenant 标签输出，本批确认字段并补 `gateway_rate_limited_total` / `gateway_budget_exceeded_total` 计数。

**不做**：
- 真实 token 计数（tiktoken / Prometheus / Langfuse）：批次3。
- 分布式令牌桶（redis）：本批进程内即可（网关无状态但单实例演示足够；多实例精度问题批次3 用外部存储解决，已记 architecture §4.4）。
- 成本突增告警、失控循环自愈：批次3（09/05）接力。

## 4. 模块落点

| 文件 | 动作 |
|---|---|
| `config.py` | 增 `TENANT_API_KEYS`、`RATE_LIMIT_GLOBAL_RPS`、`RATE_LIMIT_TENANT_RPS`、`RATE_LIMIT_BURST`、`BUDGET_TENANT_TOKENS` |
| `limiter.py` | 【新增】`TokenBucket` 按 tenant 隔离 + 全局桶；`check(tenant) -> Verdict`（限流/预算）；进程内 dict 存桶 |
| `metrics.py` | 增 `rate_limited_total` / `budget_exceeded_total` 计数；`render()` 输出新指标 |
| `controllers/proxy.py` | `_tenant_of` 扩展 API Key 解析；建连前调 `limiter.check(tenant)`，超限 429；记录预算计数 |
| `types.py` | 可选：`TokenBucketState`（本批可内联，不强制） |

## 5. 接口契约（验收基准）

- 限流：单租户超 `RATE_LIMIT_TENANT_RPS` → HTTP 429 + `Retry-After`。
- 预算：tenant est_tokens 超 `BUDGET_TENANT_TOKENS`(>0) → HTTP 429 + `{"error":"budget exceeded"}`。
- `/metrics` 暴露：`gateway_requests_total`、`gateway_tokens_total`、`gateway_rate_limited_total`、`gateway_budget_exceeded_total`（均带 tenant 标签）。
- 限流/预算在「建连前」决策，与 ③出向 fail 开关同层（不破坏无旁路）。

## 6. 验收清单

- [ ] 租户解析支持 X-Tenant / API Key（env 映射），默认 default
- [ ] 令牌桶按 tenant 隔离 + 全局桶；超限返 429 + Retry-After
- [ ] 预算熔断：超 tenant token 预算截断 429 + 结构化错误
- [ ] `/metrics` 新增限流/预算计数指标（tenant 标签）
- [ ] 限流/预算在建连前生效，不污染已放行流量
- [ ] 进程内实现不引入 redis（多实例精度问题批次3 解决，已记 architecture §4.4）
