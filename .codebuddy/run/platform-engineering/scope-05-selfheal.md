# scope-05 失控循环防护（最小版）

> 关联论证文：`05-失控循环防护.md`（设计大纲）
> 关联代码：新增 `apps/gateway/gateway/loopguard.py`，接入 `controllers/proxy.py` 建连前。
> 关联决策：用户确认「本批做最小版（tenant 级重复请求熔断，接 limiter 风格）」。

## 1. 目标
在网关层检测**同一 tenant 短时间内重复触发相同请求**的失控循环（Agent 自我循环调用），
超阈值即熔断返回 429 + 标记，防止无限烧钱/烧配额。

## 2. 边界（最小版）
- **仅 tenant 级、基于请求指纹**：指纹 = `hash(tenant + model + prompt 归一化)`。
- **不深入 Agent 链路追踪**：不跨请求关联「调用图」，只防「同一请求被重复打爆」。
- 熔断为**时间窗滑动**：窗口内同指纹计数超 `LOOP_GUARD_MAX`（默认 8）/ 窗口（默认 10s）
  即 deny，窗口滑动过期。
- 与限流同层：建连前执行，fail 开关同层，不破坏无旁路不变量。
- 指标：`gateway_loop_guarded_total{tenant}` 计数熔断次数。

## 3. 实现要点
- `loopguard.check(tenant, model, prompt) -> Verdict`：建连前调用，仿 `limiter.check` 风格。
- 进程内滑动窗口（dict[fp -> list[timestamp]]），`_LOOP_TTL` 过期清理。
- 配置：`LOOP_GUARD_MAX`（默认 8）、`LOOP_GUARD_WINDOW_SEC`（默认 10）。
- proxy 在 `limit_check` 之后、`router.decide()` 之前调用 `loopguard.check`。

## 4. 验收项（scope-05 §6）
1. 同一 tenant + 相同 prompt 在窗口内重复超阈值 → deny（429 + `loop guarded`）。
2. 不同 tenant 互不影响（隔离）。
3. 不同 prompt 不触发（指纹区分）。
4. 窗口滑动：超过 `LOOP_GUARD_WINDOW_SEC` 后计数重置，不再熔断。
5. 配置可通过 env 调整阈值。
6. 指标 `gateway_loop_guarded_total` 随熔断累加。

## 5. 不做（留后续）
- 跨进程/redis 共享指纹（多实例一致性）。
- Agent 调用图级别的循环检测（需链路 trace，依赖批次3 tracing）。
- 人在回路降级（沿用 06 的 human_review 通道，本批不新增）。
