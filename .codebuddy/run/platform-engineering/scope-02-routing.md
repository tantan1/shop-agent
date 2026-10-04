# 子 Scope · 批次1-02：路由策略与多后端故障转移

> 上游：`docs/platform-engineering/02-路由策略与引擎可替换性.md`（目录大纲）+ 总 `scope.md` §3/§5
> 性质：落地 02 篇骨架的**路由执行点 + 故障转移**，非重推路由原理（原理引 `ai-agent-distributed-system` 第3/4篇）
> 下游消费者：`python-coder`（批次1 实现）、`code-reviewer`、`test-generator`、用户验收

---

## 1. 目标（一句话可验证）

在批次0 已落地的 `router.route(model, tenant)` 基础上，补齐 02 篇 §4「多后端并存与故障转移」：每个后端可配**有序 fallback 链**；上游限流/不可达时按链切换，全程**不泄露到业务层**（响应 metadata 带回 `X-Upstream-Model` / `X-Fallback` 证据头），全挂则 fail-closed 返回结构化错误（绝不返回降级空壳冒充成功，引 02 §4）。

## 2. 设计依据（必读）

- `docs/platform-engineering/02-路由策略与引擎可替换性.md` §4（故障转移分级契约）、§1（路由键=模型名）
- 现有代码：`apps/gateway/gateway/router.py`（`_backend_for` 已分四类）、`controllers/proxy.py`（已带 `X-Upstream-Model` / `X-Fallback` 头，本批填真实值）、`config.py`（后端端点 env）、`types.py`（`RouteDecision` 已含 `fallback_allowed`）

## 3. 子 scope 边界

**做**：
- 路由表升级为「主后端 + fallback 链」：每个模型前缀/类别映射一个有序后端列表（env 配置，逗号分隔，`;` 分隔链）。
- `route()` 返回 `RouteDecision` 携带 `fallback_chain: list[str]`（base_url 列表）与命中主后端。
- 故障转移执行点落在 `controllers/proxy.py`：主后端建连失败/收到 429/5xx（限流/宕机信号）→ 沿 `fallback_chain` 依次重试，**仅重试同模型跨厂商**（不自动跨模型降级——跨模型降级需业务侧感知，引 02 §4「两种变」①）。
- 切换成功时 `decision.fallback_allowed = True` 且 `X-Fallback: true` + `X-Upstream-Model` 填实际命中后端；全链失败 → 返回结构化 503（沿用现有 fail-closed 风格），绝不返回降级模型的空壳。
- 故障转移触发条件（薄写，引分布式第4篇）：上游返回 429 / 5xx 即触发切换；配置项 `ROUTE_FALLBACK_ON_STATUS` 默认 `{429,500,502,503,504}`。

**不做**：
- 跨模型自动降级（大模型→本地小模型）：属 02 §4「两种变」②，需 05/06/09 接力，**本批不自动降级**，仅 fail-closed。
- 限流/熔断算法实现（令牌桶）：属 03 篇，本批不碰。
- Bedrock 无标准 OpenAI 兼容端点：本批仍留空回退（config 已注释），不实现 Bedrock SDK 适配。

## 4. 模块落点（不新增文件，改现有）

| 文件 | 动作 |
|---|---|
| `config.py` | 后端端点支持 fallback 链配置（如 `azure_openai_base_url` 可为逗号分隔多地址；或新增 `fallback_chain` 全局配置）。新增 `ROUTE_FALLBACK_ON_STATUS` 配置 |
| `router.py` | `RouteDecision` 增 `fallback_chain: list[str]`；`_backend_for` → 返回主+链；`fallback_allowed` 语义改为「存在可用 fallback 链」 |
| `types.py` | `RouteDecision` 加 `fallback_chain: list[str]` 字段（默认空） |
| `controllers/proxy.py` | 主后端失败→沿链重试；成功填真实 `X-Fallback`/`X-Upstream-Model`；全失败返 503 |

## 5. 接口契约（验收基准）

- `route(model, tenant) -> RouteDecision`：`fallback_chain` 为非空列表（至少含主后端自身）。
- 响应头：`X-Upstream-Model` = 实际命中后端 model 或原 model；`X-Fallback` = `true`/`false`。
- 全挂：HTTP 503 + `{"error": "all backends unavailable", "model": ...}`（结构化，非空壳）。
- 不破坏无旁路不变量：fallback 链仅含网关已配置的「网关身后后端」，绝不退回业务直连供应商。

## 6. 验收清单

- [ ] 路由表支持主后端 + 有序 fallback 链（env 配置）
- [ ] 主后端 429/5xx 触发沿链重试，不泄露故障到业务层
- [ ] 切换成功 `X-Fallback: true` + `X-Upstream-Model` 填实际后端
- [ ] 全链失败返结构化 503，不返回降级空壳冒充成功
- [ ] 跨模型自动降级不发生（仅 fail-closed，留待批次5/06 接力）
- [ ] 现有 `/v1` 非流式/流式路径行为不变（含批次0 的 SSE 钩子生效）
