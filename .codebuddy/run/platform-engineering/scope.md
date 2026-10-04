# Scope: 依据 platform-engineering 系列（01~10）全量落地生产级 Agent 平台

> 本文件由 `multi-agent-workflow` 阶段0 产出，是 10 篇文档落地的**总控 scope**。后续所有批次（架构/编码/审查/测试）必须引用本文件，不得超出其范围边界与批次编排。
> 设计依据权威源：`docs/platform-engineering/`（10 篇 + 大纲）。本次为"落地既有设计骨架"，非重新论证架构。

## 1. 目标（一句话可验证）

将 `docs/platform-engineering/01~10` 的章节骨架，按依赖批次落成 `apps/gateway/`、`apps/monitoring-agent/`（及必要的 `shop-agent` 改造）的可运行实现，覆盖**网关治理（路由/成本/合规/缓存/注入）· 监控自愈（巡检/自愈/可观测）· 安全（注入三防线）**三大主题。

## 2. 设计依据（权威源，必须 read_file 后再动手）

- 主依据：`docs/platform-engineering/大纲.md` + `01~10` 各篇（状态：**均为目录大纲/骨架**，本次为"落地骨架"非重推架构）
- 现有代码锚点：
  - `apps/gateway/`（Python/FastAPI，已有路由骨架、`/metrics` 钩子、限流 TODO）
  - `apps/monitoring-agent/`（如存在，监控自愈落点；若不存在则本批次新建）
  - `apps/shop-agent/`（业务侧 LLM SDK `base_url` 需指向网关）
- 关联系列：`docs/ai-agent-distributed-system/`（路由/成本/限流/熔断原理，薄写交叉引用，不重推）

## 3. 批次编排与依赖（核心：不一次性全做，按依赖分批 + 批内并行）

```
批次0  01 网关总论        → [你确认 scope+架构] → 编码 → [你验收]
         │  (已有 gateway-core/scope.md)
         ▼
批次1  02 路由 + 03 成本限流   (依赖01, 彼此独立 → 并行)
         │  [你验收: 路由契约 + 限流/计量接口]
         ▼
批次2  06 合规 + 07 脱敏 + 08 语义缓存 + 10 注入三防线
         │  (依赖01/02；批内非全并行，见下方 2a/2b 分阶段)
         │
        │   2a  07 脱敏引擎 + rules/ 公共包   ← 硬前置，串行先做
        │        │  (产出 Redactor 接口 + RulesStore 三级来源 + metrics registry + GovernanceError + async 骨架)
        │        ▼
        │  [★ 你验收 2a 独立门 (G2)：接口契约/规则三级来源/registry/异常体/async 骨架 + 批次0/1 回归全绿]
        │        ▼
        │   2b  06 合规 · 08 缓存 · 10 注入   ← 三路并行（均消费 2a 产物）
        │
        │  [你验收: 注入三闸落点 + 合规/脱敏/缓存钩子]
         ▼
批次3  04 巡检 + 05 自愈 + 09 业务可观测   (依赖01出口+03计量 → 并行)
         │  [你验收: 监控双通道 + 自愈安全边界 + golden signal]
         ▼
批次4  跨篇集成验证 + 大纲"待补"(决策why/交叉引用/简历要点)  → [你终验]
```

### 各批次子 scope 落点
| 批次 | 篇目 | 子 scope 路径 | 并行 agent |
|---|---|---|---|
| 0 | 01 | `.codebuddy/run/gateway-core/scope.md`（已存在） | architecture-designer → python-coder |
| 1 | 02,03 | `.codebuddy/run/platform-engineering/scope-02-routing.md`<br>`.codebuddy/run/platform-engineering/scope-03-cost.md` | python-coder ×2 并行 |
| 2a | 07 | `scope-07-pii.md`（含 `rules/` 公共包 + `metrics.py` registry 化） | python-coder ×1（**硬前置，串行**） | **★ 2a 独立验收门（G2）：通过后才开 2b** |
| 2b | 06,08,10 | `scope-06-compliance.md` `scope-08-cache.md` `scope-10-injection.md` | python-coder ×3 并行（消费 2a 的 `Redactor` + `RulesStore` + metrics registry） |

#### 2b 并行写冲突面（编排铁律，D2/D4 已处置）

三路并行必须先消除共享文件的写冲突，否则合并即返工：

| 共享文件 | 冲突来源 | 处置 |
|---|---|---|
| `metrics.py` | 06/08/10 各要新增计数器，原实现须改同一 `render()` | **2a 已 registry 化（D2）**：三路各自 `register_counter` 注册，**任何人不再改 `metrics.py`**；注册语句写在各自模块（08→`cache/store.py`，06/10→`hooks/governance.py`） |
| `hooks/governance.py` | 06 加 `guardrails_check`/`human_review`、08 加 `cache_lookup`/`cache_store`、10 加 `judge_egress`，07 已接 `PiiEngine` | **2a 预先落好类骨架**：`GovernanceHooks` 里把三路方法签名（含 `async`，见 D4）与 `GovernanceError`（D3）**全部空实现占位**，2b 三路只填各自方法体，不动他人方法与类头 |
| `controllers/proxy.py` | 06/08/10 都要改 ingress/egress 编排顺序 | **2a 预先按 B2 顺序落好调用骨架**（调用空实现的钩子），2b 三路只改自己那一步的分支，不重排链路 |
| `config.py` | 三路各加自己的 env 字段 | 追加式（各加各的字段行），冲突面小；合并时按字母序归位 |

**结论**：`metrics.py` / `hooks/governance.py` / `controllers/proxy.py` 三个文件的**结构性改动全部前移到 2a**，2b 只做「填自己的方法体」，三路真正可并行。
| 3 | 04,05,09 | `scope-04-monitor.md` `scope-05-selfheal.md`（09 业务级可观测无独立 scope，随 04 批次落地，见 `04b` 素材 §7 数据接口对齐 09b §一） | python-coder / devops ×3 并行 |
| 4 | 集成 | 无新 scope，复用验收清单 | code-reviewer + test-generator |

## 4. 确认门（人在回路，硬卡点）

以下节点**必须你显式确认**才继续，agent 不得跳过：

1. **总 scope 确认**：本文件（批次划分/依赖/范围边界）——当前待你确认。
2. **每批次架构确认**：批次 1~3 开始前，对应子 scope + `architecture.md` 产出后给你看，确认接口契约/模块边界再编码。
3. **每批次验收门**：批次实现完跑验收清单，结果给你，通过才进下一批。
4. **LLM 安全相关设计**：注入三防线（10）、合规/脱敏（06/07）的落点需人审——不可逆风险高。

**不打断你的**：批次内具体代码、文件命名、单测编写；文档已写死的方案直接落地。

## 5. 验收清单（总控，逐批勾选）

### 批次0（01 网关）
- [x] `apps/gateway/` 暴露 OpenAI 兼容 `/v1/chat/completions`
- [x] `shop-agent` 的 LLM SDK `base_url` 指向网关（非直连）
- [x] 路由表支持 tool_select/param→本地、gpt*→Azure、claude*→Bedrock、qwen*→百炼
- [x] 入向①预留 Prompt 注入第一道闸钩子；出向③ fail-closed/open 显式开关
- [x] 关键路径无直连供应商硬编码；网关无状态可水平扩容
- [x] 端点层 controller 化（`controllers/` 包，main.py 纯装配，业务逻辑在 router/hooks/metrics）
- [x] 流式 SSE 下治理钩子（run_stream）真实生效（按 `data:` 行解析，非 JSON 原样透传，修复初版字节块 json.loads 必抛异常导致钩子被绕过）
- [x] 响应 metadata 按 02 篇带回 `X-Upstream-Model` / `X-Fallback` 证据头（X-Fallback 待批次1 故障转移填充）

### 批次1（02 路由 / 03 成本限流）—— 详见 `scope-02-routing.md` / `scope-03-cost.md`
- [x] 02：路由表支持主后端 + 有序 fallback 链（`RouteDecision.fallback_chain`）；主后端 429/5xx 沿链重试（仅同模型跨厂商）；全挂返结构化 503；填真实 `X-Upstream-Model`/`X-Fallback`
- [x] 03：租户解析（X-Tenant / API Key）；进程内令牌桶按 tenant 隔离 + 全局桶，超限 429+Retry-After；预算熔断超 tenant token 截断 429；`/metrics` 补限流/预算计数（limiter.py 已落地）

### 批次2（06/07/08/10）—— 详见 `scope-06-compliance.md` / `scope-07-pii.md` / `scope-08-cache.md` / `scope-10-injection.md`
- [ ] 10：输入/输出/工具参数三道闸各守其门（确定性层先上，语义层留钩子）；编排顺序：入向/出向均为「先注入检测、后脱敏」（先防攻击、后护隐私）
- [ ] 06：网关这道闸前置脱敏 + 后置兜底 + Guardrails 集中配置；跨平面处置总表为唯一权威（引擎由 07 落地）；**Guardrails 确定性违禁词命中 deny（非流式 4xx / 流式截断+终止标记），模糊边界打 `human_review` 标记不阻断（三通道：响应头 + 结构化日志 + metrics）**
- [ ] 07：脱敏规则即数据/三级来源/双层级降级（确定性层永远在线，语义层留钩子）
- [ ] 08：语义缓存相似 prompt 命中短路，省 token 降延迟；**词频向量+余弦命中（含换说法同义用例）返非流式答案，不进模型、不计 token**；PII 整条不写门禁 + 业务维度分片 + TTL/写操作不命中
- [ ] 出向合规熔断（judge_egress）：大模型返回不合规则终止后续结果（流式=截断+终止标记不回滚；非流式=整块拦截返 4xx），属 10 注入第三道闸 + 06 Guardrails 并列，按业界 fail-closed 通用做法在批次2 实现（不在批次0 提前补）；**judge_egress 归 scope-10 为唯一权威实现，scope-06 的 Guardrails 是独立一道闸，两者在 proxy 并列调用**（F2 归属澄清）
- [ ] 顺序不变量（批次2 总控铁律）：入向 注入检测 → guardrails → 脱敏 → 缓存查询(命中短路) → 模型（B2）；出向 judge_egress → guardrails → 脱敏（先防攻击、后护隐私）；检测器必须基于未脱敏原文；缓存命中不重跑 egress 治理（B3 前提：cache_store 在治理通过后调用）
- [ ] **2a 独立验收门（G2，★ 2b 开工硬卡）**：以下全过才允许 06/08/10 并行开工，否则 2b 基于未验收接口返工——① `Redactor.redact` 接口契约冻结（签名 `redact(text: str) -> tuple[str, list[str]]`，入参脱敏文本、出参 (脱敏后文本, 命中类型列表)、异常类型），2b 只依赖签名不依赖实现细节；② `RulesStore` 三级来源（基线 YAML / 远程 URL / 本地覆盖）加载与 fallback 跑通，**无规则文件时回退基线（绝不零规则启动）**；③ metrics registry 验收（D2）：新增指标只 `register_counter` 不改 `render()`、`MetricSeries.render_lines` 输出标准 Prometheus 格式（`# HELP`/`# TYPE`/值行）且四指标空表零值正确、同名幂等；④ `GovernanceError`/`GovernanceStage`（D3）落地、`hooks.governance` 六方法 async 骨架（D4）就位、`proxy.py` 两处 `await` 通过；⑤ **批次0/1 回归全绿**（2a 硬约束）。此门结果须给你确认，通过才进 2b
- [ ] 2a 基建前置（D2/D3/D4，**已落地，2b 开工前提**）：① `metrics.py` 重构为可注册 registry，新增指标只 `register_counter` 不改 `render()`（消除三路写冲突）；② `types.py` 增 `GovernanceError`/`GovernanceStage`（带 `stage`/`exc_type`/`log_fields()`，reason 禁带原文）作为 C1 异常载体；③ `hooks/governance.py` 六个钩子（`run`/`run_stream`/`guardrails_check`/`judge_egress`/`cache_lookup`/`cache_store`）**统一 `async def`** 并落好空实现骨架，`proxy.py` 改 `await`。三项均要求批次0/1 零回归
- [ ] 无旁路不变量补齐（C1）：`controllers/proxy.py` 两处治理钩子 `except` 的**事实 fail-open**（流式原样 yield / 非流式 `except: pass`）改为按 `GATEWAY_FAIL_MODE` 分流——closed（默认）非流式 502 不回原文、流式截断+`governance_error` 终止标记；open 放行+强告警；**两模式均打 `gateway_governance_error_total` 与 `governance_error` 日志（不含原文）**。钩子侧禁止自吞异常（scope-06），判定语义唯一权威在 scope-10。**本批唯一改动批次0 已验收异常行为的点**，仅限异常分支，正常路径与批次0/1 回归须全绿

- [ ] `GATEWAY_FAIL_MODE` 全局硬下限（D5）：三个消费点（`egress.decide` 上游不可达 / C1 治理钩子异常 / 07 `degrade.py` 引擎降级）共用同一开关；**closed 下子模块分级放行一律被压制**（`degrade.py` 低敏 fail-open 不生效，`degrade.py` 决策入口先读 `settings.fail_mode`），被压制的决策记 `gateway_degrade_suppressed_total`

### 批次3（04/05/09）
- [ ] 04：健康巡检拓扑矩阵；指标/Prometheus + 日志/Loki 双通道同源事件；告警分级
- [ ] 05：自愈闭环安全边界（自动处置不闯祸）
- [ ] 09：LLM golden signal（token 成本/P99/质量退化）业务级可观测

### 批次4（集成）
- [ ] 跨批次接口契约一致（网关 ↔ monitoring-agent ↔ shop-agent）
- [ ] 大纲"待补"补全：决策 why 段落、交叉引用锚点、简历要点

## 6. 范围边界（明确不做）

- **不做**：重新论证"为何独立网关""三服务边界"等大纲已定架构前提
- **不做**：K8s NetworkPolicy / service mesh 出口策略配置（属部署文档，仅代码层保证"不直连"）
- **不做**：OCI Always Free 等受约束环境部署适配（指向 `docs/oracle-cloud-deploy.md`）
- **不做**：超出 01~10 范围的新功能；单篇未覆盖处仅留钩子 + 记假设
- **成本/合规/脱敏/缓存**：批次0 仅预留钩子；完整实现分散在批次 1~2

## 7. 关键约束

- 技术栈：Python + FastAPI（与现有栈一致；除非极致吞吐才考虑 Go，本系列不触发）
- 安全：所有 LLM 调用经网关，业务不直连供应商；遵循 `llm-agent` rule
- 网关可移植性（适用服务：`apps/gateway/`）：一套代码三环境——K8s 内以 `gateway:8001` 作统一出口；裸跑时回退直连真实供应商（本地 Ollama / 托管云）。业务服务（`shop-agent` 等）在 K8s 内**必须**经网关、不得直连供应商
- 误差控制：批次间验收门阻断误差传播；偏离 scope 立即人工介入（工作流快速修复表）

## 7.1 跨组件可观测契约（监控数据流向，供批次3 的 04/05/09 落地）

核心原则（用户确认）：**所有组件平时把日志/指标/告警交给中枢全量收集；仅在出现异常时才触达 monitoring-agent**，agent 不被常态数据淹没。

### 数据流向
```
各组件 ──/metrics──► Prometheus(scrape 拉)      ─┐
各组件 ──日志──────► Loki(组件 push)            ─┤ 中枢全量收集（平时）
各组件 ──指标超阈──► Prometheus alert rule      ─┤
                                        │         │
                                        ▼         │
                                  Alertmanager ───┘
                                        │ 仅异常时
                                        ├─webhook 推──► monitoring-agent /ingest/alert（异常才触达）
                                        └─agent 查询──► 拉取告警
monitoring-agent ──查询────────────────► Prometheus / Loki（按需关联，非实时推）
```

### 契约要点
- **指标**：组件暴露 `/metrics`（Prometheus 格式），Prometheus 主动 scrape；agent 平时不接收，仅异常经 alert 触达。
- **日志**：组件日志统一进 **Loki**（组件 push 给 Loki，不经 agent）；agent 按需查 Loki 做"谁影响谁"关联。
- **告警/异常**：组件指标触发 Prometheus **alert rule → Alertmanager**；**仅异常时** Alertmanager 经 `webhook` 推给 `monitoring-agent /ingest/alert`，或 agent 主动拉取 Alertmanager。组件**不直接 push 异常给 agent**（应急可选 `/ingest/event` 通道，但**不经 LLM 网关 `/v1`**，不污染计量）。
- **去重与冷却（分层，优先依赖中枢）**：
  - **第一道（推送层，交由中枢，零代码）**：Prometheus `alert rule` 用 `for:` 子句（如 `for: 2m`）过滤瞬时抖动；Alertmanager 用 `group_by: [alertname, instance]` 聚合同类告警为一组，`group_interval`（如 5m）控制同组重复推送节奏，`repeat_interval`（如 4h）控制已发告警的再提醒节奏。agent 收到的 webhook **天然已是低频聚合后的告警**，不是每条原始抖动都敲门。Loki 异常经 Loki ruler / Grafana 告警汇入 Alertmanager，同样享受该去重冷却。
  - **第二道（LLM 调用层，agent 补轻量守卫，中枢做不到）**：Alertmanager 只管「推不推」，管不了「agent 收到后调不调 LLM」。agent 在 `/ingest/alert` 入口加极薄守卫——**指纹** = `hash(alertname + 主组件 + 根因候选)`（同根因不同告警名也能合并），**同指纹 `LLM_COOLDOWN`（默认 10m）内不重复调 LLM**，直接复用上次 RCA 结论或标记「已知、观察中」。自愈 act 阶段仍用分布式锁防重入（幂等），与 LLM 调用层解耦。
  - 结论：**90% 去重冷却由 Prometheus/Alertmanager/Loki 承担，agent 仅补指纹+冷却小守卫防 LLM 滥用**，不过度设计。
  - **日志异常**：Loki 本身不推送；经 **Loki ruler / Grafana 告警**基于日志查异常（如 `level="error"` 频率超阈）生成 alert → **汇到 Alertmanager** 再推 agent，与指标异常同出口。
  - **LLM 异常**（Langfuse：trace 失败 / token 超预算 / 质量退化，见 09 篇 golden signal）：Langfuse 暴露 **metrics → Prometheus alert rule → Alertmanager**；或 Langfuse webhook 直推 agent `/ingest/event`（应急通道）。**不经 Alertmanager 的 LLM 异常不直连 agent 主告警口**。
  - **数据库/缓存/order-service 等**：均经各自 **exporter → Prometheus → Alertmanager**（连接数/慢查询/命中率等），统一汇出。
  - **汇总**：指标 / 日志 / LLM 异常**全部先汇到 Alertmanager**，再统一 webhook 推给 agent `/ingest/alert`；仅支持 webhook 的组件（如 Langfuse）可走应急 `/ingest/event`。agent 只接 Alertmanager 一个主告警口 + 可选应急口。
- **存活探活**：agent 主动探各组件 `/health`（gateway 已有；shop-agent/order-service/数据库/缓存 需暴露），不过 `/v1`。
- **`/ingest/alert` 接口契约（批次3 落点，含 LLM 调用层守卫）**：
  - 入参：`{ alerts: [{ alertname, instance, severity(P0/P1/P2), summary, labels, startsAt }] }`（Alertmanager webhook 标准 payload）。
  - 处理流程：① 按 `group_by` 已聚合的组为单位进入 → ② 计算指纹 `hash(alertname + 主组件(instance 解析) + 根因候选)` → ③ 查冷却表：同指纹在 `LLM_COOLDOWN` 内则跳过 LLM、复用上次 RCA / 标「观察中」；否则调 LLM 做 RCA + 自愈决策，写入冷却表 → ④ 自愈 act 经分布式锁防重入（幂等）。
  - 配置项：`LLM_COOLDOWN`（默认 600s）、`ALERT_GROUP_INTERVAL`/`ALERT_REPEAT_INTERVAL`（仅文档化中枢侧默认值，agent 不重实现）。
  - `/ingest/event`（应急口）：接受 Langfuse 等 webhook 直推，走同款指纹+冷却守卫，不经 `/v1`、不污染计量。
- **存储边界**：原始指标/日志/告警存 Prometheus/Loki/Alertmanager，**agent 不重复存原始流**；agent 仅持久化**派生产物**（拓扑矩阵、告警分级状态、自愈审计、golden signal 快照），落本地轻量存储或回写 Prometheus 作为新指标。
- **不变量**：监控数据全走可观测管道，绝不混入 LLM 流量（呼应 01 §2 旁路原则）。

## 8. 未确认假设（文档未覆盖处，按现有代码惯例处理）

- 监控落点：`apps/monitoring-agent/` **已存在**（FastAPI 骨架 + `monitoring_agent/main.py`），批次3 复用现有服务，不新建、不并入 shop-agent
- 路由表/限流/计量存储形式按 `apps/gateway` 现有配置惯例，不新建配置体系
- 各篇交叉引用锚点若缺失，按大纲"待补"在批次4 统一补
- 若某篇大纲与 `apps/*` 现有代码冲突，以"不破坏现有可运行行为"为前提局部对齐，审查报告显式标注冲突点

## 9. 阶段编排（总控）

1. 阶段0 澄清 → 本总 scope（待你确认）
2. 批次0→4 按 §3 编排推进，每批：architecture-designer（如需要）→ python-coder 并行 → code-reviewer → test-generator → **你验收**
3. 批次4 终验后，输出跨篇集成报告 + 大纲"待补"补全清单
