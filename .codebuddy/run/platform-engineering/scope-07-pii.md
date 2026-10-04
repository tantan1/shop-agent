# 子 Scope · 批次2-07：脱敏引擎工程化（共享薄引擎）

> 上游：`docs/platform-engineering/07-脱敏引擎工程化.md` + `06-合规护栏.md`（跨平面处置总表为唯一权威）
> 性质：落地 07「规则即数据 + 薄引擎 + 三级来源 + 双层级降级」的**网关进程内薄引擎**；语义层（模型判定）留钩子
> 下游消费者：批次2 的 06/08/10 均复用本引擎；`python-coder`、code-reviewer、test-generator、用户验收
> 安全等级：LLM 安全相关（确认门第4条）——本 scope 仅落地「确定性层永远在线 + 双层级降级骨架」，语义层不全网启用

---

## 1. 目标（一句话可验证）

在 `apps/gateway/` 落地**进程内薄脱敏引擎**：规则即数据（YAML 文件，语言中立），本地快照 + 异步刷新（热路径零联网），三级来源（磁盘快照→构建基线→异步刷新），双层级降级（确定性正则/字典层永远在线；语义层可降级）。本批次只实现**确定性层**，语义层（模型判定）仅留接口钩子，不默认启用。

## 2. 设计依据（必读）

- `docs/platform-engineering/07-脱敏引擎工程化.md` §1（规则即数据+薄引擎）、§2（本地缓存+异步刷新）、§3（三级来源）、§4（双层级降级）
- `docs/platform-engineering/06-合规护栏.md` §2（跨平面处置总表：网关/缓存/向量/trace/日志/审计/告警共用同一引擎）
- 现有代码：`apps/gateway/gateway/hooks/governance.py`（`run`/`run_stream` 已 passthrough，批次2 注册脱敏）、`types.py`（`Verdict`）

## 3. 子 scope 边界

**做**：
- 新增 `pii/`（包）：
  - `engine.py`：`PiiEngine` 薄引擎。`__init__(rules_path)` 加载 YAML 规则（正则列表 + 字典列表）；`redact(text: str) -> tuple[str, list[str]]` 返回（脱敏后文本, 命中类型列表）；确定性层用 `re.sub` 掩码（身份证/手机号/邮箱/银行卡等预置正则）。
  - **掩码风格（统一，E1）：保留前后缀的部分掩码**，非全掩 `***`。规则即数据——每条规则在 YAML 中自带 `mask: {keep_prefix: N, keep_suffix: M}`，引擎按之生成掩码（不足以保留时全掩兜底）。示例：手机号 `13800138000` → `138****8000`；邮箱 `abc@x.com` → `a**@x.com`；身份证 `110101199001011234` → `1101**********1234`；银行卡保留后 4 位。理由：① 客服场景需让用户辨认"是哪个号"，全掩不可用；② 保留长度/格式信息便于人工核对与日志排查；③ 与 06 §2.1 举例 `138****8888` 对齐。**掩码位数不得泄露原值**（中间段一律定长或按实际长度填充，由规则声明）。
- 新增 `rules/`（**公共包，非 PII 专属**）：
  - `store.py`：`RulesStore` 三级来源加载（**通用规则存储层，PII 与注入检测共用**）：① 磁盘快照（`<name>.snapshot.json`，运行期刷新成功即落盘 last-known-good）→ ② 构建基线（包内 `<name>.baseline.yaml`）→ ③ 中央源（env 指定 URL，仅异步刷新，本批留空不实现拉取，仅结构预留）。**绝不允许零规则启动**（基线随包存在，结构上排除 fail-open 静默）。
  - 构造签名按**规则集名**参数化：`RulesStore(name: str, baseline_path: str, snapshot_path: str, remote_url: str | None)`，使 `pii` / `injection` 两套规则各持一个实例，互不干扰。
  - 归属说明：本 scope 负责**建立** `rules/` 公共包（因 07 最先需要），但它**不属于 PII 语义**；scope-10 注入检测直接复用同一 `RulesStore`，不重复实现存储层，也不 import `pii/`。
  - `degrade.py`：引擎健康度 → 降级决策。`health` ∈ {healthy, degraded, down}；down 时按路由敏感度分级（高敏 fail-closed + 告警，低敏 fail-open + 强告警）。本批 `health` 恒 healthy（确定性层不挂），降级阶梯结构预留。
    - **与全局 `GATEWAY_FAIL_MODE` 的优先级（D5，已确认）**：两者是**不同层级、不同触发条件**的机制，必须显式定序，否则出现"全局要求 closed，但引擎分级悄悄放行低敏流量"的旁路。
      | 机制 | 层级 | 触发条件 | 决策粒度 |
      |---|---|---|---|
      | `GATEWAY_FAIL_MODE` | 网关全局（01 §5） | 上游不可达（`decide`）/ 治理钩子异常（C1） | 进程级统一 |
      | `degrade.py` | 脱敏引擎内（07 §4） | 引擎 `health=down` | 按路由敏感度分级 |
      - **规则：`GATEWAY_FAIL_MODE=closed` 是硬下限，`degrade.py` 的低敏 fail-open 不生效**。即 closed 模式下引擎 down → 一律拒绝（高敏低敏都拒），分级策略被全局开关压制；只有 `GATEWAY_FAIL_MODE=open` 时才允许 `degrade.py` 按敏感度分级放行。
      - 理由：全局开关是运维对"治理完整性 vs 可用性"的**总意图声明**，粒度更粗但优先级更高；若允许子模块分级绕过它，运维设 closed 却仍有流量绕过治理外发，`fail_mode` 就名存实亡（与 C1 要解决的问题同源——不能有任何一条路径悄悄 fail-open）。
      - 实现纪律：`degrade.py` 的决策函数**必须先读 `settings.fail_mode`**，closed 直接返回"拒绝"，不进入敏感度分支；不得让调用方自行判断（判断散落即形同虚设）。
      - 打点：被全局 closed 压制掉的分级决策要留痕——记 `gateway_degrade_suppressed_total{sensitivity}`（经 D2 registry 注册），便于运维发现"我设了 closed，所以低敏也被拒了"，而非以为是 bug。
      - 边界重申（承接 06）：引擎 `health=degraded/down` 属**引擎内可控降级**，走本节分级 + 全局压制；只有引擎降级也兜不住的**真异常**才包装为 `GovernanceError` 上抛，交 C1 的 fail_mode 分流。两条路径不重叠。
  - `rules.example.yaml`：预置正则规则示例（身份证/手机号/邮箱），作为配置示意。
- **`metrics.py` registry 化（D2，已确认，属 2a 前置基建）**：现有 `metrics.py` 是手写字符串拼接的固定四指标，`render()` 内联所有 HELP/TYPE 与样本行。批次2b 三路（06 `gateway_guardrails_review_total`、08 `gateway_cache_hits_total`、10 `gateway_governance_error_total`）**都要新增指标 = 都要改同一个 `render()`，并行必冲突**。故在 2a 一次性重构为可注册 registry：
  - `register_counter(name, help_text, labelnames) -> MetricSeries`：幂等（同名同标签返回同一实例，模块重复导入安全）、标签任意、`Lock` 保护。
  - `MetricSeries.inc(amount=1, labels={...})` / `.get(labels)` / `.clear()`；未知标签名直接 `ValueError`（防拼写错误静默产生新时间序列）。
  - `render()` 只遍历注册表，**新增指标零改动 `render()`** —— 2b 三路各自 `register_counter` 即可，互不触碰同一函数。
  - **兼容性硬约束**：既有对外函数（`record`/`tenant_tokens`/`mark_rate_limited`/`mark_budget_exceeded`/`estimate_tokens`/`render`）签名与语义不变；空注册表仍输出 `{tenant="default"} 0` 零值行（保证大盘不断线），批次0/1 验收行为零回归。
  - 不引入 `prometheus_client`（保持零依赖、热路径无额外开销）；批次3 若接真实客户端，只换 `MetricSeries` 内部实现，`register_counter`/`inc` 调用方不改。
- **治理钩子统一 async（D4，已确认，已落地骨架）**：`hooks/governance.py` 全部方法一律 `async def`，调用方一律 `await`。原 `run` 为同步、`run_stream` 为 `async` 的不对称已消除。
  - 定型签名（2a 落骨架，2b 填方法体）：
    | 方法 | 签名 | 归属 |
    |---|---|---|
    | `run` | `async run(stage: str, payload: dict) -> dict` | 07 脱敏 |
    | `run_stream` | `async run_stream(stage: str, chunk: dict) -> dict` | 07 脱敏（流式逐块） |
    | `guardrails_check` | `async guardrails_check(stage: str, text: str) -> Verdict` | 06 |
    | `judge_egress` | `async judge_egress(response: dict) -> Verdict` | 10（②出向） |
    | `cache_lookup` | `async cache_lookup(stage: str, payload: dict) -> str \| None` | 08 |
    | `cache_store` | `async cache_store(stage: str, payload: dict, answer: str) -> None` | 08 |
  - 理由：① B2 顺序要求六者串成一条链，同步异步混用在流式路径（`judge_egress` 需 await）必出错；② 任一实现将来接 sidecar/本地服务（Presidio、分类器）就必然变异步，届时改签名波及全部调用方；③ 同步实现包在 `async def` 里零成本。故一次定死，**2b 三路不得再各自决定同步性**。
  - **硬约束**：`async def` 内**禁止阻塞 IO**（规则文件走启动期加载 + 后台刷新，不在热路径同步读盘），否则卡事件循环——这与 07「热路径零联网」同源纪律。
  - 回归影响：`controllers/proxy.py` 两处 `hooks.run(...)` 改为 `await hooks.run(...)`（ingress L74 / egress L191），批次0 SSE 逐块钩子行为不变。
- `hook` 集成：在 `hooks/governance.py` 的 `run`/`run_stream` 中调用 `PiiEngine.redact`——**前置（ingress）脱敏 prompt，后置（egress）兜底掩码模型复读**（引 06 §4 前置+后置）。流式逐块：命中即就地掩码，跨块不完整 PII 返回原 chunk 打标（沿用现有 run_stream 语义）。

**不做**：
- 语义层（模型判定模糊 PII）：仅留**统一接口钩子**，不实现、不默认启用（防引入外部依赖 SPOF）。本批 `PiiEngine` 是确定性层实现，但所有实现（含未来 Presidio）须满足同一 `Redactor` 接口契约：**`redact(text: str) -> tuple[str, list[str]]`**（返回脱敏文本 + 命中类型列表），且 `redact` 热路径零联网、可返回空命中列表。调用方（`hooks/governance.run/run_stream`）只依赖该接口，不依赖具体实现类——**未来切换到 Presidio 后端（本地 sidecar 封装为同一接口）时调用方零改动**。
- 跨语言 OTel Collector sidecar：本批全 Python，进程内薄引擎即可（07 §1 单语言路径）。
- 中央源实时拉取/版本化灰度推送：仅结构预留 `PII_RULES_URL`，不实现。
- 跨平面落点（缓存/向量/trace/日志/审计/告警）的实际调用：本批只落地「网关这道闸」+ 暴露引擎供其他落点复用；其余落点接线在对应篇批次3（04/05/09）。
- **禁止引入 LangChain PII 中间件 / 触网托管 PII API**：前者是语义层实现（与"本批只做确定性层"冲突，且热路径引外部 LLM 调用语 SPOF）；后者触网破坏"热路径零联网"。确定性层引擎骨架自研、零依赖；预置正则规则**直接采用 Presidio 等开源项目的成熟正则**（抄入 `pii/pii.baseline.yaml`，不自手搓），规则即数据、社区验证优先。未来语义层钩子真要做时，接本地 Presidio sidecar 封装为同一 `Redactor` 接口（AnalyzerEngine + AnonymizerEngine），网关钩子调用、可降级、不触网；**Presidio 仅扩识别能力，其识别器配置须纳入三级来源（基线随包 + 快照兜底）管理，禁止无配置空跑（fail-open 静默放行）**——冷启动防护（绝不零规则启动）由自研三级来源纪律承担，Presidio 不解决启动纪律，反而使启动更重，故须复用同一套纪律。

## 4. 模块落点

| 文件 | 动作 |
|---|---|
| `rules/__init__.py` | 【新增】**公共**规则包标识（非 PII 专属，10 复用） |
| `rules/store.py` | 【新增】`RulesStore` 三级来源（按规则集名参数化，pii/injection 各持一实例） |
| `rules/injection.baseline.yaml` / `rules/injection.snapshot.json` | 【由 scope-10 新增】注入闸随包基线正则 + 运行期快照（归 `rules/` 公共包，与 `pii/pii.baseline.yaml` 对称；本 scope 只建 `RulesStore` 存储层，不新增注入规则文件） |
| `pii/__init__.py` | 【新增】包标识；暴露 `Redactor` 协议类型 |
| `pii/engine.py` | 【新增】`PiiEngine.redact()` 确定性层（消费 `rules.store.RulesStore`） |
| `pii/degrade.py` | 【新增】降级阶梯（结构预留，本批 healthy） |
| `pii/pii.baseline.yaml` | 【新增】随包基线正则（**生产默认**，非 example；绝不零规则启动） |
| `pii/pii.snapshot.json` | 【运行期生成】刷新成功落盘 last-known-good |
| `types.py` | **【D3 已完成，2a 交付】增 `GovernanceError(Exception)` + `GovernanceStage(str, Enum)`**（`stage`/`exc_type`/`reason` + `log_fields()`；reason 禁带原文）。06/10 复用，不重复定义 |
| `hooks/governance.py` | `run`/`run_stream` 接入 `PiiEngine`（前置+后置兜底）；**2a 同时落好 2b 三路方法的 async 空实现骨架（D4）**：`guardrails_check` / `judge_egress` / `cache_lookup` / `cache_store`，2b 只填方法体不动类头 |
| `config.py` | 增 `PII_RULES_PATH`（默认 `pii/pii.baseline.yaml`）、`PII_RULES_URL`（默认空） |
| `metrics.py` | **【D2 已完成，2a 前置】重构为可注册 registry**：`register_counter(name, help, labelnames) -> MetricSeries`（幂等注册、按标签分桶、线程安全），`render()` 遍历注册表渲染，**新增指标不再改动 `render()` 本体**。既有四指标（requests/tokens/rate_limited/budget_exceeded）已迁移，对外函数签名（`record`/`tenant_tokens`/`mark_*`/`estimate_tokens`/`render`）与空表零值输出保持不变 |

## 5. 接口契约（验收基准）

- `PiiEngine.redact("手机13800138000邮箱abc@x.com") -> ("手机138****8000邮箱a**@x.com", ["phone","email"])`（确定性层命中即**保留前后缀**掩码，掩码参数由规则声明，见 §3）。
- **统一 `Redactor` 接口（可替换性基准）**：所有脱敏实现（确定性层 `PiiEngine`、未来语义层 Presidio 封装）须满足 `redact(text: str) -> tuple[str, list[str]]`，且热路径零联网、可返回空命中。调用方仅依赖该接口签名，`pii/__init__.py` 暴露该协议类型；**将来直接以 Presidio 后端实现同一接口即可替换，网关 hook 与 proxy 不改**。
- 热路径零联网：引擎加载后 `redact` 不触网。
- 启动：无规则文件时回退基线 YAML，绝不零规则启动（fail-open 静默被排除）。
- **降级优先级（D5）**：`GATEWAY_FAIL_MODE=closed` + 引擎 `health=down` → **低敏路由也拒绝**（全局硬下限压制分级），并记 `gateway_degrade_suppressed_total`；`GATEWAY_FAIL_MODE=open` + `health=down` → 才按敏感度分级（高敏拒 + 告警 / 低敏放行 + 强告警）。`degrade.py` 内部先判全局开关再进分支。
- 钩子生效：`controllers/proxy.py` 的 ingress/egress 经 `hooks.run`/`run_stream` 真实脱敏（非 passthrough），且流式逐块生效（沿用批次0 SSE 行解析）。

## 6. 验收清单

- [ ] `PiiEngine` 确定性层对预置正则命中即掩码，返回命中类型列表
- [ ] 三级来源：快照→基线→（预留）中央源；无规则文件回退基线，不零规则启动
- [ ] 热路径 `redact` 零联网（加载后无网络调用）
- [ ] `hooks/governance.run/run_stream` 接入引擎：前置脱敏 prompt + 后置兜底模型输出
- [ ] 流式逐块脱敏生效（跨块不完整 PII 打标不误拦）
- [ ] 降级阶梯结构预留（本批 health=healthy，确定性层永远在线）
- [ ] **D5 降级优先级**：`degrade.py` 决策入口先读 `settings.fail_mode`；closed 下 `health=down` 低敏路由**也拒绝**（分级被压制）且 `gateway_degrade_suppressed_total` +1；open 下才走敏感度分级
- [ ] 语义层仅留钩子，不默认启用（不引入外部依赖）
- [x] **D2 `metrics.py` registry 化**：`register_counter` 可注册、`render()` 遍历渲染、新增指标不改 `render()`；既有四指标输出与空表零值行与重构前一致（批次0/1 零回归）
- [x] **D3 `GovernanceError` / `GovernanceStage` 落 `types.py`**（含 `log_fields()`，reason 禁带原文）
- [x] **D4 钩子统一 async**：六个方法全 `async def` 骨架就位，`proxy.py` 两处改 `await`，批次0 SSE 钩子回归通过
- [ ] D4 落地校验：2b 填充方法体时 `async def` 内无阻塞 IO（规则加载在启动期/后台，不在热路径读盘）
- [ ] **2a 独立验收门（G2，★ 2b 开工硬卡）**：以下全过才允许 06/08/10 并行开工——① `Redactor.redact` 接口契约冻结（签名 `redact(text: str) -> tuple[str, list[str]]`，入参脱敏文本、出参 (脱敏后文本, 命中类型列表)、异常类型），2b 只依赖签名不依赖实现细节；② `RulesStore` 三级来源（基线 YAML / 远程 URL / 本地覆盖）加载与 fallback 跑通，**无规则文件时回退基线（绝不零规则启动）**；③ metrics registry（D2）：新增指标只 `register_counter`、不改 `render()`、`MetricSeries.render_lines` 输出标准 Prometheus 格式（`# HELP`/`# TYPE`/值行）且四指标空表零值正确、同名幂等；④ `GovernanceError`/`GovernanceStage`（D3）落地、`hooks.governance` 六方法 async 骨架（D4）就位、`proxy.py` 两处 `await` 通过；⑤ **批次0/1 回归全绿**。此门结果须交你确认，通过才进 2b
