# 子 Scope · 批次2-10：Agent 注入检测三道防线

> 上游：`docs/platform-engineering/10-Agent注入检测-LLM网关三道防线.md` + `06-合规护栏.md`（编排顺序：先注入检测后脱敏）
> 性质：落地 10「输入/输出/工具参数三道闸各守其门」到 gateway 链路；确定性检测先上，语义层留钩子
> 下游消费者：`python-coder`、code-reviewer、test-generator、用户验收
> 安全等级：LLM 安全相关（确认门第4条）——三道闸落点 + 编排顺序需人审

---

## 1. 目标（一句话可验证）

在 `apps/gateway/` 落地 10 篇三道注入检测闸，按「先注入检测、后 PII 脱敏」的纵深顺序编排（引 10 §3）。**三道防线编号体系（D1，已确认，全仓唯一权威）**：

| 编号 | 名称 | 位置 | 落点 |
|---|---|---|---|
| **①** | 入向 prompt 检测 | pre-LLM，ingress 首位 | `gate.check(payload)` |
| **②** | 出向 judge_egress | post-LLM，egress 首位 | `hooks.judge_egress(response)` |
| **③** | 工具参数检测 | ingress 阶段，随 ① 一并执行 | `gate.check` 扫 `tool_calls[].function.arguments` |

确定性检测（正则/字典已知攻击模式）先上，语义层留钩子。

**编号纪律**：`controllers/proxy.py` 原注释把 ③ 用于「出向 fail 开关」，与本表冲突——fail 开关**不是注入防线**（它是上游可达性/降级机制，属 01 §5），本批须改注释去掉其 ③ 编号，避免与三道闸编号抢位。全仓（scope-06/07/08、`proxy.py`、`architecture.md`）引用三道防线编号一律以本表为准。

## 2. 设计依据（必读）

- `docs/platform-engineering/10-Agent注入检测-LLM网关三道防线.md` §2（三道防线各守其门）、§3（与 06 编排顺序：先注入后脱敏）、§4（硬注入拦截 vs 模糊升级）、§5（降级：高敏 fail-closed/低敏 fail-open）
- 现有代码：`apps/gateway/gateway/hooks/injection.py`（`InjectionGate` 协议 + `PassthroughGate`，本批替换为真实检测）、`hooks/governance.py`（脱敏在后）、`types.py`（`Verdict`）、`controllers/proxy.py`（①入向在 gate.check，egress 经 hooks.run("egress")）、`architecture.md` §6（已定义 `judge_egress` 占位）

## 3. 子 scope 边界

**做**：
- `hooks/injection.py`：实现 `RegexInjectionGate`（确定性层）：内置已知攻击模式正则（jailbreak/角色覆盖/`ignore previous instructions`/`[SYSTEM]` 注入等）。`check(payload) -> Verdict`：扫描 `messages[].content` 命中即 `Verdict.deny(reason="injection: <pattern>")`。**规则加载复用公共包 `rules/store.py` 的 `RulesStore` 三级来源**（由 07/2a 建立，注入侧持 `injection` 独立实例，**不 import `pii/`**；基线随包 `rules/injection.baseline.yaml`（与 `pii/pii.baseline.yaml` 对称，归 `rules/` 公共包，非独立 `injection/` 目录）→ 磁盘快照 `rules/injection.snapshot.json` 兜底 → 中央源仅预留 `INJECTION_RULES_URL`）：绝不零规则启动（无文件回退基线，排除 fail-open 静默）；运行期异步刷新、失败回退快照。本批中央源不实现拉取，仅结构预留。
- 编排顺序（关键，引 10 §3）：`controllers/proxy.py` 现有顺序是 gate.check → hooks.run(ingress 脱敏)。完整 ingress 顺序（B2，已确认；**此处 1..5 是链路步序，与上表三道防线编号 ①②③ 不同命名空间，勿混用**）：**1. 注入检测（gate.check，含①+③）→ 2. guardrails → 3. 脱敏（hooks.run ingress）→ 4. 缓存查询（cache_lookup 命中即短路）→ 5. 模型**——即先判"是否恶意/违规"再擦"敏感信息"，避免脱敏后攻击 payload 变良性绕过检测（10 §3 明确顺序）；缓存查询放脱敏后、模型前，命中省模型调用且不重跑 egress 治理（08 B3）。现有顺序 gate 在 hooks 前已满足"先检测后脱敏"，确认即可，无需改序，仅把 gate 从 passthrough 换为真实检测；guardrails/cache_lookup 由 06/08 在对应位置接入。
  - **为何顺序不可倒（脱敏破坏检测的四类具体形态）**：
    1. **夹带打断连续匹配**：`ignore 13800138000 previous instructions` 原文本可被含容错的模式命中；若先脱敏成 `ignore <PHONE> previous instructions`，占位符长度/字符类变化会打断依赖字符距离或字符类的正则，命中率下降。
    2. **占位符破坏边界**：注入模式常锚定标记边界（如 `[SYSTEM]`、`### instruction`）。脱敏若把标记内夹带的邮箱/ID 替换为 `<EMAIL>`，标记串被切成两段，边界锚点失效。
    3. **字段名伪装**：攻击者把指令塞进看似 PII 的字段（`"email": "admin@x.com; ignore all rules"`），脱敏按字段整体替换为 `<EMAIL>`，指令连同载体一起被"洗掉"——检测看不到，但审计与后续人工复盘也丢失了攻击证据。
    4. **整段洗白**：高召回脱敏对长串数字/编码类内容做整体替换，编码型注入（base64/URL-encode 的指令）可能被判为敏感串而抹平，检测器拿到的是无害占位符。
  - 因此方向不变量固定为：**入向 = 先检测后脱敏；出向 = 先检测后脱敏**（统一规则：无论入向出向，均先防攻击、后护隐私）。
    - 入向：注入检测读 hooks ingress 之前的原始 prompt，判完 deny/allow 再脱敏发给模型。
    - 出向：judge_egress 读**未脱敏原文**（模型原始输出），先判复读注入 payload / 危险内容（`rm -rf`、SQL/XSS 等），deny 则拦；放行的内容再经脱敏洗掉 PII 后外发。
    - **出向为何也是先检测后脱敏（而非先脱敏后检测）**：① 检测必须基于原文——`rm -rf /`、SQL 注入等危险内容**本身不是 PII**，脱敏换不掉它，但精确匹配类规则（如 `DROP TABLE`）怕被任何改写文本的预处理切断，基于原文最稳；② 若先脱敏再检测，占位符 `<VERB>` 会切断 `DROP TABLE` 导致漏报，与入向"占位符破坏检测"同源；③ "护隐私"的脱敏只是把 PII 洗掉，它不解决代码/命令注入，绝不能承担安全拦截职责。
    - 检测器读到的必须是**未脱敏原文**这一点，需在实现上保证：`gate.check`（入向）与 `judge_egress`（出向）接收的 payload 都是脱敏之前的原始对象，不得复用脱敏后的引用。
- **③ 工具参数检测**：在 ingress 阶段，若 payload 含 `tools`/`tool_calls` 参数，对 `function.arguments` 字符串同样过 `RegexInjectionGate.check`（参数即文本，复用同一检测）。
- **② 出向 judge_egress**（已在 architecture §6 占位）：`hooks/governance.py` 增 `async judge_egress(response) -> Verdict`；检测模型输出是否复读注入 payload（如 `[SYSTEM] 新指令`）或危险内容（`rm -rf`/SQL/XSS）。**judge_egress 必须基于未脱敏原文判定**（见 §3 出向不变量），deny 则 `controllers/proxy.py` 非流式整块拦截 4xx、流式**截断 + 终止标记**（沿用用户确认的 fail-closed 通用做法）；放行内容随后经脱敏外发。本批 judge_egress 用同一 `RegexInjectionGate` 对输出原文检测。
  - **流式语义澄清（C2，能力边界）**：SSE 已发出的 chunk **无法回收**，"整块拦截"对流式不可达成。流式的 deny 语义严格定义为：**立即停止后续 yield + 补发 `data: {"choices":[{"finish_reason":"content_policy_violation"}]}` + `data: [DONE]`**，已发送内容不回滚。这是协议层的固有限制，非实现缺陷。因此：① 验收标准对流式只校验"截断发生 + 终止标记存在"，不校验"客户端未收到任何违规内容"；② 需要零泄露的高敏场景应关闭流式（走非流式整块判定），此权衡记为已知约束。
- 降级（引 10 §5）：`GATEWAY_FAIL_MODE=closed` 时命中即拦（fail-closed）；`open` 时放行+强告警。复用现有 `decide`/fail_mode 机制，不新造。
- **`GATEWAY_FAIL_MODE` 是全局硬下限（D5，已确认）**：本开关目前有三个消费点——① `hooks/egress.decide`（上游不可达）、② C1 治理钩子异常（见下）、③ 07 `degrade.py` 引擎降级分级。**closed 模式下任何子模块的分级放行策略均被压制**（07 `degrade.py` 的"低敏 fail-open"在 closed 下不生效，细则见 scope-07 §3）。理由与 C1 同源：不允许任何一条路径悄悄 fail-open，否则运维设 closed 却仍有未治理流量外发，开关名存实亡。
- **治理钩子异常按 fail_mode 分流（C1，已确认）**：`controllers/proxy.py` 现有两处 `except` 是**事实 fail-open**——流式 L169-174 钩子抛异常即 `yield` 原样 `data`（未经治理的原文直发客户端）、非流式 L185-190 `except: pass` 保留 `upstream.content` 原样返回。这与"治理不可旁路"不变量冲突：只要治理链任一环抛异常（正则栈溢出、规则文件损坏、JSON 结构异常），未经检测/未经脱敏的内容就静默外发，且**默认 `fail_mode=closed` 也拦不住**（异常路径不走 `decide`）。本批改为按 `GATEWAY_FAIL_MODE` 分流：
  - **closed（默认）**：治理链异常 = 判定不可用 = 拒绝外发。非流式返 `502 {"error":"governance unavailable"}`（不返 `upstream.content`）；流式**立即停止 yield + 补发终止标记**（复用 C2 定义的截断语义，`finish_reason` 用 `governance_error` 与 `content_policy_violation` 区分）。
  - **open**：放行原样内容 + **强告警**（结构化日志 `governance_error{stage, exc_type, ts}` + metrics `gateway_governance_error_total{stage, mode}`），即现有行为，但从"静默"升级为"有痕"。
  - 两种模式下**异常都必须打点**（closed 也计数），避免 fail-closed 掩盖规则文件损坏这类根因；日志**不带原文**（只 `exc_type` + `stage`，与 E3 同一纪律）。
  - 说明：这是本批**唯一改动批次0 已验收异常行为**的点。改动仅限异常分支，正常路径（钩子返回正常对象）行为不变，批次0 SSE 逐块钩子、批次1 fallback/限流回归须全绿。
  - 边界澄清：`json.loads` 失败（上游返回非 JSON body）与钩子内部异常**不同类**——前者本就不该由治理承担，按 open 语义原样透传 + 打点 `governance_error{stage, exc_type="JSONDecodeError"}`；后者（钩子已进入但抛错）严格按 fail_mode 分流。实现上两段 `try` 需拆开，不得共用一个宽 `except`。
  - **异常载体（D3，已确认，已落地 `types.py`）**：钩子统一抛 `GovernanceError(stage, cause=e)`（`stage` 取 `GovernanceStage.INGRESS/EGRESS/STREAM`），proxy 侧 **`except GovernanceError` 精确捕获，不用宽 `except Exception`**——宽捕获会连编程错误（`TypeError`/`AttributeError`）一起吞成"治理不可用"，掩盖真 bug。打点直接取 `err.stage` / `err.exc_type`，日志用 `err.log_fields()`（结构上就不含原文）。`json.loads` 那段仍用 `except json.JSONDecodeError` 精确捕获，与治理段物理分开。

**不做**：
- 语义层（模型判定模糊注入）：仅留**统一接口钩子**，不实现、不默认启用（防引入外部依赖 SPOF）。本批 `RegexInjectionGate` 是确定性层实现，但所有实现须满足同一 `InjectionGate` 接口契约：**`check(payload) -> Verdict`**（扫描 `messages[].content` 与 `tool_calls[].function.arguments` 等文本，命中即 `Verdict.deny(reason=...)`），且热路径零联网、可返回 allow。`judge_egress` 亦满足同一接口（对输出文本检测）。调用方（`controllers/proxy.py`、hooks）只依赖该接口，不依赖具体实现类——**未来切换到成熟检测（Rebuff / LLM Guard 的本地层 / 本地轻量分类器）作为同接口另一实现即可替换，网关编排与 fail_mode 逻辑零改动**。
- 禁止引入 LangChain 式注入中间件 / 触网托管检测 API（Rebuff/LLM Guard 的模型层、Azure/OpenAI moderation 等）：热路径引外部调用 = SPOF，且破坏统一 `Verdict`/`fail_mode` 契约。确定性层自研 `RegexInjectionGate` 零依赖；预置攻击模式正则**优先采用社区成熟规则集**（如已知 jailbreak/角色覆盖/`ignore previous instructions`/`[SYSTEM]` 注入模式）填入 `INJECTION_PATTERNS`，不自手搓。未来语义层真要做时接**本地**轻量分类器（sidecar、可降级、不触网），不触网托管 API。
- 输出侧跨轮 context 状态感知（10 §5「需感知之前标记」）：本批无状态检测，状态管理留后续（属批次3 可观测/05 自愈范畴，不在此）。
- 人在回路升级通道（10 §4 模糊 case 升舱人工）：本批硬注入直接拦，模糊升级留钩子不接人工。

## 4. 模块落点

| 文件 | 动作 |
|---|---|
| `hooks/injection.py` | `PassthroughGate` → `RegexInjectionGate`（确定性正则检测）；支持 messages + tool 参数；规则加载复用公共 `RulesStore` |
| `rules/store.py`（2a 建立） | 复用其 `RulesStore`（通用存储层，`injection` 实例；本 scope 不重复实现，也不依赖 `pii/`） |
| `rules/injection.baseline.yaml` | 【新增】随包基线攻击模式正则（归 `rules/` 公共包，与 `pii/pii.baseline.yaml` 对称；注入闸启动纪律，绝不零规则） |
| `rules/injection.snapshot.json` | 【运行期生成】刷新成功落盘 last-known-good（失败回退） |
| `hooks/governance.py` | 填充 **`async judge_egress(response: dict) -> Verdict`** 方法体（②出向注入/合规熔断，复用 RegexInjectionGate）；**签名与空实现骨架已由 2a 落好（D4），本路只填方法体**；异常按 D3 抛 `GovernanceError` |
| `controllers/proxy.py` | 非流式 judge_egress deny → 4xx；流式 deny → 停 yield + 终止标记；**治理钩子异常两处 `except` 按 fail_mode 分流（C1）**：closed→非流式 502 / 流式截断+`governance_error` 终止标记；open→原样放行+告警；两模式均打点。`json.loads` 失败与钩子异常分开 `try` |
| `config.py` | 增 `INJECTION_RULES_URL`（默认空，仅预留中央源） |
| `metrics.py` | **不改 `render()`**——经 2a registry（D2）`register_counter("gateway_governance_error_total", ..., ("stage","mode"))`（C1 异常打点）。注册语句写在 `hooks/governance.py` 模块级，避免与 06/08 争抢 `metrics.py` |
| `pii/`（07 落地） | 脱敏引擎在注入检测**之后**调用（入向：proxy 保证 gate 先于 hooks ingress；出向：judge_egress 先于脱敏外发） |

## 5. 接口契约（验收基准）

- ① 入向：含 `ignore previous instructions` 等已知模式的 prompt → 403 `Verdict.deny`，**先于**脱敏执行。
- ③ 工具参数：`tool_calls[].function.arguments` 含注入 → deny。
- ② 出向 judge_egress：模型输出复读注入/危险内容 → 非流式 4xx / 流式**截断+终止标记**（fail-closed；已发 chunk 不回滚，见 §3 流式语义澄清）；judge_egress 基于**未脱敏原文**判定，判定放行后再脱敏外发。
- 顺序不变量：入向、出向均为**先注入检测、后 PII 脱敏**（统一规则：先防攻击、后护隐私）。入向完整顺序（B2）：注入检测 → guardrails → 脱敏 → 缓存查询(命中短路) → 模型；缓存查询在脱敏后、模型前，命中不重跑 egress 治理（08 B3）。
- 反例回归：`ignore <含PII夹带> previous instructions`、`"email": "a@b.com; ignore all rules"` 两类 payload，在开启脱敏的完整链路下仍必须 deny（证明检测读到的是未脱敏原文）。
- `GATEWAY_FAIL_MODE=open` 时命中放行+告警（不破坏 fail 开关）。
- **治理钩子异常（C1）**：构造钩子抛异常的用例——`closed` 下非流式返 502 且 body 不含上游原文、流式停止 yield 且带 `governance_error` 终止标记；`open` 下原样放行且 `gateway_governance_error_total` +1、日志有 `governance_error` 事件（不含原文）。两模式计数器都必须 +1。

## 6. 验收清单

- [ ] **①入向** `RegexInjectionGate` 对已知模式 deny，先于脱敏执行（顺序不变量）
- [ ] 反例回归：PII 夹带 / 字段名伪装两类 payload 在完整链路（含脱敏）下仍 deny
- [ ] **③工具参数检测**：tool_calls arguments 命中即拦
- [ ] **②出向** `judge_egress` 非流式整块拦截 4xx / 流式**截断+终止标记**（fail-closed；不校验已发内容回滚）
- [ ] **D1 编号一致性**：`proxy.py` 注释、scope-06/07/08、`architecture.md` 中三道防线编号与 §1 表一致；fail 开关注释不再占用 ③
- [ ] 降级：fail-closed 默认拦；fail-open 放行+告警
- [ ] **C1 治理钩子异常按 fail_mode 分流**：closed→非流式 502 不回原文 / 流式截断+`governance_error` 终止标记；open→放行+告警；两模式均打 `gateway_governance_error_total` 与 `governance_error` 日志（不含原文）
- [ ] **C1 边界**：`json.loads` 失败（`except json.JSONDecodeError`）与钩子内部异常（`except GovernanceError`）分开处理，**全链路无宽 `except Exception`**（编程错误不得被吞成"治理不可用"）
- [ ] 语义层仅留钩子，不默认启用
- [ ] 现有 `/v1` 非流式/流式路径 + 批次0 SSE 钩子 + 批次1 fallback/限流不受影响（C1 仅改异常分支，正常路径行为不变）
