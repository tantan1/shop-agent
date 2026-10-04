# 子 Scope · 批次2-06：合规护栏（网关这道闸 + 策略集中配置）

> 上游：`docs/platform-engineering/06-合规护栏.md`（跨平面处置总表为唯一权威）+ `07-脱敏引擎工程化.md`
> 性质：落地 06「卡哪里、做什么」的**网关这道闸**——前置脱敏 + 后置兜底 + 策略集中配置；引擎形态由 07 落地，本 scope 复用
> 下游消费者：`python-coder`、code-reviewer、test-generator、用户验收
> 安全等级：LLM 安全相关（确认门第4条）

---

## 1. 目标（一句话可验证）

在 gateway 这道闸落地 06 的合规护栏：**前置脱敏（请求进模型前）+ 后置兜底（响应回写前）**，由 07 的同一脱敏引擎驱动（跨平面总表唯一权威），护栏策略（哪些字段/话题拦截）集中在网关配置（业务零改动）。硬违规（明文身份证/违禁词）网关直接拦；模糊边界留钩子接人在回路（本批不接人工）。

## 2. 设计依据（必读）

- `docs/platform-engineering/06-合规护栏.md` §3（PII 脱敏 vs Guardrails 两类）、§4（前置+后置+跨平面延伸）、§5（策略集中）、§6（人在回路分工：硬违规直拦，模糊升舱）
- `docs/platform-engineering/07-脱敏引擎工程化.md`（引擎由 scope-07-pii 落地，本 scope 仅调用）
- 现有代码：`apps/gateway/gateway/hooks/governance.py`（`run` ingress / `run` egress / `run_stream`；本批接入 07 引擎 + Guardrails 拦截）、`types.py`（`Verdict`）、`controllers/proxy.py`（ingress 在 gate 后、egress 经 hooks）

### 2.1 为何必须有 egress 兜底（合规依据）

法规/平台政策不写"回复不能带敏感信息"这一句话，但多层要求共同指向同一工程动作——**响应不得含明文 PII，egress 必须兜底脱敏**：

- **法规要求去标识化 + 最小必要**：《个人信息保护法》第 6 条（最小必要）、第 9 条（采取去标识化等技术措施确保安全）、第 23/38 条（对外提供合规）；GDPR 第 5 条（数据最小化、完整性与保密性）、第 32 条（适当技术措施）；PCI-DSS（明文卡号不得出现在响应/日志）。模型原样回显用户 PII，等于主动把敏感信息二次扩散（进浏览器历史、前端日志、截图），违反"防泄露/去标识化"义务。
- **平台/行业政策禁止回显**：OpenAI / Anthropic 等使用政策禁止日志落明文 PII、要求输出侧不无意泄露；支付/医疗等行业规范要求响应报文 PII 掩码。
- **工程常识：回显零价值却放大风险**：用户自己刚发的 PII 原样发回对其零价值，但会扩大泄露面（前端缓存、第三方分析、截图留存）。业界共识是即便用户自愿提供，回复也应脱敏（不重复或掩码为 `138****8888`）。
- **ingress 不够**：前置脱敏擦的是 prompt（保护上游/模型侧），但模型已在上下文"记住"该信息，回答时仍会复读（回声泄露）——故必须 egress 兜底（保护用户侧/客户端），双闸缺一不可。典型场景：客服确认"手机号 138xxxx 已查询"、办理"身份证 110xxxx 已提交"、多轮摘要复述、工具结果回显。

> 📝 **blog 取材标记**：本节（§2.1 egress 兜底的合规依据 + 回声泄露场景）是后续写《合规护栏》blog 的核心素材——可直接展开为"为什么 ingress 不够、必须 egress 兜底"一节。素材取自本 scope，勿回写大纲（大纲 06 §4 已含论点级"后置兜底防复读"）。选题对比（自研钩子 vs LangChain/LiteLLM 中间件）另见大纲 06 待补项。

## 3. 子 scope 边界

**做**：
- 复用 07 的 `PiiEngine` 做前置脱敏 + 后置兜底（06 §4）：`hooks/governance.py` 的 `run("ingress", payload)` 调 `engine.redact` 擦 prompt；`run("egress", body)` 调 `engine.redact` 兜底模型复读敏感信息。流式 `run_stream` 逐块掩码（沿用批次0 SSE 钩子）。
- Guardrails（内容拦截，06 §3）：在 `hooks/governance.py` 增 **`async guardrails_check(stage: str, text: str) -> Verdict`**（D4：钩子统一 async，签名由 2a 骨架定死，本 scope 只填方法体）——确定性违禁词/话题字典（env `GUARDRAILS_BLOCKLIST` 配置）；命中即 `Verdict.deny`，非流式 4xx / 流式**截断+终止标记**（措辞与 10 §3 统一：已发 chunk 不回滚，详见 scope-10「流式语义澄清」；与 10 judge_egress 同源 fail-closed，但职责不同：Guardrails 管"说什么不对"，注入检测管"谁在指挥"，引 10 §3）。
- 策略集中配置（06 §5）：`config.py` 增 `GUARDRAILS_BLOCKLIST`（逗号分隔违禁词）、`PII_ENABLED`（默认 true）。业务不感知。
- 人在回路（06 §6）：硬违规直拦已实现；模糊边界（置信度在阈值附近、命中 blocklist 变体但不足 deny）本批不接人工审批，但需**可观测、可被消费**——打 `human_review` 标记（不阻断，放行）。三通道落点（E3，已确认）：
  - **响应头**：`X-Guardrails-Review: <rule_id>`（调用方/下游可读、可做二次处理）
  - **结构化日志**：`guardrails_review` 事件，字段 `{rule_id, text_hash, stage, ts}`（text_hash 为命中文本 sha256 前缀，**不带原文**，避免标记本身成为泄露面）；供审计追溯与离线人工复核队列
  - **Metrics**：`gateway_guardrails_review_total{rule, stage}` 计数器（监控大盘、阈值告警：review 量突增可能预示新型攻击潮）
  - 约束：标记**不阻断**（与 deny 区分：deny=4xx/截断；review=放行+打标）；标记内容**不含命中原文**（只 rule_id + text_hash）；本批**不接人工审核队列**（只产出标记，消费端留后续），但标记须"可被消费"（头+日志+metrics 三处都给），否则等于没做。
- 治理钩子异常按 fail_mode 分流（C1，已确认；**判定与降级语义以 scope-10 为唯一权威**，本 scope 只承接钩子侧责任）：`hooks/governance.py` 的 `run`/`run_stream`/`guardrails_check` 抛异常时，不得让 `controllers/proxy.py` 静默透传未治理内容。钩子侧要求：① 异常**向上抛**，不在钩子内 `except: return payload` 自吞（自吞等于把 fail-open 藏进治理层，proxy 的 fail_mode 就管不到）；② 异常携带 `stage`（ingress/egress/stream）供 proxy 打点分流；③ 脱敏引擎自身的降级（07 `degrade.py` 的 healthy/degraded/down）属**引擎内可控降级**，走 07 的分级策略，不算此处的"钩子异常"——只有引擎降级也兜不住的真异常才上抛。
- **异常载体类型（D3，已确认，已落地 `types.py`）**：`GovernanceError(Exception)` + `GovernanceStage(str, Enum)`（`ingress`/`egress`/`stream`）。
  - 构造：`GovernanceError(stage, cause=原始异常, reason="非敏感定位信息")`；自动派生 `exc_type = type(cause).__name__`（根因类型名，供打点区分 `RulesLoadError` / `re.error` / `KeyError`）。
  - `log_fields() -> dict` 返回 `{stage, exc_type, reason}`，proxy 直接展开进结构化日志。
  - **安全约束（硬）**：`reason` 只放规则 id / 阶段名等非敏感定位信息，**禁止塞入命中的 prompt 或响应原文**——异常必进日志，带原文等于新开泄露面（与 E3 同纪律）。
  - 钩子侧统一 `raise GovernanceError(stage, cause=e) from e`，proxy 侧只 `except GovernanceError`（**不再用宽 `except Exception`**，避免误吞编程错误）。

**不做**：
- 跨平面其他落点（缓存/向量/trace/日志/审计/告警）的实际脱敏调用：本批只落地「网关这道闸」，其余落点在对应篇批次3 接线（08 会接缓存写入门禁）。
- 语义层护栏（模型判定模糊内容）：留钩子，不启用。
- 人在回路人工审批通道：仅留标记，不接人工系统。

## 4. 模块落点

| 文件 | 动作 |
|---|---|
| `types.py`（2a 落地） | **【D3 已完成】`GovernanceError(Exception)` + `GovernanceStage(str, Enum)`**：携带 `stage`/`exc_type`/`reason` 与 `log_fields()`；本 scope 复用，不重复定义 |
| `hooks/governance.py` | `run`/`run_stream` 接入 07 `PiiEngine`（前置+后置）；增 `guardrails_check`（确定性违禁词）；增 `human_review` 标记（三通道：响应头 `X-Guardrails-Review` + 结构化日志 `guardrails_review` + metrics `gateway_guardrails_review_total`，不阻断）；**异常一律 `raise GovernanceError(stage, cause=e) from e` 上抛，禁止钩子内自吞（C1/D3）** |
| `controllers/proxy.py` | ingress：gate(注入) → guardrails → redact(脱敏) → cache_lookup(命中短路) → 模型；egress：judge_egress(注入) → guardrails(输出) → redact(兜底脱敏)，治理通过后才 cache_store；**钩子异常按 fail_mode 分流（C1，实现细则见 scope-10）** |
| `config.py` | 增 `GUARDRAILS_BLOCKLIST`、`PII_ENABLED` |
| `metrics.py` | **不改 `render()`**——经 2a registry（D2）`register_counter("gateway_guardrails_review_total", ..., ("rule","stage"))`。注册语句写在 `hooks/governance.py` 模块级，避免与 08/10 争抢 `metrics.py` |
| `pii/`（07） | 脱敏引擎，本 scope 调用 |

## 5. 接口契约（验收基准）

- 前置：含明文身份证的 prompt 进模型前被掩码（PII_ENABLED=true）。
- 后置：模型输出复读敏感信息被兜底掩码。
- Guardrails：含违禁词 → deny（非流式 4xx / 流式截断+终止标记，不回滚已发内容）。
- Guardrails 模糊边界：置信度边缘命中 → 放行 + 打 `human_review` 标记（响应头 `X-Guardrails-Review` + 日志 `guardrails_review` 事件含 rule_id/text_hash + metrics 计数 +1），**不阻断、不带原文**。
- 顺序（引 10 §3 + 06 §3，统一规则先防攻击后护隐私；ingress 含缓存查询位 B2）：ingress：注入检测 → guardrails → 脱敏 → 缓存查询(命中短路) → 模型；egress：judge_egress(注入) → guardrails(输出) → redact(兜底脱敏)，治理通过后才 cache_store。出向两条检测（注入/guardrails）均基于未脱敏原文，脱敏兜底放最后；缓存查询在脱敏后、模型前，命中不重跑 egress 治理（08 B3）。
- 策略 env 配置，业务零改动。
- 钩子异常（C1）：`hooks` 抛异常不被自吞，proxy 按 `GATEWAY_FAIL_MODE` 分流（closed 拒发 / open 放行+告警，两者均打点）；引擎级可控降级走 07 `degrade.py`，不混入此路径。

## 6. 验收清单

- [ ] 前置脱敏：明文 PII 进模型前掩码（复用 07 引擎）
- [ ] 后置兜底：模型输出复读 PII 被掩码
- [ ] Guardrails：确定性违禁词命中 deny（非流式 4xx / 流式截断+终止标记）
- [ ] Guardrails 模糊边界：`human_review` 三通道生效（响应头 `X-Guardrails-Review` + 日志 `guardrails_review` + metrics `gateway_guardrails_review_total`），放行不阻断、标记不含原文
- [ ] 顺序不变量（B2）：ingress 注入检测→guardrails→脱敏→缓存查询(命中短路)→模型；egress judge_egress→guardrails→redact（脱敏兜底最后；两条出向检测均基于未脱敏原文；缓存命中不重跑 egress 治理，B3）
- [ ] 策略 env 集中配置（PII_ENABLED / GUARDRAILS_BLOCKLIST），业务零改动
- [ ] 人在回路仅留标记钩子（三通道可观测），不接人工审批
- [ ] C1：治理钩子异常上抛不自吞（带 stage），由 proxy 按 fail_mode 分流（细则/验收见 scope-10）
- [x] D3：`GovernanceError` + `GovernanceStage` 已在 `types.py` 定义（含 `log_fields()`，reason 禁带原文）
- [ ] D3 落地校验：钩子内无 `except: return payload` 自吞；proxy 侧 `except GovernanceError` 而非宽 `except Exception`
