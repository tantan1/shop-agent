# 任务执行列表（提效 + 提质 5 项改造）

> 依据：`.codebuddy/plans/quality-efficiency-enhancements.md`
> 状态：⬜ 待办　🟨 进行中　✅ 完成　⏸️ 暂缓

## 执行原则

1. **按依赖顺序推进**，不按文档章节顺序（⑤ 依赖 ② 的沙箱）。
2. 每个任务的验收标准必须是**可测试的**（拒绝"可用/正确"等主观表述）——沿用方案 C 的 scope 质量锚点约定。
3. 所有新增门禁沿用 **`check_quality_gate` 的软跳过语义**（产物缺失 = 未启用，不卡死离线 eval）。
4. 每完成一个任务，必须完成「DoD 通用收尾」（见文末），否则漂移自检会标红。
5. 🚨 **T0 是硬前置，必须先于 T1 完成**——改造一旦开始，v0 基线永远打不到，改造成效将无法证明（依据：设计文档第六节）。

---

## 执行顺序总览

| 序号 | 任务 | 所属 | 依赖 | 优先级 |
|---|---|---|---|---|
| **T0.1** | 打 v0 基线（真实模式） | ⑥ | — | 🚨 **P0 硬前置** |
| **T0.2** | 冻结 v1 判据副本 | ⑥ | T0.1 | 🚨 **P0 硬前置** |
| **T0.3** | 补充考题（契约类 + 关键路径类） | ⑥ | — | 🚨 **P0 硬前置** |
| **T0.4** | 新增把关效果指标采集 | ⑥ | T0.1 | 🚨 **P0 硬前置** |
| T1 | 沙箱运行器 `sandbox_run.py` | ② | **T0** | P1 |
| T2 | 领域层 `check_sandbox` | ② | T1 | P1 |
| T3 | 沙箱接线 + D14 + 文档 | ② | T2 | P1 |
| T4 | 每 PR eval 回归 | ③-3b | — | P1 |
| T5 | 反模式判据换 Semgrep | ⑤ | — | P1 |
| T6 | 伪绿判据接变异测试 | ⑤ | — | P1 |
| T7 | 自洽判据接容器化执行 | ⑤ | T1 | P1 |
| T8 | 元评测（评测评测）+ D18 | ⑤ | T5,T6,T7 | P1 |
| T9 | `PHASE_CONTRACT` 与阶段链 | ① | — | P0 |
| T10 | `contract_gen.py` 生成器 | ① | T9 | P0 |
| T11 | 契约门禁 + 接线 | ① | T10 | P0 |
| T12 | D13 + 文档同步 | ① | T9–T11 | P0 |
| T13 | `critical_paths.yaml` + 校验脚本 | ④ | — | P3（试点） |
| T14 | 形式化接线 + D17 | ④ | T13 | P3 |
| T15 | 对抗式审查（降级/可选） | ③-3a | 建议 T8 后 | P2 |

---

## ⚠️ T0　前置：改造前基线冻结（**必须先于 T1 完成**）

> **为什么是硬前置**：改造一旦开始，`v0` 基线**永远打不到**——没有 v0，改造成效既无法证明，还会被「通过率下降」误判为失败。
> 依据：设计文档**第六节「评测方法论」**。

### T0.1　打 v0 基线　⬜

**目标**：在改动任何代码之前，记录当前 pipeline 的真实水平。

**操作**：**真实模式**跑一遍 `golden_tasks`（dry-run 下 `tests_pass` 为 SKIP，量不出真实质量），结果存 `benchmark_results/eval_baseline_v0.json` 并**冻结**（只读，改造期间不得覆盖）。

**验收标准**
- `eval_baseline_v0.json` 已生成，包含全部 golden tasks 的通过率与分维度判据结果
- 文件纳入版本管理并标记为「基线冻结」
- **记录当次的 prompt 版本 / 模型版本**（否则后续无法归因）

**依赖**：无

---

### T0.2　冻结 v1 判据副本　⬜

**目标**：留一把「不会变长的尺子」用于前后对比——⑤ 会改判据，改了就不等比。

**操作**：把当前五道判据（粒度 / 伪绿 / 验收 / 反模式 / 自洽）快照为 `v1`，改造期间**不得修改**；改造后的新判据记为 `v2`，与 v1 **物理隔离**。

**验收标准**
- v1 判据副本存在且可**独立调用**
- v1 跑当前 golden tasks 的结果与 v0 基线**一致**（证明副本未失真）
- v1 与 v2 **不共用文件**（物理隔离）

**依赖**：T0.1

---

### T0.3　补充考题（契约类 + 关键路径类）　⬜

**目标**：让考题集能覆盖 ① 契约驱动与 ④ 形式化，否则这两项改造**量不出效果**。

**操作**：`golden_tasks` 增加两类
- **契约类任务**：在 `scope.md` 声明 `contract`（验①）
- **关键路径任务**：如「实现租户隔离」「实现支付金额校验」（验④）

**验收标准**
- 两类任务各至少 1 道，且能被当前 pipeline 正常跑通（不报错）
- 新考题的验收点**可判定**（有明确关键词 / 验收条件）
- 考题**独立于**被验对象的实现（沿用独立性铁律）

**依赖**：无（加完需**重跑 T0.1**，让 v0 基线也覆盖新考题）

---

### T0.4　新增把关效果指标采集　⬜

**目标**：产出质量之外的第二组指标——否则「通过率下降」无法被正确解读。

**采集项**

| 指标 | 说明 | 期望 |
|---|---|---|
| 各 gate 的 BLOCKING 计数 | 质量门 / 沙箱 / 契约 / 形式化 / 对抗 各拦下多少 | **↑** |
| `fix` 回退次数 | 返工率 | **↓** |
| **人工门禁驳回率** | 设计 / 合并门禁被 human 打回的比例 | **↓** |

**验收标准**
- 三项指标可被 `run_eval_regression.py` 采集并写入结果
- 当前（改造前）三项指标已记录为**对照值**
- 「人工门禁驳回率」可区分设计门禁与合并门禁

**依赖**：T0.1

---

## T1　沙箱运行器 `scripts/sandbox_run.py`　⬜

**目标**：在隔离环境真实执行生成代码 + 测试，采集运行证据，补「静态 → 动态」验证真空。

**涉及文件**：新建 `scripts/sandbox_run.py`

**实现要点**
- 优先 Docker / podman 容器：`--network=none`、内存/CPU 上限、超时强制 kill
- 无容器环境**降级**为子进程 + `timeout` + `ulimit`（不抛异常）
- 装依赖（取自产物 requirements）→ 跑 `pytest` → 采集结果
- 输出 JSON：`{ran, exit_code, runtime_errors, wall_time, killed_by: null|timeout|oom}`

**验收标准（可测试）**
- 注入死循环样例 → `killed_by == "timeout"`
- 注入内存爆掉样例 → `killed_by == "oom"`
- 注入 import 失败样例 → `exit_code != 0` 且 `runtime_errors` 非空
- 正常样例 → `ran == true 且 exit_code == 0`
- 无 docker 环境 → 降级路径跑通、不抛异常

**依赖**：**T0（硬前置——v0 基线必须在任何改造前冻结）**

---

## T2　领域层 `check_sandbox`　⬜

**目标**：把沙箱结果接成与质量门同构的门禁函数。

**涉及文件**：`scripts/agent_orchestrator.py`

**实现要点**
- `check_sandbox(state) -> list`，与 `check_quality_gate` **同签名**
- 读 `.codebuddy/run/{task}/sandbox.json`；**文件缺失返回 `[]`（软跳过，不报错）**
- 失败条件：`exit_code != 0` 或 `killed_by ∈ {timeout, oom}` 或 `runtime_errors` 非空 → 返回 `[SANDBOX] ...` 项

**验收标准**
- 产物缺失 → 返回 `[]`，编排正常推进（不卡）
- 构造失败证据 → 返回含 `[SANDBOX]` 的 BLOCKING 项
- 函数签名与 `check_quality_gate` 一致（可并列调用）

**依赖**：T1

---

## T3　沙箱接线 + D14 + 文档同步　⬜

**目标**：把沙箱门真正挂到 `test → merge` 转换处，并加漂移自检。

**涉及文件**
- `scripts/agent_orchestrator.py`（`PHASE_TEST → PHASE_MERGE` 转换处，与 `check_quality_gate` **并列**调用 `check_sandbox`）
- `scripts/check_workflow_drift.py`（新增 D14）
- `.codebuddy/agents/agent_driver.md`、`.codebuddy/agents/multi-agent-workflow.md`

**D14 校验内容**：`check_sandbox` 实现存在 + `advance()` 在 `PHASE_TEST` 转换真实调用 + `sandbox_run.py` 存在 + 软跳过约定已文档化

**验收标准**
- `python scripts/check_workflow_drift.py` 退出码 0
- 离线 eval（`run_workflow`）跑通不受影响（无 sandbox 产物 → 软跳过）
- 构造 BLOCKING → 进合并门禁④，标 `[SANDBOX]`

**依赖**：T2

---

## T4　每 PR eval 回归　⬜

**目标**：把方案③从「CI 定期」下推到「每 PR」，退化在合并前被拦。

**涉及文件**
- `.github/workflows/ci-tests.yml`（`eval-regression` job 的 `on:` 加 `pull_request`）
- `scripts/check_workflow_drift.py`（D16，扩展原 D11）
- `.codebuddy/agents/agent_driver.md`

**验收标准**
- 提一个 PR → `eval-regression` job 被触发
- 人为让通过率低于基线阈值 → PR 检查失败
- D16 校验：job 含 `pull_request` 触发 + 引用 `run_eval_regression.py` + `benchmark_results/eval_baseline.json` 存在

**依赖**：无

---

## T5　反模式判据换 Semgrep　⬜

**目标**：用手写黑名单换 AST 级规则引擎（语义更准、规则可复用）。

**涉及文件**
- 新建 `.semgrep/rules/*.yml`（恒真断言、占位空实现、危险求值/反序列化/命令执行）
- `benchmark/eval/code_eval_harness` 判据层：改为调用 semgrep 并解析 JSON 输出

**验收标准**
- 注入 `assert True` / `pass` 占位 → 自定义规则命中
- harness 反模式判据结果与 semgrep 输出**一致**
- 规则文件纳入版本管理（可被 D18 校验到）

**依赖**：无

---

## T6　伪绿判据接变异测试　⬜

**目标**：把「静态查恒真断言」升级为「注入变异看测试红不红」——这才是「测试灵不灵」的证明。

**涉及文件**：`scripts/mutation_check.py`（已有）→ 接入 harness 判据层

**验收标准**
- 注入已知松断言（`==` 改 `in`、阈值放大）→ 判据判 **FAIL**
- 未注入 → 判 **PASS**
- 捕获率可被元评测（T8）度量

**依赖**：无

---

## T7　自洽判据接容器化执行　⬜

**目标**：`tests_pass` 从「dev 环境跑」升级为「隔离容器跑真测试」。

**涉及文件**：`benchmark/eval/code_eval_harness` 的 `EvalDriver` → `tests_pass` 判据改为调用 T1 的 `sandbox_run.py`

**验收标准**
- 真实模式：容器内跑真测试，`tests_pass` 取容器退出码
- dry-run 模式：仍标 `SKIP`（不因缺容器而误判失败）
- 与 T1 共用同一份 `sandbox_run.py`（不重复实现）

**依赖**：T1

---

## T8　元评测（评测评测）+ D18　⬜

**目标**：验证「判据本身灵不灵」——注入已知缺陷，看每道把关抓不抓得到。

**涉及文件**
- 新建 `scripts/eval_meta_check.py`（注入用例 + 断言各判据响应）
- `scripts/check_workflow_drift.py`（新增 D18）

**注入用例清单（每道都必须有对应反例）**

| 把关 | 注入什么（已知是坏的） | 期望 |
|---|---|---|
| 变异测试 | 松断言（`==`→`in`） | 判 FAIL |
| 放宽检测 | 阈值放大 10 倍 | 判 FAIL |
| SAST | 硬编码密钥 / 危险求值 | 判 FAIL |
| 质量门 | 低覆盖率 PR | 判 BLOCKING |
| 沙箱 | OOM / 超时 / import 失败 | 判 BLOCKING |
| 形式化 | 破坏不变量（租户越权） | 判 FAIL |
| D 漂移自检 | 改代码不改文档 | 判 FAIL |

**D18 校验内容**：判据层映射一致性（反模式走 semgrep 规则文件、伪绿接 `mutation_check.py`、自洽接 `sandbox_run.py`）+ 自定义 semgrep 规则文件存在

**验收标准**
- 上表 7 类注入**全部**被对应判据捕获
- 反向：全部不注入时，所有判据**全部** PASS（无误报）
- `python scripts/check_workflow_drift.py` 退出 0

**依赖**：T5, T6, T7

---

## T9　`PHASE_CONTRACT` 与阶段链　⬜

**目标**：插入契约生成阶段（确定性，非 LLM）。

**涉及文件**：`scripts/agent_orchestrator.py`
- 新增 `PHASE_CONTRACT = "contract"`
- `STAGE_AGENT[PHASE_CONTRACT]` = 确定性 runner（不走 LLM subagent）
- `advance()` 转换：
  ```
  SCOPE  → (有 contract? CONTRACT : CODE)
  DESIGN → CONTRACT
  CONTRACT → CODE
  ```

**验收标准**
- `scope.md` 声明 `contract` → 阶段链进入 CONTRACT
- 未声明 → 直接进 CODE（不卡，沿用 design 跳过语义）
- `run_workflow`（离线 eval）阶段链更新后仍跑通

**依赖**：无

---

## T10　`contract_gen.py` 确定性生成器　⬜

**目标**：从 OpenAPI 生成 client / model / mock / 属性测试，把可推导部分从 AI 手里拿走。

**涉及文件**：新建 `scripts/contract_gen.py`
- `openapi-spec-validator` 校验 spec
- `datamodel-code-generator` 生成 models（pydantic v2）
- 生成 typed client（含 retry / 超时骨架）
- `schemathesis` 生成属性测试（属「第 1 层契约镜像测试」→ **免审**）
- 生成 mock server（供 test 阶段依赖）

**验收标准**
- 给定合法 `openapi.yaml` → `models.py` 可 import、字段与 spec 一致
- client 方法签名与 spec 的 path/method **一一对应**
- schemathesis 用例可跑通
- spec 非法 → 报错退出非 0

**依赖**：T9

---

## T11　契约门禁 + 接线　⬜

**目标**：契约产物不合格即阻断。

**涉及文件**：`scripts/agent_orchestrator.py`、`scripts/contract_gen.py`

**实现要点**
- contract 阶段落盘后跑 spec 校验 + 生成物 import 干净检查
- 失败 → `[CONTRACT]` BLOCKING → 触发 `fix` 回退（≤ `MAX_AUTO_RETRIES=2`），超限转人工
- **生成器未启用（产物缺失）→ 软跳过**

**验收标准**
- 非法 spec → `[CONTRACT]` BLOCKING
- 生成器未启用 → 不卡，正常进 CODE
- 产物落盘至 `.codebuddy/run/{task}/contract/`

**依赖**：T10

---

## T12　D13 + 文档同步　⬜

**涉及文件**
- `scripts/check_workflow_drift.py`（新增 D13）
- `.codebuddy/agents/agent_driver.md`（阶段表、阶段→subagent 映射）
- `.codebuddy/agents/multi-agent-workflow.md`
- `.codebuddy/settings.json`（如需注册新 runner）

**D13 校验内容**：`PHASE_CONTRACT` 常量 + `STAGE_AGENT[PHASE_CONTRACT]` 映射 + `advance()` 三处转换 + `contract_gen.py` 存在

**验收标准**
- `python scripts/check_workflow_drift.py` 退出 0
- 离线 eval 不受影响

**依赖**：T9–T11

---

## T13　`critical_paths.yaml` + `critical_guarantees.py`　⬜（试点）

**目标**：对高后果关键路径（auth / 支付 / 租户隔离）加类型级 + 不变式硬保证。

**涉及文件**
- 新建 `critical_paths.yaml`：
  ```yaml
  - module: apps/shop-agent/src/modules/auth
    guarantees: [strict-typing, invariants]
    invariants_spec: tests/test_auth_invariants.py
  - module: apps/shop-agent/src/modules/tenant
    guarantees: [strict-typing, invariants]
  ```
- 新建 `scripts/critical_guarantees.py`：`mypy --strict` + 不变式测试（+ 可选 Alloy）

**验收标准**
- 对清单模块跑 `mypy --strict` 通过
- 人为破坏不变量（如让 tenant A 读到 tenant B 数据）→ 不变式测试判 FAIL
- 只对**关键路径**启用，普通 CRUD 不纳入

**依赖**：无

---

## T14　形式化接线 + D17　⬜（试点）

**涉及文件**
- `scripts/agent_orchestrator.py`（`PHASE_TEST → PHASE_MERGE` 与质量门/沙箱**并列**调用 `check_critical_guarantees`）
- `scripts/check_workflow_drift.py`（新增 D17）
- `.codebuddy/agents/agent_driver.md`

**验收标准**
- 无 `critical_paths.yaml` → **软跳过**，绝大多数任务零开销、不卡
- 有清单且校验失败 → `[FORMAL]` BLOCKING → 合并门禁④转人工
- `python scripts/check_workflow_drift.py` 退出 0

**依赖**：T13

---

## T15　对抗式审查　⏸️（降级/可选）

> **前置条件**：必须先定义「确定性外壳」，否则会成为整套体系里唯一"自己说自己对了"的环节。

**确定性外壳要求**
- 对抗审查提出的每个问题**必须附可复现失败用例**（输入 + 期望 + 实际）
- **复现不了的问题一律不计入 BLOCKING**

**涉及文件**
- 新建 `.codebuddy/agents/red-team-reviewer.md`
- `scripts/agent_orchestrator.py`：新增 `PHASE_REDTEAM`（`REVIEW → REDTEAM → TEST`）
- `scripts/check_workflow_drift.py`（新增 D15）
- `.codebuddy/settings.json`（注册 subagent）

**验收标准**
- 无复现用例的对抗意见 → **不产生** BLOCKING
- 有可复现用例 → 触发 `fix` 回退（≤2 次），超限转人工
- 与 `code-reviewer` 的判定**不高度同质**（抽样比对）

**依赖**：建议 T8 完成后再做

---

# DoD 通用收尾（每个任务都必须完成）

- [ ] 代码改动完成，本地验证命令跑通（不消耗 token）
- [ ] 新增接线已加入 `scripts/check_workflow_drift.py` 对应 D 校验
- [ ] `agent_driver.md` 已同步（阶段表 / 门禁说明 / 校验项清单）
- [ ] `multi-agent-workflow.md` 已同步（如涉及阶段链变更）
- [ ] `.codebuddy/settings.json` 已同步（如新增 subagent / runner）
- [ ] `python scripts/check_workflow_drift.py` 退出码为 **0**
- [ ] 离线 eval（`run_workflow`）跑通不受影响（新门禁在缺产物时软跳过）
- [ ] 不破坏 `agent_driver.md` 的三条不变量（stdout 只给指令 / 无活跃编排 exit 0 空 stdout / 只有 `on-stop` 可 `continue:false`）

---

# 验证命令汇总

```bash
# 漂移自检（所有任务完成后必跑，退出码 1 = 存在漂移）
python scripts/check_workflow_drift.py

# 离线跑完整阶段链（仅验证结构，不消耗 token）
python scripts/orchestrator_hook.py dry-run --task <task> --request "..."

# 查看编排进度 / 中止
python scripts/orchestrator_hook.py status
python scripts/orchestrator_hook.py reset

# eval 回归（对照 baseline）
python scripts/run_eval_regression.py

# 元评测（验证判据本身灵不灵）—— T8 产出
python scripts/eval_meta_check.py
```

---

# 风险提醒（执行时留意）

- **软跳过是硬要求**：任何新门禁在产物缺失时必须视为"未启用"，否则离线 eval / 缺工具环境会被卡死。
- **阶段链变更是高危操作**：T9 会改变 `run_workflow`，必须同步 `agent_driver.md` 阶段表与 `multi-agent-workflow.md`，否则 D 校验标红。
- **容器安全**：T1 的沙箱必须网络隔离 + 超时/OOM 上限，防止生成代码反向打外部或跑飞。
- **Semgrep 规则自身会漂移**：规则文件必须纳入版本管理，由 D18 兜住。
- **内部 harness 与对外 benchmark 不要合并**：避免内部回归被公开榜单口径绑架。
- **合规**：评测数据可能含真实对话 / PII，外部平台优先选可自托管，配合 `shared/redact.py` 脱敏。
