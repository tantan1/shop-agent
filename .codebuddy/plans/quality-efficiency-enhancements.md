# 提效 + 提质增强设计（契约驱动 / 沙箱执行 / 对抗审查+每PR eval / 关键路径形式化 / 评测引擎重新分层 / 评测方法论）

> 目标：在已落地的「hooks 护栏底座」（设计门禁 A、合并门禁④、硬性质量门②、eval 回归③、PR 模型①、D1–D12 漂移自检）之上，再叠加 6 项（原 4 项改造 + 第五节「评测引擎重新分层」+ 第六节「评测方法论」），把「没人 review 时不失控」往前推到「AI 生成的东西本身更对、更快」。
>
> ⚠️ **执行前必读**：**第六节是硬前置**——改造一旦开始，`v0` 基线永远打不到，改造成效将无法证明。落地前先完成 T0.1–T0.4（见任务列表）。
> 设计原则（复用既有三层结构，不另起炉灶）：
> - 领域层加阶段/门禁常量，状态层无变化，适配层加对应子命令/闸门；
> - 所有新增门禁沿用 `check_quality_gate` 的**软跳过语义**（产物缺失=未启用，不卡死离线/eval）；
> - 每新增一个接线点，必须在 `scripts/check_workflow_drift.py` 加一条校验（D13 起），保证文档↔执行不漂移；
> - 不破坏 `agent_driver.md` 已定义的不变量（stdout 只给指令、无活跃编排 exit 0 空 stdout、只有 `on-stop` 可 `continue:false`）。

---

## 一、契约驱动代码生成（提效 + 提质，最高杠杆）

**核心思想**：接口明确（有 OpenAPI / JSON-Schema）的那一块，从 AI 手里拿走，用**确定性代码生成器**产出 client / model / mock / schemathesis 测试，跳过 LLM。既快又确定，且天然满足「设计/契约→测试」可追溯，反哺现有把关闭环。

### 触发与跳过
- `scope` 阶段在 `scope.md` 声明 `contract: <openapi.yaml 路径或内联>`。
- 未声明 → `PHASE_CONTRACT` 直接跳过（同 design 的默认跳过语义），`advance()` 直进 `code`。
- 已声明 → 进入契约生成阶段。

### 新增（确定性，非 LLM）
- 阶段常量：`PHASE_CONTRACT = "contract"`。
- 生成器脚本：`scripts/contract_gen.py`（纯确定性，调 `openapi-generator` / `datamodel-code-generator` / `schemathesis`）：
  - `models.py`（pydantic v2 / dataclass，带字段校验）
  - `client.py`（typed client，含 retry/超时骨架）
  - `mock_server.py`（基于契约的 mock，供 test 阶段依赖）
  - `contract_tests.yaml`（schemathesis 属性测试，属 agent_driver 定义的「第 1 层契约镜像测试」→ **免审**）
- `STAGE_AGENT[PHASE_CONTRACT]` = 确定性 runner（不走 LLM subagent），例如封装为 `scripts/contract_gen.py --task <task> --spec <path>`。

### 接线（领域层 `advance()`）
```
SCOPE → (有 contract? CONTRACT : CODE)
DESIGN → CONTRACT
CONTRACT → CODE        # 生成完毕或跳过，均进编码
```
- `code`（python-coder / java-coder）从此**消费** `contract/` 产物，不再从零写模型/客户端 → 减少 AI 臆测、减少接口错配。

### 门禁（契约校验）
- `contract` 阶段落盘后跑 `openapi-spec-validator` + 生成物 import 干净检查。
- 失败 → 标 `[CONTRACT]` BLOCKING，触发 `fix` 回退（修正 spec 或生成参数），超限转人工。
- 产物缺失（未启用生成器）→ 软跳过，不报错。

### 漂移自检
- **D13**：`PHASE_CONTRACT` 常量 + `STAGE_AGENT[PHASE_CONTRACT]` 映射 + `advance()` 三处转换 + `contract_gen.py` 存在性。

---

## 二、沙箱执行验证（纯提质，merge 前的动态关门）

**核心思想**：`test` 阶段「测试全绿」只证明 dev 环境能跑；生成代码可能在隔离/干净环境里 import 失败、OOM、死锁、依赖缺失。沙箱执行把代码在**隔离容器**里真跑一遍，抓运行时缺陷，与静态的质量门（coverage/SAST）互补。

### 新增
- 门禁函数：`check_sandbox(state) -> list`（与 `check_quality_gate` 同签名，并列调用）。
- 运行器：`scripts/sandbox_run.py`：
  - 起一个**网络隔离 + 资源受限**容器（Docker / podman；无 docker 时降级为带 `seccomp`+`timeout`+`ulimit` 的子进程）；
  - 装 `requirements`（取自产物），跑 `pytest`；
  - 采集：`ran / exit_code / runtime_errors / wall_time / killed_by(null|timeout|oom)`。
- 证据产物：`.codebuddy/run/{task}/sandbox.json`。

### 接线（领域层 `advance()`，在 `PHASE_TEST → PHASE_MERGE` 转换处）
```
TEST → [check_quality_gate] + [check_sandbox] + [check_critical_guarantees]
       任一返回 BLOCKING → 进 MERGE 门禁（标 [QUALITY]/[SANDBOX]/[FORMAL]），由人审
       全绿 → MERGE
```
- 沿用质量门**软跳过**：无 `sandbox.json`（CI 未产出 / 无 docker 环境）→ 视为未启用，不卡。

### 门禁语义
- 退出码非 0、或 `killed_by ∈ {timeout, oom}`、或有未捕获运行时异常 → `[SANDBOX]` BLOCKING。
- 这类问题自动 `fix` 回退解决不了（环境/资源类），最终在**合并门禁④**转人工（与质量门一致）。

### 漂移自检
- **D14**：`check_sandbox` 实现 + `advance()` 在 `PHASE_TEST` 转换真实调用 + `sandbox_run.py` 存在性 + 软跳过约定文档。

---

## 三、对抗式审查 + 每 PR eval 回归

### 3a 对抗式审查（red-team，与友好审查互补）
**核心思想**：`code-reviewer` 是「帮你找 bug 的同事」；red-team 是「专门攻破你输出的对手」——主动构造越权、边界、竞态、规格偏离。两者视角正交，互补兜住单视角盲区。

- 新 subagent：`red-team-reviewer`（或 `code-reviewer --mode adversarial`），输出 `redteam.md`，含 `[BLOCKING]` 项（同 code-reviewer 约定）。
- 新阶段：`PHASE_REDTEAM`，插在 `review` 与 `test` 之间：
  ```
  REVIEW → (BLOCKING? FIX : REDTEAM)
  REDTEAM → (BLOCKING? FIX : TEST)
  FIX → REVIEW
  ```
- 触发 `fix` 回退规则、重试上限 `MAX_AUTO_RETRIES=2`、超限转人工，全部复用既有 `review` 机制（同一套 `[BLOCKING]`→`fix` 回路）。
- 产物：`.codebuddy/run/{task}/redteam.md`。

### 3b 每 PR eval 回归（把方案③从 CI 定期下推到每 PR）
**核心思想**：现有 `eval-regression` 是 CI job（基线 `benchmark_results/eval_baseline.json`），但只在 `push/main` 跑；改为**每个 PR** 都跑，退化在合并前被拦。

- `.github/workflows/ci-tests.yml` 的 `eval-regression` job 增加 `pull_request` 触发器（原 `on: push` 改为 `push` + `pull_request`）；
- PR 模型（方案①，`ORCH_GIT=1`）`merge` 开 PR 后，该 PR 的 CI 自动跑 `run_eval_regression.py` 对照基线，退化超阈值（如通过率跌 > N%）→ PR 检查失败，阻断合并；
- 可选：本地 `merge --eval` 在开 PR 前先跑一遍 eval（仅 ORCH_GIT 且 eval 启用时），早发现早修。
- 与方案③完全一致：基线、判据、harness 不变，只是触发时机从「定期」变「每 PR」。

### 漂移自检
- **D15**：`PHASE_REDTEAM` 常量 + `STAGE_AGENT[PHASE_REDTEAM]=red-team-reviewer` 映射 + `advance()` 两处转换 + subagent 文件存在性。
- **D16**：`ci-tests.yml` 的 `eval-regression` job 含 `pull_request` 触发 + 引用 `run_eval_regression.py` + 基线文件存在（扩展原 D11）。

---

## 四、关键路径类型级 / 形式化保证（auth / 支付 / 权限 / 租户隔离）

**核心思想**：对高后果关键路径，单测/AI review 仍可能漏「类型层」与「不变式」错误。用类型级约束（branded types / 穷尽匹配 / 禁止 `Any`）+ 不变式测试（property-based）+ 可选轻量形式化（Alloy/TLA+ 对协议建模），把它们变成**编译期/运行期硬保证**。

### 新增
- 清单：`critical_paths.yaml`（落仓库根或 `scripts/`）：
  ```yaml
  - module: apps/shop-agent/src/modules/auth
    guarantees: [strict-typing, invariants]
    invariants_spec: tests/test_auth_invariants.py
  - module: apps/shop-agent/src/modules/payment
    guarantees: [strict-typing, invariants, model-check]
    model_spec: specs/payment.als
  - module: apps/shop-agent/src/modules/tenant
    guarantees: [strict-typing, invariants]
  ```
- 校验脚本：`scripts/critical_guarantees.py`：
  - 读清单 → 对每个 module 跑 `mypy --strict`（禁止 `Any`/`object`、未用变量、返回 `None` 隐式）；
  - 跑 `invariants_spec` 里的不变式测试（如「tenant A 永远读不到 tenant B 数据」「金额非负且精度固定」）；
  - 若清单含 `model-check` 且 `.als` 存在 → 跑 Alloy 求解器断言协议无死锁/无越权。
- 证据产物：`.codebuddy/run/{task}/formal.json`（`{module, level, passed, detail}`）。

### 接线（领域层 `advance()`，在 `PHASE_TEST → PHASE_MERGE` 转换处，与质量门/沙箱并列）
```
TEST → [check_quality_gate] + [check_sandbox] + [check_critical_guarantees]
```
- **软跳过**：无 `critical_paths.yaml` → 视为非关键任务，不触发（绝大多数任务不受影响，零开销）。
- 失败 → `[FORMAL]` BLOCKING → 合并门禁④转人工（类型/不变式问题自动 `fix` 难解，须人审）。

### 漂移自检
- **D17**：`critical_paths.yaml` 清单（如存在）+ `check_critical_guarantees` 实现 + `advance()` 在 `PHASE_TEST` 转换真实调用 + 软跳过约定文档。

---

## 五、评测引擎重新分层（`code_eval_harness` 保留 + 判据层换成熟工具）

> 本节为后续评审结论的固化（原方案未涉及）：`code_eval_harness` **不整体替换为外部产品**，改为重新分层。

**决策结论**：harness 保留为「**编排 / 组合层**」，判据**实现层**换成严格更成熟的单项工具；对外对标另外接公开 benchmark。

### 为什么整体无替代
- 公开工具分两类：**Benchmark**（SWE-bench / Terminal-Bench / SWE-Lancer）回答「agent 能不能做对单题」；**Framework**（inspect_ai / 本 harness）回答「我的流程稳不稳」。本 harness 属后者，**不会被 benchmark 替代**。
- **「流程级回归」（固定考题集上跨任务 / 跨时间的通过率与退化趋势）没有任何开箱产品**。2026 才出现同理念工作（arXiv *Compiled AI: Deterministic Code Generation*、`ai-pipeline-evaluation-harness`）——方向被验证，但无成熟产品可买。
- 解耦设计（harness 只打分不生成 + `MockDriver` 脱离本项目自测）支撑 **L2 元评测**（验证「判据本身灵不灵」），外部工具无此能力。

### 判据层逐条重映射

| 原判据 | 改为 | 说明 |
|---|---|---|
| **反模式** | **Semgrep / CodeQL** | AST 级 + 社区规则库，严格优于手写黑名单；领域规则写成**自定义 Semgrep 规则**保留 |
| **伪绿** | **变异测试**（`scripts/mutation_check.py` 已有） | 静态查「恒真断言」只是最浅层；「测试灵不灵」只能靠**注入变异看它红不红**证明 → 把 `mutation_check` 接入判据层 |
| **自洽 tests_pass** | **容器化执行** | 即本节②沙箱（`sandbox_run.py`）；dry-run 下仍标 SKIP，真实模式跑真仓库 / 真测试基建 |
| 验收关键词 | **保留自研** | 领域定制，外部无替代 |
| 粒度 | **保留自研** | 领域定制，外部无替代 |

### 框架层
- 可选迁 **inspect_ai**（UK AISI）：task/solver/scorer 抽象、并发 / 日志 / 报告，且支持自定义确定性 scorer（能装下五道判据）。
- **默认保留自研**：`run_eval_regression.py` 很薄、维护成本可接受；仅当并发 / 报告需求变强时再迁。

### 对外对标（与内部回归并存，不冲突）
- **内部回归** → 自建 harness（灵活、判据可定制）
- **对外证明能力** → 额外接 **SWE-bench / Terminal-Bench**
- 两套各管一头，不互相替代，**不合并**。

### 与本节其他项的关系
- ② 沙箱 = 自洽判据的容器化执行载体（同一份 `sandbox_run.py` 复用）
- ③-3a 对抗审查可借 **promptfoo** 的 red-teaming 能力补强（声明式 + CI 原生）
- 三层评测（L0 CI 离线 / L1 在线 / L2 实验）中的 **L0** 即由本 harness 承载

### 漂移自检
- **D18**：判据层映射一致性——反模式走 Semgrep 规则文件、伪绿接 `mutation_check.py`、自洽接 `sandbox_run.py`；且自定义 Semgrep 规则文件存在、纳入版本管理。

---

## 六、评测方法论：改造前后如何证明「变好了」

> 本节为后续评审结论的固化，且是**硬前置**：本次改造的本质是**把把关变严**，因此**通过率会下降**（更多缺陷被抓出）。若只盯通过率这一个数，会**误判改造失败**——必须建立正确的对比口径。

### 6.1 核心陷阱：通过率被「更严的把关」污染

| 改造 | 对通过率影响 | 真实含义 |
|---|---|---|
| ② 沙箱 | **↓ 下降** | 更多运行时缺陷被抓出 → **好事** |
| ⑤ 判据换 Semgrep / 变异 | **↓ 下降** | 以前误判为过的伪绿被抓出 → **好事** |
| ③-3a 对抗审查 | **↓ 下降** | 更多问题被挑出 → **好事** |
| ① 契约驱动 | ↑ 或 → | 可推导部分确定性生成，臆测减少 → **好事** |
| ④ 形式化 | → | golden tasks 不覆盖关键路径，几乎无影响 |

**结论**：「通过率下降」≠「流程变差」，它往往是改造起效的**信号**。

### 6.2 判据版本化（新增约定）

类比已落地的「提示词版本化」（第 5 道把关）——**判据本身也必须版本化**：

- 改造前**冻结一份 v1 判据副本**，改造期间**不得修改**，专用于前后对比
- v2 判据（改造后）用于新口径，其首次跑分 = **新基线**，**不与 v0 比较**
- 否则等于「用一把会变长的尺子量身高」，前后数据不可比

### 6.3 两组指标（必须分开看）

| 组 | 指标 | 期望 | 用什么尺子量 |
|---|---|---|---|
| **产出质量** | 生成的代码 / 测试本身好不好 | **↑** | **冻结的 v1 判据** |
| **把关效果** | 各 gate 的 BLOCKING 计数、`fix` 回退次数、**人工门禁驳回率** | 拦截 **↑**、返工 **↓**、驳回 **↓** | 新增采集 |

> 「人工门禁驳回率」是**「流程是否真的减少了人审」**的直接度量，正对应《免审的代价 / 兑换法则》的主张，建议列为核心指标。

### 6.4 A/B 对比方法（6 步）

1. **改造前打 v0 基线**：当前 pipeline + 当前判据跑一遍，存 `eval_baseline_v0.json` 并**冻结**
2. **冻结 v1 判据副本**：改造期间不改，专用于前后对比
3. **补充考题**：golden_tasks 加两类——**契约类任务**（验①）、**关键路径任务**（验④）
4. **改造后各跑两次**：v1 判据（比产出质量，与 v0 可比）+ v2 判据（**建新基线**，不与 v0 比）
5. **新增采集三指标**：BLOCKING 计数 / `fix` 回退次数 / 人工门禁驳回率
6. **重新打基线**：v2 口径第一次跑 = `eval_baseline_v1.json`

### 6.5 各改造项的可评测性

| 改造 | harness 能量吗 | 说明 |
|---|---|---|
| ① 契约驱动 | ✅ | 需 golden_tasks 补**契约类任务**（声明 `contract` 的题） |
| ② 沙箱 | ✅ | 看 BLOCKING 增量 + 产出质量 |
| ③-3a 对抗审查 | ✅ | 看 BLOCKING 增量 |
| ④ 形式化 | ⚠️ 弱 | 需单独建**关键路径考题**（如「实现租户隔离」） |
| ③-3b 每 PR eval | ❌ | 属 CI 触发机制，非产出质量，harness 量不到 |
| ⑤ 判据重分层 | ❌ **不能用它量自己** | 靠 **T8 元评测**（注入已知缺陷验证判据） |

### 6.6 约束

- dry-run（空跑）下 `tests_pass` 为 SKIP，量不出真实质量 → **前后对比必须跑真实模式**（有 token 成本，建议先小样本跑通再全量）
- **本节是硬前置**：改造一旦开始，v0 基线永远打不到 → 任务列表中对应 **T0 块，必须先于 T1 完成**

---

## 落地顺序建议（按性价比 / 风险）

| 优先级 | 项 | 理由 |
|---|---|---|
| 🚨 **P0 硬前置** | **六、评测方法论（T0）** | **改造一开始 v0 基线就永远打不到**；没有它，改造成效无法证明，甚至会被「通过率下降」误判为失败 |
| P0 | 一、契约驱动代码生成 | 提效+提质双高；确定性、零 AI 幻觉；直接反哺把关闭环；接线最干净（一个新阶段） |
| P1 | 二、沙箱执行验证 | 纯提质、抓运行时盲区；与质量门同构（软跳过），风险低 |
| P1 | 三-3b、每 PR eval | 几乎零新代码（加 PR 触发 + 可选 `--eval`），把已有方案③价值最大化 |
| P1 | 五、评测引擎重新分层 | 判据层换 Semgrep/变异/容器化，严格更准且少自维护；保留 harness 组合层与元评测能力 |
| P2 | 三-3a、对抗式审查 | 多一个 subagent + 一个阶段，需调 red-team prompt 防与 code-reviewer 同质 |
| P3 | 四、关键路径形式化 | 收益最高但最重（mypy 治理 + 不变式测试 + 可选 Alloy）；只有关键模块才上，先小范围试点 |

## 风险 / 注意
- 所有新门禁必须**软跳过**，否则离线 eval / 缺工具环境会被卡死（沿用质量门先例）。
- 新增阶段（CONTRACT / REDTEAM）会改变 `run_workflow` 阶段链 → 必须同步更新 `agent_driver.md` 阶段表、`multi-agent-workflow.md`、`.codebuddy/settings.json`（如新 subagent 需注册）、`check_workflow_drift.py`（D13–D18），否则漂移自检会标红。
- 沙箱执行注意**资源与安全性**：容器必须网络隔离、设超时/OOM 上限，避免生成代码反向打外部或跑飞。
- 形式化只对「关键路径」启用，避免把强约束误用到普通 CRUD 上导致开发体验恶化。
- **引入外部评测工具的两个坑**：① Semgrep 自定义规则本身也会漂移，须纳入版本管理与回归（由 D18 兜住）；② 对外 benchmark（SWE-bench）与内部 harness **不要合并**，避免内部回归被公开榜单口径绑架。
- **合规**：评测数据可能含真实对话 / PII，外部评测平台优先选**可自托管**（如 LangFuse），配合 `shared/redact.py` 脱敏，避免数据外发。
