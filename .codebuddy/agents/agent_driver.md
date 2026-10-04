# Agent 驱动接入说明（CodeBuddy hooks 事件驱动编排）

本文件说明主 Agent 如何用 CodeBuddy hooks 把 `scripts/agent_orchestrator.py` 的领域层
接入真实 subagent（`.codebuddy/agents/` 下定义），实现"workflow.md 的可自动执行编排"。
**废除手动驱动类**：原 `Orchestrator` 主 Agent 循环壳已删除，编排改由 hook 事件驱动。

## 三层结构（职责严格切分，避免双实现）

| 层 | 文件 | 职责 |
|----|------|------|
| 领域层 | `scripts/agent_orchestrator.py` | 纯规则引擎：`advance()` / `next_instruction()` 不碰 state.json；状态序列化 `state_to_dict/state_from_dict`；`STAGE_AGENT` 模块级常量；`run_workflow` 完整阶段链 |
| 状态层 | `scripts/orchestrator_state.py` | state.json 读写、原子写、跨进程文件锁、`.current` 指针、路径安全校验 |
| 适配层 | `scripts/orchestrator_hook.py` | hook 协议：stdin 收 JSON / stdout 出 JSON / 退出码语义 / 持久化时机 |

> eval（`benchmark/eval/`）与 hooks 共用同一段 `advance()` 状态机，保证"离线度量"
> 与"线上驱动"行为一致——eval 绿灯即证明 hooks 路径正确。

## 原理
- `agent_orchestrator.py` 提供 `advance(state)` 状态机（纯函数）：决定下一阶段、是否跳过、是否回退。
- `orchestrator_hook.py` 把 CodeBuddy 的 `PreToolUse` / `PostToolUse` / `Stop` 事件接到
  `advance()` 上：PostToolUse 捕获 Task 产出并推进，Stop 仅在编排未完成时阻止早停。
- 两层严格对应 `multi-agent-workflow.md`，保证文档↔执行不漂移。

## 事件 → 子命令映射（在 `.codebuddy/settings.json` 注册）

| 事件 | matcher | 子命令 | 作用 |
|------|---------|--------|------|
| `PreToolUse` | `Task` | `on-pre-task` | 派发前：校验阶段、打印指令 |
| `PostToolUse` | `Task` | `on-post-task` | 捕获产出 → `advance()` 推进 |
| `Stop` | — | `on-stop` | 编排未完成则 `continue:false` 阻止早停 |

## 主 Agent 执行模型（由 hook 自动驱动，主 Agent 无需手写循环）

```
start ──► PreToolUse(on-pre-task) 打印指令
      └─► 主 Agent 用 Task 调真实 subagent
PostToolUse(on-post-task) 捕获产出 → advance() → 推进到下一阶段
      └─► 循环直到 phase=done 或 needs_human
Stop(on-stop) 未完成则阻止停止，提示下一步指令
```

## 阶段 → subagent 映射（`agent_orchestrator.STAGE_AGENT`）

| stage | subagent | 对应 workflow.md |
|-------|----------|------------------|
| scope | scope | 阶段0 目标澄清 |
| design | architecture-designer | 阶段1 架构设计（条件触发，默认跳过；触发后进入设计门禁，需人工 approve）|
| contract | contract-gen | 阶段1.5 契约生成（确定性 runner，不走 LLM；scope 声明 contract 时触发）|
| code | python-coder | 阶段2/3 编码 |
| review | code-reviewer | 阶段4 审查（阻塞自动回退）|
| fix | python-coder | 阶段4 回退修复（重试≤2 次）|
| redteam | red-team-reviewer | 阶段4.5 对抗审查（可选，无复现用例不产生 BLOCKING）|
| test | test-generator | 阶段5 测试生成 |

## 测试生成审核约定（防假绿）
- test-generator 生成的**第2层（自由生成型）测试**，每个测试文件顶部必须写 `Rule: <本文件验证的业务规则>` docstring。
- 主 Agent / 审查者**只审这句规则描述**（不逐行读测试代码），30 秒/文件判断"这规则是不是我要的"。
- 若测试是契约/设计的镜像（第1层，如档 B 的契约用例），免审——审核已发生在契约定义时。
- 禁止 `assert True` / `pass` / 注释掉失败断言等伪通过；code-reviewer 须将其标为 `[BLOCKING]`。

## 阻塞判定约定
- `code-reviewer` 的输出若含 `[BLOCKING]` 行，视为阻塞 → `advance()` 自动触发 `fix` 回退
  （最多 `MAX_AUTO_RETRIES=2` 次），超限转人工（`needs_human=True`）。
- 产物落盘：`.codebuddy/run/{task}/`（scope.md / design.md / code.diff / review.md / tests.py / summary.json）。

## 硬性质量门（方案②）：test 后的覆盖率 / SAST / 沙箱门禁

`test` 阶段落盘后，`agent_orchestrator.check_quality_gate` 在执行伪绿检测之外，
再跑一道**硬性质量门**，命中即标 `[QUALITY]` BLOCKING，最终在合并门禁（方案④）处
转人工（覆盖率不足 / 高危 SAST 无法靠自动 `fix` 回退解决，必须人审）：

- **覆盖率门禁**：读取 `.codebuddy/run/{task}/coverage.json`（pytest-cov 的
  `--cov-report=json` 产物），`totals.percent_covered` 低于 `QUALITY_COVERAGE_GATE`
  （默认 80%）即阻塞。
- **SAST 门禁**：读取 `.codebuddy/run/{task}/sast.json`（bandit `-f json` 产物），
  `results[].issue_severity` 为 HIGH/MEDIUM 即阻塞。
- **沙箱门禁**：读取 `.codebuddy/run/{task}/sandbox.json`（`scripts/sandbox_run.py` 产物），
  `exit_code != 0` 或 `killed_by ∈ {timeout, oom}` 或 `runtime_errors` 非空即标
  `[SANDBOX]` BLOCKING，最终在合并门禁处转人工。

> 三个产物文件**任一缺失即视为该闸门未启用（软跳过）**，不报错——这样离线 eval /
> 缺工具环境不会被质量门卡死；只有真实 CI（由 test-generator 或 CI 脚本产出上述文件）
> 才真正生效。门槛常量 `QUALITY_COVERAGE_GATE` 定义在 `agent_orchestrator.py`。

## 不变量（改 hook 脚本时勿破坏）
1. 调试日志一律 stderr；stdout 只承载给 Agent 的指令/JSON。
2. 无活跃编排时命令一律 exit 0 且 **stdout 为空**（零干预）——否则普通会话结束会被 Stop 拦住。
3. 只有 `on-stop` 可能输出 `continue:false`，且受 `stop_forced` 上限约束（防 Agent 被无限驱动）。

## 设计门禁（方案 A）：design → code 强制人工 review

`design` 阶段默认跳过（依据既有文档时直进 code，不触发门禁）。一旦触发 fresh
设计（即 `architecture-designer` 真的从零产出 `design.md`），编排**不自动进 code**，
而是在 `.codebuddy/run/{task}/state.json` 的 `hook.await_design_approval` 置位，
进入门禁：

- PostToolUse：design 产出后返回"🛑 设计门禁"提示，要求人工 review `design.md`；
  门禁解除前，任何推进（含误派发的 code）都被拦截。
- Stop：门禁未过时 `continue:false` 拦截停止，强制先审设计。
- 放行：人工 review 确认后运行 `python scripts/orchestrator_hook.py approve --task <task>`，
  清除门禁，主 Agent 方可派发 `python-coder` 进入编码。

> 门禁是**纯 hook 层**的人机交互闸（存于 envelope 的 `hook` 段），不污染领域状态机，
> 因此 `benchmark/eval` 的 `run_workflow` 仍走 design→code 直连、不受影响（eval 自动批准）。

### 操作手册（人工视角）

**何时会撞上门禁**
- **触发**：task 被判定"需要架构"，`architecture-designer` 真正从零产出 `design.md`（fresh design）。
- **不触发**：design 阶段被默认跳过（大多数任务）；或 scope 阶段检测到**既有 design 文档**（会载入它作权威依据，走 design→code 直连，不经门禁）。
- **怎么知道撞上了**：对话出现 `🛑 设计门禁：design.md 已生成` 提示；或 `status` 显示 `await_design_approval=true`。

**标准流程**
1. 打开 `.codebuddy/run/{task}/design.md`。
2. 按 `scope.md` 验收标准**逐条核对**——设计须逐条追溯 scope 的验收/非功能约束（见方案 C）。
3. 无误 → 运行放行命令（下方）。
4. 想直接改设计 → 人工编辑 `design.md` 文件（门禁不拦你改文件），改完再 approve。
5. 方向性错误 → 驳回重出（见下）。

**放行（approve）**
```bash
python scripts/orchestrator_hook.py approve --task <task>
```
- 作用：清 `await_design_approval`，主 Agent 才能派 `python-coder` 进编码。
- 幂等：门禁未拉起时跑 `approve` 是安全 no-op（exit 0，不置位、不报错）。
- 忘了 task 名？先 `status` 看当前活跃 task。

**驳回 / 重出设计**
- 无独立 `reject` 命令；门禁本身已拦住编码，不会自动污染后续阶段。两种处理：
  - 小修：直接改 `design.md` 文件，改完 `approve`。
  - 方向性错误：先 `reset` 该 task，再 `start` 重跑（因 `advance` 幂等键 `design@N` 已归档，不 reset 会跳过重复推进）。

**排查**
- 误派了 code 但被拦：门禁拦截推进并返回提示，不写 `code` 阶段；`approve` 后正常走。
- 想确认门禁真拦住：`approve` 前跑 `status`，`await_design_approval` 应为 `true`；此期间 Stop 会被 🛑 提示拦截。
- 确属误触发：reset 后 `start`，确保 design 被跳过（或提供既有 design 文档让其直连）。

**与 eval 的关系**：eval（`run_workflow`）自动批准 design，不卡门禁；因此**只有本地 hook 路径需要人工 approve**，离线度量不受影响。

## 合并门禁（方案④）：test → done 强制人工 merge

`test` 阶段完成、全链路产物（scope/design/code/review/test）落盘后，编排**不自动收尾**，
而是在 `state.json` 的 `hook.await_merge_approval` 置位，进入合并门禁——这是 code→review→test
全自动跑完后的**唯一人工关卡**（设计门禁在更前，二者同构）：

- PostToolUse：test 产出后返回"🛑 合并门禁"提示，要求人工 review 全链路产物；
  门禁解除前，任何推进（含误派发的后续动作）都被拦截。
- Stop：门禁未过时 `continue:false` 拦截停止，强制先 review 再合并。
- 放行：人工 review 确认后运行 `python scripts/orchestrator_hook.py merge --task <task>`，
  清除门禁、置 `phase=done`、释放 `.current` 指针，编排正式完成。

> 合并门禁同样**纯 hook 层**（存于 envelope 的 `hook` 段），领域状态机只多一个
> `PHASE_MERGE` 过渡阶段（test → merge → done）；`run_workflow` 在离线模式自动批准
> merge，故 eval / dry-run 不受影响（eval 自动批准）。

### 操作手册（人工视角）

**何时会撞上门禁**
- **触发**：任何走到 test 完成的任务（包括 design 被默认跳过的多数任务）。
- **怎么知道撞上了**：对话出现 `🛑 合并门禁：test 阶段已完成` 提示；或 `status` 显示 `待人工合并放行：是`。

**标准流程**
1. 打开 `.codebuddy/run/{task}/` 下全部产物：scope.md / design.md（如有）/ code.diff / review.md / tests.py。
2. 按 `scope.md` 验收标准**逐条核对**代码与测试是否真满足；特别留意 `[BLOCKING]` 与伪绿是否已清除。
3. 无误 → 运行放行命令（下方）。
4. 方向性错误 → 先 `reset` 该 task，再 `start` 重跑。

**放行（merge）**
```bash
python scripts/orchestrator_hook.py merge --task <task>
```
- 作用：清 `await_merge_approval`、置 `phase=done`、释放 `.current`，编排完成。
- 幂等：门禁未拉起时跑 `merge` 是安全 no-op（exit 0，不报错）。
- 忘了 task 名？先 `status` 看当前活跃 task。

**与 design 门禁的关系**：design 门禁卡"编码前"，合并门禁卡"收尾前"，二者独立——
design 被跳过（多数任务）时仍会有合并门禁；design 触发时两个门禁都要过。

## PR 模型（方案①）：task → branch → commit → PR

编排产出从"落在工作区"升级为"一个 branch + 提交 + PR"，对齐成熟公司的 agentic
SDLC：Agent 的产物作为 PR 进入真实研发管线，由人类在 merge 点把关，而非 session 内直接落地。

行为（**默认关闭**，需环境变量 `ORCH_GIT=1` 启用；启用前与旧行为完全一致）：
- `start`：为 task 建/切专属分支 `orch/<task>`；
- 每次 PostToolUse 推进：把 `.codebuddy/run/<task>/` 产物提交进该分支；
- `merge`（合并门禁放行）：推送分支并开 PR（`gh pr create`），人类在 PR 处 review + 合并。

防御式设计：单次 git/gh 调用失败只记日志、不抛异常、不阻断编排；未启用时所有函数
安全 no-op。具体实现见 `scripts/orchestrator_git.py`，接线在 `orchestrator_hook.py`。

## scope.md 质量锚点（方案 C）

`scope` 阶段是后续所有阶段的**唯一质量锚点**（提示词已强制）：

- 必须给出**可测试的验收标准**（拒绝"可用/正确"等主观表述）；
- 必须列出**非功能约束**（性能/安全/兼容/可观测性），review 据此判 `[BLOCKING]`；
- 必须给出**架构边界初判**（涉及 app/模块、接口边界、复用既有资产）。

`architecture-designer` 的 `design.md` 须**逐条追溯** scope 的验收标准与非功能约束；
`code-reviewer` 严格对照 scope 验收标准/非功能约束判 `[BLOCKING]`。设计层级的问题
（架构坏、非功能没覆盖）因此能被 review 阶段兜住，而非绕过。

## 本地验证（不消耗 token）
```bash
# 开启一次编排
python scripts/orchestrator_hook.py start --task order-service-mcp --request "补充 MCP 连接池指标"
# 离线跑完整阶段链，仅验证结构
python scripts/orchestrator_hook.py dry-run --task order-service-mcp --request "..."
# 查看进度 / 中止
python scripts/orchestrator_hook.py status
python scripts/orchestrator_hook.py reset
# 设计门禁放行（design 触发后、编码前，由人工 review 后执行）
python scripts/orchestrator_hook.py approve --task order-service-mcp
# 合并门禁放行（test 完成后、收尾前，由人工 review 后执行）
python scripts/orchestrator_hook.py merge --task order-service-mcp
```

## 第二道回路：CI 失败自动修复（无需主 Agent 在场）
上面的 hook 编排是"对话内驱动"，需要主 Agent 在场。CI 侧另有一条完全无人回路：
```
ci-tests.yml 失败
  └→ ci-autofix.yml (workflow_run 触发)
       ├→ triage:      scripts/ci_failure_triage.py 结构化分诊
       ├→ autofix:     按 kind 派 codebuddy Agent 修复 → 复跑单测 → 开修复 PR
       └→ notify-human: 无法归因时开 ci-escalation issue
```
本地验证分诊逻辑：
```bash
python -m pytest scripts/test_ci_failure_triage.py -q
```

## 第三道防线：文档-执行漂移自检
本文件与 `multi-agent-workflow.md` 描述的阶段映射、阈值、hooks 注册、子命令清单，
由 `scripts/check_workflow_drift.py` 在 CI（ci-tests.yml 的 `workflow-drift` job）自动校验。
**真相源以代码/工作流为准**，文档漂移即标红。此外，golden tasks 流程级回归
（`scripts/run_eval_regression.py`，CI 的 `eval-regression` job，基线
`benchmark_results/eval_baseline.json`）与漂移自检并列，每次 PR 校验「编排流程」
未被改坏——这是代替人工 review 的客观凭证（方案③）。
```bash
python scripts/check_workflow_drift.py          # 退出码 1 = 存在漂移
```
  校验项：D1 阶段→agent 映射、D2 subagent 文件存在性、D3 覆盖率门禁、
  D4 重试上限（`MAX_AUTO_RETRIES = 2`）、D5 派单上限、D6 文档引用文件存在性、
  D7 hooks 注册（settings.json 三事件）、D8 子命令一致性、D9 合并门禁（PHASE_MERGE 迁移 +
  `merge` 子命令/Stop 拦截 + 本文档合并门禁说明）、D10 硬性质量门（QUALITY_COVERAGE_GATE
  常量 + check_quality_gate 实现 + advance(PHASE_TEST) 真实调用）、D11 eval 回归
  （ci-tests.yml 引用 `run_eval_regression.py` + baseline 文件 `benchmark_results/eval_baseline.json` 存在）、
  D12 PR 模型（orchestrator_git.py 实现 ensure_branch/commit_task/open_pr + orchestrator_hook.py 接线 + 本文档 PR 模型说明）、
  D13 PHASE_CONTRACT 与阶段链（PHASE_CONTRACT + STAGE_AGENT 映射 + advance() 三处转换 + contract_gen.py 存在）、
  D14 沙箱门禁（check_sandbox 实现 + advance(PHASE_TEST) 真实调用 + sandbox_run.py 存在 + 软跳过约定）、
  D15 对抗式审查（PHASE_REDTEAM + STAGE_AGENT 映射 + advance() 转换 + red-team-reviewer.md 存在）、
  D16 每 PR eval 回归（ci-tests.yml pull_request 触发 + run_eval_regression.py + baseline 存在）、
  D17 形式化接线（critical_paths.yaml + check_critical_guarantees + advance() 调用 + 软跳过）、
  D18 元评测（eval_meta_check.py + semgrep 规则 + 判据层映射一致性）。
  改了 `STAGE_AGENT` 或任何阈值后，必须同步更新本文件对应表格与 `.codebuddy/settings.json`。

手动对某次历史失败重跑分诊（不改代码）：
```bash
gh run download <run_id> -D artifacts
python scripts/ci_failure_triage.py --artifacts artifacts --out triage.json --summary-out triage.md
```
