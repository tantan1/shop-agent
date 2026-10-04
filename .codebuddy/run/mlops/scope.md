# Scope: MLOps 闭环模块（监控 → 人工复核 → 训练 → 评测 → 发布）

> 本文件是 `shop-agent` 新增 `modules/mlops` 模块的**顶层总控 scope**，由「两层 scope」方案的第 0 层产出。
> 后续所有子 scope（A~F）、架构设计、编码、审查、测试**必须引用本文件**，不得超出其边界；顶层独占的接口契约 / 评测阈值 / 统一数据状态机 / 跨切面非功能，子 scope 只能 conform，**不得另立标准**。
> 设计依据：本期为「从零设计 + 封装现有脚本」，无既有设计文档，将触发设计门禁（architecture-designer 产出 design.md 后需人工 review 放行）。

> **落地切片**：完整企业版（本文件）工作量约 6–10 周。已据此裁剪出 **1 周 MVP 切片**并落地于 `.codebuddy/run/mlops-mvp/scope.md`——该切片为唯一 conforming 层，复用本顶层独占领的「统一状态机」与「接口契约所有权」，并显式声明将 PII 脱敏 / 回滚演练 / 完整指标 / 并发锁 / 监控自动打标后移二期。编码、审查、测试以 MVP scope 为直接锚点；本顶层其余非功能项为二期待办，不在本期验收范围内。

## 1. 目标（一句话可验证）

在 `shop-agent` 内新增 `modules/mlops` 模块，打通「监控指标驱动筛选 → 人工复核标注 → 人工确认训练 → PyTorch+qwen3 小模型训练（封装现有脚本）→ 评测达标判定 → 满意则发布更新」的闭环，并提供独立页面供人工在回路操作，全程状态可追溯、阈值可配置、HITL 决策不可跳过。

## 2. 设计依据与既有资产（权威源 + 复用面）

- **监控数据来源**：`apps/shop-agent/src/modules/monitoring/metrics.py` 已产出 Prometheus 指标（`shop_agent_agent_chat_total`、`shop_agent_tokens_total`、`shop_agent_conversations_total`、`shop_agent_exceptions_total`、`shop_agent_l2_*`、`shop_agent_redis_available` 等）。MLOps 的「监控驱动筛选」消费其异常率/质量退化信号作为打标触发源。
- **可复用训练/评测脚本（封装，不重写）**：
  - 评测：`benchmark/eval/run_eval.py`、`benchmark/arena_models_qwen3*.json`、`data/lscale_M_holdout.json`、`eval_sft_before_after.json`
  - 训练范式参考：`docs/shop-agent-blog-series/17-LLaMA-Factory微调踩坑记录.md`、现有 qwen3 部署 `apps/vllm-qwen3/`
- **复用工程设施**：`src/core/permissions.py`（角色/工具权限/审计）、`src/core/security.py`、`src/modules/auth/`（身份）、`monitoring/metrics.py` 的指标注册范式、`modules/monitoring/router.py` 的 FastAPI 路由范式。
- **目标落点**：`apps/shop-agent/src/modules/mlops/`（Python/FastAPI，与 shop-agent 同栈）。

## 3. 范围（In scope，逐条可独立验收）

- [ ] **A. 监控驱动筛选**：从 monitoring 指标/异常流中判定「需人工复核的数据样本集」，生成待复核工单（不直接用原始指标做训练，先做样本化）。
- [ ] **B. 人工复核标注 + 统一状态机**：独立页面展示待复核样本，人工标注（正确/错误/需修正 + 修正标签），状态按顶层状态机流转。
- [ ] **C. 训练任务编排**：封装现有训练脚本，以「任务规格」驱动一次 PyTorch+qwen3 小模型微调，产出版本化模型产物；不重写训练逻辑。
- [ ] **D. 评测达标判定**：封装 `run_eval.py` 对训练产物跑评测，对照顶层阈值输出 `pass/fail` 结构化结果，不自动发布。
- [ ] **E. 模型发布更新**：仅当人工确认「满意」且评测 `pass` 时，将模型产物发布到 serving（如触发 `apps/vllm-qwen3` 重载），并更新系统生效。
- [ ] **F. 独立 MLOps 页面**：独立路由（页面或 API+静态页）承载 A~E 的全部人工操作与状态可视化，与既有 chat 页面解耦。

### 验收标准（每条范围项配可测试条件，禁止主观表述）

| 范围项 | 验收标准（输入 → 预期输出/行为） |
|---|---|
| A | 给定 monitoring 异常率超 `flag_threshold` 的窗口 → 系统生成 ≥1 条 `pending_review` 工单，工单含样本 id、指标快照、触发原因；无异常时不生成工单（误触发率 0 容忍由阈值配置保证，不靠人工肉眼）。 |
| B | 人工在页面将某工单由 `pending_review` 置 `reviewing` 并标注后 → 状态变为 `labeled`，标注内容持久化；非 `reviewing` 持有者不能并发修改（乐观锁/行锁）。 |
| C | 收到 `training_confirmed` 工单 + 合法任务规格 → 启动一次训练，产出 `models/mlops/<version>/` 产物 + 训练日志；训练超时（> `train_timeout_s`）自动终止并置 `rejected`；训练脚本退出非 0 同样置 `rejected` 并保留日志。 |
| D | 训练产物进入 `evaluated` → 评测脚本输出 `{metric, value, threshold, pass}` 列表；任一核心指标未达阈值 → `pass=false`；评测结果与阈值同屏可查。 |
| E | 仅当 `pass=true` **且** 人工显式点「发布/满意」→ 触发 serving 更新并将工单置 `published`；若 `pass=false` 或人工未确认 → **禁止**任何发布动作（fail-closed）。 |
| F | 独立页面可完成「查看待复核 → 标注 → 确认训练 → 查看评测 → 确认发布」全流程；无 chat 页面依赖；未登录/无权限角色访问被拒（401/403）。 |

## 4. 非范围（Out of scope，防范围蔓延）

- 不做训练算法/损失函数/模型结构的重新研究（复用 LLaMA-Factory + qwen3 既定范式）。
- 不做监控指标采集本身的改造（monitoring 模块已存在，MLOps 只消费）。
- 不做通用 MLOps 平台（多租户、流水线 DAG 编排引擎），本期为单闭环、单模型族（qwen3 小模型）。
- 不做自动超参搜索 / 自动数据增强；训练超参由任务规格显式给定。
- 不做训练数据的自动产出（数据来自 monitoring 筛选 + 人工标注，不接外部数据湖）。

## 5. 非功能约束（review 阶段据此判 [BLOCKING]）

- **安全 / HITL 不可跳过**：`labeled → training_confirmed`、`evaluated → published` 两个跃迁**必须由已认证人工显式触发**；任何自动化路径不得替代。所有状态跃迁写审计日志（谁/何时/从→到/原因）。复用 `src/core/permissions.py` 的角色校验。
- **数据保护**：进入训练集的样本若含 PII，须先脱敏/匿名化（复用现有脱敏能力，见 gateway 07 篇约定），训练产物与日志不得含原始 PII。
- **可观测性**：训练任务进度、评测结果、状态机跃迁均暴露 Prometheus 指标（命名 `shop_agent_mlops_*`）+ 结构化日志；复用 `monitoring/metrics.py` 的 `register` 范式，不另起指标体系。
- **资源护栏**：训练任务强制 `train_timeout_s` + 资源上限（GPU 卡数 / 内存），超时即终止，防止无人看管占用。
- **兼容性**：训练/评测严格复用现有脚本（`run_eval.py`、LLaMA-Factory 配置），MLOps 仅做编排与契约适配，不 fork 训练逻辑；模型族锁定 qwen3 小模型。
- **幂等与回滚**：发布动作可回滚（保留上一生效版本引用）；重复触发同一 `published` 工单幂等。

## 6. 架构边界初判（接口契约由顶层独占，子 scope 只 conform）

- **涉及模块**：新建 `modules/mlops/`（router / services / state_store / schemas / training_adapter / eval_adapter）；消费 `modules/monitoring`；触发 `apps/vllm-qwen3` serving 更新。
- **顶层独占的统一数据状态机（唯一权威）**：

  ```
  pending_review ──(人工认领)──▶ reviewing
  reviewing ──────(人工标注)───▶ labeled
  labeled ────────(人工确认训练)▶ training_confirmed
  training_confirmed ─(启动训练)▶ training
  training ───────(训练完成)────▶ evaluated
  training ───────(超时/失败)───▶ rejected ──(回到 labeled)
  evaluated ──────(人工满意+pass)▶ published
  evaluated ──────(人工打回/不达标)▶ rejected ──(回到 labeled)
  ```
  **硬规则**：状态跃迁表是顶层唯一权威；子 scope 不得增删状态或新增跃迁，只能细化「某跃迁内的校验逻辑」。

- **顶层独占的接口契约（签名级，子 scope 实现细节可不同，但契约不可变）**：
  1. `MonitoringFeed`：monitoring → mlops 的打标触发事件，`{sample_id, metric_snapshot, reason, ts}`（子 scope A 细化采集方式）。
  2. `TrainingSpec`：mlops → 训练封装层，`{base_model, dataset_ref, hyperparams, output_version, resource_limit, timeout_s}`（子 scope C 细化）。
  3. `EvalResult`：评测层 → 决策层，`{version, metrics: [{name, value, threshold, pass}], overall_pass: bool}`（子 scope D 细化指标集）。
  4. `PublishRequest`：决策层 → serving，`{version, rollback_ref}`（子 scope E 细化触发方式）。
  5. `StateTransition`：UI/API → 状态机，`{from, to, actor, reason}`（子 scope B/F 消费，须带认证 actor）。

- **顶层独占的评测阈值契约（子 scope D 填具体值，不得另立阈值维度）**：
  - 阈值以配置文件（如 `mlops/thresholds.yaml`）集中存放，顶层定义**键名与语义**（如 `core_accuracy_min`、`regression_max`、`p99_latency_budget`），子 scope D 只填数值。
  - 阈值变更需走配置评审，不在代码里硬编码魔法数。

- **复用资产**：`permissions.py`（鉴权）、`security.py`（认证上下文）、`monitoring/metrics.py`（指标注册范式）、`auth`（身份）、DB（状态持久化，沿用 shop-agent 现有 ORM/连接）。

## 7. 两层 scope 拆分与阶段编排

> 顶层拥有：状态机、接口契约、评测阈值契约、跨切面非功能。以下子 scope **只 conform，不另立标准**。

| 子 scope | 负责面 | 复用顶层契约 |
|---|---|---|
| A 监控驱动筛选 | 指标→样本化工单 | `MonitoringFeed`、状态机 `pending_review` 入口 |
| B 人工复核标注 | 页面标注 + 状态机 `reviewing/labeled` | `StateTransition`、状态机 |
| C 训练任务编排 | 封装现有脚本跑训练 | `TrainingSpec`、状态机 `training_confirmed/training` |
| D 评测达标判定 | 跑评测对照阈值 | `EvalResult`、阈值契约、状态机 `evaluated` |
| E 模型发布更新 | 触发 serving 更新 | `PublishRequest`、状态机 `published/rejected`、fail-closed |
| F 独立 MLOps 页面 | 承载 A~E 人工操作与可视化 | 全部契约 + 权限/审计 |

**建议编排（每子 scope 独立 design.md + 编码 + 审查；顶层是合并门）**：
1. 阶段0（本文件）：顶层 scope 确认 ✅ → 进入设计门禁。
2. 阶段1 `architecture-designer`：产出 `modules/mlops/design.md`，逐条追溯本 scope 验收标准与非功能约束（触发设计门禁，需人工 review 放行）。
3. 子 scope 落地顺序（依赖链）：F 先定页面/路由骨架与 `StateTransition` API → A 接 monitoring → B 标注 → C 训练封装 → D 评测 → E 发布；C/D/E 可复用同一 `training_adapter`/`eval_adapter` 抽象并行细化。
4. 阶段末 `code-reviewer` + `test-generator`：重点核对「HITL 不可跳过」「fail-closed」「阈值不硬编码」「状态机不变量」四条顶层铁律。

## 8. 未确认假设（文档未覆盖处，按现有代码惯例处理）

- 训练执行形态：进程内子进程 vs 独立 worker，默认先用「shop-agent 进程内异步子进程 + 资源上限」，若 GPU 隔离需要再拆 worker（在子 scope C 定）。
- serving 更新触发方式：默认「写版本引用 + 调用 vLLM 重载钩子」，具体在子 scope E 结合 `apps/vllm-qwen3` 落地。
- 状态持久化存储：默认沿用 shop-agent 现有 DB/ORM，不新建存储体系。
- 页面形态：默认「FastAPI 新路由 + 内联静态页（HTML/JS）」，不引入前端框架；若后续要并入统一前端再议。
