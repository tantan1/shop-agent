# Scope: MLOps 闭环模块 · 1 周 MVP 切片

> 本文件是「顶层 scope」(`.codebuddy/run/mlops/scope.md`) 按"两个确认点接收"裁剪后的**落地切片**，是本次编码/测试/上线的唯一质量锚点。
> 顶层 scope 定义的**统一状态机**与**接口契约所有权**本切片继续 conform；其"范围边界/非功能"中被明确后移的项，本切片不再实现（见第 4 节）。
> 目标：1 周内交付一条**跑通的 HITL 闭环**（人工在回路不可跳过 + 发布 fail-closed），复用现有训练/评测/serving 脚本，不做企业级加固。

## 1. 目标（一句话可验证）

在 `shop-agent` 新增 `modules/mlops`，实现「人工建批 → 标注 → 确认训练(封装现有脚本) → 评测(封装现有脚本) → 人工满意才发布(fail-closed)」闭环，并提供极简独立页面；全程状态可追溯、HITL 不可跳过、发布受评测结果门禁约束。

## 2. 设计依据与复用（权威源 + 复用面）

- **顶层 scope**：`.codebuddy/run/mlops/scope.md`（状态机/契约所有权唯一权威，本切片 conform）。
- **训练脚本（封装，不重写）**：`scripts/train/sft/train_unified_sft.py`（`--smoke` 30 样本/5 步验证链路，`--merge-out` 产版）。
- **评测脚本（封装，不重写）**：`scripts/eval/param/eval_sft_before_after.py`（base vs sft，输出 `eval_sft_before_after.json`，含 `overall` 指标）。
- **权限/审计**：`src/modules/auth/dependencies.py` 的 `require_admin` / `get_current_client`；`src/ports/audit.py` 的 `audit.log`。
- **DB/ORM**：`src/shared/database.py` 的 `Base` / `get_db` / `get_async_session`。
- **页面挂载**：`src/main.py` 的 `StaticFiles` 挂载范式（已挂 `/mlops-ui`）。
- **落点目录**：`apps/shop-agent/src/modules/mlops/`。
- **顶层口径调整（F4·一致性）**：顶层 scope 的 A 项为「监控指标驱动自动筛选」，本切片以 **Langfuse tool-call 样本**作为 A 的数据源（非单纯"后移"，属切片对顶层的口径重定义）；顶层其余状态机/契约本切片继续 conform，故不另改顶层权威。

## 3. 范围（In scope，逐条可独立验收）

- [ ] **A. 复核任务创建（数据源=Langfuse）**：提供 `POST /mlops/tasks`（人工建批，含 `sample_id`=Langfuse trace/observation id / `metric_snapshot` / `reason`），生成 `pending_review` 工单。*`sample` 即「一次工具选择决策」，取自 Langfuse 已自动记录的 tool-call trace（shop-agent 已接入 Langfuse v4 且带 `mask_for_langfuse` 脱敏），`sample_id` 指向其 trace_id / observation_id，由 MVP 闭环从 Langfuse API 拉取「已评分的 tool-call」作为样本（拉取/映射细节见 design 阶段）。MVP 不做监控自动打标触发，仍为人工发起。*
- [ ] **B. 状态机 + 人工复核（质检门禁语义）**：`pending_review → reviewing → labeled → ... → published/rejected` 全状态在 `state_machine.py` 声明；提供认领/标注接口，状态跃迁经 `can_transition` 校验 + `audit.log` 留痕。*标注语义为「人工质检门禁」：`label` ∈ {correct, wrong, needs_fix} 判定工具选择是否正确；correct/wrong 信号来源为 **Langfuse 的 score**（规则初判：工具执行失败/用户明确否定→wrong，下游任务成功→correct，自动经 Langfuse Score API 打；人工确认 correct/wrong 走 Langfuse UI 或本 MVP 页面打分），**标注结果不直接回流成训练数据集**（与训练集解耦；回流见第 4 节后移项）。*
- [ ] **C. 训练编排（封装，单任务串行语义）**：`labeled → training_confirmed → training`；`confirm-training` 触发 subprocess 跑 `train_unified_sft.py`，产物落 `outputs/mlops/<version>` 并 merge 回 base（使 `trained_model_path` 即完整模型）；超时即终止置 `rejected`。实际 cmd（见 `services._run_training`）：`python scripts/train/sft/train_unified_sft.py --model models/Qwen3-1.7B --data <dataset_ref> --output-dir outputs/mlops/<version> [--smoke] --merge --merge-out outputs/mlops/<version>`。**本切片训练为单任务语义：约定同一时刻仅一个训练 subprocess 运行，不实现并发锁/队列（多任务并行见第 4 节后移项）。**
- [ ] **D. 评测（fail-closed）**：`training → evaluated`；subprocess 跑 `scripts/eval/param/eval_sft_before_after.py --base models/Qwen3-1.7B --sft <trained_model_path> --data <eval_data> --device <cuda|cpu> --max-samples N`，对照阈值（field_f1 / value_exact_match_rate ≥ 0.6）判 `eval_pass`；指标缺失/未达标即 `eval_pass=False`。
- [ ] **E. 发布（fail-closed）**：仅当 `eval_pass==True` 且人工 `publish` 才置 `published`；写 `models/active_model.txt`；可选调 `scripts/publish_model.sh` 重启 serving（短停机）。
- [ ] **F. 极简页面**：`/mlops-ui/`（静态 HTML + fetch API），承载建批/标注/确认训练/评测/发布全流程，复用 `monitoring` 路由 + `require_admin` 鉴权。

### 验收标准（可测试，禁止主观表述）

| 范围项 | 验收标准（输入 → 预期输出/行为） |
|---|---|
| A | `POST /mlops/tasks` 带合法 key → 返回 `pending_review` 工单，库中存在对应行；无 key → 401。 |
| B | `claim`(`pending_review`→`reviewing`) 后须 `reviewing` 态；`label` 仅 `reviewing` 可调用：label=correct→`labeled` 并留 audit 行（可训练），label=wrong/needs_fix→`rejected`（质检门禁，不可训练）；非 `reviewing` 调 `label` 被拒；`labeled`→`published` 被拒（非法跃迁）。 |
| C | `confirm-training` 后状态进入 `training` 并启动后台训练；`--smoke` 链路跑完产 `outputs/mlops/<v>` 并置 `evaluated`；超时(>`timeout_s`)置 `rejected`。 |
| D | `eval` 后 `eval_result` 非空且 `eval_pass` 为布尔；阈值未达 → `eval_pass=false`。 |
| E | `evaluated`+`eval_pass=false` 时 `publish` 返回 4xx 且不改变状态（fail-closed）；`eval_pass=true`+人工 `publish` 才置 `published` 且写入 `active_model.txt`。 |
| F | 页面可完成全链路操作；非 admin key 访问写操作为 403。 |

## 4. 范围边界（Out of scope，本切片明确后移二期）

- **不做**监控指标自动打标（A 改为人工建批）。
- **不做** 训练数据 PII 脱敏/匿名化——本项目数据为合成 shop 参数，风险低，记二期；**注意** 数据源 Langfuse 已在采集层经 `mask_for_langfuse` 统一脱敏（见 `langfuse_callback.py`），trace/score 不含原始 PII，仅训练数据集本身不做额外脱敏。
- **不做** 回滚演练 / 零停机发布——仅保留 `active_model.txt` 引用，手动回滚。
- **不做** 完整 `shop_agent_mlops_*` 指标注册——仅 `shop_agent_mlops_transitions_total` / `shop_agent_mlops_training_duration_seconds` 两个关键计数器（见 `metrics.py`）。
- **不做** 并发乐观锁 / 审计全维度 / 多轮 code review / 自动 serving 重载编排。
- **不做** 多任务并行训练（受控并发）：本切片训练为**单任务串行语义**，代码层每个任务是独立后台进程但**不提供 GPU 资源池管理 / `CUDA_VISIBLE_DEVICES` 按卡分配 / `max_concurrent` 并发上限 / 等待队列**，故约定一次只跑一个训练（代码层多任务会并发起进程但将互抢 GPU 显存导致不稳定/OOM）。**受控并行训练属二期能力**——接 MLflow(训练后端)+ 自研训练调度器时实现：GPU 池分配、按卡绑定、并发上限、失败重排；这同时是「真跑业务/长期用」时的吞吐扩展点。
- **回流明确（目标偏差提示，F3）**：本切片 correct/wrong 信号仅作「质检门禁」判定（决定样本是否可训练），**不回流进训练数据集**；「工具选择信号 → SFT 训练数据」为二期能力（标注样本经 Langfuse 导出/转换喂 `dataset_ref`）。故 MVP 跑通验证的是 HITL 闭环与门禁，**不等于工具选择信号已用于模型训练**——这是合理裁剪，但需明确知晓。
- **不做** 工具选择决策依据采集（selection 级 Langfuse span）：MVP 复用 Langfuse 自动捕获的**最终 tool-call trace** 已足以判定 correct/wrong（选了什么工具 + 执行成败）；P0/P1/P2 多级决策的中间分数（embedding/线性头 logits，`tool_select_stages.py` 的 `scored_candidates`）未进 Langfuse。若需「为什么选错」归因分析，**二期**在各 selection stage 加 `@observe` span 把 `scored_candidates` 记成 Langfuse metadata（`langfuse_callback.py` 已提供 `observe` 装饰器，数行即可）。

## 5. 非功能约束（本切片硬约束，review 据此判 [BLOCKING]）

- **HITL 不可跳过（铁律 1）**：`labeled → training_confirmed`、`evaluated → published`、所有 `reject`、标注确认，必须由 `require_admin` 认证的调用方经对应接口触发；状态机层 `can_transition` 强制校验，任何自动化路径不得替代。
- **发布 fail-closed（铁律 2）**：`evaluated → published` 仅当 `eval_pass is True`；否则 `can_transition` 拒绝，且 `publish` 接口不得改变状态/写 active 文件。
- **审计留痕**：每次状态跃迁写 `audit.log(event="mlops.transition", decision=allow/deny)`，含 actor 与 from→to。
- **复用约束**：训练/评测严格 subprocess 封装现有脚本，不 fork 逻辑；模型族锁定 qwen3 小模型。
- **可观测（最小）**：`shop_agent_mlops_transitions_total{transition,decision}` + `shop_agent_mlops_training_duration_seconds`。
- **资源护栏**：训练 subprocess 受 `timeout_s` 超时控制，超时即 `rejected`，不无限占用。

## 6. 架构边界初判（接口契约由顶层独占，本切片 conform）

- **模块**：`modules/mlops/` = `router.py`(路由) / `services.py`(状态机驱动+subprocess 编排) / `state_machine.py`(顶层状态机唯一权威实现) / `models.py`(ORM `ReviewTask` + 建表) / `schemas.py`(请求体) / `metrics.py`(最小指标)。
- **状态机**：完全 conform 顶层 scope 第 6 节定义的状态集合与跃迁（新增任何状态/跃迁须回顶层确认）。
- **接口契约映射（顶层 → 本切片实现）**：
  - `TrainingSpec` → `ConfirmTrainingRequest{dataset_ref, hyperparams, output_version, smoke, timeout_s}`
  - `EvalResult` → `(eval_result JSON, eval_pass bool)`，阈值来自 `config.MLOPS_EVAL_THRESHOLDS`
  - `PublishRequest` → `PublishRequest{rollback_ref}`（rollback 二期）
  - `StateTransition` → 各写接口的 `require_admin` actor + `can_transition` 校验
  - `MonitoringFeed` → **本期未用**（A 改人工建批）
- **发布落点**：`config.MLOPS_ACTIVE_MODEL_FILE`(默认 `models/active_model.txt`)；`config.MLOPS_PUBLISH_CMD` 留空时只写文件不重启，填入 `scripts/publish_model.sh` 即触发重启。

## 7. 阶段编排（1 周切片）

1. 阶段0（本文件）：MVP scope 落地 ✅
2. 编码：A~F 模块（已完成首版，待按本审查 F1/F2 修复 label 质检门禁与 rejected 死跃迁后复审）
3. 测试：`tests/test_mlops_state_machine.py` 覆盖铁律 1/2 + 非法跃迁（已完成）
4. 审查：对照本 scope 验收标准 6 条与非功能铁律判 [BLOCKING]
5. 上线：本地起服务 + `POST` 跑通 smoke 训练 + 评测 + 发布写 active 文件

## 8. 未确认假设（按现有代码惯例处理）

- `MLOPS_PYTHON` 默认 `python`；CUDA 训练环境需指向 `venv_cuda` 的解释器（配置覆盖）。
- `MLOPS_EVAL_DEVICE` 默认 `cuda`；无 GPU 时改 `cpu` 但仍可跑小样本。
- `init_tables()` 在 `main.py` lifespan 内幂等建表，失败不影响其他模块。
- 页面 `/mlops-ui/` 与 API `/api/v1/mlops` 同源，无 CORS 问题（复用项目既有挂载范式）。
- **数据来源（MVP 约定，具体接入见 design 阶段）**：**复用已接入的 Langfuse（v4.x + `mask_for_langfuse` 统一脱敏）作为唯一数据源**。shop-agent 的 ReAct agent 执行已挂 `CallbackHandler`，工具选择（tool-call）自动成 trace；`sample` = 一条带 correct/wrong score 的 tool-call observation，`sample_id` 即其 trace_id / observation_id。correct/wrong 判定：①规则初判（工具执行报错/用户明确否定→wrong，下游任务成功→correct）自动经 Langfuse Score API；②人工在 Langfuse UI 或本 MVP 页面确认/修正。MVP 闭环通过 Langfuse client 拉取「已评分 tool-call」形成复核样本（拉取/字段映射见 design 阶段）。训练数据集仍由 `confirm-training` 的 `dataset_ref` 独立指定（默认 `data/llamafactory/shop_unified_v1.json`），与标注样本解耦；评测数据固定 `data/llamafactory/shop_param_v1.json`。*design 阶段须明确：Langfuse fetch/score 的 API 字段映射、拉取过滤条件、selection 级 span（见第 4 节）是否纳入。*
