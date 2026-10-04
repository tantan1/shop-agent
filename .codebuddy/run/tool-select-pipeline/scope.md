# scope：工具选择四级流水线 —— Phase 1（Stage 接口 + Pipeline 调用器）

## 0. 设计依据（方案 C 特例：依据既有设计文档落地）

| 项 | 内容 |
|---|---|
| 权威文档 | `docs/architecture/tool-select-pipeline-design.md`（设计）<br>`docs/architecture/tool-select-implementation-plan.md`（实施计划） |
| 文档状态 | 两份均已对齐现状代码并经人工评审；本次**不重新论证架构** |
| 权威声明 | 相关架构决策以上述两份文档为唯一权威；本任务只做「实现与衔接」，禁止新建与之一致的冲突方案 |
| 跨篇依赖 | 无（两份文档自洽） |

## 1. 目标

在 `apps/shop-agent/src/modules/chat/core/` 下交付工具选择流水线的**基础设施**（统一 Stage 接口 + Pipeline 调用器），使四层工具选择从硬编码串联变为可配置迭代式编排，**且不改动任何现有调用方行为**。

## 2. 范围（In scope）

1. `tool_select_stage.py`：统一 Stage 接口与数据结构
2. `tool_select_pipeline.py`：Pipeline 调用器（早停 / 候选传递 / 降级 / 超时）
3. 单元测试：每层独立 mock，验证 Pipeline 编排逻辑

## 3. 非范围（Out of scope）

1. **不实现 P0~P3 各具体 Stage**（`RuleFilterStage` / `FaissRecallStage` / `LinearHeadStage` / `LlmFallbackStage`）—— 属实施计划 Phase 4 迁移，本次仅在注册表留登记入口
2. **不改 `react_agent.py`** 及任何现有调用方（Phase 1 期间新 Pipeline 独立存在）
3. **不改 `schemas.py`** 的 `ToolPlan` / `PlannedAction` 结构（复用，不改）
4. 不做外部配置化（YAML / `ToolSelectConfig`）—— 属 Phase 3
5. 不做 ToolHead 服务化 —— 属 Phase 2
6. 不做跨层融合：设计 §4.3 明确各层分数量纲不同、不可比，Pipeline **不计算任何融合值**

## 4. 验收标准（可测试、可客观判定）

| # | 范围项 | 验收条件（输入 → 预期行为） |
|---|--------|--------------------------|
| A1 | Stage 接口 | `ToolSelectStage(ABC)` 定义抽象方法 `run(query, scope, context) -> StageResult`；`name` 经构造注入 |
| A2 | 数据结构 | `CandidateScore(tool, score, source)`、`StageResult(tool, confidence, source, scored_candidates, error)` 字段齐全 |
| A3 | 注册表 | 模块级 `STAGE_REGISTRY: Dict[str, Type[ToolSelectStage]]` 存在，并提供 `register_stage()` 登记入口 |
| A4 | 早停 | 阈值 `{p1:0.85}`，某层返回 `tool="x", confidence=0.9` → `select()` 返回 `stop_condition="plan_complete"`、`actions[0].name=="x"`，且**后续层 `run()` 被调用次数为 0** |
| A5 | 无早停降级 | 所有层 `tool=None` → 返回 `stop_condition="need_llm"`、`actions==[]` |
| A6 | default_tool 降级 | `fallback={"on_all_stages_failed":"default_tool"}` 且无层早停 → 返回 `stop_condition="plan_complete"`、`actions[0].name == all_tools[0]` |
| A7 | 候选传递 | 前一层 `scored_candidates` 工具名集合为 `[a,b]` → 下一层 `run()` 收到的 `scope == [a,b]` |
| A8 | 错误处理 skip | 某层抛异常 + `on_stage_failure="skip"` → 继续下一层，最终结果由后续层决定 |
| A9 | 错误处理 abort | 某层抛异常 + `on_stage_failure="abort"` → 返回 `need_llm`，且后续层 `run()` 未被调用 |
| A10 | 超时强制 | 某层 `run()` 耗时 > `timeouts[name]`（毫秒）→ 该层按 error 处理；配 `skip` 时继续下一层 |
| A11 | 不融合 | 无层早停时 Pipeline 不产出任何跨层融合分数，仅记 `tool_select_no_early_exit` 观测日志 |
| A12 | 测试独立性 | 单测用 mock Stage，不依赖 P0~P3 真实实现，且全部通过 |

## 5. 非功能约束

| 类别 | 约束 |
|---|---|
| 性能 | 每层超时由 `timeouts` 强制（`asyncio.wait_for`），不得出现无界等待；编排本身不引入阻塞式 I/O |
| 兼容 | 复用 `schemas.py` 的 `ToolPlan` / `PlannedAction`，字段与语义不变；`stop_condition` 仅取 `plan_complete` / `need_llm` |
| 可观测 | 每层打结构化日志（stage / tool / confidence / scored_candidates 数量 / error）；无早停时打 `tool_select_no_early_exit` 观测日志 |
| 安全 | 不引入动态 import（`importlib`）；Stage 解析走显式注册表，可静态跳转、可审计 |
| 规范 | 遵循 `pyproject.toml` 的 ruff 规则（line-length 120、target py310）；新增代码通过 `python -m pytest` |

## 6. 架构边界初判

| 维度 | 判定 |
|---|---|
| 涉及 app / 模块 | `apps/shop-agent`（Python/FastAPI 主栈）；新建文件落在 `apps/shop-agent/src/modules/chat/core/` |
| 新建文件 | `tool_select_stage.py`（接口 + 数据结构 + 显式注册表）、`tool_select_pipeline.py`（调用器）、`tests/core/test_tool_select_pipeline.py`（单测） |
| 对外接口边界 | `ToolSelectPipeline.select(query) -> ToolPlan`、`ToolSelectStage.run(query, scope, context) -> StageResult`。两者均为**新增**，不改任何既有对外接口 |
| 复用既有资产 | `schemas.py::ToolPlan` / `PlannedAction`（复用、不改结构）；阶段常量 `STAGE_P0_RULE` 等；工具全集源自 `src/core/permissions.py::_ALL_TOOLS`（`FrozenSet`，Phase 1 由调用方注入，本任务不绑定来源） |
| 不触及 | `react_agent.py`、`react_agent_selection.py`、`llm_service.py`、`local_model_service.py`、`eval_embedding_baseline.py` |

## 7. 未确认假设

1. **`ALL_TOOLS` 来源**：设计文档写作全局 `ALL_TOOLS`，实际仓库工具全集在 `src/core/permissions.py::_ALL_TOOLS`（`FrozenSet`）。Phase 1 假设由调用方注入 `all_tools` 参数，不在本任务内绑定具体来源（Phase 3 配置化时确定）。
2. **调用方异步上下文**：`select()` 为 `async`。假设 `react_agent._plan_tools` 调用点处于事件循环中；若为主线程同步上下文，需调用方用 `asyncio.run` / 事件循环包裹 —— 属 Phase 4 迁移时确认。
3. **具体 Stage 行为细节**（P1 的 Top-K 取值、P2 的 softmax 输出形态）以设计文档 §4.2 / §4.3 为准；本任务不实现这些 Stage，故不在此细化。
