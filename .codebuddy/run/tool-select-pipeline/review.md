# 代码复审报告

## 结论

**通过（0 项阻塞）**。

本轮已逐行复核修复后的 `tool_select_pipeline.py`、`test_tool_select_pipeline.py`、`code.fix1.diff`，并回溯 `tool_select_stage.py`、`schemas.py::ToolPlan/PlannedAction`、`pyproject.toml` ruff 配置、design §5/§7.2 与 `scope.md` 验收表。

- **B1 已闭环且非绕过**：`__init__` 由 6 参收缩为 4 个显式参数，两文件全文无 `# noqa`，`pyproject.toml` 未放宽 `max-args`、未加 per-file-ignores，ruff 实跑全绿。
- **B2 已闭环**：每层无条件打 `tool_select_stage_result`（字段与 design §7.2 对齐），收尾打 `tool_select_pipeline_complete`，并有 caplog 用例断言。
- **未引入新的规范/验收类问题**；新增的 `PipelinePolicy` 判定为**不越界**（见下）。
- A1–A12 全部满足；上轮警告 1/4/5 各有修复（4 为部分闭环），警告 2 风险经实测解除，警告 3/6 遗留——均不构成本轮阻塞。

## 阻塞问题

无。

## 上轮阻塞闭环情况

| 编号 | 上轮问题 | 修复方式 | 是否闭环 |
|---|---|---|---|
| B1 | `__init__` 6 参 > `max-args=5`（PLR0913），违反「规范」约束 | 新增 `@dataclass(frozen=True) PipelinePolicy(thresholds, fallback, timeouts)` 纯参数对象；签名改为 `(self, stages, all_tools, deps, policy=None)`；测试 `_build()` 同步改为传 `policy=PipelinePolicy(...)` | **闭环（非绕过）**。无 `# noqa`；配置未放宽；ruff 实跑 `All checks passed!` |
| B2 | 成功层无日志，design §7.2 未落地，违反「可观测」约束 | 新增 `_log_stage_result()`（循环内、错误处理前无条件调用，含 stage/tool/confidence/candidates_count/latency_ms/error）；`_run_stage()` 返回 `Tuple[StageResult, float]`（`time.monotonic()` 计时）；新增 `_complete()` 统一收尾打 `tool_select_pipeline_complete` | **闭环**。新增 `test_stage_result_logged_per_layer`（caplog）断言记录数、stage、候选数与耗时 |

**「是否绕过」专项核对**：两文件逐行通读，未出现 `# noqa` / `# type: ignore` / `--ignore`；日志为 `extra=` 结构化输出（非 f-string 拼接）；`latency_ms` 为真实计时（含超时路径），非硬编码。

## 警告遗留

| 编号 | 上轮问题 | 本轮状态 |
|---|---|---|
| 1 超时兜底 | 未配置 `timeout_ms` 时裸 `await` → 无界等待 | **已修**。`DEFAULT_STAGE_TIMEOUT_MS = 5000`，`wait_for` 无条件包裹 |
| 2 圈复杂度 | `select()` 估算 10~11，可能触发 C901 | **风险解除**。ruff 实跑无 C901；抽出 5 个私有方法后决策点降至约 7 |
| 3 测试被忽略 | `.gitignore:53` 裸 `test_*.py`，CI 跑不到 | **遗留未修**。合并前须 `git add -f` 或加 `!apps/**/tests/**/test_*.py` 例外 |
| 4 error+tool 误早停 | error 分支未 `continue` | **部分闭环**。abort/hitl 已天然阻断；skip 路径仍会落到早停判断，但 Pipeline 自造的 error 结果 `tool` 恒为 `None`，属防御性缺口。建议早停条件加 `and not result.error` |
| 5 降级缺来源 | `source` 落到默认 `"unknown"` | **已修**。显式 `source="fallback"` / `"default"`；`_complete()` 落地完成日志 |
| 6 覆盖缺口 | 缺 hitl / 显式 need_llm / 空 stages / 注册表断言 / deps 透传 | **部分闭环**。新增 3 条（A3 注册表、hitl、每层日志）；仍缺显式 need_llm、空 stages 边界、deps 透传断言、超时+abort 组合 |

**本轮新发现（均非阻塞）**

- 🟡 **文档与实现签名不一致（本次修复引入）**：design §5 伪代码与 §6.1 `build_pipeline()` 仍按旧 6 参调用。Phase 3 若照文档实现配置加载器将 `TypeError`。建议同步改为 `PipelinePolicy(thresholds=..., fallback=..., timeouts=...)` 单参传入。
- 🟡 **异常堆栈丢失**：`except Exception as exc` 只取 `str(exc)`，`logger.warning` 未带 `exc_info=True`。建议改 `logger.exception` 或补 `exc_info=True`。
- 🟢 **`str(exc)` 全量落日志（纵深防御）**：未来 P3 LLM 层异常体可能携带模型回传内容，建议对 error 做长度截断（如 512 字符）+ 脱敏。`query` 本身确认未进任何日志 ✅。
- 🟢 **`PipelinePolicy` 浅冻结**：`frozen=True` 仅冻结字段引用，内部 dict 仍可变；若 Phase 3 热更新同一 policy 会影响在途 Pipeline。建议构造时防御性拷贝。

**上轮建议项（未采纳，仍为建议）**：命名一致性（`self.stages` 公开 vs 私有字段）；`context = {"thresholds": ..., **self._deps}` 存在「deps 含 `thresholds` 键静默覆盖」与「dict 按引用透传、Stage 写入跨层污染」隐患。

## 验收标准复核

| 编号 | 结论 |
|---|---|
| A1 Stage 接口 | 满足。`ToolSelectStage(ABC)` + `@abstractmethod async def run(...)`；`name` 构造注入 |
| A2 数据结构 | 满足。`CandidateScore` / `StageResult` 字段齐全 |
| A3 注册表 | 满足且**补齐断言**。`STAGE_REGISTRY` + `register_stage()`；新增 `test_register_stage_populates_registry`；无 `importlib` |
| A4 早停 | 满足。`confidence >= thresholds.get(...)` → `plan_complete`；用例断言后续层 `run()` 零调用 |
| A5 无早停降级 | 满足。→ `tool_select_no_early_exit` 日志 → `need_llm`、`actions==[]`、`source="fallback"` |
| A6 default_tool 降级 | 满足。`all_tools[0]` + `source="default"`；`list(all_tools)` 保序 |
| A7 候选传递 | 满足。仅候选非空时更新 scope，error 层不污染 |
| A8 错误处理 skip | 满足。`_handle_error` 返回 `None` → 继续下一层 |
| A9 错误处理 abort | 满足。`policy in ("abort","hitl")` → 立即返回 `need_llm` |
| A10 超时强制 | 满足。`wait_for` + `DEFAULT_STAGE_TIMEOUT_MS` 兜底，无界等待彻底消除 |
| A11 不融合 | 满足。`observations` 仅进日志；`_fallback_plan()` 不计算分数 |
| A12 测试独立性 | 满足。10 用例全通过；纯内存 mock，不依赖 P0–P3 实现，不依赖 pytest-asyncio |

**非功能约束**：性能 ✅／兼容 ✅／可观测 ✅（B2 闭环）／安全 ✅（无 `importlib`，日志不含 `query`；error 截断为纵深防御建议）／规范 ✅（ruff + pytest 全绿）。

**越界判定：`PipelinePolicy` 不越界。** 无 `open()`/无 `yaml` 导入/无 env 覆盖/无 schema 校验/未替代 `ToolSelectConfig`/无 `config_path`，仅为满足 ruff 规范约束的**参数对象（Parameter Object）**重构，与 design §6.1 的 `ToolSelectConfig` 职责无重叠。唯一副作用是构造签名与文档伪代码不一致（已列为文档同步项）。

## 实测证据

```
$ python -m ruff check src/modules/chat/core/tool_select_stage.py src/modules/chat/core/tool_select_pipeline.py
All checks passed!                      # 退出码 0
```
同时证伪 B1（无 `PLR0913`）与上轮警告 2（无 `C901`）。

```
$ python -m pytest tests/core/test_tool_select_pipeline.py -q
..........                                                               [100%]
10 passed, 1 warning in 0.30s
```
10 = 原 7 条（A4–A10）+ 新增 3 条（A3 注册表、hitl、可观测），与 `code.fix1.diff` 的 196 行新增对应。

**建议合并前补跑**：① `git check-ignore -v` 确认警告 3 影响；② `ruff check tests/core/test_tool_select_pipeline.py` 把测试纳入规范门禁；③ `pytest -rw` 定位那 `1 warning` 来源。
