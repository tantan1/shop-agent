# 红队审查报告

## 结论

发现 **2 项可复现缺陷（B1 / B2）**，均已在本轮修复并补齐回归测试；经**变异测试**验证：缺陷注入时新增用例精确变红，恢复后全绿。当前实现无遗留阻塞问题。

## 阻塞问题处置

| 编号 | 缺陷 | 复现条件 | 修复 | 状态 |
|---|---|---|---|---|
| B1 | 空消息异常（`str(exc) == ""`）被判为成功层 | `raise RuntimeError()`（无参）→ `StageResult.error=""` → `if result.error:` 为假 → 走成功分支 | `_run_stage` 改为 `error=str(exc) or type(exc).__name__`，保证 error 恒为真值 | **已修** |
| B2 | 早停不校验工具名，产出注册表外工具的 `plan_complete` | P3 层（`threshold: 0.0`）返回 `tool="search-web"`（不在 `all_tools`） | `__init__` 预置 `self._tool_set`；早停条件增加 `result.tool in self._tool_set` | **已修** |

### B1 细节（abort/hitl 策略静默失效）

- 根因：`except Exception as exc: StageResult(..., error=str(exc))`。`str()` 对无参异常返回 `""`，是假值。
- 实际后果：配 `on_stage_failure=abort` 时，空消息异常**不触发终止**，后续层继续调用（违反 A9）；且 `tool_select_stage_error` 告警日志一条都不打（违反可观测约束）——线上该层持续失败时完全不可见。
- 对照组：`RuntimeError("boom")`（带消息）行为正确 → 证明是「空消息」特例，非 abort 逻辑本身问题。

### B2 细节（越界工具名被包装成确定性计划）

- 根因：早停条件无成员校验；`PlannedAction.name: str` 不校验取值域。
- 危害：下游 `react_agent.py:441-449` 按 `plan.to_tool_names()` 过滤工具对象，越界名字匹配不到任何工具 → 可用工具集为空，而 `stop_condition="plan_complete"` 又告知执行器「计划已完成，无需 LLM 再决策」。**P3 LLM 兜底层恰是最可能幻觉出工具名的一层**，故该缺陷在 P3 上 100% 可触发。
- 违反契约：`schemas.py::PlannedAction.name` 明确「工具名，必与 ToolService 注册表一致」。
- 对照组：同名 Stage 返回注册表内的 `"request-return"`（同样 `confidence=0.0`、`threshold=0.0`）→ 正常早停。证明是「越界」特例，非阈值 0.0 的问题。

## 变异测试（防假绿验证）

注入 B1/B2 缺陷后跑全量用例，确认新增测试**必须失败**；恢复后全绿：

```
注入缺陷（error=str(exc) 且去掉 _tool_set 校验）: 2 failed, 11 passed
恢复修复状态                                    : 13 passed
```

失败的 2 条正是 `test_empty_message_exception_is_still_an_error` 与 `test_tool_outside_registry_does_not_early_stop`，证明其断言真实有效、非假绿。

## 非阻塞观察（未修，不阻断）

1. 🟡 **`context` 按引用共享**：`context = {"thresholds": ..., **self._deps}` 每层透传同一 dict，Stage 写入会跨层污染；且 `deps` 若含 `thresholds` 键会静默覆盖阈值。建议每轮拷贝或校验冲突键。
2. 🟡 **`scope` 浅拷贝**：`scope = list(self._all_tools)` 仅防顶层列表被改；若 Stage 就地修改传入的 `scope` 列表（如 `scope.append`），仍会影响 Pipeline。建议传 `tuple` 或每层重拷贝。
3. 🟡 **A6 顺序依赖**：`_fallback_plan()` 取 `all_tools[0]`，而工具全集源自 `FrozenSet`（`src/core/permissions.py::_ALL_TOOLS`），转 `list` 顺序不保证稳定 → 默认工具可能漂移。Phase 3 绑定来源时应固定顺序。
4. 🟡 **`timeout_ms: 0` 语义**：`timeouts.get(name) or DEFAULT` 用 `or`，配置 `0` 会回落默认值。设计 §6.3 已强制正数，行为自洽且更保守，无需改。
5. 🟡 **异常堆栈丢失**：仅记录 `str(exc)`，`logger.warning` 未带 `exc_info=True`。P3 上线后定位成本高，建议改 `logger.exception`。
6. 🟢 **`str(exc)` 全量落日志**：P3 异常体可能携带模型回传内容，建议对 error 做长度截断 + 脱敏（纵深防御）。`query` 本身确认未进任何日志 ✅。
7. 🟢 **`CancelledError` 未被吞**：Python 3.8+ `asyncio.CancelledError` 继承 `BaseException`，`except Exception` 不会误吞，取消语义正确 ✅；`wait_for` 超时后内层协程被 cancel，Stage 若持不可取消资源需自行处理（属 Stage 契约，非 Pipeline 缺陷）。
8. 🟢 **空 `stages` / 空 `all_tools`**：静态推演安全——空 `stages` 不进循环 → `no_early_exit` 日志 → `need_llm`；空 `all_tools` 时 `default_tool` 分支被 `and self._all_tools` 短路 → 回落 `need_llm`。均不崩溃，建议补边界用例。

## 已尝试攻破但实现安全

- `select()` 无共享可变状态（局部变量 + 只读 `self._policy`），可并发调用 ✅
- `observations` 仅进日志，不参与任何决策，无跨层融合 ✅
- 无 `importlib` 动态加载，Stage 解析走显式注册表 ✅
- 早停后更贵层级零调用（成本漏斗成立）✅
