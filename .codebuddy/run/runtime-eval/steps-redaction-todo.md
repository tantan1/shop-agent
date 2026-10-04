# 待办：ChatResponse.steps 敏感字段脱敏（独立于 runtime_eval 的后续任务）

> ## ⚠️ 执行顺序（必读）
> **先 runtime-eval（本次任务）→ 后本文件（后续独立任务）。**
> 1. **本次 `runtime-eval` 先做**：只构建评估层，且评估层**不依赖**有缺陷的 `steps`——靠 design §4.2 多源降级（优先 Langfuse/日志源）+ §4.3 丢弃 `params`/`output_data`，即便 `steps` 当前仍带敏感字段暴露，评估层也不会落盘或放大泄漏。
> 2. **本文件之后再做**：是**服务端侧**治理（改 `routers.py:252` 出口脱敏 / 改 `schemas.py`），属于**改被测服务**，超出本任务 scope 边界，故单列待办、后续独立立项。
> 3. 本文件**不是**本次任务的 action item，仅为设计 §4.0 的论据引用（"内部信号混用户面响应是安全隐患"）。

> **状态**：🔴 **待办（pending）**，不在本次 `runtime-eval` 任务范围内（见 scope.md §3 非范围：不改被测服务）。
> **来源**：本次设计评审时发现的安全隐患，先行记录，后续独立立项处理。
> **评估层处置**：`runtime_eval` 已规避依赖，本文件是服务端治理任务，不影响评估层交付。

---

## 1. 问题描述

`ChatResponse.steps` 字段原样出现在**对外 HTTP 响应**中，且出口无脱敏：

- 字段定义：`apps/shop-agent/src/modules/chat/schemas.py:95`
  `steps: List[Dict[str, Any]]`（处理步骤详情）
- 出口：`apps/shop-agent/src/modules/chat/routers.py:252`
  `return success_response(data=response.model_dump())`
- `success_response`（`src/shared/responses.py:32-34`）仅做 `model_dump()`，**无任何脱敏**。

结果是终端调用方可以拿到 Agent 的内部执行细节，含敏感业务参数与内部决策依据。

## 2. 需要去掉 / 脱敏的字段（逐个列明）

`steps` 的元素形态有两种（见 design.md F3）：`AgentStepResult`（`agent/schemas.py:35-45`）与 remote_api 路径的临时 dict（`orchestrator_remote.py:209-216`）。

### 2.1 必须完全移除（高风险）

| 字段 | 位置 | 风险 |
|---|---|---|
| **`params`** | `IntentResult.params`（`schemas.py:209`，"远程API调用参数"）；经 `step1_understand.py:55` 的 `result.model_dump()` 整体进入 `output_data`；remote_api 路径见 `orchestrator_remote.py:214` | **最高**。工具原始入参，可能含订单号 / 手机号 / 收货地址 / 用户标识等业务 PII |
| **`output_data`（原始整体）** | `AgentStepResult.output_data`（`agent/schemas.py:41`） | 未结构化的任意 dict，内容随步骤变化，**不可控**，其中最敏感者即 `params` |
| **`input_data`** | `AgentStepResult.input_data`（`agent/schemas.py:40`） | 步骤原始输入，可能含用户 query 原文与上下文片段 |
| **`error_message`** | `AgentStepResult.error_message`（`agent/schemas.py:43`） | 可能泄露内部路径、堆栈、依赖名、SQL 片段 |

### 2.2 建议移除（中风险，属内部决策逻辑）

| 字段 | 位置 | 风险 |
|---|---|---|
| `complexity_reason` | `IntentResult`（`schemas.py:218`） | 复杂性判定的**依据说明**，暴露内部决策逻辑与提示词思路 |
| `similarity_score` | `IntentResult`（`schemas.py:211-213`） | FAISS 余弦相似度，暴露检索内部阈值与召回质量，可被用于探测/逆向 |
| `complexity` | `IntentResult`（`schemas.py:214-216`） | 内部路由分档（simple / multi_step / needs_agent） |

### 2.3 可保留（低风险，有排障与评估价值）

| 字段 | 说明 |
|---|---|
| `step_name` | 步骤名（如 `Tool调用(直接)`） |
| `step_order` | 步骤序号 |
| `status` | success / failed / skipped |
| `duration_ms` | 阶段耗时 |
| `action` | 工具名（如 `query-order`），本身非敏感 |
| `intent` | 意图标签（如 `call_remote_api`） |

> 注：`action` / `intent` 正是 `runtime_eval` 判 `tool_correct` / `intent_correct` 所需的信号。若这两项也被移除，评估层走 §4.2 的 Langfuse / 日志源即可，不阻塞。

## 3. 建议的改造方案（三选一，按推荐度排序）

### 方案 A（推荐）：`steps` 默认不返回，仅调试模式 + 管理员鉴权
- 生产响应 `steps` 置空数组；
- 通过 `?debug=1` 且 `require_admin` 鉴权通过时才返回**脱敏后**的 steps；
- 评估/排障走 Langfuse（已有完整 trace），不依赖用户面响应。

### 方案 B：出口统一脱敏
- 在 `agent_chat` 返回前对 `steps` 做字段白名单裁剪（只留 §2.3）；
- `params` 等敏感字段用 `src/shared/redact.py` 的既有脱敏函数处理后再出（若业务确需保留形态）；
- 改动小，但仍向外部暴露内部执行结构。

### 方案 C：保留 steps，但仅保留白名单字段
- 与 B 类似，区别在于不改出口而是改 `ChatResponse` 构造处（`orchestrator_remote.py:205` 等各出参点）；
- 风险：出参点多，易遗漏，不如在出口统一收口。

## 4. 建议的验收方式

1. 对 `agent_chat` 发典型请求，断言响应 `steps` 中**不含** `params` / `input_data` / `output_data` / `error_message` / `complexity_reason` / `similarity_score` 任一字段。
2. 构造含手机号、订单号的 query，断言响应体全文无原始 PII（复用 `src/shared/redact.py` 的判定口径）。
3. 回归：确认 Langfuse 侧 trace 仍含完整信息（内部可观测不受影响）。

## 5. 关联事项：建议补 `X-Trace-Id` 响应头

业界通行做法：响应头返回仅一个 trace ID 字符串（不含任何内部细节），便于排障与关联。
- 成本极低、风险远低于返回完整 `steps`；
- 一旦具备，`runtime_eval` 可直接按 trace_id 精确关联（见 design.md §4.1），无需依赖 conversation_id 过滤日志。
