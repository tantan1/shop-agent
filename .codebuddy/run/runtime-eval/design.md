# design.md — shop-agent 运行时可观测性评估层（benchmark/runtime_eval）

> 上游：`.codebuddy/run/runtime-eval/scope.md`
> 阶段：design（待人工 review 放行后进入编码）

## 0. 设计前提：三条实地核实的关键事实（修正 scope 的假设）

| # | 事实 | 对设计的影响 |
|---|------|-------------|
| **F1** | `ChatResponse` **没有** `tool_calls` / `intent` 字段。用户面只有 `message`；内部信号仅存在于 `steps: List[dict]`（意图在 `steps[0].output_data`，工具在 `steps[*].output_data.action`）与可观测面（Langfuse / 结构化日志） | `tool_calls` / `intent` **不能从用户面直读**。按 §4.0 分层原则，内部信号**优先走可观测面**（Langfuse → 日志），`steps` 仅作为 `--debug` 兜底源（§4.2），且取不到时判 SKIP 而非 FAIL |
| **F2** | 真实工具词表是 `query-order` / `check-shipping` / `request-return` / `check-balance` / `coupon-inquiry`（**连字符**命名） | 内置考题的 `expected_tool` 必须用真实词表；scope 验收 4 的 `search_order` 仅作判据纯函数自测的合成值 |
| **F3** | `steps` 有两种形状：pipeline 路径含 `duration_ms`，remote_api 路径**无** `duration_ms` | 归一化须防御式兼容，`duration_ms` 允许为 `None`；端到端延迟以**客户端实测**为准 |

## 1. 架构概述

### 1.1 风格：管道-过滤器 + 纯函数判据内核

本质是**一次性求值流水线**：考题 → 采集 → 归一化 → 判定 → 汇总 → 落盘。数据流单向、无状态。

- **I/O 与判据物理隔离**：唯一的非纯函数是 `collector`（HTTP）。判据、归一化、脱敏、汇总全部为纯函数——这是「判据不得调 LLM、同输入同输出」可被静态检查守护的前提。
- 与 `code_eval_harness` **模式同构**（固定考题 / Driver 适配 / 确定性判据 / summarize），**代码独立、互不 import**（验收 10）。
- **不引入**插件注册中心（六道判据封闭固定，注册中心只服务想象中的扩展）；**不用 async**（串行请求即可，且要避让服务端限流）。

### 1.2 目录结构

```
benchmark/runtime_eval/
  __init__.py
  models.py         RuntimeTask / RuntimeArtifact / StepInfo / Verdict
  tasks.py          固定考题集（≥10 道，真实工具词表）
  collector.py      ★唯一 I/O：HTTP 采集 → 原始响应
  extract.py        纯：原始响应 → 归一化信号（F1/F3）
  redact.py         纯：PII 扫描与脱敏（内置正则，不 import src）
  checks.py         纯：六道判据
  engine.py         纯：run_all / evaluate_one / summarize
  persistence.py    JSONL + report.json + report.md 落盘
  logging_setup.py  结构化日志
  run.py            CLI 入口与退出码
```

### 1.3 依赖方向（单向无环）

```
run → engine → {collector, extract, checks, tasks, persistence, logging_setup}
                   checks → redact
```

**技术栈：仅 Python 标准库**（`urllib.request` / `json` / `re` / `time` / `dataclasses`）。不引 requests/httpx——减少依赖即减少漂移面。

## 2. 核心数据结构

```python
@dataclass RuntimeTask:
    task: str                    # 题目标识
    query: str                   # 发给服务的问句
    expected_tool: str           # 期望工具（真实词表，F2）
    expected_intent: str         # 期望意图标签
    forbid_patterns: list[str]   # 响应禁止出现的模式
    budget_ms: int = 5000        # 延迟预算
    tags: list[str] = []         # 场景标签

@dataclass StepInfo:
    name: str
    duration_ms: float | None    # F3：允许 None
    action: str | None           # 工具名

@dataclass RuntimeArtifact:
    task: str
    request_id: str              # 客户端生成，关联主键
    trace_id: str | None         # 响应/日志中取不到则为 None
    steps: list[StepInfo]
    tool_calls: list[str]        # 多源采集（F1 / §4.2）：Langfuse → 日志 → steps(debug)
    intent: str | None
    response: str                # 原文（PII 判据用）
    response_redacted: str       # 脱敏后（落盘用）
    latency_ms: float            # 客户端实测
    tokens: int | None
    fallback_hit: bool
    error: str | None            # 非 None 表示采集失败
```

## 3. 六道判据与判定语义

| 判据 | 输入 | PASS | FAIL | SKIP |
|---|---|---|---|---|
| `tool_correct` | tool_calls vs expected_tool | 期望工具在调用列表中 | 未调用期望工具 | expected_tool 为空 |
| `intent_correct` | intent vs expected_intent | 相等（大小写不敏感） | 不等 | 任一方为空 |
| `pii_leak` | response 正则扫描 | 无 PII 命中 | 命中手机号/身份证/银行卡 | — |
| `latency_slo` | latency_ms vs budget_ms | ≤ budget | > budget | budget_ms ≤ 0 |
| `forbid` | response vs forbid_patterns | 无命中 | 命中禁止模式 | 列表为空 |
| `no_fallback` | fallback_hit | False | True | — |

**四态语义**：`PASS` / `FAIL` / `SKIP`（判据不适用，不计入通过判定）/ `ERROR`（采集失败，整题记 ERROR 并计入报告）。
**硬门槛（一票否决）**：`tool_correct`、`pii_leak`、`no_fallback`。其余三道 FAIL 亦判整题不过，但 SKIP 不否决。

> **信号缺失即 SKIP**：`tool_calls` / `intent` 取不到时（见 §4.2 多源降级全 miss），`tool_correct` / `intent_correct` 判 **SKIP 而非 FAIL**，并标注 `signal_missing: true`。避免把"采集不到"误判成"行为错误"。
**整题判定**：`passed = 无 ERROR 且 六道判据均非 FAIL`。

## 4. 在线采集器

### 4.0 分层原则（本节的架构前提）

**用户面响应 ≠ 可观测数据面**，采集器必须分层取数，不得混用：

| 数据面 | 内容 | 用于判据 |
|---|---|---|
| **用户面**（`message`） | 给终端用户的回复文本 | `pii_leak` / `forbid` / 语义类 |
| **可观测面**（trace / 结构化日志） | 工具调用、意图标签、阶段耗时、token | `tool_correct` / `intent_correct` / 耗时分解 |

**理由**：内部信号混在用户面响应里是安全隐患（详见 `steps-redaction-todo.md`）。采集器若写死从用户面 `steps` 取内部信号，就等于**依赖一个安全漏洞**——一旦 `steps` 被锁死，采集器立刻失效。因此内部信号一律走可观测面，用户面只取 `message`。

### 4.1 请求与关联键

- **请求**：`POST {base_url}/api/v1/chatagent/agent/chat`，body 含 `message` / `user_id` / **自设 `conversation_id`**，附 `X-API-Key`。
- **关联键**：客户端生成 `request_id`（uuid4）作本地主键；**同时自设 `conversation_id=<request_id>`** 作为服务端侧关联键。
  - 已核实：`conversation_id` 是客户端可控入参（`schemas.py:59`），且**主链路日志会记录它**（`orchestrator.py:432-436`、`services.py:201`）——故按 conversation_id 捞 trace **零服务端改动即可工作**。
  - `trace_id` 响应不带（已知缺口）；若后续服务端补 `X-Trace-Id` 响应头，优先改用它，采集器只需新增一个分支。

### 4.2 多源降级采集（核心设计）

内部信号按优先级依次尝试，任一源拿到即止：

| 优先级 | 信号源 | 生效条件 | 可获取 | 实现方式 |
|---|---|---|---|---|
| 1 | **Langfuse trace** | ① 服务端配了 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`；② 评估侧可访问 Langfuse API | 工具 / 意图 / 阶段耗时 / token | HTTP 查 Langfuse API（Basic auth，可用 `urllib` + `base64` 实现，不引第三方） |
| 2 | **结构化日志** | 能读取服务日志（文件或 Loki） | 工具 / 意图 / 耗时 | 按 conversation_id 过滤日志行 |
| 3 | **响应 `steps`** | 仅 `--debug` 模式**且**字段存在 | 工具 / 意图（**丢弃 params 等敏感字段**） | 解析响应体 |
| 4 | **最小模式** | 总是可用 | 仅 `message` + `latency_ms` | 客户端计时 |

**优先级 1 的启用与查询（已核实，务必知悉）**：
- 被测服务侧启用条件（已核实）：`langfuse_callback.py:29-31` 与 `:83-86` 以环境变量 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` 决定启停；两 key 存在即启用 tracing、handler 返回非 None；未配则 `create_langfuse_handler` 返回 None 静默禁用（`orchestrator.py:340-341` 判空跳过）。`config.py` 中无 Langfuse 配置项——**但这不构成障碍**，因启用只需 env 注入，无需改被测服务代码。
- **如何"确保能用到"（在线评估的环境前置）**：评估 / CI 环境启动被测服务时**注入 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`（指向可用实例）**即可启用 tracing，本仓库 `langfuse_callback.py` 自动生效，**零被测服务代码改动**（符合 scope 边界）。
  - **决策锁定（方案 A）**：本任务**采用 env 注入方式启用 Langfuse**，不增加 `config.py` 配置项、**不改动被测服务任何代码**（符合 scope 边界：不改动被测服务）。第二部分「在线评估」时，由运行/CI 环境注入 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` 即可让 tracing 生效，评估层优先级 1 自动可用。
- **评估层侧必须首版实现查询**：评估层读取相同 `LANGFUSE_*` 凭据，按 `session_id=conversation_id` 检索该请求 trace，解析工具 / 意图 / 阶段耗时 / token。优先级 1 **不再是可选增强**，而是默认主信号源。
- 关联天然成立：`orchestrator.py:332-333` 以 `session_id=conversation_id` 建 trace，故可直接按 conversation_id 检索，**无需服务端改动**。
- **静默降级保底**：若环境未注入 key（Langfuse 不可用），采集器**静默降级**到优先级 2/3/4，不得报错或中断；目标是**默认走 Langfuse**，降级仅作兜底。

**实现复杂度控制**：Langfuse API 查询（分页、鉴权、API 版本）是本节最复杂的部分，但为"确保能用到 Langfuse"必须首版实现。约束：仅用 `urllib` + `base64` 做 Basic auth 查询、分页拉取、按 `session_id` 过滤，封装在 `collector/langfuse_source.py` 单一模块；查询失败一律降级而非抛错，避免主链路被拖复杂。

**信号缺失的处置（关键保护）**：若 `tool_calls` / `intent` 均取不到，`tool_correct` / `intent_correct` 判 **SKIP（不判 FAIL）**，并在报告标注 `signal_missing: true`。这样 `steps` 被锁死或 Langfuse 不可用时，评估优雅降级为"只评用户面质量"（PII / 禁止模式 / 延迟），**不污染通过率**。

### 4.3 归一化与脱敏

- **归一化（F3）**：`duration_ms` 缺失置 `None`；端到端 `latency_ms` 取客户端 `time.monotonic()` 实测（不依赖服务端字段）。
- **脱敏**：`extract` 归一化时**只保留 `step_name` / `duration_ms` / `action`，一律丢弃 `params` / `output_data` 原始内容**。落盘前 `response` 经 `redact.redact_text()` 转为 `response_redacted`。评估层自身不成为泄漏点。

### 4.4 错误处理与 mock

- 连接失败 / 超时 / 非 2xx / JSON 解析失败 → `error` 置位，整题记 `ERROR`，**不中断整轮**。
- `--mock` 模式：不依赖真实服务，由内置 fixture 直接产出 artifacts，用于无服务时跑通全链路并自测判据。

## 5. CLI 契约

```
python -m benchmark.runtime_eval.run --base-url http://127.0.0.1:8000 [--out DIR] [--mock] [--tasks-file PATH]
```

- 退出码：`0` 全部通过；`1` 存在 FAIL；`2` 运行期异常。
- 产物：`{out}/traces.jsonl`（脱敏）、`{out}/runtime_eval_report.json`、`{out}/runtime_eval_report.md`。

## 6. 验收标准追溯（scope 10 条 → 设计决策）

| scope 验收 | 设计保证 |
|---|---|
| 1 考题集 ≥10 且必含三字段 | `tasks.py` 内置 12 道；`load_tasks()` 校验缺字段抛 `ValueError` |
| 2 `validate_trace` | `models.validate_trace(d)` 校验 `request_id` 与 `tool_calls` 存在 |
| 3 采集器返回非空 response 且 latency>0 | §4.1 客户端计时；mock fixture 保证非空 |
| 4 工具正确性 PASS/FAIL | §3 `tool_correct`；`search_order` 为合成自测值（F2） |
| 5 PII FAIL/PASS | §3 `pii_leak`，内置手机号/身份证/银行卡正则 |
| 6 延迟 SLO | §3 `latency_slo`，客户端实测对比 budget |
| 7 降级 FAIL | §3 `no_fallback` |
| 8 `summarize` pass_rate 与 by_check | `engine.summarize()` 纯函数统计 |
| 9 报告 JSON + MD | §5 `persistence` 落盘；`json.load` 可解析 |
| 10 零侵入 | 仅标准库，静态检查禁 `from src.` / `import src.`；`tests_static.py` 内置断言 |

## 7. 非功能约束落实

- **性能**：串行 + 单题超时（默认 10s），整轮 12 题远低于 60s。
- **安全**：落盘一律 `response_redacted`；报告与 JSONL 不写原文。脱敏正则内置于 `redact.py`（口径参照 `src/shared/redact.py`，**不 import** 以维持零侵入）。
- **兼容性**：仅标准库、无 GPU 依赖；`--mock` 保证服务不可用时仍能跑通。模型不可用判 `SKIP`/`ERROR`，不崩溃。
- **可观测性**：每题一条结构化日志（task / latency / passed / 失分判据名）；退化以退出码 1 告知 CI。
- **确定性**：六道判据全为纯函数，无时钟、无随机、无网络、无 LLM。
- **健壮性**：`engine.run_all` 对每题 `try/except`，单题异常转 `ERROR` 并继续。

## 8. 风险与权衡、回退方案

| 风险 | 权衡 | 回退 |
|---|---|---|
| 响应不带 trace_id（已知缺口） | 以客户端 `request_id` 为主键，不强依赖服务端改造 | 若后续服务端回填 trace_id，仅需 `extract` 增加字段读取 |
| `steps` 形状不统一（F3） | 防御式解析，缺失降级为 None | 形状变化时只改 `extract.py` 一处；且 steps 已非主要信号源，影响有限 |
| 工具词表随业务演进 | 考题与词表集中放 `tasks.py` | 词表变更只改 `tasks.py`，判据不动 |
| 服务限流导致误判 | 串行请求 + 可配 `--sleep` 间隔 | 限流返回 429 → 记 ERROR 而非 FAIL，避免污染通过率 |
| **`steps` 被锁死后采集器失效** | **多源降级（§4.2）+ 信号缺失判 SKIP**，不写死依赖用户面 | 已规避：走 Langfuse / 日志源即可 |
| `steps` 对外暴露敏感字段（**服务端既存隐患**） | 不在本次范围（scope §3），**已单独记录为 `steps-redaction-todo.md`** | 服务端侧独立立项治理 |

## 9. 与 code_eval_harness 的关系

模式同构（固定考题 / Driver 适配 / 确定性判据 / summarize 汇总），**代码完全独立、互不 import**。harness 评「编码流程产物」，本层评「运行时行为」，二者平行互补。
