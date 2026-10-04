# scope：shop-agent 运行时合成探测评估（benchmark/runtime_eval）

## 1. 目标

建成一套对**运行中的 shop-agent** 做合成探测评估的确定性评估层：固定考题驱动真实请求 → 采集每次请求的 trace 与关键信号 → 确定性判据逐条打分 → 输出通过率与失分原因报告，可供 CI 调用做回归门禁。

> **术语澄清**：本层是「**对在线服务的合成探测 + CI 门禁**」——被测对象在线，评估以**批处理脚本**方式执行。
> **不是**「在线评估平台」（生产流量持续采样 + 异步打分 + 常驻服务 + 看板告警）。后者不在本次范围，见 §3。

## 2. 范围（In scope）

1. **RuntimeTask 考题模型 + 固定考题集**：定义考题字段（query / 期望工具 / 期望意图 / 禁止模式 / 延迟预算 / 场景标签），提供不少于 10 道覆盖工具选择、意图识别、PII 脱敏、降级回退的考题。
2. **Trace schema 与落盘**：定义每次请求的运行时产物结构（request_id / trace_id、各阶段耗时、工具调用、意图标签、响应、token、是否降级），以 JSONL 落盘。
3. **在线采集器**：对运行中的服务发送考题请求，采集响应与 trace，产出 RuntimeArtifact。
4. **确定性判据**：工具正确性、意图正确性、PII 泄漏、延迟 SLO、禁止模式、无异常降级，共六道判据（纯函数）。
5. **评估引擎与汇总**：驱动「考题 → 采集 → 判据 → 打分」，输出 per-task 明细与总体通过率。
6. **报告输出**：JSON 与 Markdown 报告落到 `benchmark_results/`。

## 3. 非范围（Out of scope）

- 不改动 `code_eval_harness`（编码流程评估）任何代码。本层与其平行，只复用其「固定考题 + Driver + 确定性判据」模式。
- 不重构 apps/shop-agent 既有监控栈（Prometheus / Langfuse / SkyWalking）。本层只读取其已产生的指标与 trace。
- 不做模型训练与微调，不替换模型。
- 不做分布式压测（k6 已有）。本层关注单请求正确性而非吞吐。
- 不改动生产鉴权逻辑。
- **不做在线评估平台**：不做常驻服务、不做生产流量采样与异步打分、不做看板与告警。本层以批处理脚本形态运行，由 CI 或人工触发。
- **不替代既有监控告警**：Prometheus + Alertmanager 的既有告警链路不动，本层不参与实时告警，只提供周期性回归凭证。

## 4. 验收标准（可测试，逐条给输入 → 预期输出）

1. 考题集：`load_tasks()` 返回不少于 10 道 RuntimeTask，每道必含 `query` / `expected_tool` / `expected_intent`；缺任一字段即抛错。
2. Trace schema：`validate_trace(d)` 对缺 `request_id` 或缺 `tool_calls` 的 dict 返回 False，对完整 dict 返回 True。
3. 采集器：对 mock 模式服务发 1 道考题，返回 RuntimeArtifact 且 `response` 非空、`latency_ms > 0`。
4. 判据-工具正确性：artifact 的 `tool_calls` 为 `search_order` 且 `expected_tool` 为 `search_order` → PASS；`tool_calls` 改为 `refund_apply` → FAIL。
5. 判据-PII：响应含 11 位手机号（形如 `13800138000`）→ `pii_leak` 判 FAIL；脱敏形态 `138****8000` → PASS。
6. 判据-延迟：`latency_ms=5000` 而 `budget_ms=2000` → `latency_slo` 判 FAIL；`latency_ms=800` → PASS。
7. 判据-无降级：`fallback_hit=True` → `no_fallback` 判 FAIL。
8. 汇总：给定 3 道通过 + 1 道不通过，`summarize()` 返回 `pass_rate == 0.75`，且 `by_check` 含各判据计数。
9. 报告：跑完生成 `benchmark_results/runtime_eval_report.json` 与同名 `.md`；JSON 可被 `json.load` 解析，md 含总体通过率。
10. 零侵入：本层不 import shop-agent 业务代码（仅通过 HTTP 采集）。对 `benchmark/runtime_eval/` 做静态检查，不得出现 `from src.` 或 `import src.`。

## 5. 非功能约束（review 据此判 [BLOCKING]）

- **性能**：单道考题「采集 + 判定」不超过 5s（mock 模式）；整轮 10 道题不超过 60s。
- **安全**：落盘前响应必须脱敏。报告与 JSONL 中不得出现原始手机号 / 身份证 / 银行卡号。
- **兼容性**：Python 3.10+；不依赖 GPU；无真实模型时用 mock 模式跑通——模型不可用应标记 SKIP 而非 FAIL，不得崩溃。
- **可观测性**：评估层自身输出结构化日志（每道考题一条，含 task / latency / passed / 失分判据名）；退化时以非零退出码告知 CI。
- **确定性**：判据必须是纯函数，同输入同输出，判据内不得调用 LLM。
- **健壮性**：服务不可达或超时时，该题判 ERROR 并计入报告，不得让整轮评估异常终止。

## 6. 架构边界初判

- **新建目录**：`benchmark/runtime_eval/`，与 `benchmark/eval/` 平行，两者互不 import。
- **涉及 app**：apps/shop-agent。仅通过其 HTTP 端点 `/api/v1/chatagent/agent/chat` 采集，不改其业务代码。
- **复用既有资产**：
  - 脱敏口径参照 `apps/shop-agent/src/shared/redact.py`；本层内置等价正则，**不跨层 import**（维持零侵入）。
  - trace 关联优先读取响应或日志中的 `trace_id`（`src/shared/logger.py` 已把 trace_id 绑进日志）。已知缺口：当前 HTTP 响应头与响应体不带 trace_id，故本层以请求侧生成的 `request_id` 作为关联主键，不阻塞本任务。
- **对外接口**：CLI `python -m benchmark.runtime_eval.run --base-url <url> --out <dir>`。
- **与 harness 的关系**：模式同构（固定考题 / Driver 适配 / 确定性判据 / 汇总），代码独立。
