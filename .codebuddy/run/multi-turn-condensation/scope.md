# scope.md — 多轮对话融合（指代消解 / 实体消歧）实现

> 设计依据（权威）：`docs/architecture/multi-turn-condensation-design.md`（已定稿、两轮架构评审通过）
> 任务类型：依据既有设计文档落地（实现与衔接，不重新论证架构）
> 工作流：阶段0(scope) → 阶段3(python-coder) → 阶段4(code-reviewer) → 阶段5(test-generator)

## 目标
依据设计文档实现多轮对话融合：新增 `conversation_condenser.py`，并将指代消解/实体消歧接入 RAG 检索与工具选择两路，同时保持 §4 红线区一律锁原文。

## 验收清单（对应设计文档 §9 实施步骤）
- [ ] 新增 `conversation_condenser.py`：`_gate` / `_load_recent_history`（与 step4 同源：get_chat_messages→summarize→redact PII）/ `_load_entity_slots`（读降级）/ `_save_entity_slots`（写 best-effort、SETEX 刷新 TTL）/ `condense_question`（gated、返回统一 `dict{standalone_query, correction}`、fallback raw、`<AMBIGUOUS>`→反问）。
- [ ] `react_agent.run()` 顶部：先 `_load_entity_slots` 合并进 `ctx.intent_result.params`（覆盖 `:752`/`:1071` 两路注入）；再接 `condense_question` 取 `result["standalone_query"]` 改 `:598` 入参；`result["correction"]` 暂存，选型后补算 `supersedes_prev`；本轮末 `_save_entity_slots`。
- [ ] `react_agent.py:891/:610/:752/:1071/:812` 保持原文语义（注入的是已合并槽位的 `intent_result.params`，非 LLM 改写文本）。
- [ ] 新增 `AgentContext.retrieval_query: str | None`；`step3_retrieve` 优先读它，空则回退 `ctx.request.message`（修复 F2 死代码）；`step1` 开时历史折进 `step1` prompt、`rewritten_queries` 经 `retrieval_query` 喂 step3；`step1` 关(ecommerce) 入口 gated 独立调用写入；`step4` `user_question` 保持 raw。
- [ ] `langfuse_mlops.py`：确认捕获 `query` 为 raw，`plan` 来自消歧选型。
- [ ] 单测覆盖 §9 步骤6 清单（无触发 / 单实体代词 / 多实体 `<AMBIGUOUS>` / rephrase 纠正(无否定词) / LLM 失败 fallback / `order_id` 跨轮 Redis 两路回填 / 原文依赖不泄漏 §4 五处 / 两路径产出同一 `standalone` 契约 / Redis key·TTL 行为），总体覆盖率 ≥60%。

## 范围边界（不做项）
- 方案 B（本地 qwen3-1.5B）**不做**。
- 不改 P0–P3 阈值 / fallback / `candidate_ranking` 逻辑。
- 不解决 ReAct `MemorySaver` 每轮新建(:1075) 的跨请求历史问题（不在本设计范围）。
- 不新建 DB 表（Redis 实体槽位复用现有 Redis）。
- 不新增对外 REST API（仅内部 `AgentContext.retrieval_query` 字段 + `_capture_cross_turn_correction` 改读 `correction`）。
- `index.html:79` "🛠纠正"按钮移出主输入栏到标注面板：仅做最小改动，若位置不确定则留 TODO 交人工。

## 关键约束
- 技术栈 Python/FastAPI；所有 LLM 调用经 `LLM_GATEWAY_URL`，禁止直连供应商。
- **§4 红线**：condensation 只能改变喂给 `pipeline.select()` / `step1`+`step3` 的那一个 query 字符串；§4 五处必须锁原文。
- `step1_rewrite_llm` / `chat_step1()` / `STEP1_REWRITE_MODEL` 已落地，`condense_question` 复用同便宜档。
- Redis 实体槽位 Key：`chat:entity_slots:{conversation_id}`，TTL 1800s（可配 `ENTITY_SLOTS_TTL_SECONDS`），读失败静默 `{}`、写失败 best-effort。

## 未确认假设
- 文档未覆盖的具体阈值按 `apps/` 现有代码惯例处理：`SHORT`（约 40 字 / 200 字符）、`_load_recent_history` 的 `n_turns=3`、摘要阈值。
- `supersedes_prev` 补算逻辑（prev plan 取来源）按 `react_agent.py` 现有 plan 存取方式实现。
- 若实现中暴露文档未定义的接口缺口，记录于 PR 描述并交 `code-reviewer` 核对，不擅自扩写架构。
