# Scope: 依据 platform-engineering/01 落地 LLM 流量网关核心能力

> 本文件由 `multi-agent-workflow` 阶段0 产出，后续所有阶段（架构/编码/审查/测试）必须引用本文件，不得超出其范围边界。

## 1. 目标（一句话可验证）

将 `docs/platform-engineering/01-LLM流量网关设计.md` 的章节骨架落成 `apps/gateway/` 的可运行实现：以独立 FastAPI 服务作为所有 LLM 流量的**唯一出口**，提供统一入口 + 多供应商路由（OpenAI 兼容 `/v1/chat/completions`）。

## 2. 设计依据（权威源，必须 read_file 后再动手）

- 主依据：`docs/platform-engineering/01-LLM流量网关设计.md`（状态：**目录大纲，章节骨架已定**——本次为"落地骨架"，非重新论证架构）
- 跨篇依赖（按序落地，不并行）：
  - `02-路由策略与引擎可替换性.md`（路由表 / 引擎可替换）
  - `03-成本治理与限流.md`（令牌桶限流 / 租户隔离）
  - `06-合规护栏.md`、`07-脱敏引擎工程化.md`、`08-语义缓存.md`、`10-Agent注入检测-LLM网关三道防线.md`（注入三道闸落点）
- 现有代码锚点：`apps/gateway/`（Python/FastAPI，已有 `gateway/` 包、`Dockerfile`、`pyproject.toml`）

## 3. 验收清单（可逐项勾选）

- [ ] `apps/gateway/` 暴露 OpenAI 兼容 `/v1/chat/completions` 端点
- [ ] 业务侧 `LLM_GATEWAY_URL` 指向网关；`shop-agent` 的 LLM SDK `base_url` 已改为指向网关（非直连供应商）
- [ ] 路由表支持 `tool_select`/`param`→本地、`gpt-*`→Azure、`claude*`→Bedrock、`qwen*`→百炼 的映射（引用 01 §4 / 02）
- [ ] 入向检测点位（①）预留 Prompt 注入第一道闸钩子，命中即拦（引用 01 §2.1 / 10）
- [ ] 出向处置边界（③）实现 fail-closed / fail-open 显式决策开关（引用 01 §5）
- [ ] 关键路径无 `unwrap` 式直连供应商硬编码；网关无状态可水平扩容
- [ ] `code-reviewer` 核对：实现未偏离 01 既定决策（尤其"独立服务而非内嵌""出口策略强制经网关"）

## 4. 范围边界（明确不做）

- **不做**：成本记账/租户账单（03）、合规护栏/脱敏（06/07）、语义缓存（08）的完整实现——仅预留钩子与接口，留待对应篇目子任务
- **不做**：重新论证"为何独立网关"（01 已定），不引入新架构概念
- **不做**：K8s NetworkPolicy / service mesh 出口策略配置（属部署文档，本次仅代码层保证"不直连"）

## 5. 关键约束

- 技术栈：Python + FastAPI（与 `apps/gateway` 现有栈一致）
- 兼容性：只认 OpenAI 兼容端点，背后 Ollama/vLLM 无感（引用 01 §4）
- 安全：所有 LLM 调用经网关，业务不直连供应商；遵循 `llm-agent` rule
- 环境：K8s 内 `gateway:8001`；裸跑回退真实供应商（一套代码三环境）

## 6. 未确认假设（文档未覆盖处，按现有代码惯例处理）

- 路由表存储形式（env / YAML / DB）按 `apps/gateway` 现有配置惯例，不新建配置体系
- 限流/成本的中间数据结构先留占位，待 03 子任务补全
- 若 01 大纲与 `apps/gateway` 现有代码冲突，以"不破坏现有可运行行为"为前提局部对齐，并在审查报告中显式标注冲突点

## 7. 阶段编排（建议）

1. 阶段0 澄清 → 本 scope（已完成）
2. 阶段1 `architecture-designer` → 输出 `architecture.md`：衔接设计，引用 01 §2/§4，标注与现有 `apps/gateway` 代码的衔接点
3. 阶段3 `python-coder` → 实现路由 + 入口端点 + 三道闸钩子（引用 02/10 接口约定）
4. 阶段4 `code-reviewer` → 重点核对 LLM 安全清单与"未偏离 01 决策"
5. 阶段测试 `test-generator` → 路由命中 / 直连拦截 / fail 开关的单测
6. 后续按 02→03→06/07/08/10 顺序开新子任务，复用本 scope 的"设计依据"锚定
