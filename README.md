# Shop-Agent 智能客服系统

基于 **RAG（检索增强生成）** 与 **Agent 编排** 构建的智能客服平台，面向电商场景提供 AI 对话服务。

## 功能概览

### 1. Agent 编排与路由

系统采用 **Orchestrator 模式**（`AgentOrchestrator`），按以下流程统一处理请求：

- **意图识别** — 基于 FAISS 向量匹配进行本地意图分类（`IntentRecognizer`），识别 `query-order`、`check-shipping`、`request-return`、`check-balance`、`coupon-inquiry` 五种业务意图，支持否定词过滤（咨询类直接走 RAG）和 LLM 兜底模式
- **路由分发** — 意图命中后根据复杂性检测结果分发到两条路径：
  - **直接 Tool 调用**：简单意图，参数抽取后直接调用对应工具，零额外 LLM 开销
  - **ReAct Agent**：多步意图（含推理/条件判断/退货类），由 `ReActAgent` 自主决策是否查 RAG、调用业务 Tool，融合结果回复
- **RAG Agent 兜底**：未命中意图的通用问答，走 `GeneralAgentExecutor` 的多步骤流水线（问题理解 → 内容审查 → 知识检索 → 回答生成）
- **人在回路**：退款审批场景，Agent 自动暂停等待管理员通过 `/agent/refund/confirm` 确认

### 2. ReAct Agent（工具调用 + RAG 融合）

基于 LangChain `create_agent` + LangGraph `MemorySaver` 的 ReAct 循环，配套三层工具选择策略：

- **P0 意图前置过滤**：`IntentResult.action` → 缩减候选工具池到 2-5 个
- **P2 FAISS 语义重排**：用户 query × 工具描述余弦相似度 + 意图加权 → Top-3/5
- **P1 本地模型确认**：本地 `Qwen2.5-1.5B` 从 Top-3/5 中选出最相关的工具（兜底消歧）

Agent 行为由 **Skill SOP 内联注入** 驱动：启动时 `SkillLoader` 从 `skills/*/SKILL.md` 加载 YAML frontmatter + Markdown 正文到 `SkillRegistry`，运行时命中 Skill 后将 SOP 正文注入 system prompt，实现"增加新业务只改配置"。

### 3. RAG 智能对话

支持两种 RAG 路径：
- **简单 RAG**（`/chat`）：Embedding → Milvus 向量检索 → LLM 生成
- **Agent RAG**（`/agent/chat` − `GeneralAgentExecutor`）：多步骤流水线，含问题理解、安全审查、知识检索（Milvus 2.6 原生混合检索 Dense + Sparse BM25）、回答生成 + 基于规则的质量评估（回答长度、上下文完整性、低质量模式检测）

支持 BGE-Reranker 重排序（`RerankerService`）对检索结果按相关性重新打分和低相关截断，支持 LLM 相关性过滤。

### 4. 文档知识库管理

支持向 Milvus 向量知识库导入文档，提供单条插入（`POST /chatagent/documents`）、批量插入（`POST /chatagent/documents/batch`）和文件上传（`POST /chatagent/documents/upload`，支持 `.txt` / `.md` / `.csv` / `.json` / `.xml` / `.html` / `.py` / `.java` 等常见文本格式）三种方式。内部使用两层切分策略：`SemanticChunker` 语义切分（`percentile` 模式，阈值 85）→ Token 安全兜底（超限 chunk 用 `RecursiveCharacterTextSplitter` 二次切分）。

### 5. 商品嵌入与搜索

支持将商品标题向量化存入 Milvus（`POST /chatagent/items/embed`），批量嵌入（`POST /chatagent/items/embed/batch`），以及文件上传嵌入（`POST /chatagent/items/embed/file`，`.txt/.tsv/.csv`）。支持 Milvus 混合检索搜索商品（`POST /chatagent/items/search`），按 item_id 去重。

### 6. 企业信息查询

基于 MySQL 的企业信息检索服务，Repository 层支持模糊匹配、精确匹配、地区/行业筛选等多维度查询，路由前缀为 `/reports`。

### 7. 问题缓存去重

基于 Redis Stack 的向量相似度搜索（`RedisCacheService`），支持 SHA256 精确哈希匹配与向量余弦相似度匹配双重策略。缓存回答经质量评估（基于规则打分 ≥ 6）后存储。同时支持对话历史存储（按 conversation_id 分 key）和高频问题统计（Sorted Set），包含问题脱敏处理（移除电话号码、邮箱、身份证号、详细地址）。

### 8. Prometheus 可观测性

通过 `prometheus-fastapi-instrumentator` 自动暴露 HTTP 请求指标，同时定义了业务自定义指标（API 调用、数据库查询、Milvus 检索、Embedding 请求、Redis 缓存、Agent 对话轮次、Token 消耗、异常统计），并实现 LangChain 标准回调处理器（`PrometheusCallbackHandler`）追踪 LLM 调用、Embedding 请求、Agent 执行、Tool 调用等事件。

### 9. Langfuse 全链路追踪

通过 `langfuse.langchain.CallbackHandler` + `propagate_attributes()` 上下文管理器（v4.x 规范）实现 LLM 调用、Agent 执行、意图识别、参数抽取等全链路追踪，每个请求创建独立 trace（session_id / tags / trace_name），应用 shutdown 时 flush 确保数据不丢失。

### 10. 参数抽取

支持三种模式逐级兜底（`IntentRecognizer.extract_params`）：
- **local**：纯正则 + 关键词（毫秒级，零 API），失败后降级到 local_model → llm
- **local_model**：transformers 本地小模型（`Qwen2.5-0.5B-Instruct`，4bit 量化），失败降级到 llm
- **llm**：Qwen structured output，最精准

---

## 系统架构

### 编排队列

```
用户请求
    │
    ▼
┌─────────────────────────────────────────────┐
│         AgentOrchestrator.chat_with_agent()  │
│                                              │
│  意图识别 (IntentRecognizer)                  │
│  ├── 否定词过滤 → RAG 兜底                   │
│  ├── FAISS 向量匹配 → call_remote_api        │
│  └── 默认 → rag_answer                       │
│                                              │
│  路由分发                                     │
│  ├── call_remote_api                        │
│  │   ├── 简单意图 → ToolService.dispatch()   │
│  │   └── 多步意图 → ReActAgent.run()         │
│  └── rag_answer                              │
│      └── GeneralAgentExecutor.execute()      │
└─────────────────────────────────────────────┘
    │
    ▼
 ChatResponse
```

### 模块划分

| 模块 | 职责 |
|------|------|
| `src/core` | 全局配置管理（Pydantic Settings），含 LLM/Embedding/Milvus/意图识别/参数抽取等全部配置项 |
| `src/shared` | 异步数据库引擎（SQLAlchemy 2.0 + aiomysql）、统一异常体系（BusinessException/401/403/404/422/500）、结构化日志（structlog）、统一响应格式（BaseResponse） |
| `src/modules/auth` | Bearer Token API Key 认证鉴权（HTTPBearer + FIXED_API_KEY 比对） |
| `src/modules/chat` | 智能客服核心模块 |
| `src/modules/chat/agent` | Agent 编排（`AgentOrchestrator`）+ 通用执行器（`GeneralAgentExecutor`）+ ReAct Agent（`ReActAgent`）+ Skill 加载器（`SkillLoader` / `SkillRegistry`）+ 提示词管理（`PromptTemplateManager`） |
| `src/modules/chat/core` | LLM 服务、Embedding 服务（local BGE）、Milvus 混合检索服务、Redis 缓存服务、意图识别器（FAISS）、文档服务（SemanticChunker）、Reranker 服务（BGE-Reranker-base）、工具注册与服务（`ToolService`）、本地模型服务（`LocalModelService`） |
| `src/modules/items` | 企业信息查询（模型/Schema/Repository），路由注册为 `/reports` |
| `src/modules/monitoring` | Prometheus 指标定义 + LangChain 回调 + Langfuse 回调 |

---

## 关键设计决策

### 大模型：通义千问（Qwen）

通过 OpenAI 兼容模式接入阿里云 DashScope（`dashscope.aliyuncs.com/compatible-mode/v1`），默认模型为 `qwen3.6-flash-2026-04-16`。`LLMService` 采用单例模式，Agent 步骤 4 回答生成使用 `temperature=0.3`。

### 嵌入模型：本地 BGE（可切换云端）

默认使用本地 `BAAI/bge-small-zh-v1.5`（sentence-transformers），通过 `EMBEDDING_PROVIDER=local` 配置。

### 意图识别：FAISS 本地优先

默认使用本地 FAISS 向量匹配（`INTENT_RECOGNITION_MODE=local`），使用类级别共享的 `faiss.IndexFlatIP` 索引（BGE 归一化向量，内积 = 余弦相似度），5 种业务意图各 4 条示例短语，相似度阈值 0.65。支持否定词过滤（含"退货政策/退款流程/怎么退"等 10 种模式）、复杂性检测（含"为什么/怎么办/帮我处理"等 13 种触发模式 + 退货类永走 Agent + 边缘分数阈值 0.85）。支持 `llm` 模式兜底。

### Milvus 混合检索策略

使用 Milvus 2.6 原生混合检索（Dense HNSW + Sparse BM25），RRF 融合（`rrf_k=60`），COSINE 相似度度量。Reranker 开启时从 Milvus 多取 `top_k * 4` 条供 BGE-Reranker 重排序。检索结果默认经 LLM 相关性过滤。Step 1 预计算的 question embedding 复用于 step 3，避免重复调用 Embedding API。

### BGE-Reranker 重排序

`RerankerService` 基于 `BAAI/bge-reranker-base` CrossEncoder 对 Milvus 检索结果重新打分，支持相关性阈值截断和 Top-K 截取。同步 CPU 推理通过 `asyncio.to_thread` / `loop.run_in_executor` 放入线程池，避免阻塞事件循环。

### Agent 安全设计

安全审查是 `GeneralAgentExecutor` 流水线中的步骤 2，电商 domain 下默认禁用（`enabled=False`）。启用时通过 LLM 结构化输出或 JSON 解析识别敏感内容，解析失败使用敏感关键词兜底（从配置读取）。高风险时返回安全警告模板并阻止继续。审查步骤 LLM 异常时采用保守策略（默认高风险）。

### 文本分块策略

使用 `langchain_experimental.text_splitter.SemanticChunker` 作为主切分策略（`percentile` 模式，阈值 85），通过向量相似度检测话题边界进行语义切分。超限 chunk 用 `RecursiveCharacterTextSplitter`（`chunk_size=3276, chunk_overlap=200`，分隔符 `["\n\n", "\n", "。", ".", " ", ""]`）做 Token 安全兜底。Embedding 模型最大输入 4096 tokens，取 80% 安全余量（3276 chars）。

### 本地模型：参数抽取

配置 `LOCAL_PARAM_MODEL = "./models/Qwen2.5-0.5B-Instruct"`，支持 `auto`/`cpu` 设备选择、4bit 量化（需 bitsandbytes）。通过 `LocalModelService` 封装，用于意图命中后的参数抽取，支持 max_retries=2 的重试机制。

### 统一异常体系

分层异常类：`BusinessException(400)` → `AuthenticationException(401)` / `AuthorizationException(403)` / `NotFoundException(404)` / `ValidationException(422)` / `DatabaseException(500)`。通过 FastAPI 三层异常处理器注册（业务异常 → HTTP 异常 → 通用异常），统一拦截并返回 `ErrorResponse` 格式。

### 统一响应格式

所有接口返回统一结构：
```json
{ "success": true, "code": 200, "message": "操作成功", "data": {...} }
```
通过 `success_response()` / `error_response()` 构建，基于 Pydantic `BaseResponse` / `SuccessResponse` / `ErrorResponse` 模型。

### 结构化日志与脱敏

使用 `structlog` 实现 JSON 格式结构化日志（开发环境可选彩色控制台输出）。关键设计：

- `logging_middleware`：FastAPI 中间件自动记录每个请求的方法、URL、客户端 IP、状态码、处理耗时
- `APILogger`：封装业务事件（`log_business_event`）、API 调用（`log_api_call`）、数据库操作（`log_database_operation`）的专用日志方法
- **日志脱敏**：API Key 仅记录前 8 位（`api_key[:8] + "****"`）

### 异步数据库访问

采用 SQLAlchemy 2.0 异步引擎 + `aiomysql` 驱动，配置 `pool_pre_ping=True`（连接预检查）和 `pool_recycle=3600`（连接定期回收）。通过 FastAPI `Depends(get_db)` 依赖注入管理会话生命周期，`get_db()` 生成器在 `finally` 块中确保会话关闭。

### 内存对话历史管理

`GeneralAgentExecutor` 使用实例变量 `_history`（Dict）存储对话历史，按 `conversation_id` 分区，通过 `asyncio.Lock`（类变量，所有实例共享）保证异步安全。历史过期时间 24 小时，每 10 次调用触发一次过期清理。对话历史限制条数由 `max_history_turns` 配置控制。

### Token 消耗控制

- 每篇检索文档截断到 800 字符后传给 LLM（`_MAX_DOC_CHARS = 800`）
- 对话历史每条截断到 300 字符
- LLM 相关性过滤最多传 20 条文档给 LLM 判断
- Step 4 prompt 中多个占位符复用同一 `rag_context` 值

### 单品单例服务模式

`LLMService`、`EmbeddingService`、`MilvusService`、`RedisCacheService`、`RerankerService`、`LocalModelService` 均采用单例模式（通过 `__new__` + `get_instance()` 实现），避免重复初始化连接。

### 人在回路：退款审批

`ReActAgent` 处理退款请求时，Agent 自动暂停并返回 `status="waiting_for_confirmation"`，管理员通过 `POST /chatagent/agent/refund/confirm` 确认或拒绝。中断上下文存储在模块级 `_INTERRUPT_STORE` 字典中，支持后续恢复执行。

### Skill SOP 注入

`SkillLoader` 从 `skills/` 目录递归扫描 `SKILL.md` 文件（YAML frontmatter + Markdown 正文），自动构建 `INTENT_TOOL_MAP`（P0 过滤用）和工具描述列表（P2 FAISS 索引用）。新增 Skill 只需创建新目录 + `SKILL.md`，无需改代码。运行时命中 Skill 后将 SOP 正文内联注入 ReAct Agent 的 system prompt。

### 基础设施容器化

通过 `docker-compose.yml` 一键编排以下服务：

- **etcd**（`quay.io/coreos/etcd:v3.5.25`）— Milvus 元数据存储
- **MinIO**（`minio/minio:RELEASE.2024-12-18T13-15-44Z`）— Milvus 对象存储 + Langfuse bucket
- **Milvus Standalone**（`milvusdb/milvus:v2.6.14`）— 向量数据库，端口 19530
- **Redis Stack**（`redis/redis-stack-server:7.2.0-v14`）— 向量缓存 + 对话历史，端口 6379
- **Prometheus** — 监控指标采集，端口 9090
- **Grafana** — 可视化仪表盘，端口 3001（映射到容器 3000）
- **Langfuse Worker + Web**（`docker.io/langfuse/langfuse:3`）— LLM 追踪平台，Web 端口 3000
- **ClickHouse** — Langfuse 分析数据库
- **PostgreSQL** — Langfuse 主数据库

所有服务均配置健康检查，Milvus 依赖 etcd + MinIO，Langfuse 依赖 postgres + minio + redis + clickhouse。
