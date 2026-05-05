# Shop-Agent 智能客服系统

基于 **RAG（检索增强生成）** 与 **多步骤 Agent 编排** 构建的智能客服平台，提供 AI 对话服务。

## 设计理念

> **能用工程手段替代 LLM 调用的，就不要用 LLM。** 通过分层架构实现模块内高内聚、模块间低耦合，每个组件的引入必须解决一个明确的、可验证的痛点。

系统围绕四个关键问题设计：
- **意图混合**：用户可能在同一句话里混合咨询与业务操作，需要编排层统一路由分发
- **信息源异构**：知识库（非结构化）和业务数据（结构化）需在同一回答里整合
- **延迟硬约束**：多步 Agent 流水线动辄秒级延迟，需缓存绕过重复 LLM 调用
- **安全边界**：敏感内容（如用药、诊断）不能由 AI 直接回答，须内置安全审查步骤

## 功能概览

### 1. RAG 智能对话

将用户提问向量化后，在 Milvus 知识库中检索相似文档，将检索结果作为上下文注入大模型，生成基于知识库的回答，同时返回引用的源文档内容。

### 2. 客服多步骤 Agent

设计了四步流水线式 Agent 编排：

- **问题重写** — 提取医疗关键词，将口语化提问改写为检索查询
- **安全审查** — 检测是否涉及用药、诊断、处方、急危重症等敏感内容，输出风险等级（low / medium / high）与结构化风险评估
- **知识检索** — 使用重写后的查询，通过嵌入模型 + Milvus 向量检索相关知识文档
- **答案生成** — 基于检索结果与安全审查结论，结合对话历史生成回复
- **质量评估** — 基于 LLM 对回答进行打分（相关性、准确性、完整性、可操作性），分数 ≥ 6 的回答才有资格被缓存

支持普通响应与 **SSE 流式响应** 两种模式，流式接口按步骤推送事件（`step_start` → `step_complete` → `content` → `done`），让前端实时感知 Agent 内部进展。

### 3. 文档知识库管理

支持向 Milvus 向量知识库导入文档，提供单条插入、批量插入和文件上传（`.txt` / `.md` / `.csv` / `.json` / `.xml` / `.html` / `.py` / `.java` 等常见文本格式）三种方式，自动完成文本分块、向量化和入库。

### 4. 企业信息查询

基于 MySQL 的企业信息检索服务，数据模型定义完整（企业名称、信用代码、法人、注册资本、经营范围、经营状态、风险信息、地区、行业），Repository 层支持模糊匹配、精确匹配、地区/行业筛选等多维度查询。

### 5. 问题缓存去重

基于 Redis Stack 的向量相似度搜索，对高频问题自动缓存回答，避免重复调用 LLM。支持精确 SHA256 哈希匹配与向量余弦相似度匹配双重策略，相似度阈值 0.85。缓存回答经质量评估（分数 ≥ 6）后才会被存储。

### 6. Prometheus 可观测性

通过 `prometheus-fastapi-instrumentator` 自动采集 HTTP 请求指标，同时定义了业务自定义指标（API 调用、数据库查询、Milvus 检索、Embedding 请求、Redis 缓存、Agent 对话轮次、Token 消耗、异常统计），并实现 LangChain 标准回调处理器（`PrometheusCallbackHandler`）统一追踪 LLM 调用、Embedding 请求、Agent 执行、工具调用等事件。

---

## 系统架构

### 四层架构

```
┌──────────────────────────────────────────────────────────────┐
│  接入层（Access Layer）                                       │
│  FastAPI REST API + Bearer Token 认证 + 请求日志中间件         │
│  + 三层异常处理器（业务异常 → HTTP 异常 → 通用异常）            │
├──────────────────────────────────────────────────────────────┤
│  编排层（Orchestration Layer）                                │
│  HospitalAgentExecutor：四步流水线编排                         │
│  问题重写 → 安全审查 → 知识检索 → 答案生成 → 质量评估           │
│  + 内存对话历史管理（asyncio.Lock + 24h 过期）                 │
├──────────────────────────────────────────────────────────────┤
│  执行层（Execution Layer）                                    │
│  ┌───────────┐ ┌──────────────┐ ┌────────────────┐          │
│  │ LLM 服务  │ │ 嵌入服务     │ │ Milvus 服务    │          │
│  │   Qwen    │ │ Doubao Embed │ │  向量存储+检索  │          │
│  └───────────┘ └──────────────┘ └────────────────┘          │
│  ┌──────────────────────┐  ┌──────────────────────┐         │
│  │  Redis 缓存服务       │  │  Prometheus 监控      │         │
│  │  向量相似度 + 对话历史 │  │  + LangChain 回调     │         │
│  └──────────────────────┘  └──────────────────────┘         │
├──────────────────────────────────────────────────────────────┤
│  基础设施层（Infrastructure Layer）                           │
│  MySQL (业务数据)   Milvus (向量库)   Redis Stack (缓存+向量) │
│  etcd (Milvus 元数据)   MinIO (Milvus 对象存储)               │
└──────────────────────────────────────────────────────────────┘
```

### 模块划分

| 模块 | 职责 |
|------|------|
| `src/core` | 全局配置管理（Pydantic Settings）、依赖注入辅助 |
| `src/shared` | 异步数据库引擎、统一异常体系、结构化日志、统一响应格式 |
| `src/modules/auth` | Bearer Token API Key 认证鉴权 |
| `src/modules/chat` | 智能客服核心：RAG 对话 + Agent 编排 + 文档管理 |
| `src/modules/chat/agent` | 客服 Agent 四步流水线（问题重写、安全审查、检索、生成）+ 质量评估 |
| `src/modules/chat/core` | LLM 服务（Qwen）、嵌入服务（Doubao）、Milvus 向量服务、Redis 缓存服务 |
| `src/modules/items` | 企业信息查询（模型/Schema/Repository 完整，Service 层待完成）|
| `src/modules/monitoring` | Prometheus 指标定义 + LangChain 回调处理器（LLM/Embedding/Agent/Tool 全链路追踪）|

---

## 关键设计决策

### 大模型选择：通义千问（Qwen）

项目文本生成统一使用通义千问，通过 OpenAI 兼容模式接入（`dashscope.aliyuncs.com/compatible-mode/v1`），模型为 `qwen3.5-plus-2026-02-15`，`temperature=0.7`。LLM 调用封装在 `LLMService` 单例中，所有 Agent 步骤及 RAG 对话均使用 `chat_qwen()` 方法。

### 嵌入模型：火山引擎 Doubao Embedding

使用 Doubao 多模态嵌入模型 `doubao-embedding-vision-251215`，通过 Ark SDK 的 `multimodal_embeddings.create` 接口生成向量，维度为 2048。`ArkEmbeddings` 实现了 LangChain 的 `Embeddings` 接口，支持文本嵌入、图文混合嵌入，异步方法通过 `asyncio.to_thread` 封装。

### Agent 安全设计

安全审查是 Agent 流水线中的**第二步**，作为内置步骤而非可选插件：

- 通过 LLM 结构化输出识别敏感内容（用药、诊断、处方、急危重症、未成年人等），输出 JSON 格式的 `is_safe`、`risk_level`、`risk_categories`、`warning_message`
- LLM 返回的 JSON 解析失败时，使用**关键词匹配兜底**（预定义敏感关键词列表），默认标记为中等风险
- 高风险（`risk_level="high"`）或 `can_proceed=False` 时直接拦截，返回安全警告模板并引导就医
- 安全审查步骤如果 LLM 调用本身抛出异常（网络错误等），采用保守策略：默认高风险并阻止继续

### Milvus 向量检索策略

- **索引类型**：HNSW（`M=16, efConstruction=200`）
- **查询参数**：`ef` 随 `top_k` 动态调整（`ef = max(50, top_k * 2)`）
- **相似度度量**：COSINE，与嵌入模型的向量输出匹配
- **维度适配**：连接时自动检测集合维度（`embedding_dimension=2048`），不匹配则删除旧集合并重建索引
- **集合字段**：id（INT64 自增）、text（VARCHAR 65535）、embedding（FLOAT_VECTOR 2048）、metadata（JSON）

### 单例服务模式

`LLMService`、`EmbeddingService`、`MilvusService`、`RedisCacheService` 均采用单例模式（通过 `__new__` + `get_instance()` 实现），避免重复初始化连接。`ChatAgentService` 内部通过 `_initialize()` 方法集中管理各服务的懒加载，调用时通过属性访问器自动创建。

### 文本分块策略

使用 LangChain `RecursiveCharacterTextSplitter`，`chunk_size=1000`，`chunk_overlap=200`，`length_function=len`，以字符级递归分割，确保相邻块有上下文重叠。

### 统一异常体系

定义了分层异常类：`BusinessException(400)` → `AuthenticationException(401)` / `AuthorizationException(403)` / `NotFoundException(404)` / `ValidationException(422)` / `DatabaseException(500)`。通过 FastAPI 三层异常处理器注册（业务异常 → HTTP 异常 → 通用异常），统一拦截并返回 `ErrorResponse` 格式。

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
- **日志脱敏**：API Key 仅记录前 8 位（`api_key[:8] + "****"`），缓存命中时用户问题仅记录前 30 字符

### Prometheus 监控与 LangChain 回调

通过 `prometheus-fastapi-instrumentator` 自动暴露 HTTP 请求指标（总数、耗时分布、进行中请求数），同时自定义了以下业务指标：

- **API/DB/Milvus/Embedding/Redis 指标**：调用次数（Counter）+ 耗时分布（Histogram），按模块、操作类型、状态等维度区分
- **Agent 指标**：对话轮次（按 success/failed）、Token 使用量（prompt/completion）
- **异常指标**：按异常类型和模块维度统计

`PrometheusCallbackHandler` 实现了 LangChain 标准 `BaseCallbackHandler`，自动追踪 LLM 调用（`on_llm_start/end/error`）、Embedding 请求（`on_embedding_start/end`）、Agent 执行（`on_agent_action/finish`）、Tool 调用（`on_tool_start/end/error`）、Chain 执行和 Retriever 操作，全程采集耗时和 Token 数据。

### 异步数据库访问

采用 SQLAlchemy 2.0 异步引擎 + `aiomysql` 驱动（`mysql+aiomysql://`），配置 `pool_pre_ping=True`（连接预检查）和 `pool_recycle=3600`（连接定期回收）。通过 FastAPI `Depends(get_db)` 依赖注入管理会话生命周期，`get_db()` 生成器在 `finally` 块中确保会话关闭。

### 内存对话历史管理

`HospitalAgentExecutor` 使用类变量 `_history`（Dict）存储对话历史，按 `conversation_id` 分区，通过 `asyncio.Lock` 保证异步安全。历史过期时间 24 小时，每 10 次调用触发一次过期清理，防止内存泄漏。

### SSE 流式事件协议

客服 Agent 流式接口返回 `text/event-stream`，事件以 JSON 行格式推送：

- `step_start`：步骤开始（如 "正在分析您的问题..."）
- `step_complete`：步骤完成（含输出数据）
- `content`：流式回答内容块（`is_final` 标记是否结束）
- `done`：整体完成（含 conversation_id、documents_used、safety_passed）
- `error`：错误信息

### 基础设施容器化

通过 `docker-compose.yml` 一键编排以下服务：

- **etcd**（`quay.io/coreos/etcd:v3.5.25`）— Milvus 元数据存储
- **MinIO**（`minio/minio:RELEASE.2024-12-18T13-15-44Z`）— Milvus 对象存储
- **Milvus Standalone**（`milvusdb/milvus:v2.6.14`）— 向量数据库，端口 19530
- **Redis Stack**（`redis/redis-stack-server:7.2.0-v14`）— 向量缓存 + 对话历史，端口 6379

所有服务均配置健康检查，网络统一在 `milvus` 网桥下。
