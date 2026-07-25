# Shop-Agent 智能客服系统

基于 **RAG（检索增强生成）** 与 **Agent 编排** 构建的智能客服平台，面向电商场景提供 AI 对话服务。

## 多应用架构

```mermaid
flowchart LR
    Client[客户端 / Web] --> ShopAgent[智能客服核心<br/>shop-agent]
    
    ShopAgent --> OrderService[订单业务服务<br/>order-service]
    ShopAgent --> Monitoring[监控代理<br/>monitoring-agent]
    
    ShopAgent --> Gateway
    Monitoring --> Gateway
    
    Gateway --> LLM[云端大模型<br/>通义千问]
    
    subgraph 数据与基础设施
        direction TB
        Infra[向量数据库 / Redis / MySQL / 对象存储 / 监控栈]
    end
    
    ShopAgent --> Infra
    OrderService --> Infra
    Monitoring --> Infra
```

| 应用 | 职责 | 技术栈 |
|------|------|--------|
| `apps/shop-agent` | 智能客服核心服务（Agent 编排、RAG、ReAct、MCP/A2A 协议） | Python 3.10+, FastAPI, LangChain/LangGraph |
| `apps/gateway` | LLM 统一网关（路由、负载均衡、故障转移、限流），所有 LLM 流量的唯一出口 | Python 3.10+, LiteLLM Router |
| `apps/monitoring-agent` | 监控与可观测性代理 | Python 3.10+ |
| `apps/order-service` | 订单/物流/售后业务服务 | Rust |

## 功能概览

### 1. Agent 编排与路由

系统采用 **Orchestrator 模式**（`AgentOrchestrator`），按以下流程统一处理请求：

- **输入归一化** — 同义词归一化 `SynonymNormalizer`（静态映射 <1ms + 文本标准化，默认开启；LLM 兜底按配置关闭）
- **情绪检测** — `SentimentService` 级联分类器（关键词 <1ms → 本地模型 ~30ms → 云端 LLM 兜底），舆情风险立即升级不走后续管线
- **Token 预算截断** — 用户输入超限时智能截断（`keep_both_ends` 策略，保留首尾内容）
- **意图识别** — 基于向量匹配的本地意图分类，识别订单查询、物流查询、退货退款、余额查询、优惠券查询五种业务意图，支持否定词过滤（咨询类模式直接走 RAG）和 LLM 兜底模式
- **路由分发** — 意图命中后，先做纠纷协调检测，再根据复杂性分发：
  - **纠纷协调**：买方诉求 + 卖方立场并行分析 → 调停裁决，含分布式锁防重复执行
  - **ReAct Agent**：多步意图（含推理/条件判断/退货类），Agent 自主决策
  - **直接 Tool 调用**：简单意图，参数抽取后直接调用对应工具，零额外 LLM 开销
  - **RAG Agent 兜底**：未命中意图的通用问答，走 4 步流水线（问题理解 → 内容审查 → 知识检索 → 回答生成）
- **人在回路**：退款审批场景，Agent 自动暂停返回待确认状态，管理员通过接口确认或拒绝

### 2. ReAct Agent（工具调用 + RAG 融合）

基于 LangChain `create_agent` + LangGraph `MemorySaver` 的 ReAct 循环，配套三层工具选择策略：

- **P0 意图前置过滤**：Skill 注册表自动构建候选映射 → 缩减候选工具池
- **P1 语义重排**：用户 query × 工具描述向量相似度 + 意图加权 → Top-K
- **P2 本地模型确认**：本地小模型从候选集中选出最相关的工具，不可用时回退到云端兜底

Agent 行为由 **Skill SOP 内联注入** 驱动：启动时 `SkillLoader` 从 `skills/*/SKILL.md` 加载定义到注册表，运行时命中 Skill 后将 SOP 注入 system prompt，实现"增加新业务只改配置"。

### 3. RAG 智能对话

支持两种 RAG 路径：

- **简单 RAG**：Embedding → 向量检索 → LLM 生成，含输入归一化 + 输出内容安全过滤
- **Agent RAG**：4 步流水线：
  - Step 1 问题理解/改写（默认禁用）
  - Step 2 内容安全审查（默认禁用，含本地优先 + 云端复核的双层策略）
  - Step 3 知识检索（混合检索 + 图查询并行）
  - Step 4 回答生成 + 规则质量评估 + 输出安全过滤

增强能力：
- **预计算向量复用**：入口预计算 question embedding，供检索复用，避免重复调用 Embedding API
- **Reranker 重排序**：对检索结果按相关性重新打分和截断，失败时不阻塞主流程
- **LLM 相关性过滤**：检索结果经语义相关性判断，过滤无关文档
- **图查询增强**：商品关系图谱与向量检索并行执行，结果注入 prompt，解决纯向量检索无法处理的关联推荐问题

### 4. 文档知识库管理

支持向向量知识库导入文档，提供单条插入、批量插入和文件上传三种方式。内部两层切分策略：语义切分 → Token 安全兜底二次切分。

### 5. 商品嵌入与搜索

支持将商品标题向量化存入向量数据库，批量嵌入，及文件上传嵌入。支持混合检索搜索商品，按商品 ID 去重。

### 6. 企业信息查询

基于关系数据库的企业信息模型，Repository 层支持模糊匹配、精确匹配、地区/行业筛选等多维度查询。

### 7. 问题缓存去重

基于 Redis 的向量相似度搜索，支持精确哈希匹配与向量余弦相似度匹配双重策略。缓存回答经质量评估后存储。同时支持对话历史存储和高频问题统计，包含问题脱敏处理。

### 8. 内容安全过滤

纯规则引擎（零 LLM 成本），提供：
- **输入过滤**：拦截明显恶意/非法内容 + Prompt Injection 检测
- **输出过滤**：按领域配置关键词表，LLM 输出在返回用户前强制扫描
- 纵深防御位置：编排器入口 → Agent 输入审查 → 执行器输出过滤

### 9. 可观测性

- **Prometheus**：自动暴露 HTTP 请求指标，业务自定义指标（API 调用、数据库查询、向量检索、Embedding 请求、缓存、Agent 对话、Token 消耗、异常统计），LangChain 标准回调处理器追踪 LLM/Agent/Tool 事件
- **SkyWalking**：gRPC 上报分布式链路追踪，与 Prometheus 互补，不可用时优雅降级
- **Langfuse**：全链路追踪 LLM 调用、Agent 执行、意图识别、参数抽取、工具匹配，shutdown 时 flush 确保数据不丢失

### 10. 参数抽取

支持四种模式，逐级兜底：
- **local_strict**：纯正则 + 关键词，毫秒级，零 API，不降级
- **local**：正则 → 失败降级本地模型 → 失败降级 LLM（默认）
- **local_model**：本地小模型 → 失败降级 LLM
- **llm**：结构化输出，最精准

### 11. MCP 协议（Model Context Protocol）

基于 FastMCP 框架将 Skill 体系通过 MCP 协议对外暴露，使外部 AI 客户端可以调用本系统的业务工具。

- **核心原则：MCP 对外，不对内** — 外部 Client 通过 JSON-RPC 调用，内部同进程直调
- `tools/list` — 从 Skill 注册表自动生成 tool 列表及参数 schema
- `tools/call` — 映射到 `ToolService.dispatch(action, params)`
- 传输模式：`stdio` / `sse` / `streamable-http`，默认 `stdio`

### 12. A2A 协议（Agent-to-Agent）

自研 A2A 协议，使外部 Agent 系统能够以标准化的方式发现能力、提交异步任务、共享对话上下文。

- **能力发现**：`GET /.well-known/agent-card.json` 无需认证，动态生成，首次构建后缓存
- **异步任务管理**：提交任务立即返回 `task_id`，后台异步执行，支持取消
- **Webhook 回调通知**：任务完成后自动 POST 到回调地址，含 HMAC 签名
- **对话上下文共享**：列出对话摘要、获取历史消息，供其他 Agent 查询
- **健康检查**：返回 LLM、Vector DB、Redis、MCP Server 等依赖状态

---

## shop-agent架构

### 编排队列

```mermaid
flowchart TD
    A[用户请求] --> B[AgentOrchestrator.chat_with_agent]

    B --> C[输入归一化<br/>SynonymNormalizer]
    C --> C1[L1 静态同义词表 <1ms]
    C --> C2[L2 文本标准化<br/>全角→半角、繁→简]

    C1 --> D
    C2 --> D

    D[Token 预算截断<br/>TokenEstimator]
    D --> D1["keep_both_ends 策略"]

    D1 --> E[情绪检测<br/>SentimentService]
    E --> E1[L1 关键词 <1ms]
    E --> E2[L2 本地模型 ~30ms<br/>舆情风险立即升级]

    E1 --> F
    E2 --> F

    F[意图识别<br/>IntentRecognizer]
    F --> F1[否定词过滤 → RAG 兜底]
    F --> F2[向量匹配 → call_remote_api]
    F --> F3[默认 → rag_answer]

    F1 --> G
    F2 --> G
    F3 --> G

    G{路由分发}

    G -->|纠纷协调| H[DisputeCoordinator]
    H --> H1[BuyerAgent]
    H --> H2[SellerAgent]
    H1 --> H3[MediatorAgent 调停裁决]
    H2 --> H3

    G -->|call_remote_api| I{意图复杂度}
    I -->|简单意图| I1[ToolService.dispatch<br/>零额外 LLM 开销]
    I -->|多步意图| I2[ReActAgent.run<br/>自主决策]

    G -->|rag_answer| J[GeneralAgentExecutor.execute]
    J --> J1[预计算向量 缓存复用]
    J1 --> J2[Step1 问题理解<br/>默认跳过]
    J2 --> J3[Step2 安全审查<br/>默认跳过]
    J3 --> J4["Step3 检索 ∥ 图查询<br/>并行"]
    J4 --> J5[Step4 回答生成 + 质量评估]

    style A fill:#e1f5fe
    style G fill:#fff3e0
    style H fill:#fce4ec
    style J fill:#e8f5e9
```

### 模块划分

| 模块 | 职责 |
|------|------|
| `src/core` | 全局配置管理、Token 预估器、速率限制器 |
| `src/shared` | 异步数据库引擎、统一异常体系、结构化日志、统一响应格式 |
| `src/modules/auth` | Bearer Token API Key 认证鉴权 |
| `src/modules/chat` | 智能客服核心模块，含 A2A 协议路由 |
| `src/modules/chat/agent` | Agent 编排、通用执行器、ReAct Agent、Skill 加载器、提示词管理、纠纷协调器 |
| `src/modules/chat/core` | LLM 服务、Embedding 服务、向量检索服务、Redis 缓存服务、意图识别器、文档服务、Reranker 服务、工具注册与服务、本地模型服务、内容安全过滤、同义词归一化、情绪检测、图查询服务、参数抽取器、MCP Server、A2A 任务服务、A2A Webhook 服务、Agent Card 构建器 |
| `src/modules/items` | 企业信息查询，路由前缀 `/reports` |
| `src/modules/monitoring` | Prometheus 指标定义 + LangChain 回调 + Langfuse 回调 + SkyWalking 分布式追踪客户端 |

---

## 关键设计决策

### 大模型：通义千问（Qwen）

通过 OpenAI 兼容模式接入阿里云 DashScope，默认模型为 `qwen3.7-plus-2026-05-26`。`LLMService` 采用单例模式，Agent 回答生成使用 `temperature=0.3`。

### 嵌入模型：本地 BGE（可切换云端）

默认使用本地 `BAAI/bge-small-zh-v1.5`，通过配置切换。全链路多个消费方（意图匹配 + RAG 检索 + 语义缓存）统一不加指令前缀。

### 意图识别：本地向量优先

默认使用本地向量匹配，5 种业务意图各若干示例短语，支持否定词过滤（咨询类模式直接走 RAG）、复杂性检测（退货类永走 Agent）和 LLM 兜底。

### 混合检索策略

使用原生混合检索（Dense HNSW + Sparse BM25），RRF 融合，开启 Reranker 时多取候选供重排序。检索结果经语义相关性过滤。预计算的 question embedding 复用于检索，避免重复调用 Embedding API。

### Reranker 重排序

基于 `BAAI/bge-reranker-base` CrossEncoder 对检索结果重新打分，支持相关性阈值截断和 Top-K 截取。CPU 推理放在独立线程池执行，避免阻塞事件循环。Rerank 失败时不阻塞主流程，保留原始检索结果。

### 同义词归一化：三级设计

编排器入口处统一处理用户输入：
- **L1 静态同义词表**（<1ms）：将常见变体归一化为标准术语
- **L2 文本标准化**：全角→半角、繁体→简体、多余空格/标点清理
- **L3 LLM 归一化**：调用 LLM 覆盖长尾表达（默认关闭）

95% 的 case 在 L1+L2 完成，零 LLM 成本。

### 情绪检测：级联分类器

三级级联：
- **L1 关键词**（<1ms）：匹配明确情绪信号
- **L2 本地模型**（~30ms）：零样本分类
- **L3 云端 LLM**（~300ms）：边界情况兜底，极少触发

情绪等级超过阈值建议升级，紧急舆情风险强制立即升级并跳过后续管线。含 Session 情绪跟踪器（滑动窗口 + 趋势方向）。

### 纠纷协调器：多 Agent 三方协调

检测到愤怒情绪 + 退货意图时触发：
- **FactCollector**：收集客观事实（查订单/查物流/查政策）
- **BuyerAgent ∥ SellerAgent**：并行分析买方诉求与卖方立场
- **MediatorAgent**：串行等待两者结果后调停裁决

所有 Agent 复用同一 LLM 实例，仅 prompt 不同。支持 Mock 事实数据（无远程 API 时自动降级）。

**分布式锁防重**：执行前获取分布式锁（TTL 5 分钟），防止重复执行。锁使用原子释放，Redis 不可用时降级为无锁通过。

### 安全设计

四层纵深防御：
- **输入过滤**（编排器入口）：规则引擎极保守拦截，零 LLM 成本
- **Agent 输入审查**：默认禁用，启用时分两级——本地优先，失败升级云端复核
- **输出过滤**：强制规则引擎扫描 LLM 输出，block 关键词直接拦截，replace 关键词脱敏
- **审查异常保守策略**：审查步骤异常时默认高风险，确保安全优先

### 文本分块策略

语义切分作为主切分策略，通过向量相似度检测话题边界。超限 chunk 用 Token 安全兜底二次切分。Embedding 模型最大输入 4096 tokens，取 80% 安全余量。

### 参数抽取：本地优先逐级兜底

配置本地小模型路径，支持自动/CPU 设备选择、4bit 量化。同一模型还作为工具选择本地兜底。

### 图增强

商品关系图谱（同品牌/兼容配件/替代品），图查询与向量检索并行执行（不增加端到端延迟），结果注入回答生成 prompt。未启用或查询无结果时静默降级。

### 统一异常体系

分层异常类：业务异常(400) → 认证(401) / 授权(403) / 未找到(404) / 校验(422) / 数据库(500)。统一拦截并返回标准格式。

### 统一响应格式

所有接口返回统一结构，基于 Pydantic 模型构建。

### 结构化日志与脱敏

JSON 格式结构化日志，FastAPI 中间件自动记录每个请求的方法、URL、客户端 IP、User-Agent、状态码、处理耗时。API Key 仅记录前 8 位。

### 速率限制与 Token 消耗管控

两层限流：
- **请求次数限流**：Redis 滑动窗口 + 内存降级
- **Token 消耗限流**：基于 tokenizers 引擎，Pre-check → 调用 → Post-report 三阶段，LRU 缓存常用文本编码

用户输入超限时使用 `keep_both_ends` 策略智能截断。

### 对话历史管理

基于 Redis 的对话历史存储：
- 按会话 ID 分 key，每条消息截断防撑爆 Redis
- 过期时间控制，最大返回条数限制
- 回答生成时从 Redis 获取历史，截断后注入 prompt
- 不可用时返回空历史静默降级

### Token 消耗优化

- 检索文档截断后传给 LLM
- 对话历史每条截断后注入
- 相关性过滤限制文档数量
- 回答生成 prompt 中多个占位符复用同一检索上下文
- Prompt token 预算守卫：生成前估算，超限时逐篇丢弃检索文档，极端情况全部丢弃，纯模型常识回答
- 入口预计算一次 question embedding，复用于缓存检查 + 检索

### 单品单例服务模式

LLM 服务、Embedding 服务、向量数据库服务、Redis 缓存服务、Reranker 服务、本地模型服务、内容安全过滤服务均采用单例模式，避免重复初始化连接和模型加载。

### 人在回路：退款审批

处理退款请求时，通过状态标志位传递中断上下文。Agent 返回待确认状态含订单号/原因，管理员确认或拒绝。防御重复调用。

### Skill SOP 注入

运行时命中 Skill 后将 SOP 正文内联注入 system prompt（含情绪 tone 提示、输入截断提醒），实现"增加新业务只改配置"。

### MCP Server：工具体系对外开放

基于 FastMCP 实现，启动时自动扫描 Skill 注册表中所有 Skill，动态生成带类型注解的 async 工具函数：
- **自动注册**：新增 Skill 只需在 `skills/` 目录下创建定义文件，重启后自动暴露为新 Tool
- **参数 Schema 管理**：Skill 的参数定义集中管理，客户端可通过 `tools/list` 获取参数结构
- **配置项**：`MCP_ENABLED`（默认 `False`）、`MCP_SERVER_NAME`、`MCP_TRANSPORT`

### A2A 协议：自研架构决策

- **无外部 SDK 依赖**：基于 FastAPI + asyncio 自研完整协议栈
- **内存存储 → 升级路径**：当前使用内存存储，代码注释标注了持久化升级接口
- **Agent Card 加速**：首次构建后缓存，命中 <1ms；启动时预热，避免首个请求阻塞
- **A2A vs MCP 定位差异**：A2A 面向服务间互操作（异步任务 + 上下文共享），MCP 面向工具调用（同步 tools/list + tools/call），两者互补

### 基础设施容器化

通过 `docker-compose.yml` 一键编排以下服务：

- **etcd** — 元数据存储
- **MinIO** — 对象存储
- **Milvus Standalone** — 向量数据库
- **Redis Stack** — 向量缓存 + 对话历史 + 速率限制
- **Prometheus** — 监控指标采集
- **Grafana** — 可视化仪表盘
- **Langfuse** — LLM 追踪平台
- **ClickHouse** — Langfuse 分析数据库
- **PostgreSQL** — Langfuse 主数据库

所有服务均配置健康检查。

---

## 快速开始

### 环境要求

- Python 3.10+
- Rust 1.70+（order-service）
- Docker & Docker Compose
- NVIDIA GPU + CUDA 11.8+（本地模型/微调可选）

### 安装

1. 克隆仓库并进入项目目录
2. 启动基础设施：etcd、MinIO、Milvus、Redis Stack、监控栈等
3. 安装 shop-agent 依赖
4. 配置环境变量：填入 API Key、数据库地址、Redis 地址等
5. 启动服务

### 验证

- 健康检查：访问 `/api/chatagent/health`
- 智能对话（Agent 路由）：POST `/api/chatagent/agent/chat`
- 简单 RAG：POST `/api/chatagent/chat`

### 运行测试

各应用目录下运行测试命令。

---

## 版本与兼容性

| 组件 | 版本/要求 |
|------|-----------|
| Python | 3.10+ |
| FastAPI | 0.100+ |
| LangChain | 0.1+ |
| LangGraph | 0.0+ |
| Milvus | 2.6+ |
| Redis Stack | 7.2+ |
| Rust (order-service) | 1.70+ |
| NVIDIA Driver | 525+（本地模型推理） |
| CUDA | 11.8+（本地模型推理） |

---

## 本地模型微调

参数抽取模型使用 LLaMA-Factory 做 QLoRA 微调。

### 训练 / 评测流程

1. 生成基础训练集
2. 训练 + 合并 LoRA + 评测对比 base vs sft
3. 单独跑评测（模型已存在时）

### GPU 验证

判断 GPU 是否真正参与训练/推理，以 `nvidia-smi` 为准。

判断标准：
- `Memory-Usage` 显存占几 GB → 模型已加载到 GPU
- `GPU-Util` 在训练 steps 阶段呈 50%–95% 抖动 → GPU 在算

> Windows 任务管理器默认显示 **"3D"** 引擎占用率，CUDA 计算跑在独立的 **Cuda / Compute** 引擎上。需切换显示才能看到真实负载。

### 训练提速配置

- `per_device_train_batch_size: 8` + `gradient_accumulation_steps: 4`（有效 batch 仍为 32）
- `preprocessing_num_workers: 4`（Windows 安全）
- **注意**：`dataloader_num_workers > 0` 在 Windows + spawn 多进程下会因 transformers 局部闭包无法 pickle 而崩溃，故保持默认 0。
