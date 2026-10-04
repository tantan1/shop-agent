from typing import Dict

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 应用配置
    DEBUG_MODE: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # 独立探针服务（liveness/readiness，与业务端口隔离；K8s 探针指向此端口）
    PROBE_ENABLED: bool = True
    PROBE_HOST: str = "0.0.0.0"
    PROBE_PORT: int = 8001

    # 数据库配置
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_USER: str = "root"
    DB_PASSWORD: str = ""  # 禁止源码硬编码口令；生产环境必须由环境变量/secret 注入
    DB_NAME: str = "fastapi_dev"

    # Redis 配置（对齐 k8s 注入的 REDIS_HOST/REDIS_AUTH）
    # 注意：REDIS_HOST 接 env（k8s service 注入为 "redis"，正常）；
    # 但不可接 env 的 REDIS_PORT —— k8s 会把 REDIS_PORT 污染为 "tcp://<ip>:<port>"
    # （service 环境变量歧义），无法解析成 int。故字段名避开 REDIS_PORT，
    # 用 REDIS_PORT_NUM 固定 6379，不被 env 污染。
    REDIS_HOST: str = "localhost"
    REDIS_PORT_NUM: int = 6379
    REDIS_PASSWORD: str = ""

    # 日志配置
    LOG_LEVEL: str = "DEBUG"
    LOG_FORMAT: str = "json"

    # 固定API密钥配置
    FIXED_API_KEY: str

    # 通义千问API配置
    TONGYI_API_KEY: str = ""

    # 火山引擎 Doubao API配置
    VOLCENGINE_API_KEY: str = ""

    # 聊天模型配置（云端模型，用于 Agent 回答生成等复杂任务）
    CHAT_MODEL: str = Field(default="")

    # P2 工具选择器专用模型（更轻量更快，qwen-turbo 延迟约为主模型 40%）
    TOOL_SELECTOR_MODEL: str = Field(default="")

    # P2 工具选择器本地模型路径（设置后优先用本地模型替代云端 API）
    # 推荐: Qwen2.5-1.5B-Instruct（速度和准确度的最佳平衡点）
    TOOL_SELECTOR_LOCAL_MODEL: str = ""  # 如 ./models/Qwen2.5-1.5B-Instruct
    TOOL_SELECTOR_LOCAL_DEVICE: str = "cpu"  # cpu | auto
    TOOL_SELECTOR_LOCAL_LOAD_IN_4BIT: bool = False

    # 本地小模型配置（用于参数抽取，transformers 直接加载，无需部署）
    LOCAL_PARAM_MODEL: str = Field(
        default="./models/Qwen2.5-0.5B-Instruct"
    )  # 轻量级中文模型，~1GB，CPU 可跑
    LOCAL_PARAM_DEVICE: str = "auto"  # cpu | cuda | auto（auto 优先 GPU）
    LOCAL_PARAM_MAX_TOKENS: int = 256  # 参数抽取很短，256 足够
    LOCAL_PARAM_LOAD_IN_4BIT: bool = True  # 4bit 量化，节省内存（需 bitsandbytes）

    # 本地小模型后端: vllm | ollama | transformers
    # vllm        = 进程外调 vLLM OpenAI 兼容端点（默认，qwen3-unified 承载 param+tool_select）
    # ollama      = 进程外调 Ollama
    # transformers= 进程内加载（需 torch，生产镜像未装，仅本地调试用）
    LOCAL_MODEL_BACKEND: str = "vllm"
    # Ollama OpenAI 兼容端点（k8s 内用服务名，本机用 localhost）
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    # unified 模型名（Ollama 内 `ollama create <name> -f Modelfile` 命名）
    OLLAMA_PARAM_MODEL: str = "unified"
    OLLAMA_TOOL_SELECTOR_MODEL: str = "unified"
    OLLAMA_TIMEOUT: int = 60  # Ollama HTTP 超时（秒）

    # vLLM OpenAI 兼容端点（LOCAL_MODEL_BACKEND=vllm 时生效）
    # 注意：需含 /v1 后缀或不含均可，代码会规范化
    VLLM_BASE_URL: str = "http://host.docker.internal:8003/v1"
    VLLM_PARAM_MODEL: str = "qwen3-unified"
    VLLM_TOOL_SELECTOR_MODEL: str = "qwen3-unified"
    VLLM_TIMEOUT: int = 60
    AGENT_TIMEOUT: int = 120  # ReAct 主循环整体墙钟超时（秒），超时返回降级响应（防上游 hang 耗尽事件循环）

    # P2 本地工具选择（小模型专项辅助层）开关与阈值。
    # 默认关闭：工具选择已由 P0 规则 + P1 意图加权软过滤完成，本地 1.7B 仅作补充确认。
    # 开启后仅当候选工具数 > P2_LOCAL_CLASSIFY_MIN_CANDIDATES 才介入，避免对少量候选做无意义二次推理。
    ENABLE_P2_LOCAL_CLASSIFY: bool = False
    P2_LOCAL_CLASSIFY_MIN_CANDIDATES: int = 4

    # P2 线性头（PyTorch 训练的 ToolHead，替代生成式 chat_classify）。
    # 权重由 scripts/eval/tool_select/eval_embedding_baseline.py 训练产出，
    # 输入为 BAAI/bge-small-zh-v1.5 归一化后的 query embedding，输出为训练词表上的 logits。
    # 默认词表为生产 5 个 skill（data/lscale_prod.json：query-order / check-shipping /
    # check-balance / coupon-inquiry / request-return），与生产候选名一一对应，
    # 故 P2 可对真实候选正常打分（不再因词表 0 重合而退化为透传）。
    # P2 线性头（ONNX 导出权重 + 元信息，随镜像打包进 src/modules/chat/agent/assets/）。
    # 默认相对路径基于容器 WORKDIR=/code 解析；若该路径不存在，分类器自动回退到
    # 模块同级 assets/ 目录下的同名文件，保证 Docker 构建后一定能定位到打包权重。
    P2_HEAD_MODEL_PATH: str = Field(default="src/modules/chat/agent/assets/p2_linear_head.onnx")
    P2_HEAD_META_PATH: str = Field(default="src/modules/chat/agent/assets/head_meta.json")
    P2_HEAD_EMBED_MODEL: str = Field(default="/models/bge-small-zh-v1.5")
    P2_HEAD_CLASSES_SOURCE: str = Field(default="data/lscale_prod.json")
    P2_HEAD_TOP_K: int = Field(default=3)

    # Embedding 模型（本地 BGE/Sentence-Transformers）
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-small-zh-v1.5")
    # Embedding 本地模型路径（优先级最高，存在则直接加载；不存在则报错，不走 HF 下载）
    # 如 E:/workspace/shop-agent/models/BAAI/bge-small-zh-v1.5
    EMBEDDING_MODEL_LOCAL_PATH: str = Field(default="")
    # Embedding 后端: local=进程内 sentence-transformers（本地开发） | ollama=进程外 Ollama API（k8s 部署） | vllm=进程外 vLLM OpenAI 兼容端点
    EMBEDDING_PROVIDER: str = Field(default="vllm")
    # Ollama embedding 模型名（EMBEDDING_PROVIDER=ollama 时生效，`ollama create` 注册的名字）
    OLLAMA_EMBEDDING_MODEL: str = Field(default="bge-small-zh-v1.5")

    # BGE-Reranker 本地模型路径（用于 RAG 检索结果重排序）
    # 优先从本地路径加载，不存在则报错（不走 HF 下载）
    RERANKER_LOCAL_MODEL_PATH: str = Field(default="")  # 如 E:/workspace/shop-agent/models/BAAI/bge-reranker-base

    # 重排后端: local=进程内 sentence-transformers | vllm=远程调 vLLM bge-reranker（直连）
    RERANKER_PROVIDER: str = "vllm"
    # vLLM bge-reranker（RERANKER_PROVIDER=vllm 时生效，直连容器名）
    VLLM_RERANK_BASE_URL: str = "http://vllm-bge-reranker:8000"
    VLLM_RERANK_MODEL: str = "bge-reranker-base"

    # vLLM bge-small-zh-v1.5 嵌入（EMBEDDING_PROVIDER=vllm 时生效，直连容器名）
    # 生产 embedding 端点（vllm-bge-small-zh 容器）服务 /models/bge-small-zh-v1.5（512 维），
    # 与 P2 线性头训练所用 embedding 完全一致，实现"工具选择四层统一到一个 embedding 模型"。
    VLLM_EMBEDDING_BASE_URL: str = "http://vllm-bge-small-zh:8000"
    VLLM_EMBEDDING_MODEL: str = "/models/bge-small-zh-v1.5"

    # 本地小模型配置（参数抽取、工具选择等）——全部走 vLLM 统一服务，不在 shop-agent 进程内加载
    LOCAL_PARAM_MODEL_PATH: str = Field(default="")
    TOOL_SELECTOR_LOCAL_MODEL_PATH: str = Field(default="")

    # 本地小模型后端：统一用 vLLM（openai 兼容端点），不走 transformers 进程内加载
    LOCAL_MODEL_BACKEND: str = "vllm"

    # 向量数据库提供者: milvus | pgvector
    VECTOR_STORE_PROVIDER: str = "milvus"

    # Milvus向量数据库配置
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # PostgreSQL pgvector 配置（VECTOR_STORE_PROVIDER=pgvector 时生效）
    PGVECTOR_HOST: str = "localhost"
    PGVECTOR_PORT: int = 5432
    PGVECTOR_DB: str = "shop_agent"
    PGVECTOR_USER: str = "postgres"
    PGVECTOR_PASSWORD: str = ""  # 禁止源码硬编码口令；生产环境必须外部注入
    PGVECTOR_TABLE: str = "documents"
    # 远程业务API配置（意图识别触发远程调用时使用）
    REMOTE_API_BASE_URL: str = ""
    REMOTE_API_TIMEOUT: int = 10

    # 订单服务（Rust + PostgreSQL）：提供售后举证 / 订单数据，替代原 Mock
    ORDER_SERVICE_URL: str = "http://order-service:8080"
    ORDER_SERVICE_TIMEOUT: int = 5

    # 意图识别模式: local=关键词+向量(免费), llm=通义千问(精准)
    INTENT_RECOGNITION_MODE: str = "local"

    # 参数抽取模式: local=正则+关键词(免费,毫秒级), local_model=transformers本地小模型(免费,智能), llm=通义千问structured output(精准)  # noqa: E501
    PARAM_EXTRACTION_MODE: str = "local"

    # FAISS 意图向量匹配参数
    INTENT_VECTOR_SIMILARITY_THRESHOLD: float = 0.65  # 余弦相似度阈值（BGE归一化向量用内积）
    INTENT_WRITE_THRESHOLD: float = 0.78  # 写类意图（如 request-return）的高阈值，防止误触发副作用操作
    AMBIGUITY_SIMILARITY_THRESHOLD: float = 0.72  # 歧义带阈值：低于此值判为可能存在歧义，需 ReAct 处理

    # 同义词归一化配置
    # L1+L2: 静态同义词表 + 文本标准化（默认开启，零LLM成本，零延迟）
    SYNONYM_NORMALIZE_ENABLED: bool = True
    # L3: LLM归一化（默认关闭，需API调用，约500-1500ms延迟，覆盖长尾表达）
    SYNONYM_NORMALIZE_LLM_ENABLED: bool = False

    # NebulaGraph 图数据库配置（商品关系图谱，增强 RAG 的结构化知识）
    NEBULA_GRAPH_ADDRS: str = "127.0.0.1:9669"  # graphd 地址，逗号分隔多地址
    NEBULA_USER: str = "root"
    NEBULA_PASSWORD: str = ""  # 禁止源码硬编码口令；生产环境必须外部注入
    NEBULA_SPACE: str = "shop_graph"  # 图空间名
    NEBULA_TIMEOUT: int = 3000  # 连接超时 ms
    NEBULA_POOL_SIZE: int = 4  # 连接池大小
    NEBULA_GRAPH_ENABLED: bool = True  # 是否启用图查询增强

    # Step2 输入安全审查本地小模型配置
    # 开启后 Step2 优先用本地小模型做合规分类（省 API 费），非合规才升级云端 LLM 复核
    STEP1_SAFETY_LOCAL_MODEL_ENABLED: bool = False

    # Token 预估器配置（用于 Token 消耗限流）
    # Qwen3 全系列共用 tokenizer，指向本地 tokenizer.json 即可
    TOKENIZER_PATH: str = "./models/Qwen3-1.7B/tokenizer.json"
    # Token 消耗限流默认值（每窗口 max_tokens）
    TOKEN_LIMIT_MAX_TOKENS: int = 100000  # 每分钟最大 token 消耗
    TOKEN_LIMIT_WINDOW_SECONDS: int = 60  # 窗口 60 秒
    TOKEN_LIMIT_ENABLED: bool = True  # 是否启用 token 消耗限流
    # 用户输入长度管控（基于 token 而非字符数，与 LLM 实际消耗一致）
    MAX_USER_MESSAGE_TOKENS: int = 2000  # 单条用户消息的最大 token 数（~1300 中文字/4000 英文字）
    # 截断策略: keep_both_ends | keep_start_only | keep_end_only
    # keep_both_ends: 保留首 40% + 尾 20%，中间插入省略标记（推荐，核心意图在首部，关键细节在尾部）
    # keep_start_only: 仅保留开头（适合客服场景）
    TRUNCATION_STRATEGY: str = "keep_both_ends"
    # 截断提示语（{original_tokens}/{truncated_tokens}/{max_tokens} 会被替换）
    TRUNCATION_WARNING_TEMPLATE: str = (
        "⚠️ 您的输入较长（原始 {original_tokens} token，已自动保留核心 {truncated_tokens} token）。"
        "如需更精准的回答，建议精简描述后重新提问。\n\n"
    )

    # MCP Server 配置
    MCP_SERVER_NAME: str = "shop-agent"
    MCP_ENABLED: bool = False  # 是否启用 MCP Server
    MCP_TRANSPORT: str = "stdio"  # stdio | sse | streamable-http

    # MCP Client 配置 —— Agent 作为 Client 消费远程 MCP Server 的工具
    # JSON 数组，每个元素包含 name、url、headers（可选）
    # 示例: '[{"name":"order-system","url":"http://localhost:3002/mcp"}]'
    MCP_CLIENT_SERVERS: str = ""
    MCP_CLIENT_ENABLED: bool = False  # 是否启用 MCP Client 模式

    # 基于角色的工具权限控制（默认开启；admin 角色放行全部工具，见 core/permissions.py）
    PERMISSION_ENABLED: bool = True

    # 速率限制（可调，压测时提高以测真实编排层吞吐；默认值与历史一致）
    GLOBAL_RATE_LIMIT: int = 30  # 全局中间件：req / 60s / IP
    CHAT_RATE_LIMIT: int = 15  # /agent/chat 端点级：req / 60s / IP

    # LLM 适配器类型：langchain（默认）| mock（压测 0 Token 消耗）
    LLM_ADAPTER_TYPE: str = "langchain"

    # Mock LLM（压测 0 Token 消耗）：LLM_ADAPTER_TYPE=mock 时生效
    # MOCK_LLM_LATENCY_MIN/MAX 模拟 LLM 延迟（ms）；MOCK_LLM_ERROR_RATE 模拟错误率；MOCK_LLM_OUTPUT_TOKENS 单次输出 token  # noqa: E501
    MOCK_LLM_LATENCY_MIN: int = 500
    MOCK_LLM_LATENCY_MAX: int = 800
    MOCK_LLM_ERROR_RATE: float = 0.01
    MOCK_LLM_OUTPUT_TOKENS: int = 200

    # TTS 配置
    TTS_PROVIDER: str = "edge"  # edge | baidu
    BAIDU_TTS_API_KEY: str = ""
    BAIDU_TTS_SECRET_KEY: str = ""

    # 数字人配置
    AVATAR_PROVIDER: str = "static"  # static | baidu
    BAIDU_AVATAR_API_KEY: str = ""
    BAIDU_AVATAR_SECRET_KEY: str = ""

    @property
    def database_url(self) -> str:
        """构建数据库连接URL

        优先使用环境注入的完整 DATABASE_URL（.env 中已配置为 PostgreSQL，
        见 .env 的 DATABASE_URL=postgresql://postgres:...@postgres:5432/postgres），
        否则回退到 MySQL 拼接（兼容历史默认配置）。
        """
        import os

        env_url = os.getenv("DATABASE_URL")
        if env_url:
            # 应用使用 SQLAlchemy 异步引擎，需 asyncpg 驱动；
            # .env 的 DATABASE_URL 为通用 postgresql://（同步 scheme），
            # 此处统一改写为 async 驱动，避免 SQLA 误用 psycopg2。
            if env_url.startswith("postgresql://"):
                env_url = "postgresql+asyncpg://" + env_url[len("postgresql://"):]
            elif env_url.startswith("postgres://"):
                env_url = "postgresql+asyncpg://" + env_url[len("postgres://"):]
            return env_url
        return f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @model_validator(mode="after")
    def _enforce_secret_hygiene(self) -> "Settings":
        """生产环境（非 DEBUG）禁止空密码/密钥，fail-closed 启动即报错。

        D 维度修复：移除源码硬编码弱口令（123456/postgres/nebula）后，
        密码必须外部注入；开发环境（DEBUG_MODE=True）仅告警。
        """
        passwords = {
            "DB_PASSWORD": self.DB_PASSWORD,
            "PGVECTOR_PASSWORD": self.PGVECTOR_PASSWORD,
            "NEBULA_PASSWORD": self.NEBULA_PASSWORD,
            "REDIS_PASSWORD": self.REDIS_PASSWORD,
        }
        missing = [name for name, val in passwords.items() if not val]
        if missing and not self.DEBUG_MODE:
            raise ValueError(
                "生产环境禁止空密码，请通过环境变量/secret 注入：" + ", ".join(missing)
            )
        if missing:
            import warnings

            warnings.warn(
                "以下密码未配置（空值），仅允许在 DEBUG 开发环境使用："
                + ", ".join(missing),
                stacklevel=2,
            )
        # 生产环境（非 DEBUG）禁止 DEBUG 级别日志，避免敏感信息与海量日志外泄。
        # D 维度修复：DEBUG_MODE=True / LOG_LEVEL=DEBUG 作为生产默认值是高风险配置。
        if not self.DEBUG_MODE and self.LOG_LEVEL.upper() == "DEBUG":
            raise ValueError(
                "生产环境（DEBUG_MODE=False）禁止 LOG_LEVEL=DEBUG，请改用 INFO/WARNING 等。"
            )
        if self.DEBUG_MODE and self.LOG_LEVEL.upper() == "DEBUG":
            import warnings

            warnings.warn(
                "当前为 DEBUG_MODE=True 且 LOG_LEVEL=DEBUG，仅在开发与排障时允许；"
                "生产部署必须设置 DEBUG_MODE=False 并关闭调试日志。",
                stacklevel=2,
            )
        return self

    class Config:
        env_file = (".env", ".env.prod")  # 多个环境文件，后者优先
        extra = "ignore"  # 忽略未知的环境变量


    # ===== MLOps 闭环模块配置（训练端已解耦到独立 mlops-trainer 服务） =====
    MLOPS_ENABLED: bool = True
    # 空字符串 = 本容器自带训练栈本地执行（dev）；生产指向独立训练 worker。
    # 解耦后训练/评测由 mlops-trainer（独立 GPU 容器）承担，在线 serving 不再 spawn 训练进程。
    MLOPS_TRAINER_URL: str = ""
    # 训练/评测产物共享目录：shop-agent 与 mlops-trainer 通过同名挂载共享同一绝对路径。
    # 两容器均挂载到 /code/mlops_artifacts，故该绝对路径在两容器内一致。
    MLOPS_ARTIFACTS_DIR: str = "/code/mlops_artifacts"
    MLOPS_PYTHON: str = "python"  # 仅本地兜底（MLOPS_TRAINER_URL 为空）时使用的解释器
    MLOPS_BASE_MODEL: str = "/code/models/Qwen3-1.7B"
    MLOPS_OUTPUT_DIR: str = "outputs/mlops"
    MLOPS_EVAL_DATA: str = "/code/data/llamafactory/shop_param_v1.json"
    MLOPS_EVAL_DEVICE: str = "cuda"
    MLOPS_EVAL_MAX_SAMPLES: int = 200
    MLOPS_EVAL_THRESHOLDS: Dict[str, float] = Field(
        default_factory=lambda: {"field_f1": 0.6, "value_exact_match_rate": 0.6}
    )
    MLOPS_PUBLISH_CMD: str = ""  # 留空则只写 active 模型文件，需人工重启 serving；可填 scripts/publish_model.sh
    MLOPS_ACTIVE_MODEL_FILE: str = "models/active_model.txt"
    # A2A 任务完成时自动把执行 trace（用户 query + 模型输出 + 工具调用轨迹）灌入
    # MLOps 复核任务作为 sample_content，使标注界面无需手工粘贴即可看到待标注内容。
    MLOPS_AUTO_CAPTURE_FROM_A2A: bool = True
    # 工具选择监控：流水线出错 / 退化到 need_llm / 低置信度 时，自动把该次选择灌入
    # MLOps 复核任务，记录「选了什么 / 候选 / 置信度 / 由哪层选出 / 全部可选工具」，
    # 标注界面提供工具下拉框供标注员选「应该是什么」。
    MLOPS_TOOL_SELECT_MONITOR_ENABLED: bool = True
    # 不仅捕获可疑项，也捕获每一次选择（用于积累标注语料）。
    # 默认开启：标注语料的分布必须与线上真实分布一致。若只在「报错/低置信/不确定」时
    # 采集，样本池会系统性偏向难题，缺失 P0/P1/P2 高置信早停（即成本漏斗真正生效）的
    # 样本，导致训练集分布偏斜、且无法评估便宜层级的真实准确率。代价是 Langfuse 数据量
    # 上升，但这是训练数据资产，收益远大于存储成本。
    MLOPS_TOOL_SELECT_MONITOR_ALL: bool = True
    # 低置信度阈值：主工具置信度低于该值即视为可疑并捕获。
    MLOPS_TOOL_SELECT_LOW_CONF_THRESHOLD: float = 0.85
    # 执行/反馈期业务回灌（工具执行失败 + 用户纠正）开关；设计 2（Langfuse 版）。
    MLOPS_TOOL_SELECT_FEEDBACK_CAPTURE: bool = True
    # 版本化数据集落盘目录。留空则用默认可写路径（`scripts/data/tool_select`）。
    # 生产建议指向持久化卷：容器内文件系统会随重建丢失，数据集必须落卷。
    MLOPS_DATASET_DIR: str = ""


config = Settings()
