"""
Prometheus 自定义指标定义
"""

import inspect
import time
from functools import wraps

from prometheus_client import Counter, Gauge, Histogram, Info

# ============ 应用信息 ============
app_info = Info("shop_agent", "Shop Agent application information")

# ============ HTTP 请求指标 (由 instrumentator 自动处理) ============
# 这些指标由 prometheus-fastapi-instrumentator 自动生成：
# - http_requests_total (counter): HTTP 请求总数
# - http_request_duration_seconds (histogram): 请求耗时分布
# - http_requests_in_progress (gauge): 正在处理的请求数

# ============ 业务自定义指标 ============

# API 调用统计 (按模块/接口维度)
# 注意：不包含 endpoint 标签以避免高基数问题
# 如需追踪具体端点，使用 FastAPI instrumentator 自动生成的 http_requests_total 指标
api_call_counter = Counter(
    "shop_agent_api_calls_total",
    "API 调用总次数",
    ["module", "method", "status"],  # 标签：模块、方法、状态 (避免高基数)
)

# API 调用耗时
api_duration_histogram = Histogram(
    "shop_agent_api_duration_seconds",
    "API 调用耗时分布",
    ["module"],  # 只按模块区分，避免高基数
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
    ),  # 更精细的 bucket 配置
)

# 数据库查询统计
db_query_counter = Counter(
    "shop_agent_db_queries_total",
    "数据库查询总次数",
    ["operation", "table"],  # 操作类型(select/insert/update/delete), 表名
)

# 数据库查询耗时
db_query_duration = Histogram(
    "shop_agent_db_query_duration_seconds",
    "数据库查询耗时",
    ["operation", "table"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, float("inf")),
)

# Milvus 向量检索统计
milvus_search_counter = Counter(
    "shop_agent_milvus_searches_total", "Milvus 向量检索次数", ["collection"]
)

# Milvus 检索耗时
milvus_search_duration = Histogram(
    "shop_agent_milvus_search_duration_seconds",
    "Milvus 检索耗时",
    ["collection"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, float("inf")),
)

# Embedding 请求统计
embedding_request_counter = Counter(
    "shop_agent_embedding_requests_total",
    "Embedding 请求次数",
    ["provider", "status"],  # 提供商(dashscope/local), 状态(success/error)
)

# Embedding 请求耗时
embedding_request_duration = Histogram(
    "shop_agent_embedding_request_duration_seconds",
    "Embedding 请求耗时",
    ["provider"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, float("inf")),
)

# Embedding Token 使用量统计
embedding_token_counter = Counter(
    "shop_agent_embedding_tokens_total",
    "Embedding Token 使用总量",
    ["provider", "type"],  # 提供商, 类型(text/image/total)
)

# Redis 缓存命中/未命中统计
redis_cache_counter = Counter(
    "shop_agent_redis_cache_total",
    "Redis 缓存操作统计",
    ["operation", "result"],  # operation: get/set/delete, result: hit/miss/success/error
)

# Redis 依赖可用性 (Gauge: 1=可用, 0=降级/不可用)
# 用于 fail-soft 模式下观测「Redis 挂掉但 chat 仍可服务」的降级态，
# 替代原先靠 Pod 被摘除(up==0)才能发现的粗粒度信号。
redis_available = Gauge("shop_agent_redis_available", "Redis 依赖是否可用 (1=可用, 0=降级/不可用)")

# Agent 对话轮次统计
agent_conversation_counter = Counter(
    "shop_agent_conversations_total",
    "Agent 对话总轮次",
    ["status"],  # success/failed
)

# Agent Token 使用量统计
agent_token_counter = Counter(
    "shop_agent_tokens_total",
    "Agent Token 使用总量",
    ["type"],  # prompt/completion
)

# Agent Chat 请求计数
agent_chat_counter = Counter(
    "shop_agent_agent_chat_total",
    "Agent Chat 请求总数",
    ["status"],  # success/error
)

# Agent Chat 请求耗时（护栏 P99 延迟数据源，Phase 5 接入）
# 单位：毫秒(ms)。bucket 覆盖常规(<1s)到极端(<30s)场景，P99 由 bucket 累积估算。
agent_chat_duration_ms = Histogram(
    "shop_agent_agent_chat_duration_ms",
    "Agent Chat 请求耗时(ms)",
    buckets=(100, 250, 500, 1000, 2000, 3000, 5000, 8000, 12000, 20000, 30000),
)

# ============ L2 短期记忆指标 ============

l2_save_triggered_total = Counter(
    "shop_agent_l2_save_triggered_total",
    "L2 保存触发次数",
    ["trigger"],  # turn / timeout / batch
)

l2_save_success_total = Counter(
    "shop_agent_l2_save_success_total",
    "L2 保存成功次数",
    ["trigger"],
)

l2_save_failure_total = Counter(
    "shop_agent_l2_save_failure_total",
    "L2 保存失败次数",
    ["trigger", "error"],  # error: milvus_unavailable / llm_error / redis_error
)

l2_save_duration_ms = Histogram(
    "shop_agent_l2_save_duration_ms",
    "L2 保存耗时（摘要生成 + Milvus 写入）",
    ["trigger"],
    buckets=(50, 100, 200, 500, 1000, 2000, 5000),
)

l2_summary_tokens = Histogram(
    "shop_agent_l2_summary_tokens",
    "L2 摘要 token 数",
    buckets=(50, 100, 200, 500, 1000, 2000),
)

# 异常统计
exception_counter = Counter(
    "shop_agent_exceptions_total",
    "异常发生次数",
    ["type", "module"],  # 异常类型, 模块
)

# ============ 辅助装饰器 ============


def track_api_call(module: str):
    """
    API 调用追踪装饰器
    用法: @track_api_call('chat')
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            endpoint = func.__name__
            method = "unknown"

            # 尝试从 kwargs 或 args 获取 request 对象以确定 HTTP 方法
            for arg in list(args) + list(kwargs.values()):
                if hasattr(arg, "method"):
                    method = arg.method
                    break

            try:
                result = await func(*args, **kwargs)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                exception_counter.labels(type=type(e).__name__, module=module).inc()
                raise
            finally:
                duration = time.time() - start_time
                api_call_counter.labels(
                    module=module, endpoint=endpoint, method=method, status=status
                ).inc()
                api_duration_histogram.labels(module=module, endpoint=endpoint).observe(duration)

        # 同步函数包装器
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            endpoint = func.__name__

            try:
                result = func(*args, **kwargs)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                exception_counter.labels(type=type(e).__name__, module=module).inc()
                raise
            finally:
                duration = time.time() - start_time
                api_call_counter.labels(
                    module=module, endpoint=endpoint, method="sync", status=status
                ).inc()
                api_duration_histogram.labels(module=module, endpoint=endpoint).observe(duration)

        # 判断是异步还是同步函数
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def track_db_query(operation: str, table: str):
    """
    数据库查询追踪装饰器
    用法: @track_db_query('select', 'users')
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                db_query_counter.labels(operation=operation, table=table).inc()
                db_query_duration.labels(operation=operation, table=table).observe(duration)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                db_query_counter.labels(operation=operation, table=table).inc()
                db_query_duration.labels(operation=operation, table=table).observe(duration)

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def track_milvus_search(collection: str = "default"):
    """
    Milvus 检索追踪装饰器
    用法: @track_milvus_search('item_embeddings')
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                milvus_search_counter.labels(collection=collection).inc()
                milvus_search_duration.labels(collection=collection).observe(duration)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                milvus_search_counter.labels(collection=collection).inc()
                milvus_search_duration.labels(collection=collection).observe(duration)

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ============ MCP Client 指标 ============
# 工具调用计数（按 tool + status）
mcp_call_total = Counter(
    "shop_agent_mcp_calls_total",
    "MCP 工具调用次数",
    ["tool", "status"],  # status: success / error
)

# 工具调用耗时（按 tool）
mcp_call_duration = Histogram(
    "shop_agent_mcp_call_duration_seconds",
    "MCP 工具调用耗时分布",
    ["tool"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")),
)

# 连接状态（按 server）：1=已连接，0=断开/降级
mcp_connection_status = Gauge(
    "shop_agent_mcp_connection_status",
    "MCP 远程服务连接状态 (1=已连接, 0=断开/降级)",
    ["server"],
)

# 当前活跃 session（连接）数
mcp_sessions_active = Gauge(
    "shop_agent_mcp_sessions_active",
    "当前活跃的 MCP 连接(session)数",
)

# 已发现的 MCP 工具总数
mcp_tools_total = Gauge(
    "shop_agent_mcp_tools_total",
    "已从远程 MCP Server 发现的工具总数",
)

# Schema 失配告警（档 B）：远程 inputSchema 与项目期望契约不一致时累加。
# mismatch_type: field_missing（字段名漂移）/ type_drift（类型漂移）/ required_mismatch（必填缺漏）
mcp_schema_mismatch_total = Counter(
    "shop_agent_mcp_schema_mismatch_total",
    "MCP 工具 schema 与项目期望契约失配次数（告警，不阻断调用）",
    ["tool", "mismatch_type"],
)

# ============ 记忆系统指标 ============
memory_recall_counter = Counter(
    "shop_agent_memory_recall_total",
    "记忆召回总次数",
    ["layer"],  # l2 / l3 / profile
)

# 记忆召回耗时
memory_recall_duration = Histogram(
    "shop_agent_memory_recall_duration_seconds",
    "记忆召回耗时",
    ["layer"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, float("inf")),
)

# 记忆块统计（按类型）
memory_block_counter = Counter(
    "shop_agent_memory_blocks_total",
    "记忆块总数（按类型）",
    ["block_type"],  # summary / preference / profile / fact ...
)

# 遗忘任务指标
forgetting_job_counter = Counter(
    "shop_agent_forgetting_job_total",
    "遗忘任务执行次数",
    ["action"],  # archive / delete
)

forgetting_job_duration = Histogram(
    "shop_agent_forgetting_job_duration_seconds",
    "遗忘任务耗时",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, float("inf")),
)

# MRAG 记忆上下文大小（字符数）
memory_context_size = Histogram(
    "shop_agent_memory_context_chars",
    "MRAG 记忆上下文字符数",
    ["layer"],  # short_term / long_term / profile
    buckets=(50, 100, 200, 500, 1000, 2000, 5000),
)

# ============ 工具选择四层 Pipeline 监控（仅保留 4 个核心指标） ============
# 1. exit_total{stage, stop_condition}    - 成本漏斗分布（最关键）
# 2. stage_duration_ms{stage}              - 瓶颈定位 p50/p95
# 3. candidates_in/out{stage}              - 漏斗收窄验证
# 4. stage_total{outcome="error/timeout"}  - 异常率零容忍

# 各层执行计数（按 outcome 区分命中/未命中/错误/超时）
tool_select_stage_total = Counter(
    "shop_agent_tool_select_stage_total",
    "工具选择各层(stage)执行计数",
    ["stage", "outcome"],  # outcome: hit / miss / error / timeout
)

# 各层执行耗时（毫秒）
# bucket 设计覆盖 P0<1ms 到 P3>5s，p50/p95 从 buckets 插值可得
tool_select_stage_duration_ms = Histogram(
    "shop_agent_tool_select_stage_duration_ms",
    "工具选择各层执行耗时(ms)",
    ["stage"],
    buckets=(0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
)

# 进入各层的候选工具数量（漏斗上游）
tool_select_candidates_in = Histogram(
    "shop_agent_tool_select_candidates_in",
    "进入各层的候选工具数量",
    ["stage"],
    buckets=(1, 2, 3, 5, 10, 15, 20, 30, 50),
)

# 各层输出(收窄后)的候选工具数量（漏斗下游）
tool_select_candidates_out = Histogram(
    "shop_agent_tool_select_candidates_out",
    "各层收窄后输出的候选工具数量",
    ["stage"],
    buckets=(0, 1, 2, 3, 5, 10, 15, 20, 30),
)

# 终止工具选择的层级分布（成本漏斗：期望在 P0/P1/P2 提前结束，而非总到 P3）
tool_select_exit_total = Counter(
    "shop_agent_tool_select_exit_total",
    "终止工具选择的层级分布(成本漏斗)",
    ["stage", "stop_condition"],  # stage: p0/p1/p2/p3/fallback
)

# 各层产出的置信度分布（调早停阈值的依据：看阈值是否可达、是否过于宽松）
tool_select_confidence = Histogram(
    "shop_agent_tool_select_confidence",
    "各层产出的候选置信度",
    ["stage"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.78, 0.85, 0.9, 0.95, 0.99, 1.0),
)

# 最终结果分布（来源 / 终止条件 / 候选集规模）
# 与 exit_total 的区别：exit 记「在哪层终止」，final 记「最终产出的形态」，
# 用于判断候选集是否收敛、是否退化成 need_llm。
tool_select_final_total = Counter(
    "shop_agent_tool_select_final_total",
    "工具选择最终结果分布",
    ["source", "stop_condition", "candidates"],  # candidates: 1 / 2 / 3+
)
