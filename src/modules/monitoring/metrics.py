"""
Prometheus 自定义指标定义
"""
from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time
import inspect

# ============ 应用信息 ============
app_info = Info('shop_agent', 'Shop Agent application information')

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
    'shop_agent_api_calls_total',
    'API 调用总次数',
    ['module', 'method', 'status']  # 标签：模块、方法、状态 (避免高基数)
)

# API 调用耗时
api_duration_histogram = Histogram(
    'shop_agent_api_duration_seconds',
    'API 调用耗时分布',
    ['module'],  # 只按模块区分，避免高基数
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)  # 更精细的 bucket 配置
)

# 数据库查询统计
db_query_counter = Counter(
    'shop_agent_db_queries_total',
    '数据库查询总次数',
    ['operation', 'table']  # 操作类型(select/insert/update/delete), 表名
)

# 数据库查询耗时
db_query_duration = Histogram(
    'shop_agent_db_query_duration_seconds',
    '数据库查询耗时',
    ['operation', 'table'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, float('inf'))
)

# Milvus 向量检索统计
milvus_search_counter = Counter(
    'shop_agent_milvus_searches_total',
    'Milvus 向量检索次数',
    ['collection']
)

# Milvus 检索耗时
milvus_search_duration = Histogram(
    'shop_agent_milvus_search_duration_seconds',
    'Milvus 检索耗时',
    ['collection'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, float('inf'))
)

# Embedding 请求统计
embedding_request_counter = Counter(
    'shop_agent_embedding_requests_total',
    'Embedding 请求次数',
    ['provider', 'status']  # 提供商(dashscope/local), 状态(success/error)
)

# Embedding 请求耗时
embedding_request_duration = Histogram(
    'shop_agent_embedding_request_duration_seconds',
    'Embedding 请求耗时',
    ['provider'],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, float('inf'))
)

# Redis 缓存命中/未命中统计
redis_cache_counter = Counter(
    'shop_agent_redis_cache_total',
    'Redis 缓存操作统计',
    ['operation', 'result']  # operation: get/set/delete, result: hit/miss/success/error
)

# 活跃用户数 (Gauge 类型，可增可减)
active_users = Gauge(
    'shop_agent_active_users',
    '当前活跃用户数'
)

# Agent 对话轮次统计
agent_conversation_counter = Counter(
    'shop_agent_conversations_total',
    'Agent 对话总轮次',
    ['status']  # success/failed
)

# Agent Token 使用量统计
agent_token_counter = Counter(
    'shop_agent_tokens_total',
    'Agent Token 使用总量',
    ['type']  # prompt/completion
)

# 异常统计
exception_counter = Counter(
    'shop_agent_exceptions_total',
    '异常发生次数',
    ['type', 'module']  # 异常类型, 模块
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
            method = 'unknown'
            
            # 尝试从 kwargs 或 args 获取 request 对象以确定 HTTP 方法
            for arg in list(args) + list(kwargs.values()):
                if hasattr(arg, 'method'):
                    method = arg.method
                    break
            
            try:
                result = await func(*args, **kwargs)
                status = 'success'
                return result
            except Exception as e:
                status = 'error'
                exception_counter.labels(type=type(e).__name__, module=module).inc()
                raise
            finally:
                duration = time.time() - start_time
                api_call_counter.labels(
                    module=module,
                    endpoint=endpoint,
                    method=method,
                    status=status
                ).inc()
                api_duration_histogram.labels(
                    module=module,
                    endpoint=endpoint
                ).observe(duration)
        
        # 同步函数包装器
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            endpoint = func.__name__
            
            try:
                result = func(*args, **kwargs)
                status = 'success'
                return result
            except Exception as e:
                status = 'error'
                exception_counter.labels(type=type(e).__name__, module=module).inc()
                raise
            finally:
                duration = time.time() - start_time
                api_call_counter.labels(
                    module=module,
                    endpoint=endpoint,
                    method='sync',
                    status=status
                ).inc()
                api_duration_histogram.labels(
                    module=module,
                    endpoint=endpoint
                ).observe(duration)
        
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


def track_milvus_search(collection: str = 'default'):
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


def track_embedding(provider: str = 'dashscope'):
    """
    Embedding 请求追踪装饰器
    用法: @track_embedding('dashscope')
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                status = 'error'
                # 记录异常类型到异常计数器
                exception_counter.labels(type=type(e).__name__, module=provider).inc()
                raise
            finally:
                duration = time.time() - start_time
                embedding_request_counter.labels(provider=provider, status=status).inc()
                embedding_request_duration.labels(provider=provider).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            try:
                return func(*args, **kwargs)
            except Exception as e:
                status = 'error'
                # 记录异常类型到异常计数器
                exception_counter.labels(type=type(e).__name__, module=provider).inc()
                raise
            finally:
                duration = time.time() - start_time
                embedding_request_counter.labels(provider=provider, status=status).inc()
                embedding_request_duration.labels(provider=provider).observe(duration)
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator
