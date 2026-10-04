"""
LLM 服务模块
支持通义千问和火山引擎 Doubao 模型，含 Token 消耗限流
"""

import asyncio
import contextvars
import os
import time
from typing import Dict, List, Optional, Type, TypeVar

from langchain_openai import ChatOpenAI
from prometheus_client import Counter, Histogram
from pydantic import BaseModel

from src.modules.chat.config import chat_config
from src.modules.monitoring.langchain_callback import get_prometheus_callback
from src.ports import metrics
from src.shared.logger import APILogger

logger = APILogger("llm_service")

# 类型变量用于泛型返回
T = TypeVar("T", bound=BaseModel)

# ── 无旁路不变量（平台工程 01 §L3）────────────────────────────────────
# 所有出网 LLM 流量必须收敛到唯一出口（gateway）。此处不再提供公网
# fallback 默认值：缺失 LLM_GATEWAY_URL 即拒启，避免"忘记注入环境变量
# 就静默直连供应商"这一旁路。本地裸跑需显式声明逃生阀。
_DIRECT_EGRESS_FLAG = "ALLOW_DIRECT_LLM_EGRESS"
_DIRECT_EGRESS_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 逃生阀在生产环境默认无效：需显式 OVERRIDE_PROD=true 才允许，防止误配绕过网关。
_DIRECT_EGRESS_OVERRIDE_PROD = "ALLOW_DIRECT_LLM_EGRESS_OVERRIDE_PROD"
_PROD_ENV_VALUES = ("production", "prod", "prd")

# 监控：逃生阀被触发的次数（即便因生产护栏被拒绝也计数，便于运维发现误配）。
_direct_egress_counter = Counter(
    "shop_agent_llm_direct_egress_total",
    "LLM 出口逃生阀被触发的次数（直连公网供应商，绕过网关卡口链）",
    ["allowed"],  # allowed=1/0
)

# 监控：LLM 调用延迟分布（Histogram）。
_llm_call_duration_seconds = Histogram(
    "shop_agent_llm_call_duration_seconds",
    "LLM 调用端到端延迟分布（含重试）",
    ["model", "status"],  # status=success/timeout/error
)

# 监控：LLM 调用重试次数。
_llm_call_retries_total = Counter(
    "shop_agent_llm_call_retries_total",
    "LLM 调用因网络异常重试的总次数",
    ["model"],
)


class GatewayNotConfigured(RuntimeError):
    """未配置 LLM 网关出口且未显式开启直连逃生阀（或被生产护栏拦截）。"""


def _direct_egress_allowed() -> bool:
    """逃生阀是否真正允许直连。

    需 ``ALLOW_DIRECT_LLM_EGRESS=true``，且（非 production 环境 OR 显式
    ``OVERRIDE_PROD=true``）。生产环境误设该变量会被忽略，仍走拒启路径。
    """
    flag = (os.getenv(_DIRECT_EGRESS_FLAG) or "").strip().lower()
    if flag not in ("1", "true", "yes"):
        return False
    env = (os.getenv("ENVIRONMENT") or os.getenv("APP_ENV") or "").strip().lower()
    is_prod = env in _PROD_ENV_VALUES
    override = (os.getenv(_DIRECT_EGRESS_OVERRIDE_PROD) or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    return (not is_prod) or override


def resolve_llm_base_url() -> str:
    """解析 LLM 出口地址，fail-closed。

    优先返回 ``LLM_GATEWAY_URL``；未配置时仅当逃生阀真正允许
    （见 :func:`_direct_egress_allowed`）才降级为公网直连并计数告警，
    否则抛 :class:`GatewayNotConfigured` 拒绝启动。
    """
    gateway_url = (os.getenv("LLM_GATEWAY_URL") or "").strip()
    if gateway_url:
        return gateway_url

    if _direct_egress_allowed():
        _direct_egress_counter.labels(allowed="1").inc()
        logger.warning(
            "LLM_GATEWAY_URL 未配置，已按 %s 逃生阀直连公网供应商；"
            "该模式绕过网关卡口链（脱敏/护栏/注入闸/计量），仅限本地调试",
            _DIRECT_EGRESS_FLAG,
        )
        return _DIRECT_EGRESS_BASE_URL

    # 逃生阀未真正允许：区分"未设"与"生产被拦截"，均拒启并计数。
    _direct_egress_counter.labels(allowed="0").inc()
    env = (os.getenv("ENVIRONMENT") or os.getenv("APP_ENV") or "").strip().lower()
    if env in _PROD_ENV_VALUES and (os.getenv(_DIRECT_EGRESS_FLAG) or "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        logger.error(
            "生产环境检测到 %s=true 但未设 %s=true，逃生阀被拒；"
            "如确需在应急场景直连，请同时设置 %s=true",
            _DIRECT_EGRESS_FLAG,
            _DIRECT_EGRESS_OVERRIDE_PROD,
            _DIRECT_EGRESS_OVERRIDE_PROD,
        )
    raise GatewayNotConfigured(
        "LLM_GATEWAY_URL 未配置：所有出网 LLM 流量必须经由网关。"
        f"如需本地裸跑直连，请显式设置 {_DIRECT_EGRESS_FLAG}=true"
        f"（生产环境还需 {_DIRECT_EGRESS_OVERRIDE_PROD}=true）。"
    )


# ── Token 限流上下文（contextvars，跨异步调用传递）───────────────────
_rate_limit_key_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "llm_rate_limit_key", default=""
)
_token_limit_enabled_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "llm_token_limit_enabled", default=False
)


class TokenLimitExceeded(Exception):
    """Token 消耗超限异常。"""

    def __init__(self, msg: str, remaining: int = 0, reset_seconds: int = 0):
        super().__init__(msg)
        self.remaining = remaining
        self.reset_seconds = reset_seconds


def set_rate_limit_context(key: str, enabled: bool = True):
    """设置当前请求的 Token 限流上下文（在 router/中间件中调用）。"""
    _rate_limit_key_ctx.set(key)
    _token_limit_enabled_ctx.set(enabled)


def clear_rate_limit_context():
    """清除 Token 限流上下文。"""
    _rate_limit_key_ctx.set("")
    _token_limit_enabled_ctx.set(False)


# ── 横切关注点装饰器（平台工程 04）────────────────────────────────────

def _retry_on_network_error(max_attempts: int = 3):
    """LLM 调用网络异常重试装饰器（指数退避：1s / 2s / 4s）。"""
    import httpx
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
    )


# ── Gateway 健康检查 ───────────────────────────────────────────────────

_HEALTH_PATH = "/health"

async def _ping_gateway(base_url: str, timeout: float = 5.0) -> bool:
    """Ping LLM Gateway 健康端点，返回是否可用。"""
    import httpx
    url = base_url.rstrip("/") + _HEALTH_PATH
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            return resp.status_code < 500
    except Exception:
        return False


class LLMService:
    """大语言模型服务"""

    _instance = None
    _qwen_llm: Optional[ChatOpenAI] = None
    _tool_selector_llm: Optional[ChatOpenAI] = None
    _gateway_healthy: bool = True
    _last_gateway_check: float = 0.0
    _gateway_check_interval: float = 30.0  # 秒

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "LLMService":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self):
        """初始化 LLM 服务"""
        # 启动期即校验出口地址：无论是否配置 api_key，都必须经由网关或显式逃生阀。
        # 不依赖下方 api_key 分支 —— 否则缺 api_key 时会静默跳过校验（A-1）。
        try:
            resolve_llm_base_url()
        except GatewayNotConfigured:
            # 拒启语义不可被通用容错吞掉（A-2）：单独重抛，保持 fail-closed。
            logger.error("LLM 网关出口未配置，拒绝启动（无旁路不变量）")
            raise
        try:
            # 初始化通义千问模型。
            # LLM key 统一由 Gateway 持有，shop-agent 不持有真实 key：若无 TONGYI_API_KEY，
            # 用占位串满足 SDK 非空要求，真实鉴权由网关侧 TONGYI_API_KEY 兜底。
            _api_key = chat_config.tongyi_api_key or "gateway-managed"
            if chat_config.chat_model:
                self._qwen_llm = ChatOpenAI(
                    model=chat_config.chat_model,
                    api_key=_api_key,
                    base_url=resolve_llm_base_url(),
                    temperature=0.7,
                    max_tokens=8096,
                    extra_body={"enable_thinking": False},
                )
                logger.info(f"通义千问模型初始化成功: {chat_config.chat_model}")

        except Exception as e:
            logger.error(f"LLM 服务初始化失败: {str(e)}")
            raise

        # 启动时 ping 一次网关，fail-closed 不阻断启动（降级为 unhealthy 状态）。
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._check_gateway_health_task())
        except RuntimeError:
            # 非 async 上下文（如同步启动脚本），同步执行一次
            asyncio.run(self._check_gateway_health_task())

    async def _invoke_with_retry(self, llm, messages, config, max_retries=3):
        """带重试的 LLM 调用（网络异常指数退避重试 + Prometheus 指标）。"""
        import time as _time
        retry_count = 0

        def _before_sleep(retry_state):
            nonlocal retry_count
            retry_count += 1
            attempt = retry_state.attempt_number
            logger.warning(
                f"LLM 调用网络异常，准备第 {attempt} 次重试",
                retry_count=retry_count,
                error=str(retry_state.outcome.exception())[:120] if retry_state.outcome else "unknown",
            )

        retry_decorator = _retry_on_network_error(max_attempts=max_retries)
        t0 = _time.perf_counter()
        try:
            result = await retry_decorator(llm.ainvoke)(messages, config=config)
            duration = _time.perf_counter() - t0
            model_name = getattr(getattr(llm, "model_name", None), "model_name", "unknown")
            _llm_call_duration_seconds.labels(model=model_name, status="success").observe(duration)
            if retry_count > 0:
                _llm_call_retries_total.labels(model=model_name).inc(retry_count)
            return result
        except Exception as e:
            duration = _time.perf_counter() - t0
            model_name = getattr(getattr(llm, "model_name", None), "model_name", "unknown")
            status = "timeout" if "timeout" in str(e).lower() else "error"
            _llm_call_duration_seconds.labels(model=model_name, status=status).observe(duration)
            if retry_count > 0:
                _llm_call_retries_total.labels(model=model_name).inc(retry_count)
            raise

    async def _check_gateway_health_task(self):
        """后台任务：ping 网关并更新健康状态（不阻塞主流程）。"""
        try:
            healthy = await _ping_gateway(resolve_llm_base_url())
            type(self)._gateway_healthy = healthy
            type(self)._last_gateway_check = time.time()
            if not healthy:
                logger.warning("LLM Gateway 健康检查失败")
        except Exception as e:
            logger.debug(f"Gateway 健康检查异常: {str(e)[:100]}")

    def check_gateway_health(self) -> dict:
        """查询 Gateway 健康状态（用于监控/告警）。"""
        import time
        healthy = type(self)._gateway_healthy
        last_check = type(self)._last_gateway_check
        return {
            "healthy": healthy,
            "last_check_seconds_ago": round(time.time() - last_check, 1) if last_check else None,
            "gateway_url": resolve_llm_base_url() if healthy else "unknown",
        }

    @property
    def qwen_llm(self) -> ChatOpenAI:
        """获取通义千问 LLM（懒加载）。

        LLM key 统一由 Gateway 持有，shop-agent 不持有真实 key：无 TONGYI_API_KEY 时用
        占位串，真实鉴权由网关侧 TONGYI_API_KEY 兜底。
        """
        if self._qwen_llm is None:
            _api_key = chat_config.tongyi_api_key or "gateway-managed"
            if not chat_config.chat_model:
                raise ValueError("CHAT_MODEL 未配置")
            self._qwen_llm = ChatOpenAI(
                model=chat_config.chat_model,
                api_key=_api_key,
                base_url=resolve_llm_base_url(),
                temperature=0.7,
                max_tokens=8096,
                extra_body={"enable_thinking": False},
            )
        return self._qwen_llm

    @property
    def tool_selector_llm(self) -> ChatOpenAI:
        """获取工具选择器专用轻量 LLM（更快、更便宜）。

        P2 工具选择任务极其简单（从 3-5 个工具名选一个），
        不需要主 Agent 的大模型，用小模型可降低延迟 50%+ 且不牺牲准确率。
        工具选择器主路径走本地 vLLM（LocalModelService），本属性仅作云端 fallback。
        """
        if self._tool_selector_llm is None:
            # 无 tongyi_api_key 时降级到主模型（本地 vLLM 由网关兜底，无需云端 key）
            if not chat_config.tongyi_api_key:
                return self.qwen_llm  # 降级到主模型
            model = getattr(chat_config, "tool_selector_model", None)
            if model is None:
                return self.qwen_llm  # 未配置则回退主模型
            self._tool_selector_llm = ChatOpenAI(
                model=model,
                api_key=chat_config.tongyi_api_key,
                base_url=resolve_llm_base_url(),
                temperature=0.0,  # 工具选择不需要创造性
                max_tokens=8096,
                extra_body={"enable_thinking": False},
            )
            logger.info(f"工具选择器模型初始化成功: {model}")
        return self._tool_selector_llm

    async def _mock_simulate(self, messages: List[Dict[str, str]]) -> str:
        """Mock LLM 仿真：模拟延迟 + 错误率（压测 0 Token 消耗）。

        由 LLM_ADAPTER_TYPE=mock 触发，模拟真实 LLM 的行为特征而不产生任何
        外部调用或 Token 计费，用于发现编排层/网关的吞吐瓶颈。
        """
        import random

        latency_min = getattr(chat_config, "mock_llm_latency_min", 100) / 1000.0
        latency_max = getattr(chat_config, "mock_llm_latency_max", 500) / 1000.0
        error_rate = getattr(chat_config, "mock_llm_error_rate", 0.01)

        delay = random.uniform(latency_min, latency_max)
        await asyncio.sleep(delay)

        if random.random() < error_rate:
            raise TimeoutError("Mock LLM timeout (simulated)")

        last_user = messages[-1]["content"] if messages else ""
        return f"[Mock] 收到：{last_user[:50]}（模拟回复）"

    async def chat_qwen(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        track_metrics: bool = True,
        langfuse_handler=None,
        **kwargs,
    ) -> str:
        """
        使用通义千问模型聊天

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            track_metrics: 是否追踪指标（默认开启，LangChain回调会自动统计Token）
            langfuse_handler: 可选的 Langfuse CallbackHandler（由调用方传入，携带 session_id/user_id/tags）
            **kwargs: 其他参数

        Returns:
            模型回复内容

        Raises:
            TokenLimitExceeded: Token 消耗超限
        """
        try:
            # ── Mock 仿真（LLM_ADAPTER_TYPE=mock，压测 0 Token 消耗）──
            if getattr(chat_config, "LLM_ADAPTER_TYPE", "langchain") == "mock":
                return await self._mock_simulate(messages)
            # ── Token 消耗预检 ──
            estimated_tokens = 0
            if _token_limit_enabled_ctx.get():
                from src.core.config import config as core_config
                from src.core.rate_limiter import get_rate_limiter
                from src.core.token_estimator import get_token_estimator

                rl_key = _rate_limit_key_ctx.get("")
                estimator = get_token_estimator()
                estimated_tokens = estimator.estimate_messages(messages)
                limiter = get_rate_limiter()

                max_tokens = getattr(core_config, "TOKEN_LIMIT_MAX_TOKENS", 100000)
                window = getattr(core_config, "TOKEN_LIMIT_WINDOW_SECONDS", 60)
                allowed, remaining, reset = limiter.check_tokens(
                    rl_key, estimated_tokens, max_tokens=max_tokens, window_seconds=window
                )
                if not allowed:
                    logger.warning(
                        "Token 消耗超限 key=%s estimated=%d remaining=%d",
                        rl_key,
                        estimated_tokens,
                        remaining,
                    )
                    raise TokenLimitExceeded(
                        f"Token 消耗超限（预估 {estimated_tokens}，剩余 {remaining}）",
                        remaining=remaining,
                        reset_seconds=reset,
                    )

            # 构建配置
            config = {}
            if track_metrics:
                callbacks = [get_prometheus_callback()]
                if langfuse_handler:
                    callbacks.append(langfuse_handler)
                config["callbacks"] = callbacks

            response = await self._invoke_with_retry(self.qwen_llm, messages, config)

            # ── Token 消耗上报 ──
            if estimated_tokens > 0:
                self._report_token_usage(response, estimated_tokens)

            return response.content
        except TokenLimitExceeded:
            raise
        except Exception as e:
            logger.error(f"通义千问调用失败: {str(e)}")
            raise

    async def chat_qwen_structured(
        self,
        messages: List[Dict[str, str]],
        output_schema: Type[BaseModel],
        temperature: float = 0.0,
        track_metrics: bool = True,
        max_retries: int = 2,
        langfuse_handler=None,
        **kwargs,
    ) -> BaseModel:
        """
        使用通义千问模型进行结构化输出（带重试和降级）

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            output_schema: Pydantic Schema，用于结构化输出
            temperature: 温度参数（结构化输出通常用0）
            track_metrics: 是否追踪指标
            max_retries: 最大重试次数（ValidationError 时）
            langfuse_handler: 可选的 Langfuse CallbackHandler
            **kwargs: 其他参数

        Returns:
            结构化对象（Pydantic Model 实例）

        Raises:
            TokenLimitExceeded: Token 消耗超限
        """
        last_error = None

        # ── Mock 仿真（LLM_ADAPTER_TYPE=mock，压测 0 Token 消耗）──
        if getattr(chat_config, "LLM_ADAPTER_TYPE", "langchain") == "mock":
            await self._mock_simulate(messages)
            return output_schema()

        # ── Token 消耗预检（仅一次，不在重试循环内）──
        estimated_tokens = 0
        if _token_limit_enabled_ctx.get():
            from src.core.config import config as core_config
            from src.core.rate_limiter import get_rate_limiter
            from src.core.token_estimator import get_token_estimator

            rl_key = _rate_limit_key_ctx.get("")
            estimator = get_token_estimator()
            estimated_tokens = estimator.estimate_messages(messages)
            limiter = get_rate_limiter()

            max_tokens = getattr(core_config, "TOKEN_LIMIT_MAX_TOKENS", 100000)
            window = getattr(core_config, "TOKEN_LIMIT_WINDOW_SECONDS", 60)
            allowed, remaining, reset = limiter.check_tokens(
                rl_key, estimated_tokens, max_tokens=max_tokens, window_seconds=window
            )
            if not allowed:
                logger.warning(
                    "Token 消耗超限(结构化) key=%s estimated=%d remaining=%d",
                    rl_key,
                    estimated_tokens,
                    remaining,
                )
                raise TokenLimitExceeded(
                    f"Token 消耗超限（预估 {estimated_tokens}，剩余 {remaining}）",
                    remaining=remaining,
                    reset_seconds=reset,
                )

        for attempt in range(max_retries + 1):
            try:
                # 构建配置
                config = {}
                if track_metrics:
                    callbacks = [get_prometheus_callback()]
                    if langfuse_handler:
                        callbacks.append(langfuse_handler)
                    config["callbacks"] = callbacks

                # 使用 with_structured_output 获取支持结构化输出的 LLM
                structured_llm = self.qwen_llm.with_structured_output(output_schema)

                response = await self._invoke_with_retry(structured_llm, messages, config)

                # ── Token 消耗上报 ──
                if estimated_tokens > 0:
                    self._report_token_usage(response, estimated_tokens)

                return response

            except TokenLimitExceeded:
                raise
            except Exception as e:
                last_error = e
                error_type = type(e).__name__

                # ValidationError：模型输出不符合 Schema
                if "ValidationError" in error_type or "validation" in str(e).lower():
                    if attempt < max_retries:
                        logger.warning(
                            f"结构化输出验证失败（尝试 {attempt + 1}/{max_retries}）: {str(e)[:200]}"
                        )
                        continue
                    else:
                        logger.error(f"结构化输出重试耗尽: {str(e)[:200]}")

                logger.error(f"通义千问结构化调用失败: {error_type} - {str(e)[:200]}")
                raise

        # 不应该到达这里，但以防万一
        raise last_error or Exception("结构化输出未知错误")

    def _report_token_usage(self, response, estimated_tokens: int):
        """从 LLM 响应中提取真实 token 数并上报到限流器。"""
        try:
            rl_key = _rate_limit_key_ctx.get("")
            if not rl_key:
                return

            # 尝试从多种来源提取 token_usage
            actual = 0
            # 来源1: response_metadata（LangChain OpenAI 通常在这里）
            meta = getattr(response, "response_metadata", {}) or {}
            usage = meta.get("token_usage") or meta.get("usage")
            if isinstance(usage, dict):
                actual = usage.get("total_tokens") or (
                    usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                )

            # 来源2: usage_metadata（LangChain >= 0.3 新格式）
            if not actual:
                um = getattr(response, "usage_metadata", None)
                if um and isinstance(um, dict):
                    actual = um.get("total_tokens", 0) or (
                        um.get("input_tokens", 0) + um.get("output_tokens", 0)
                    )

            # 来源3: additional_kwargs（某些模型/适配器）
            if not actual:
                ak = getattr(getattr(response, "message", None), "additional_kwargs", {}) or {}
                if isinstance(ak, dict):
                    u = ak.get("usage", {})
                    if isinstance(u, dict):
                        actual = u.get("total_tokens", 0) or (
                            u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
                        )

            if actual <= 0:
                return

            # G 维度：上报独立成本指标（token 计数器，不依赖 Langfuse/SkyWalking）。
            # 金额缺单价时传 0，仅累计 token 用量；端口 stub 进程内聚合，可换 Prometheus/OTel。
            model_val = meta.get("model") if isinstance(meta, dict) else None
            model = model_val if isinstance(model_val, str) else "unknown"
            try:
                metrics.record_cost(model=str(model), tokens=actual)
            except Exception:
                pass

            from src.core.rate_limiter import get_rate_limiter

            limiter = get_rate_limiter()
            limiter.report_tokens(rl_key, actual, estimated_tokens)
        except Exception:
            pass  # 上报失败不影响主流程

    def create_structured_llm(self, output_schema: Type[BaseModel]) -> ChatOpenAI:
        """
        创建支持结构化输出的 LLM 实例

        Args:
            output_schema: Pydantic Schema

        Returns:
            配置好的 ChatOpenAI 实例（已绑定 with_structured_output）
        """
        return self.qwen_llm.with_structured_output(output_schema)

    async def chat_qwen_with_prompt(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        langfuse_handler=None,
    ) -> str:
        """使用 prompt 字符串调用通义千问"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self.chat_qwen(messages, temperature, langfuse_handler=langfuse_handler)

    def close(self):
        """关闭服务"""
        self._qwen_llm = None
        self._tool_selector_llm = None
        LLMService._instance = None
