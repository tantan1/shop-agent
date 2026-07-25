"""网关配置（pydantic-settings）。

后端端点全部 env 配置，平等可替换（01 §4）。换云厂商只改这里，业务无感。

生产级路由/故障转移由 LiteLLM Router 接管（见 gateway/litellm_router.py）：
  - `litellm_config_path` 指向 model_list YAML 时为权威来源（推荐生产用法）；
  - 未配该路径时，由下方 legacy 字段自动生成 model_list（向后兼容，标记 deprecated）。
"""
from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict

from .types import GatewayMode

# 百炼(阿里)默认端点：现有 shop-agent 直连地址，作裸跑回退默认值
_BAI_LIAN_DEFAULT = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 本地统一模型 served 名（vLLM 提供），所有本地小模型任务以该名发往 vLLM
_VLLM_MODEL = "qwen3-unified"


def split_chain(raw: str) -> list[str]:
    """将逗号分隔的地址串解析为去重保序的地址列表（02 §4 fallback 链）。

    保留 str 而非 list[str]，因 pydantic-settings 会对 list 字段尝试 JSON 解析而拒绝 `url1,url2`。
    """
    out: list[str] = []
    for part in (raw or "").split(","):
        p = part.strip()
        if p and p not in out:
            out.append(p)
    return out


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 基础
    ollama_base_url: str = "http://ollama:11434"
    openai_base_url: str = "https://api.openai.com/v1"
    # vLLM（本地 GPU 模型）。路由键 local/*、param、tool_select、qwen3-unified 等走此后端。
    # 地址需含 OpenAI 兼容路径前缀 /v1（与 openai_base_url 形态一致）。
    vllm_base_url: str = "http://vllm-qwen3:8000/v1"
    # mock LLM 上游（压测/联调用，独立 Go 服务零依赖）。路由键 mock* 走此后端。
    mock_base_url: str = "http://mock-llm:8080/v1"
    # 压测开关：开启后所有云端 API（azure/openai/bedrock/百炼）一律改路由到 mock 上游，
    # 本地 vLLM 不受影响。压测时免去真实云调用，直接打 mock-llm（配合 mock_base_url）。
    mock_cloud_override: bool = False

    # 路由后端（01 §4 / 02 §4）。每项可配逗号分隔多地址，首个为主后端，其余为有序 fallback 链
    # azure 默认给 Azure 形态占位地址（与 openai_base_url 不同，使 gpt-* 天然形成 2 元素跨厂商链），部署时配真值
    # 【legacy / deprecated】生产环境改用 litellm_config_path 的 model_list YAML 驱动，下列字段仅作
    # 未配置 YAML 时的自动回退，且不再演进（新增后端请走 YAML）。
    azure_openai_base_url: str = "https://your-azure-openai.openai.azure.com"  # 占位，部署时配真值
    bedrock_base_url: str = ""                                 # AWS Bedrock 无标准 OpenAI 兼容端点，留空待批次1 接
    bai_lian_base_url: str = _BAI_LIAN_DEFAULT
    # 百炼/通义 API Key（统一在网关侧持有；上游 LLM 调用由网关用此 key 鉴权，
    # 下游服务不应持有 LLM key，只走网关）。百炼与通义共用同一 key，统一经
    # TONGYI_API_KEY env 注入（与下游 shop-agent 的字段名一致）。
    tongyi_api_key: str = ""

    # 生产级路由驱动（01 §4 / 02 §4）：LiteLLM Router 的 model_list YAML 路径。
    # 配了该路径即作为路由权威来源；为空则退化为上方 legacy 字段自动生成 model_list。
    # YAML 结构见 gateway/litellm_config.example.yaml。
    litellm_config_path: str = ""
    # Router 级负载均衡/重试策略（可整体在 YAML 内联；此处为 env 覆盖入口）。
    # 取值：simple-shuffle（默认）| least-busy | latency-based-routing | usage-based-routing | cost-based-routing
    litellm_routing_strategy: str = "simple-shuffle"
    # Router 故障转移重试次数（替代原网关手写 fallback_chain 遍历，02 §4 语义不变）。
    litellm_num_retries: int = 2
    # 单次上游调用超时（秒），兜底防悬挂。
    litellm_timeout_sec: float = 180.0

    # 处置边界（01 §5）
    gateway_fail_mode: str = "closed"  # fail-closed 默认

    # 业务侧指向本网关的地址（仅用于文档/回退说明）
    llm_gateway_url: str = "http://gateway:8001"

    # 02 §4 故障转移触发状态码（限流/宕机信号）
    # 【deprecated】生产环境故障转移由 LiteLLM Router（litellm_num_retries + timeout）原生接管，
    # 本字段仅保留供 fail-closed 校验参考与 legacy 回退路径使用，不再驱动网关手写重试。
    route_fallback_on_status: str = "429,500,502,503,504"

    # 03 成本治理：租户解析 / 限流 / 预算熔断
    tenant_api_keys: str = ""  # 逗号分隔 `tenant:key`，用于 API Key → tenant 映射
    rate_limit_global_rps: float = 1000.0   # 全局令牌桶 refill 速率（请求/秒）
    rate_limit_tenant_rps: float = 10.0    # 单租户令牌桶 refill 速率
    rate_limit_burst: int = 20             # 桶容量（突发）
    budget_tenant_tokens: int = 0          # 单租户 token 预算（0=不熔断）

    # 05 失控循环防护（最小版）
    loop_guard_max: int = 8               # 窗口内同指纹最大命中数（<=0 关闭）
    loop_guard_window_sec: float = 10.0   # 滑动窗口时长（秒）

    # 07 脱敏引擎（批次2/2a + 批次3-07 工程化）
    pii_enabled: bool = True                          # 脱敏总开关（默认开）
    pii_rules_path: str = "gateway/pii/pii.baseline.yaml"  # 随包基线，可被覆盖
    pii_rules_url: str = ""                           # 中央源（本批仅预留，不拉取）
    pii_refresh_interval_sec: float = 60.0           # 背景规则刷新间隔（0=关闭热更新）

    # 10 注入检测闸（批次2/2b）
    injection_rules_path: str = "gateway/rules/injection.baseline.yaml"  # 随包基线
    injection_rules_url: str = ""                      # 中央源（仅预留，不拉取）

    # 06 合规护栏（批次2/2b）：确定性违禁词/话题字典（逗号分隔，业务零改动）
    guardrails_blocklist: str = ""                     # 例："违禁词A,违禁词B"

    # 日志（阶段二 2.2/2.4）：LOG_LEVEL/LOG_FORMAT 与 shop-agent 契约一致
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")

    # 08 语义缓存（批次2/2b）：三层兜底默认值（可被 cache-policy.yml 分片覆盖）
    cache_enabled: bool = True                         # 缓存总开关（默认开）
    cache_ttl: int = 3600                             # 缓存条目 TTL（秒），内置默认
    similarity_threshold: float = 0.85                # 词频向量余弦命中阈值（内置默认）
    cache_policy_path: str = "cache-policy.yml"  # 业务分片策略文件（相对 cache 包目录解析）

    @property
    def fail_mode(self) -> GatewayMode:
        return GatewayMode.from_env(self.gateway_fail_mode)

    @property
    def fallback_status(self) -> set[int]:
        out: set[int] = set()
        for part in self.route_fallback_on_status.split(","):
            part = part.strip()
            if part.isdigit():
                out.add(int(part))
        return out or {429, 500, 502, 503, 504}

    def tenant_of_key(self, api_key: str) -> str | None:
        """按 TENANT_API_KEYS 映射 API Key → tenant；未命中返回 None。"""
        for pair in self.tenant_api_keys.split(","):
            pair = pair.strip()
            if not pair or ":" not in pair:
                continue
            tenant, key = pair.split(":", 1)
            if key.strip() == api_key:
                return tenant.strip()
        return None

    def build_model_list(self) -> list[dict]:
        """legacy 回退：未配 litellm_config_path 时，按现有后端字段生成 LiteLLM model_list。

        映射规则复刻 router._backend_for_raw（01 §4 四类映射）：
          - qwen3-unified / tool_select / param / local/* / models/* → vLLM（本地 GPU）
          - gpt-*  → Azure（主）+ openai（同模型跨厂商备选）
          - claude* → Bedrock（未配回退 openai）
          - qwen*  → 百炼
          - mock*  → mock 上游
          - 其余    → openai
        每个 model_name 在 model_list 中可有多条 deployment（实现 02 §4 故障转移/负载均衡）。
        """
        deployments: list[dict] = []
        _V = _VLLM_MODEL

        def _add(
            model_name: str,
            litellm_model: str,
            api_base: str,
            provider: str,
            api_key: str = "",
        ) -> None:
            # 凭证/地址缺失时跳过该 deployment，避免 Router 构建因缺 api_key 崩溃
            # （如 azure/openai/bedrock 未配置时，LiteLLM 初始化 AsyncAzureOpenAI
            # 会抛 Missing credentials）。真实链路（百炼 qwen / 本地 vLLM）仍保留。
            base_missing = not api_base or str(api_base).startswith("http://_unset_")
            key_required = provider in ("azure", "openai", "bedrock")
            if base_missing or (key_required and not api_key):
                return
            # 显式 custom_llm_provider 避免 LiteLLM 对未知模型名（如 qwen3-unified）做
            # provider 推断失败；裸 model 名 + provider 即可让 Router 按 openai 兼容调用。
            params: dict = {
                "model": litellm_model,
                "custom_llm_provider": provider,
                "api_base": api_base,
            }
            if api_key:
                params["api_key"] = api_key
            deployments.append({"model_name": model_name, "litellm_params": params})

        # 本地统一模型（含本地小模型任务键）：openai 兼容 vLLM，裸模型名 qwen3-unified
        for alias in (_V, "tool_select", "param", "local/*", "models/*"):
            _add(alias, _V, self.vllm_base_url, "openai")
        # mock 上游：openai 兼容 mock 服务
        _add("mock", "mock", self.mock_base_url, "openai")
        _add("mock*", "mock", self.mock_base_url, "openai")
        # gpt-*：Azure 主 + openai 同模型备选
        _add("gpt-*", "gpt-4o", self.azure_openai_base_url, "azure")
        _add("gpt-*", "gpt-4o", self.openai_base_url, "openai")
        # claude*：Bedrock（未配回退 openai）
        _add("claude*", "claude-3", self.bedrock_base_url or self.openai_base_url,
             "bedrock" if self.bedrock_base_url else "openai")
        # qwen*（除 qwen3-unified）：百炼（openai 兼容），api_key 由网关统一持有
        _add("qwen*", "qwen3.7-plus-2026-05-26", self.bai_lian_base_url, "openai",
             api_key=self.tongyi_api_key)
        # 具体大模型名（云端百炼真实模型，避免 qwen* 通配被降级为 qwen3.7-plus-2026-05-26）
        _add("qwen3.7-plus-2026-05-26", "qwen3.7-plus-2026-05-26",
             self.bai_lian_base_url, "openai", api_key=self.tongyi_api_key)
        # 默认兜底
        _add("default", "gpt-4o", self.openai_base_url, "openai")
        return deployments


settings = Settings()


def get_settings() -> Settings:
    return settings
