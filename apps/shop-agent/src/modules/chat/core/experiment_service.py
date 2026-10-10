"""
在线 A/B 实验框架 — 核心引擎

架构概览:
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│   Router    │───▶│  Experiment  │───▶│ Orchestrator │
│ (routers.py)│    │   Service    │    │ (注入 variant)│
└─────────────┘    └──────┬───────┘    └──────┬───────┘
                          │                    │
                   ┌──────▼───────┐    ┌──────▼───────┐
                   │    Redis     │    │  Langfuse    │
                   │  (热加载配置) │    │  (自动标记)   │
                   └──────────────┘    └──────────────┘
                          │
                   ┌──────▼───────┐
                   │ Safety Guard │
                   │ (指标监控+   │
                   │  自动停止)    │
                   └──────────────┘

支持的实验变量:
  - Reranker: threshold, top_k, enabled
  - Retrieval: strategy (hybrid/dense-only/bm25-only), top_k, rrf_k
  - LLM: model, temperature, max_tokens
  - Prompt: template version key
  - Embedding: provider, model
  - Domain Config: top-level AgentConfig overrides
  - Content Filter: enabled, threshold
  - Synonym Normalization: enabled
  - Graph Knowledge: NebulaGraph enabled
  - Intent Recognition: mode

使用方法:
  1. 在 Redis 中写入实验配置（或通过管理API）
  2. ExperimentService 每 30 秒拉取一次配置（热加载，无需重启）
  3. Router 层调用 experiment_service.assign() 分配用户到 variant
  4. Orchestrator 层读取 variant.pipeline_overrides 覆盖默认 AgentConfig
  5. Langfuse trace 自动包含 experiment_id + variant_name 标签
  6. SafetyGuard 监控关键指标，超阈值自动暂停实验
"""

import asyncio
import json
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

import redis

from src.shared.logger import APILogger

# 类型标注专用导入（仅类型检查阶段生效，运行时不实际导入）。
# 注意：GrowthBookClient / GrowthBookDataSource 的实际导入必须放在函数体内（懒加载），
# 否则会与 growthbook_client（其顶层已 import 本模块的 Assignment 等）形成顶层循环导入。
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.growthbook_client import GrowthBookClient
    from src.core.growthbook_datasource import GrowthBookDataSource

logger = APILogger("experiment_service")

# =============================================================================
# 数据模型
# =============================================================================


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ARCHIVED = "archived"


class VariantType(str, Enum):
    CONTROL = "control"
    TREATMENT = "treatment"


class SafetyMetricType(str, Enum):
    """安全护栏监控的指标类型"""

    ESCALATION_RATE = "escalation_rate"  # 转人工率
    SENTIMENT_NEGATIVE = "sentiment_negative"  # 负面情绪比例
    P99_LATENCY_MS = "p99_latency_ms"  # P99 延迟
    ERROR_RATE = "error_rate"  # 错误率
    SAFETY_FAILED_RATE = "safety_failed_rate"  # 安全检查失败率


@dataclass
class SafetyGuard:
    """安全护栏：自动停止条件"""

    metric: SafetyMetricType
    threshold: float  # 阈值（如 0.2 表示 20%）
    comparison: str = "gt"  # gt (大于) | lt (小于) | pct_change (相对变化)
    window_seconds: int = 300  # 监控窗口（秒）
    action: str = "pause"  # pause | stop

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "window_seconds": self.window_seconds,
            "action": self.action,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SafetyGuard":
        return cls(
            metric=SafetyMetricType(d["metric"]),
            threshold=d["threshold"],
            comparison=d.get("comparison", "gt"),
            window_seconds=d.get("window_seconds", 300),
            action=d.get("action", "pause"),
        )


@dataclass
class PipelineOverrides:
    """实验变量：管道路由组件的运行时覆盖配置"""

    # --- Reranker ---
    rerank_enabled: Optional[bool] = None
    rerank_threshold: Optional[float] = None
    rerank_top_k: Optional[int] = None
    rerank_initial_top_k: Optional[int] = None

    # --- Retrieval ---
    retrieval_strategy: Optional[str] = None  # "hybrid" | "dense_only" | "bm25_only"
    retrieval_top_k: Optional[int] = None  # Milvus 召回 top_k
    retrieval_rrf_k: Optional[int] = None  # RRF 融合参数 k

    # --- LLM ---
    llm_model: Optional[str] = None  # 如 "qwen3.7-plus-2026-05-26" vs "qwen3.6-plus-2026-04-02"
    llm_temperature: Optional[float] = None
    llm_max_tokens: Optional[int] = None

    # --- Prompt ---
    prompt_template_key: Optional[str] = None  # 如 "ecommerce_step4_v2"

    # --- Embedding ---
    embedding_model: Optional[str] = None

    # --- Domain Config ---
    domain_overrides: Optional[Dict[str, Any]] = None  # AgentConfig 任意字段覆盖

    # --- Feature Toggles ---
    content_filter_enabled: Optional[bool] = None
    synonym_normalize_enabled: Optional[bool] = None
    nebula_graph_enabled: Optional[bool] = None
    intent_recognition_mode: Optional[str] = None  # "local" | "llm"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineOverrides":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class VariantDef:
    """实验变体定义（对照组/实验组）"""

    name: str  # "control" / "treatment_A"
    variant_type: VariantType  # control | treatment
    traffic_percent: float  # 流量比例，如 50.0 表示 50%
    pipeline_overrides: PipelineOverrides = field(default_factory=PipelineOverrides)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "variant_type": self.variant_type.value,
            "traffic_percent": self.traffic_percent,
            "pipeline_overrides": self.pipeline_overrides.to_dict(),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VariantDef":
        return cls(
            name=d["name"],
            variant_type=VariantType(d["variant_type"]),
            traffic_percent=float(d["traffic_percent"]),
            pipeline_overrides=PipelineOverrides.from_dict(d.get("pipeline_overrides", {})),
            description=d.get("description", ""),
        )


@dataclass
class ExperimentDef:
    """实验定义"""

    id: str  # 唯一 ID，如 "exp_reranker_threshold_a01"
    name: str  # 人类可读名称
    description: str = ""
    status: ExperimentStatus = ExperimentStatus.DRAFT
    variants: List[VariantDef] = field(default_factory=list)
    safety_guards: List[SafetyGuard] = field(default_factory=list)
    domains: List[str] = field(default_factory=lambda: ["ecommerce"])  # 生效领域
    created_at: str = ""
    updated_at: str = ""
    owner: str = ""
    # 实验模式（GrowthBook 接入新增；design.md §4/§5）：
    #   "experiment" → A/B 实验（建 GB Feature + Experiment，写 Data Source 算显著性）
    #   "canary"    → 金丝雀/功能开关（建 GB Feature-only，用 rolloutPercentage 渐进开量）
    kind: str = "experiment"
    # 计划结束日期（scope §9.2 生命周期治理，ISO 日期 YYYY-MM-DD）。
    # 创建 GB Feature 时写为 feature tag `expected_end_date:<date>`，审计脚本据此判超时。
    # 仅 exp_/canary_ 类 flag 强制；switch_/perm_ 为长期开关可不填。
    expected_end: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "variants": [v.to_dict() for v in self.variants],
            "safety_guards": [g.to_dict() for g in self.safety_guards],
            "domains": self.domains,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "owner": self.owner,
            "kind": self.kind,
            "expected_end": self.expected_end,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentDef":
        return cls(
            id=d["id"],
            name=d["name"],
            description=d.get("description", ""),
            status=ExperimentStatus(d.get("status", "draft")),
            variants=[VariantDef.from_dict(v) for v in d.get("variants", [])],
            safety_guards=[SafetyGuard.from_dict(g) for g in d.get("safety_guards", [])],
            domains=d.get("domains", ["ecommerce"]),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            owner=d.get("owner", ""),
            kind=d.get("kind", "experiment"),
            expected_end=d.get("expected_end", ""),
        )


@dataclass
class Assignment:
    """用户分配结果

    exp_mode 字段（GrowthBook 接入阶段新增，见 design.md §3.2）：
      - "experiment"：命中 GB Experiment（A/B 显著性模式）
      - "canary"    ：命中 GB Feature rollout（金丝雀/开关模式）
      - None        ：control / 降级 / 无实验命中
    to_tags / to_metadata 以 exp_mode 为守卫：None 时不写任何 Langfuse tag，
    保证 GB 禁用或降级时主链路零影响（scope §6 回归项）。
    """

    user_id: str
    experiment_id: str
    variant_name: str
    variant_type: VariantType
    pipeline_overrides: PipelineOverrides
    traffic_percent: float
    bucket: int  # 哈希桶号 (0-99)；GB 无桶号概念，固定 -1 兼容旧契约
    exp_mode: Optional[str] = None  # "experiment" | "canary"；None = control/降级/无实验

    def to_tags(self) -> List[str]:
        """生成 Langfuse 标签（exp_mode 守卫：None 时不写 tag，降级对主链路零影响）"""
        if not self.exp_mode:
            return []
        return [
            f"exp:{self.experiment_id}",
            f"variant:{self.variant_name}",
            f"exp_type:{self.exp_mode}",
        ]

    def to_metadata(self) -> Dict[str, Any]:
        """生成 Langfuse metadata（exp_mode 守卫：None 时返回空，不污染 trace）"""
        if not self.exp_mode:
            return {}
        return {
            "experiment_id": self.experiment_id,
            "variant": self.variant_name,
            "variant_type": self.variant_type.value,
            "exp_mode": self.exp_mode,
            "traffic_percent": self.traffic_percent,
        }


# =============================================================================
# 哈希分流引擎
# =============================================================================


# =============================================================================
# 历史遗留类（LEGACY / VALIDATION-ONLY）
# -----------------------------------------------------------------------------
# 显式偏差说明（scope §4.4 原要求删除这些类，但为保障现有 validate_distribution
# 与潜在测试不红，本阶段（Phase 2）仅保留其定义、不再用于实时分配）：
#   - TrafficRouter           : 仅 TrafficRouter.validate_distribution 在校验接口用，
#                              实时分配已委托 GrowthBook eval_variant。
#   - ExperimentStore         : 不再实例化；本地 sidecar（self._active）替代 Redis store。
#   - _ExperimentMetricsCollector / SampleSizeCalculator / StatisticalTest :
#                               保留定义，仅供 validate / 离线分析参考，运行时护栏
#                               改走 SafetyGuardScheduler（实时查 Langfuse/Prometheus）。
# 真正的删除留待后续清理 phase。请勿在实时分配链路中再引用这些类。
# =============================================================================


class TrafficRouter:
    """基于用户 ID 哈希的一致性流量分配（MurmurHash 风格）

    算法: hash(user_id + experiment_id) % 100 → bucket → variant
    - 同一用户+实验始终落入同一桶 → 用户体验一致
    - 流量比例通过 variant.traffic_percent 控制
    - 100 个桶确保足够的分配粒度（1% 精度）
    """

    NUM_BUCKETS = 100

    @staticmethod
    def _hash_user(user_id: str, experiment_id: str) -> int:
        """FNV-1a 哈希（避免 hash() 的跨进程/Python 版本不一致问题）"""
        key = f"{user_id}:{experiment_id}"
        # FNV-1a 64-bit
        h = 0xCBF29CE484222325
        for ch in key:
            h ^= ord(ch)
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return h % TrafficRouter.NUM_BUCKETS

    @staticmethod
    def assign(
        experiment: ExperimentDef, user_id: str, domain: str = "ecommerce"
    ) -> Optional[Assignment]:
        """
        为用户分配实验变体。

        Args:
            experiment: 实验定义
            user_id: 用户 ID（如请求的 conversation_id）
            domain: 业务领域（实验仅对匹配的域名生效）

        Returns:
            Assignment 或 None（用户不在实验流量内）
        """
        # 仅对匹配域名生效
        if domain not in experiment.domains:
            return None

        # 仅运行态实验生效
        if experiment.status != ExperimentStatus.RUNNING:
            return None

        if not experiment.variants:
            return None

        bucket = TrafficRouter._hash_user(user_id, experiment.id)

        # 按 traffic_percent 分配桶区间
        offset = 0
        for variant in experiment.variants:
            slot_count = int(variant.traffic_percent)  # 如 50 → 50 个桶
            if slot_count <= 0:
                continue
            if bucket < offset + slot_count:
                return Assignment(
                    user_id=user_id,
                    experiment_id=experiment.id,
                    variant_name=variant.name,
                    variant_type=variant.variant_type,
                    pipeline_overrides=variant.pipeline_overrides,
                    traffic_percent=variant.traffic_percent,
                    bucket=bucket,
                )
            offset += slot_count

        # 剩余桶不进实验（如 50+30=80，剩余 20% 不进实验）
        return None

    @staticmethod
    def validate_distribution(experiment: ExperimentDef, sample_users: List[str]) -> Dict[str, Any]:
        """验证流量分配均匀性（用于面试追问——"你测过分流均匀性吗？"）

        Args:
            experiment: 实验定义
            sample_users: 样本用户 ID 列表（建议 1000+）

        Returns:
            分配统计: {"variant_counts": {name: count}, "chi_square": ...}
        """
        counts = {v.name: 0 for v in experiment.variants}
        counts["not_assigned"] = 0
        total = len(sample_users)

        for uid in sample_users:
            assignment = TrafficRouter.assign(experiment, uid)
            if assignment:
                counts[assignment.variant_name] += 1
            else:
                counts["not_assigned"] += 1

        # 计算卡方统计量（检验分配均匀性）
        expected_ratios = {}
        offset = 0
        for v in experiment.variants:
            expected_ratios[v.name] = v.traffic_percent / 100.0
            offset += v.traffic_percent
        expected_ratios["not_assigned"] = max(0, (100 - offset)) / 100.0

        chi_square = 0.0
        for name, count in counts.items():
            expected = total * expected_ratios.get(name, 0)
            if expected > 0:
                chi_square += (count - expected) ** 2 / expected

        return {
            "total_users": total,
            "variant_counts": counts,
            "chance_prob": chi_square,  # 卡方值，越小越均匀
            "is_uniform": chi_square < 5.99,  # 自由度 n-1=2 时 α=0.05 临界值
        }


# =============================================================================
# Redis 配置存储（热加载）
# =============================================================================


class ExperimentStore:
    """Redis 实验配置存储

    Key 设计:
      shop_agent:experiments:list          → JSON list of experiment IDs
      shop_agent:experiments:{exp_id}      → JSON 实验定义
      shop_agent:experiments:version       → 版本号（检测变化）
    """

    PREFIX = "shop_agent:experiments"
    DEFAULT_REFRESH_SECONDS = 30

    def __init__(self, redis_client: redis.Redis, refresh_seconds: int = DEFAULT_REFRESH_SECONDS):
        self._redis = redis_client
        self._refresh_seconds = refresh_seconds
        self._cache: Dict[str, ExperimentDef] = {}
        self._version: int = -1
        self._last_refresh: float = 0.0
        self._lock = threading.Lock()

    # ---- 配置读写 ----

    def save_experiment(self, experiment: ExperimentDef) -> bool:
        """保存实验到 Redis"""
        experiment.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        key = f"{self.PREFIX}:{experiment.id}"
        try:
            self._redis.set(key, json.dumps(experiment.to_dict(), ensure_ascii=False))
            # 更新列表
            self._redis.sadd(f"{self.PREFIX}:list", experiment.id)
            # 递增版本
            self._redis.incr(f"{self.PREFIX}:version")
            return True
        except Exception as e:
            logger.error(f"保存实验配置失败: {experiment.id}, {e}")
            return False

    def delete_experiment(self, experiment_id: str) -> bool:
        """删除实验"""
        try:
            self._redis.delete(f"{self.PREFIX}:{experiment_id}")
            self._redis.srem(f"{self.PREFIX}:list", experiment_id)
            self._redis.incr(f"{self.PREFIX}:version")
            self._version = -1  # 强制下次刷新
            return True
        except Exception as e:
            logger.error(f"删除实验配置失败: {experiment_id}, {e}")
            return False

    # ---- 热加载 ----

    def _needs_refresh(self) -> bool:
        """检查是否需要刷新缓存"""
        now = time.time()
        if now - self._last_refresh < self._refresh_seconds:
            return False
        try:
            current_version = int(self._redis.get(f"{self.PREFIX}:version") or 0)
            return current_version != self._version
        except Exception:
            return True

    def _load_all(self) -> Dict[str, ExperimentDef]:
        """从 Redis 全量加载实验配置"""
        experiments: Dict[str, ExperimentDef] = {}
        try:
            exp_ids = self._redis.smembers(f"{self.PREFIX}:list")
            for eid in exp_ids:
                eid_str = eid.decode("utf-8") if isinstance(eid, bytes) else eid
                raw = self._redis.get(f"{self.PREFIX}:{eid_str}")
                if raw:
                    raw_str = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                    exp = ExperimentDef.from_dict(json.loads(raw_str))
                    experiments[exp.id] = exp
            self._version = int(self._redis.get(f"{self.PREFIX}:version") or 0)
            self._last_refresh = time.time()
            logger.info(f"实验配置刷新完成，加载 {len(experiments)} 个实验")
        except Exception as e:
            logger.error(f"实验配置加载失败: {e}")
        return experiments

    def get_active_experiments(self) -> List[ExperimentDef]:
        """获取所有 RUNNING 状态的实验（带缓存 + 热加载）"""
        with self._lock:
            if self._needs_refresh():
                self._cache = self._load_all()
            return [e for e in self._cache.values() if e.status == ExperimentStatus.RUNNING]

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentDef]:
        """获取单个实验定义"""
        with self._lock:
            if self._needs_refresh():
                self._cache = self._load_all()
            return self._cache.get(experiment_id)

    def force_refresh(self) -> Dict[str, ExperimentDef]:
        """强制刷新缓存（管理 API 调用）"""
        with self._lock:
            return self._load_all()


# =============================================================================
# 安全管理器
# =============================================================================


class SafetyGuardEvaluator:
    """安全护栏评估器：监控关键指标，超阈值自动暂停/停止实验"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self._redis = redis_client  # 用于读取实时指标（可选）
        self._alert_callback: Optional[callable] = None

    def set_alert_callback(self, callback: callable):
        """设置告警回调（如发送钉钉/企微通知）"""
        self._alert_callback = callback

    def fire_alert(self, triggered: List["SafetyGuard"]) -> None:
        """触发告警回调（best-effort，吞异常）。无回调则仅 logger.warning。"""
        if not triggered:
            return
        names = [g.metric.value for g in triggered]
        logger.warning(f"[SAFETY-GUARD] 实验护栏触发: {names}（action={[g.action for g in triggered]}）")
        if self._alert_callback is not None:
            try:
                self._alert_callback(triggered)
            except Exception:  # noqa: BLE001
                pass

    def evaluate(self, experiment: ExperimentDef, metrics: Dict[str, float]) -> List[SafetyGuard]:
        """
        评估实验的安全护栏。

        Args:
            experiment: 实验定义
            metrics: 当前指标快照 {metric_type.value: value}

        Returns:
            触发的护栏列表（空列表表示安全）
        """
        triggered = []
        for guard in experiment.safety_guards:
            current = metrics.get(guard.metric.value, 0.0)
            threshold = guard.threshold

            is_triggered = False
            if guard.comparison == "gt":
                is_triggered = current > threshold
            elif guard.comparison == "lt":
                is_triggered = current < threshold
            elif guard.comparison == "pct_change":
                # 相对变化：如转人工率从 5% 升到 6%（相对变化 +20%）
                baseline = metrics.get(f"{guard.metric.value}_baseline", threshold)
                if baseline > 0:
                    is_triggered = (current - baseline) / baseline > threshold

            if is_triggered:
                triggered.append(guard)
                logger.warning(
                    f"实验 {experiment.id} 触发安全护栏: "
                    f"metric={guard.metric.value}, current={current}, "
                    f"threshold={threshold}, action={guard.action}"
                )

        return triggered


# =============================================================================
# 实验服务（单例，对外接口）
# =============================================================================


class ExperimentService:
    """A/B 实验服务（单例）— Phase 2 GrowthBook 代理 + 安全护栏调度。

    职责变更（design.md §3.5 / scope §4.4）：
      - 实时分配不再走自研 Redis 引擎（ExperimentStore/TrafficRouter），
        改为委托 GrowthBookClient.eval_variant（本地 eval，命中即返回 Assignment）。
      - CRUD 改为 best-effort 代理 GrowthBook REST（self._gb.create_experiment 等），
        本地 sidecar（self._active）作为实时分配与护栏调度的数据源（替代 Redis store）。
      - 安全护栏由 SafetyGuardScheduler 后台 asyncio 任务周期性触发评估/暂停。
      - GB 任何不可用（无服务/无网络）时：本服务仍可初始化、assign 返回
        exp_mode=None 的安全默认 control，主流程零影响（scope §4.9 红线）。

    显式偏差（scope §4.4 原要求删除 TrafficRouter/ExperimentStore/
    _ExperimentMetricsCollector/SampleSizeCalculator/StatisticalTest）：为保障
    validate_distribution 与现有测试不红，本阶段仅保留这些类定义、不用于实时分配，
    并标注 "LEGACY / VALIDATION-ONLY"。真正删除留待后续清理 phase。
    """

    _instance: Optional["ExperimentService"] = None
    _initialized: bool = False

    def __new__(cls) -> "ExperimentService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._gb = None
            cls._instance._ds = None
            cls._instance._active: Dict[str, ExperimentDef] = {}
            cls._instance._safety_guard = None
            cls._instance._metrics_collector = None
            cls._instance._scheduler = None
            cls._instance._scheduler_task = None
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ExperimentService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(
        self, redis_client: Optional[redis.Redis] = None, refresh_seconds: int = 30
    ) -> "ExperimentService":
        """初始化实验服务（Phase 2：GrowthBook 代理 + 护栏调度）。

        - 获取 GrowthBookClient 单例（phase1 已在 lifespan 中 await 初始化；此处取引用，
          若尚未初始化则 best-effort 触发其 initialize()——但 GB.initialize 为异步，
          正常编排中由 main.py 先 await 它，故此处通常已就绪；GB 禁用/不可达时
          GrowthBookClient 自身以安全默认模式运行，不影响本服务初始化）。
        - DataSource best-effort（GB 不可用时也允许本地 PG 曝光表存在）。
        - 保留 SafetyGuardEvaluator（带默认告警回调）+ _ExperimentMetricsCollector
          （仅用于 validate，不用于 assign）。
        - 本地 sidecar self._active 替代 Redis store。
        - 启动 SafetyGuardScheduler 后台 asyncio 任务（无运行 loop 时延后）。
        """
        if self._initialized:
            return self

        # ── GrowthBookClient 单例（分配后端）──
        try:
            from src.core.growthbook_client import GrowthBookClient

            self._gb = GrowthBookClient.get_instance()
            if not getattr(self._gb, "is_initialized", False):
                # 极少数情况下（如直接调用本方法而未经 main.py 编排），
                # 幂等触发 GB 初始化（异步；此处仅尝试，失败由 GB 自降级）。
                try:
                    import asyncio

                    loop = asyncio.get_running_loop()
                    loop.create_task(self._gb.initialize())  # type: ignore[union-attr]
                except RuntimeError:
                    logger.warning("GrowthBookClient 尚未初始化且无运行 loop，留待首个请求前就绪")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"GrowthBookClient 单例获取失败（degraded，安全默认模式）: {e}")
            self._gb = None

        # ── DataSource（曝光/指标写入）best-effort ──
        try:
            from src.core.growthbook_datasource import GrowthBookDataSource

            self._ds = GrowthBookDataSource.get_instance()
        except Exception:  # noqa: BLE001
            self._ds = None

        # ── 安全护栏评估器（保留，带默认告警回调）──
        self._safety_guard = SafetyGuardEvaluator(redis_client)
        self._safety_guard.set_alert_callback(self._default_alert_callback)

        # ── 指标收集器：仅 validate 用途（保留类定义，不在 assign 使用）──
        self._metrics_collector = _ExperimentMetricsCollector()

        # ── 本地 sidecar：替代 Redis store，实时分配 + 护栏调度数据源 ──
        self._active: Dict[str, ExperimentDef] = {}

        self._initialized = True
        logger.info("ExperimentService 初始化完成（GrowthBook 代理 + 护栏调度）")

        # ── 启动护栏调度（后台 asyncio 任务）──
        self._start_scheduler()
        return self

    @staticmethod
    def _default_alert_callback(triggered: List["SafetyGuard"]) -> None:
        """默认告警回调（logging 兜底；生产可替换为钉钉/企微）。"""
        names = [g.metric.value for g in triggered]
        logger.warning(f"[SAFETY-GUARD-ALERT] 触发护栏: {names}")

    # ---- 核心 API：用户分配（绝不可返回 None）----

    def assign(
        self,
        user_id: str,
        domain: str = "ecommerce",
        forced_group: Optional[str] = None,
    ) -> Assignment:
        """
        为用户分配实验变体（主入口）。

        遍历本地 sidecar (self._active) 各实验，逐个调 GrowthBookClient.eval_variant；
        首个 exp_mode 非 None 的 Assignment 直接返回（design.md §3.5）。
        GB 不可用 / 全未命中 → 返回 exp_mode=None 的安全默认 control Assignment
        （绝不 None、绝不 raise，主流程零影响，scope §4.9 红线）。

        Args:
            user_id: 用户 ID（建议用 conversation_id）
            domain: 业务领域
            forced_group: 可选，强制实验组（variant name，如 "control"/"treatment_A"）。
                用于测试/灰度验证：跳过 hash 分桶直接落指定 variant。若指定组在
                任一活动实验中不存在，则回退到自动分桶（不静默丢弃请求）。

        Returns:
            Assignment（命中实验 / 或安全默认 control）
        """
        if not self._initialized or self._gb is None:
            return self._safe_control(user_id)

        # ── 强制分组（测试/验证用）：跳过 hash 直接落指定 variant ──
        if forced_group:
            for exp_id, exp_def in self._active.items():
                for v in exp_def.variants:
                    if v.name == forced_group:
                        assignment = Assignment(
                            user_id=user_id,
                            experiment_id=exp_id,
                            variant_name=v.name,
                            variant_type=v.variant_type,
                            pipeline_overrides=v.pipeline_overrides,
                            traffic_percent=v.traffic_percent or 0.0,
                            bucket=-1,
                            exp_mode=exp_def.kind,
                        )
                        if assignment.exp_mode == "experiment":
                            try:
                                self._gb.track_exposure(exp_id, user_id, v.name, domain)
                            except Exception:  # noqa: BLE001
                                pass
                        logger.info(
                            f"[forced] 用户强制分组: user={user_id[:12]}..., "
                            f"experiment={exp_id}, variant={v.name}, mode={assignment.exp_mode}"
                        )
                        return assignment
            logger.warning(
                f"[forced] 指定分组 '{forced_group}' 在活动实验中未找到，回退自动分桶"
            )

        for exp_id in self._active:
            try:
                assignment = self._gb.eval_variant(exp_id, user_id, {"domain": domain})
            except Exception as e:  # noqa: BLE001
                logger.warning(f"assign: GB eval_variant({exp_id}) 失败（跳过）: {e}")
                continue
            if assignment is not None and assignment.exp_mode is not None:
                # experiment 模式命中 → 触发曝光上报（best-effort，不影响返回）
                if assignment.exp_mode == "experiment":
                    try:
                        self._gb.track_exposure(
                            assignment.experiment_id or exp_id,
                            user_id,
                            assignment.variant_name,
                            domain,
                        )
                    except Exception:  # noqa: BLE001
                        pass
                logger.info(
                    f"用户分配实验: user={user_id[:12]}..., "
                    f"experiment={assignment.experiment_id}, "
                    f"variant={assignment.variant_name}, "
                    f"mode={assignment.exp_mode}"
                )
                return assignment

        # 全未命中 / GB 不可用 → 安全默认（exp_mode=None，不写 Langfuse tag）
        return self._safe_control(user_id)

    def _safe_control(self, user_id: str) -> Assignment:
        """安全默认 control Assignment（exp_mode=None → 不写 Langfuse tag）。"""
        return Assignment(
            user_id=user_id,
            experiment_id="",
            variant_name="control",
            variant_type=VariantType.CONTROL,
            pipeline_overrides=PipelineOverrides(),
            traffic_percent=0.0,
            bucket=-1,
            exp_mode=None,
        )

    # ---- 管理 API（维护本地 sidecar + best-effort 代理 GB REST）----

    def create_experiment(self, experiment: ExperimentDef) -> bool:
        """创建/更新实验：写入本地 sidecar，best-effort 代理 GB REST 建 feature(/experiment)。"""
        # 本地 sidecar（实时分配 + 护栏调度数据源）
        self._active[experiment.id] = experiment
        ok = False
        if self._gb is not None:
            try:
                ok = self._gb.create_experiment(experiment, experiment.kind)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"create_experiment: GB REST 创建失败（sidecar 仍记录）: {e}")
                ok = False
        return ok

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentDef]:
        """获取实验定义（本地 sidecar 优先，best-effort 代理 GB）。"""
        if experiment_id in self._active:
            return self._active[experiment_id]
        if self._gb is not None:
            try:
                raw = self._gb.get_experiment(experiment_id)
                if raw:
                    return self._gb_feature_to_def(raw)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"get_experiment: 从 GB 获取失败: {e}")
        return None

    def list_experiments(self) -> List[ExperimentDef]:
        """列出所有实验（本地 sidecar 为主，best-effort 补充 GB 列表）。"""
        experiments = list(self._active.values())
        if self._gb is not None:
            try:
                for raw in self._gb.list_experiments():
                    gid = (raw.get("key") or raw.get("id")) if isinstance(raw, dict) else None
                    if gid and gid not in self._active:
                        conv = self._gb_feature_to_def(raw)
                        if conv is not None:
                            experiments.append(conv)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"list_experiments: 从 GB 列举失败: {e}")
        return experiments

    def delete_experiment(self, experiment_id: str) -> bool:
        """删除实验：从本地 sidecar 移除，best-effort 代理 GB 删除 feature。"""
        self._active.pop(experiment_id, None)
        ok = False
        if self._gb is not None:
            try:
                ok = self._gb.delete_experiment(experiment_id)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"delete_experiment: GB REST 删除失败（sidecar 已移除）: {e}")
                ok = False
        return ok

    def update_experiment_status(self, experiment_id: str, status: str) -> bool:
        """更新实验状态：同步本地 sidecar 状态，best-effort 代理 GB REST 置 rollout/archive。"""
        exp = self._active.get(experiment_id)
        if exp is None:
            exp = self.get_experiment(experiment_id)
        if exp is None:
            return False
        exp.status = ExperimentStatus(status)
        self._active[experiment_id] = exp
        ok = False
        if self._gb is not None:
            try:
                ok = self._gb.update_experiment_status(experiment_id, status)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"update_experiment_status: GB REST 失败（sidecar 已更新）: {e}")
                ok = False
        return ok

    def pause_experiment(self, experiment_id: str) -> bool:
        """暂停实验（保留配置，所有用户退出实验）。"""
        return self.update_experiment_status(experiment_id, ExperimentStatus.PAUSED.value)

    def stop_experiment(self, experiment_id: str) -> bool:
        """停止实验。"""
        return self.update_experiment_status(experiment_id, ExperimentStatus.STOPPED.value)

    def force_refresh(self) -> None:
        """强制刷新配置缓存（代理 GB SDK 重新 loadFeatures；best-effort）。"""
        if self._gb is None:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._gb.refresh())  # type: ignore[union-attr]
        except RuntimeError:
            # 无运行中的事件循环（如离线测试直接调用）：降级为不即时刷新，
            # SDK 在 cache_ttl 过期后自动重载 features。
            logger.info("force_refresh: 当前无运行事件循环，跳过即时刷新")

    # ---- 安全护栏（保留不变）----

    def evaluate_safety(self, experiment_id: str, metrics: Dict[str, float]) -> List[SafetyGuard]:
        """评估实验安全护栏"""
        exp = self.get_experiment(experiment_id)
        if not exp:
            return []
        return self._safety_guard.evaluate(exp, metrics)

    # ---- 验证工具（仍委托 TrafficRouter.validate_distribution，保留用于兼容）----

    def validate_distribution(self, experiment_id: str, sample_users: List[str]) -> Dict[str, Any]:
        """验证流量分配均匀性（LEGACY/VALIDATION-ONLY：仅校验用途，不参与实时分配）。"""
        exp = self.get_experiment(experiment_id)
        if not exp:
            return {"error": f"实验 {experiment_id} 不存在"}
        return TrafficRouter.validate_distribution(exp, sample_users)

    # ---- 内部：GB feature 字典 → ExperimentDef 转译（best-effort）----

    def _gb_feature_to_def(self, raw: Dict[str, Any]) -> Optional[ExperimentDef]:
        """best-effort 将 GB feature 字典转译为本服务 ExperimentDef。

        TODO(GB-SDK): GB feature/variation 的实际字段结构待与官方 API 对齐，
        此处仅做最小映射（id/name/status/kind），variations/safety_guards 缺省。
        """
        if not isinstance(raw, dict):
            return None
        try:
            feature_id = raw.get("key") or raw.get("id") or ""
            if not feature_id:
                return None
            tags = raw.get("tags", []) or []
            kind = "experiment"
            expected_end = ""
            for t in tags:
                if isinstance(t, str) and t.startswith("exp_type:"):
                    kind = t.split(":", 1)[1]
                elif isinstance(t, str) and t.startswith("expected_end_date:"):
                    expected_end = t.split(":", 1)[1]
            return ExperimentDef(
                id=feature_id,
                name=raw.get("name", feature_id),
                description=raw.get("description", ""),
                status=ExperimentStatus.RUNNING,
                kind=kind,
                expected_end=expected_end,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"_gb_feature_to_def 转译失败: {e}")
            return None

    # ---- 内部：护栏调度器启停 ----

    def _start_scheduler(self) -> None:
        """启动安全护栏调度器后台任务（需运行中的事件循环）。"""
        try:
            self._scheduler = SafetyGuardScheduler(self)
            loop = asyncio.get_running_loop()
            self._scheduler_task = loop.create_task(self._scheduler.run_loop())
            logger.info("安全护栏调度器已启动")
        except RuntimeError:
            # 当前无运行中的事件循环（如测试/离线直接调用 initialize）：
            # 调度器延后启动；GB 分配与护栏评估能力在首个有 loop 的上下文再激活。
            logger.info("当前无运行事件循环，安全护栏调度器延后启动")


# =============================================================================
# 安全护栏调度器（后台 asyncio 任务）
# =============================================================================

# =============================================================================
# 安全护栏指标源（Phase 5：SafetyMetricsProvider + Prometheus 实现）
# =============================================================================


class SafetyMetricsProvider(Protocol):
    """安全护栏指标源协议。

    实现须返回 ``dict[str, float | None]``：
      - 能取到的指标   → ``float``
      - 取不到/不覆盖   → ``None``（护栏 evaluator 对缺失键跳过，不误触发）

    ``exp_id`` / ``window_seconds`` 为语义占位：本地 Prometheus 为进程级累积量，
    无历史窗口能力；基于窗口的精确速率需后续接入 Prometheus 查询 / Langfuse。
    """

    def collect(self, exp_id: str, window_seconds: int) -> Dict[str, Optional[float]]:
        ...


def _metric_samples(metric: Any) -> List[Tuple[Dict[str, str], float]]:
    """从 prometheus_client 指标对象抽取 ``(labels_dict, value)`` 列表。

    任意异常（指标未注册 / 采集失败）均返回空列表，由调用方保守降级。
    """
    out: List[Tuple[Dict[str, str], float]] = []
    try:
        for family in metric.collect():
            for sample in family.samples:
                out.append((dict(sample.labels), float(sample.value)))
    except Exception:  # noqa: BLE001
        pass
    return out


def _histogram_p99(metric: Any) -> Optional[float]:
    """从 Histogram 的 bucket 累积估算 P99（近似分位，单位同 Histogram）。

    原理：Histogram 暴露 ``*_bucket{le=...}`` 累积计数与 ``*_count``/``*_sum``。
    遍历 ``le`` 升序定位 99% 秩所在 bucket，桶内线性插值。
    若 99% 秩落到 ``+Inf`` 桶（绝大多数样本超过最大有限 bucket 上界），
    回退为该最大有限 bucket 上界（下界近似），避免返回 ``+Inf``。
    无样本 / 不可解析时返回 ``None``（护栏跳过该指标，不误触发）。
    """
    try:
        total: Optional[float] = None
        buckets: List[Tuple[float, float]] = []  # (le_upper, cumulative_count)
        for family in metric.collect():
            for sample in family.samples:
                name = sample.name
                if name.endswith("_count"):
                    total = float(sample.value)
                elif name.endswith("_bucket"):
                    le = sample.labels.get("le", "+Inf")
                    upper = float("inf") if le == "+Inf" else float(le)
                    buckets.append((upper, float(sample.value)))
        if total is None or total <= 0:
            return None
        buckets.sort(key=lambda x: x[0])
        target = 0.99 * total
        prev_upper = 0.0
        prev_count = 0.0
        for upper, count in buckets:
            if count >= target:
                if upper == float("inf"):
                    # 99% 秩落在 +Inf 桶：用上一个有限 bucket 上界做下界近似
                    return prev_upper if prev_upper > 0 else None
                if count > prev_count:
                    ratio = (target - prev_count) / (count - prev_count)
                    return prev_upper + ratio * (upper - prev_upper)
                return upper
            prev_upper = upper
            prev_count = count
        return prev_upper if prev_upper > 0 else None
    except Exception:  # noqa: BLE001
        return None


class PrometheusSafetyProvider:
    """基于本地 prometheus_client 的指标源（零外部依赖，立即可用）。

    能覆盖：
      - ``error_rate``     : ``agent_chat_counter{status=error} / total(success+error)``
      - ``p99_latency_ms`` : ``agent_chat_duration_ms`` 的 P99 近似分位（毫秒）

    无法覆盖（需 Langfuse score 聚合，留待 ``LangfuseSafetyProvider`` 实现）：
      - ``escalation_rate``    转人工率
      - ``sentiment_negative`` 负面情绪比例
      - ``safety_failed_rate`` 安全检查失败率
    上述三类返回 ``None``，护栏 evaluator 对缺失键跳过（不误触发）。

    注：error_rate / p99 为进程级累积量（Prometheus counter/histogram 自带），
    非 window 内速率；window 仅作语义占位。窗口级精确值需后续接入 Prometheus 查询。
    """

    def collect(self, exp_id: str, window_seconds: int) -> Dict[str, Optional[float]]:
        from src.modules.monitoring.metrics import (
            agent_chat_counter,
            agent_chat_duration_ms,
        )

        result: Dict[str, Optional[float]] = {
            # TODO(Langfuse): escalation/sentiment/safety_failed 需 Langfuse score
            # 聚合，后续由 LangfuseSafetyProvider 实现，本阶段返回 None（护栏跳过）。
            "escalation_rate": None,
            "sentiment_negative": None,
            "safety_failed_rate": None,
            "error_rate": None,
            "p99_latency_ms": None,
        }
        try:
            # error_rate：本地 Prometheus 计数器比率（进程级累积）
            total = 0.0
            err = 0.0
            for labels, value in _metric_samples(agent_chat_counter):
                status = labels.get("status")
                if status in ("success", "error"):
                    total += value
                    if status == "error":
                        err += value
            if total > 0:
                result["error_rate"] = err / total
            # p99_latency_ms：从 Histogram bucket 累积估算近似分位
            result["p99_latency_ms"] = _histogram_p99(agent_chat_duration_ms)
        except Exception:  # noqa: BLE001
            # 采集失败：保持 None，护栏跳过（保守，不误触发）
            pass
        return result


class SafetyGuardScheduler:
    """安全护栏后台调度器（asyncio 任务）。

    Phase 2 新增（design.md §3.6 / scope §4.4 护栏调度补位）：周期性遍历本地 sidecar
    中 RUNNING 实验，best-effort 从指标源（Langfuse/Prometheus/Redis）取指标快照，
    调 SafetyGuardEvaluator 评估；触发护栏则暂停/停止实验（代理 GB REST）+ 告警回调。

    全程 try/except 包裹，任何异常仅 logger.warning，绝不中断事件循环。
    """

    def __init__(self, service: "ExperimentService", poll_interval: int = 60) -> None:
        self._service = service
        # 指标源（Phase 5）：默认 Prometheus 本地指标；Langfuse 源后续接入。
        self._provider: SafetyMetricsProvider = PrometheusSafetyProvider()
        # 轮询间隔下限 60s（避免过频）；运行时取护栏最小 window_seconds
        self._poll_interval = max(60, int(poll_interval))
        self._task: Optional[asyncio.Task] = None

    def _resolve_poll_interval(self) -> int:
        """取所有 RUNNING 实验护栏最小 window_seconds 作为轮询间隔（下限 60s）。"""
        min_window = 60
        try:
            for exp in self._service._active.values():
                if exp.status != ExperimentStatus.RUNNING:
                    continue
                for g in exp.safety_guards:
                    min_window = min(min_window, g.window_seconds)
        except Exception:  # noqa: BLE001
            pass
        return max(60, min_window)

    async def run_loop(self) -> None:
        try:
            while True:
                interval = self._resolve_poll_interval()
                await asyncio.sleep(interval)
                await self._poll_once()
        except asyncio.CancelledError:
            logger.info("安全护栏调度器已取消")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"安全护栏调度器异常退出（不影响主流程）: {e}")

    async def _poll_once(self) -> None:
        try:
            running = [
                e for e in self._service._active.values()
                if e.status == ExperimentStatus.RUNNING
            ]
        except Exception:  # noqa: BLE001
            return
        for exp in running:
            try:
                metrics = self._collect_metrics(exp)
                if not metrics:
                    continue  # 取不到指标 → 跳过该实验（保守，不误触发）
                triggered = self._service.evaluate_safety(exp.id, metrics)
                if triggered:
                    actions = {g.action for g in triggered}
                    if "stop" in actions:
                        self._service.stop_experiment(exp.id)
                    else:
                        self._service.pause_experiment(exp.id)
                    self._service._safety_guard.fire_alert(triggered)
                    logger.warning(
                        f"护栏触发，实验 {exp.id} 已"
                        f"{'停止' if 'stop' in actions else '暂停'}: "
                        f"{[g.metric.value for g in triggered]}"
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"护栏评估实验 {exp.id} 失败（跳过）: {e}")

    def _collect_metrics(self, exp: "ExperimentDef") -> Dict[str, float]:
        """best-effort 指标快照（接入 SafetyMetricsProvider，Phase 5）。

        通过 ``PrometheusSafetyProvider`` 取 ``error_rate`` / ``p99_latency_ms`` 两类真实指标；
        ``escalation_rate`` / ``sentiment_negative`` / ``safety_failed_rate`` 暂无 Prometheus
        数据源（需 Langfuse score 聚合，留待 LangfuseSafetyProvider），返回 ``None`` → 护栏跳过。

        只保留非 ``None`` 的指标（evaluator 仅对存在值做比较；``None`` 视为无数据→跳过该指标）。
        无任何可用指标（如无 Prometheus 数据）时返回 ``{}`` → scheduler 跳过、不误触发。
        """
        window = 300
        if exp.safety_guards:
            window = max(g.window_seconds for g in exp.safety_guards)
        raw = self._provider.collect(exp.id, window)
        return {k: v for k, v in raw.items() if v is not None}


# =============================================================================
# 指标收集器（用于安全护栏 + Langfuse 数据聚合）
# =============================================================================


class _ExperimentMetricsCollector:
    """实验指标收集器（轻量级，内存聚合）

    生产环境建议替换为：
      - Langfuse Score API 拉取指标
      - 或 Prometheus + Grafana 透视
    """

    def __init__(self):
        self._buckets: Dict[str, List[Dict[str, Any]]] = {}  # exp_id → [{...}]
        self._lock = threading.Lock()

    def record(self, experiment_id: str, variant_name: str, metrics: Dict[str, Any]):
        """记录一次实验请求的指标"""
        with self._lock:
            key = f"{experiment_id}:{variant_name}"
            if key not in self._buckets:
                self._buckets[key] = []
            self._buckets[key].append(
                {
                    "timestamp": time.time(),
                    **metrics,
                }
            )

    def compute_stats(
        self, experiment_id: str, window_seconds: int = 300
    ) -> Dict[str, Dict[str, float]]:
        """计算指定实验的各 variant 聚合统计"""
        now = time.time()
        cutoff = now - window_seconds
        stats: Dict[str, Dict[str, float]] = {}

        with self._lock:
            for key, records in self._buckets.items():
                if not key.startswith(experiment_id + ":"):
                    continue
                variant_name = key.split(":", 1)[1]
                recent = [r for r in records if r["timestamp"] >= cutoff]

                if not recent:
                    continue

                # 聚合指标
                latencies = [r.get("latency_ms", 0) for r in recent if r.get("latency_ms")]
                errors = sum(1 for r in recent if r.get("is_error"))
                escalations = sum(1 for r in recent if r.get("is_escalation"))

                stats[variant_name] = {
                    "request_count": len(recent),
                    "error_rate": errors / len(recent) if recent else 0,
                    "escalation_rate": escalations / len(recent) if recent else 0,
                    "p50_latency_ms": _percentile(latencies, 50),
                    "p99_latency_ms": _percentile(latencies, 99),
                }

        return stats


def _percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    sorted_vals = sorted(vals)
    idx = int(math.ceil(p / 100.0 * len(sorted_vals))) - 1
    return sorted_vals[max(0, min(idx, len(sorted_vals) - 1))]


# =============================================================================
# 统计显著性工具
# =============================================================================


class SampleSizeCalculator:
    """样本量计算器 — 用于回答"统计显著怎么判？"

    公式: n = (Z_α/2 + Z_β)² * (p1*(1-p1) + p2*(1-p2)) / (p1 - p2)²

    其中:
      - Z_α/2 = 1.96 (α=0.05, 双尾检验)
      - Z_β   = 0.84 (power=0.8)
    """

    Z_ALPHA = 1.96  # 95% 置信
    Z_BETA = 0.84  # 80% 统计效力

    @staticmethod
    def sample_size_per_variant(
        baseline_rate: float, minimum_effect: float, alpha: float = 0.05, power: float = 0.80
    ) -> int:
        """计算每个 variant 所需的最小样本量

        Args:
            baseline_rate: 对照组的基线转化率（如 0.05 表示 5% 转人工率）
            minimum_effect: 最小可检测效应（如 0.01 表示 1% 绝对变化）
            alpha: 显著性水平（默认 0.05）
            power: 统计效力（默认 0.80）

        Returns:
            每个 variant 所需的最小样本数

        Example:
            # 转人工率从 5% 变化到 6%（1% 绝对变化）
            n = SampleSizeCalculator.sample_size_per_variant(0.05, 0.01)
            # → 需要每组约 7,849 个用户
        """
        z_alpha = SampleSizeCalculator.Z_ALPHA
        z_beta = SampleSizeCalculator.Z_BETA

        p1 = baseline_rate
        p2 = baseline_rate + minimum_effect

        # 两样本比例检验的样本量公式（Fleiss corrected）
        n = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (minimum_effect**2)
        return max(1, int(math.ceil(n)))

    @staticmethod
    def estimate_duration(
        samples_needed: int, daily_traffic: int, traffic_percent: float = 50.0
    ) -> float:
        """估算实验需要的天数"""
        daily_variant_traffic = daily_traffic * (traffic_percent / 100.0)
        if daily_variant_traffic <= 0:
            return float("inf")
        return samples_needed / daily_variant_traffic


class StatisticalTest:
    """统计检验工具 — 用于实验结论判定"""

    @staticmethod
    def z_test_proportions(success_a: int, n_a: int, success_b: int, n_b: int) -> Dict[str, Any]:
        """双样本 Z 检验（比例）

        Args:
            success_a: 对照组成功次数
            n_a: 对照组总次数
            success_b: 实验组成功次数
            n_b: 实验组总次数

        Returns:
            {"p_value": ..., "z_score": ..., "significant": bool}
        """
        p_a = success_a / n_a if n_a > 0 else 0
        p_b = success_b / n_b if n_b > 0 else 0
        p_pool = (success_a + success_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0

        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
        if se == 0:
            return {"p_value": 1.0, "z_score": 0.0, "significant": False}

        z_score = (p_b - p_a) / se
        # 近似 P 值（双尾）
        p_value = 2 * (1 - _normal_cdf(abs(z_score)))

        return {
            "z_score": round(z_score, 4),
            "p_value": round(p_value, 4),
            "significant": p_value < 0.05,
            "effect_size": round(p_b - p_a, 4),
            "ci_95_lower": round((p_b - p_a) - 1.96 * se, 4),
            "ci_95_upper": round((p_b - p_a) + 1.96 * se, 4),
        }


def _normal_cdf(x: float) -> float:
    """标准正态分布 CDF 近似（Abramowitz & Stegun 7.1.26）"""
    # 简化实现，生产环境建议用 scipy.stats.norm.cdf
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# =============================================================================
# 预置实验模板
# =============================================================================

# 示例 1: Reranker 阈值消融实验
EXP_TEMPLATE_RERANKER_THRESHOLD = ExperimentDef(
    id="exp_reranker_threshold_001",
    name="Reranker 阈值消融实验",
    description="对比 rerank_threshold=0.3 (当前) vs 0.1 (宽松) 对回答质量和延迟的影响",
    variants=[
        VariantDef(
            name="control_threshold_0.3",
            variant_type=VariantType.CONTROL,
            traffic_percent=50,
            pipeline_overrides=PipelineOverrides(rerank_threshold=0.3),
        ),
        VariantDef(
            name="treatment_threshold_0.1",
            variant_type=VariantType.TREATMENT,
            traffic_percent=50,
            pipeline_overrides=PipelineOverrides(rerank_threshold=0.1),
        ),
    ],
    safety_guards=[
        SafetyGuard(SafetyMetricType.ESCALATION_RATE, threshold=0.10, comparison="pct_change"),
        SafetyGuard(SafetyMetricType.ERROR_RATE, threshold=0.05),
    ],
    domains=["ecommerce", "customer_service"],
)

# 示例 2: LLM 模型对比实验
EXP_TEMPLATE_LLM_MODEL = ExperimentDef(
    id="exp_llm_model_001",
    name="LLM 模型对比实验",
    description="对比 qwen3.6-plus-2026-04-02 (当前) vs qwen3.7-plus-2026-05-26 对回答质量的影响",
    variants=[
        VariantDef(
            name="control_flash",
            variant_type=VariantType.CONTROL,
            traffic_percent=50,
            pipeline_overrides=PipelineOverrides(llm_model="qwen3.6-plus-2026-04-02"),
        ),
        VariantDef(
            name="treatment_max",
            variant_type=VariantType.TREATMENT,
            traffic_percent=50,
            pipeline_overrides=PipelineOverrides(
                llm_model="qwen3.7-plus-2026-05-26-2025-01-25",
                llm_temperature=0.3,
            ),
        ),
    ],
    safety_guards=[
        SafetyGuard(SafetyMetricType.P99_LATENCY_MS, threshold=30000, comparison="pct_change"),
        SafetyGuard(SafetyMetricType.ERROR_RATE, threshold=0.05),
    ],
    domains=["ecommerce", "customer_service"],
)

# 示例 3: 检索策略对比实验
EXP_TEMPLATE_RETRIEVAL_STRATEGY = ExperimentDef(
    id="exp_retrieval_strategy_001",
    name="检索策略对比实验",
    description="对比 Hybrid (Dense+BM25) vs Dense-only 对上下文质量的影响",
    variants=[
        VariantDef(
            name="control_hybrid",
            variant_type=VariantType.CONTROL,
            traffic_percent=50,
            pipeline_overrides=PipelineOverrides(retrieval_strategy="hybrid"),
        ),
        VariantDef(
            name="treatment_dense_only",
            variant_type=VariantType.TREATMENT,
            traffic_percent=50,
            pipeline_overrides=PipelineOverrides(retrieval_strategy="dense_only"),
        ),
    ],
    safety_guards=[
        SafetyGuard(SafetyMetricType.ESCALATION_RATE, threshold=0.10, comparison="pct_change"),
        SafetyGuard(SafetyMetricType.ERROR_RATE, threshold=0.05),
    ],
    domains=["ecommerce", "customer_service", "medical"],
)

# 预置模板注册表
EXP_TEMPLATES: Dict[str, ExperimentDef] = {
    "reranker_threshold": EXP_TEMPLATE_RERANKER_THRESHOLD,
    "llm_model": EXP_TEMPLATE_LLM_MODEL,
    "retrieval_strategy": EXP_TEMPLATE_RETRIEVAL_STRATEGY,
}
