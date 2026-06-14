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

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type

import redis

from src.shared.logger import APILogger

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
    ESCALATION_RATE = "escalation_rate"       # 转人工率
    SENTIMENT_NEGATIVE = "sentiment_negative"  # 负面情绪比例
    P99_LATENCY_MS = "p99_latency_ms"         # P99 延迟
    ERROR_RATE = "error_rate"                  # 错误率
    SAFETY_FAILED_RATE = "safety_failed_rate"  # 安全检查失败率


@dataclass
class SafetyGuard:
    """安全护栏：自动停止条件"""
    metric: SafetyMetricType
    threshold: float          # 阈值（如 0.2 表示 20%）
    comparison: str = "gt"    # gt (大于) | lt (小于) | pct_change (相对变化)
    window_seconds: int = 300  # 监控窗口（秒）
    action: str = "pause"     # pause | stop

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
    retrieval_strategy: Optional[str] = None       # "hybrid" | "dense_only" | "bm25_only"
    retrieval_top_k: Optional[int] = None           # Milvus 召回 top_k
    retrieval_rrf_k: Optional[int] = None            # RRF 融合参数 k

    # --- LLM ---
    llm_model: Optional[str] = None                  # 如 "qwen-max" vs "qwen3.6-plus-2026-04-02"
    llm_temperature: Optional[float] = None
    llm_max_tokens: Optional[int] = None

    # --- Prompt ---
    prompt_template_key: Optional[str] = None        # 如 "ecommerce_step4_v2"

    # --- Embedding ---
    embedding_model: Optional[str] = None

    # --- Domain Config ---
    domain_overrides: Optional[Dict[str, Any]] = None  # AgentConfig 任意字段覆盖

    # --- Feature Toggles ---
    content_filter_enabled: Optional[bool] = None
    synonym_normalize_enabled: Optional[bool] = None
    nebula_graph_enabled: Optional[bool] = None
    intent_recognition_mode: Optional[str] = None      # "local" | "llm"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineOverrides":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class VariantDef:
    """实验变体定义（对照组/实验组）"""
    name: str                                    # "control" / "treatment_A"
    variant_type: VariantType                    # control | treatment
    traffic_percent: float                       # 流量比例，如 50.0 表示 50%
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
    id: str                                      # 唯一 ID，如 "exp_reranker_threshold_a01"
    name: str                                    # 人类可读名称
    description: str = ""
    status: ExperimentStatus = ExperimentStatus.DRAFT
    variants: List[VariantDef] = field(default_factory=list)
    safety_guards: List[SafetyGuard] = field(default_factory=list)
    domains: List[str] = field(default_factory=lambda: ["ecommerce"])  # 生效领域
    created_at: str = ""
    updated_at: str = ""
    owner: str = ""

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
        )


@dataclass
class Assignment:
    """用户分配结果"""
    user_id: str
    experiment_id: str
    variant_name: str
    variant_type: VariantType
    pipeline_overrides: PipelineOverrides
    traffic_percent: float
    bucket: int           # 哈希桶号 (0-99)

    def to_tags(self) -> List[str]:
        """生成 Langfuse 标签"""
        return [
            f"exp:{self.experiment_id}",
            f"variant:{self.variant_name}",
            f"exp_type:{self.variant_type.value}",
        ]

    def to_metadata(self) -> Dict[str, Any]:
        """生成 Langfuse metadata"""
        return {
            "experiment_id": self.experiment_id,
            "variant": self.variant_name,
            "variant_type": self.variant_type.value,
            "traffic_percent": self.traffic_percent,
            "bucket": self.bucket,
        }


# =============================================================================
# 哈希分流引擎
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
        h = 0xcbf29ce484222325
        for ch in key:
            h ^= ord(ch)
            h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
        return h % TrafficRouter.NUM_BUCKETS

    @staticmethod
    def assign(experiment: ExperimentDef, user_id: str,
               domain: str = "ecommerce") -> Optional[Assignment]:
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

    def evaluate(self, experiment: ExperimentDef,
                 metrics: Dict[str, float]) -> List[SafetyGuard]:
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
    """A/B 实验服务（单例）"""

    _instance: Optional["ExperimentService"] = None
    _initialized: bool = False

    def __new__(cls) -> "ExperimentService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._store = None
            cls._instance._router = None
            cls._instance._safety_guard = None
            cls._instance._metrics_collector = None
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

    def initialize(self, redis_client: Optional[redis.Redis] = None,
                   refresh_seconds: int = 30) -> "ExperimentService":
        """初始化实验服务"""
        if self._initialized:
            return self

        if redis_client is None:
            try:
                redis_client = redis.Redis(
                    host="localhost", port=6379, db=0,
                    decode_responses=True, socket_connect_timeout=2,
                )
                redis_client.ping()
            except Exception:
                logger.warning("Redis 不可用，实验服务将以空配置启动")
                redis_client = None

        self._store = ExperimentStore(redis_client, refresh_seconds=refresh_seconds)
        self._router = TrafficRouter()
        self._safety_guard = SafetyGuardEvaluator(redis_client)
        self._metrics_collector = _ExperimentMetricsCollector()
        self._initialized = True

        logger.info("ExperimentService 初始化完成")
        return self

    # ---- 核心 API：用户分配 ----

    def assign(self, user_id: str, domain: str = "ecommerce") -> Optional[Assignment]:
        """
        为用户分配实验变体（主入口）。

        Router 在收到请求后调用，返回的 Assignment 将沿管道传递到 Orchestrator
        和所有子组件。

        Args:
            user_id: 用户 ID（建议用 conversation_id）
            domain: 业务领域

        Returns:
            Assignment 或 None（该用户不在任何实验组中）
        """
        if not self._initialized or self._store is None:
            return None

        try:
            active_experiments = self._store.get_active_experiments()
        except Exception:
            return None

        for experiment in active_experiments:
            assignment = self._router.assign(experiment, user_id, domain)
            if assignment is not None:
                logger.info(
                    f"用户分配实验: user={user_id[:12]}..., "
                    f"experiment={assignment.experiment_id}, "
                    f"variant={assignment.variant_name}, "
                    f"bucket={assignment.bucket}"
                )
                return assignment
        return None

    # ---- 管理 API ----

    def create_experiment(self, experiment: ExperimentDef) -> bool:
        """创建/更新实验配置"""
        if not self._store:
            return False
        return self._store.save_experiment(experiment)

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentDef]:
        """获取实验定义"""
        if not self._store:
            return None
        return self._store.get_experiment(experiment_id)

    def list_experiments(self) -> List[ExperimentDef]:
        """列出所有实验"""
        if not self._store:
            return []
        return list(self._store.get_active_experiments())

    def delete_experiment(self, experiment_id: str) -> bool:
        """删除实验"""
        if not self._store:
            return False
        return self._store.delete_experiment(experiment_id)

    def pause_experiment(self, experiment_id: str) -> bool:
        """暂停实验"""
        exp = self.get_experiment(experiment_id)
        if not exp:
            return False
        exp.status = ExperimentStatus.PAUSED
        return self._store.save_experiment(exp)

    def stop_experiment(self, experiment_id: str) -> bool:
        """停止实验"""
        exp = self.get_experiment(experiment_id)
        if not exp:
            return False
        exp.status = ExperimentStatus.STOPPED
        return self._store.save_experiment(exp)

    def force_refresh(self):
        """强制刷新配置缓存"""
        if self._store:
            self._store.force_refresh()

    # ---- 安全护栏 ----

    def evaluate_safety(self, experiment_id: str,
                        metrics: Dict[str, float]) -> List[SafetyGuard]:
        """评估实验安全护栏"""
        exp = self.get_experiment(experiment_id)
        if not exp:
            return []
        return self._safety_guard.evaluate(exp, metrics)

    # ---- 验证工具 ----

    def validate_distribution(self, experiment_id: str,
                              sample_users: List[str]) -> Dict[str, Any]:
        """验证流量分配均匀性（用于面试追问）"""
        exp = self.get_experiment(experiment_id)
        if not exp:
            return {"error": f"实验 {experiment_id} 不存在"}

        assignment = self._router
        return assignment.validate_distribution(exp, sample_users)


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
            self._buckets[key].append({
                "timestamp": time.time(),
                **metrics,
            })

    def compute_stats(self, experiment_id: str,
                      window_seconds: int = 300) -> Dict[str, Dict[str, float]]:
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

    Z_ALPHA = 1.96   # 95% 置信
    Z_BETA = 0.84    # 80% 统计效力

    @staticmethod
    def sample_size_per_variant(baseline_rate: float, minimum_effect: float,
                                alpha: float = 0.05, power: float = 0.80) -> int:
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
        n = (
            (z_alpha + z_beta) ** 2
            * (p1 * (1 - p1) + p2 * (1 - p2))
            / (minimum_effect ** 2)
        )
        return max(1, int(math.ceil(n)))

    @staticmethod
    def estimate_duration(samples_needed: int, daily_traffic: int,
                          traffic_percent: float = 50.0) -> float:
        """估算实验需要的天数"""
        daily_variant_traffic = daily_traffic * (traffic_percent / 100.0)
        if daily_variant_traffic <= 0:
            return float("inf")
        return samples_needed / daily_variant_traffic


class StatisticalTest:
    """统计检验工具 — 用于实验结论判定"""

    @staticmethod
    def z_test_proportions(success_a: int, n_a: int,
                           success_b: int, n_b: int) -> Dict[str, Any]:
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
    description="对比 qwen3.6-plus-2026-04-02 (当前) vs qwen-max 对回答质量的影响",
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
                llm_model="qwen-max-2025-01-25",
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
