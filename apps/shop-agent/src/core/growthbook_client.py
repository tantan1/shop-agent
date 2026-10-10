"""
GrowthBook 客户端单例封装（shop-agent 进程内分配后端）

职责（design.md §3.1 / scope §4.3 / §4.9）：
  - 持有 growthbook SDK 单例（GrowthBookClient v3.2.0），本地内存 eval（无每次请求网络往返）。
  - 启动门禁：阻塞 loadFeatures（超时 + 指数退避）；GB 不可达时 stale 快照上岗 / 安全默认上岗，
    绝不抛异常导致进程起不来。
  - eval_variant：本地 eval；失败/未命中 → 安全默认 control Assignment（exp_mode=None），
    绝不返回 None、绝不抛异常。
  - track_exposure：触发曝光上报 → Data Source 写入（best-effort 吞异常）。
  - 缓存持久化：v3 SDK 已移除旧版 cacheConnection/RedisConnection；本模块自实现「层1 快照」
    （Redis 字符串 + 本地文件回退），供重启 stale 上岗。
  - 熔断 + 陈旧告警：连续失败达阈值 circuit open，用缓存/安全默认；health() 暴露 degraded/
    circuit_open/cache_age。

关键质量属性（scope §4.9）：可用性 > 一致性；韧性 > 实时性；数据不出内网。
GB 禁用或不可达时，主流程零影响（返回安全默认 control）。

SDK 版本对齐（§13 缺口①②就地确认）：
  - 已安装 growthbook==3.2.0，API 与 design.md 伪代码（旧版 GrowthBook + cache_connection）不同：
    * 新模型：GrowthBookClient(Options(...)) + UserContext；eval_feature(key, user_context) 同步本地 eval；
      initialize() 异步加载；曝光经 set_event_logger(fn(event_name, data, user_context)) 回调，
      experiment_viewed 事件名。
    * 已无 cache_connection / RedisConnection；缓存持久化由本模块自实现快照替代（见 _persist_snapshot / _try_inject_cached_snapshot）。
  - eval_feature 的 user/attributes 传参：v3 用 UserContext(attributes={"id": user_id, ...})，
    每调用传入，避免实例级可变状态（并发安全）。已按此实现。
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

# 注意：单例类命名为 GrowthBookClient，故 SDK 客户端导入为别名避免命名冲突。
from growthbook import GrowthBookClient as GBGrowthBookClient, Options, UserContext

from src.core.config import config
from src.core.growthbook_datasource import GrowthBookDataSource
from src.modules.chat.core.experiment_service import (
    Assignment,
    ExperimentDef,
    PipelineOverrides,
    VariantType,
)
from src.shared.logger import APILogger

logger = APILogger("growthbook_client")

# 启动门禁 / 熔断参数
_INIT_MAX_ATTEMPTS = 3          # loadFeatures 重试次数（指数退避 1,2,4s）
_LOAD_TIMEOUT = 8.0             # 单次 loadFeatures 超时（秒）
_CIRCUIT_THRESHOLD = 5          # 连续失败达此值 → circuit open
_STALE_ALERT_MULTIPLIER = 3     # cache_age > cache_ttl * 该倍数 → 陈旧告警

# ── scope §9 Flag 生命周期治理：命名前缀强制（§9.1）─────────────────────────
# 任何 create_experiment 的 feature key 必须以以下前缀之一开头，否则拒绝创建
# （Redis sidecar key 与 GB feature key 必须一致，故不自动加前缀，而是 fail-loud）。
_FLAG_PREFIXES = ("exp_", "canary_", "switch_", "perm_")
# 带预期结束日期的前缀（§9.2）：此类 flag 应在创建时带 expected_end_date tag。
_FLAG_PREFIXES_WITH_END_DATE = ("exp_", "canary_")


def validate_flag_prefix(feature_key: str) -> Optional[str]:
    """校验 flag 命名前缀（scope §9.1）。

    返回 None 表示合法；否则返回人类可读的错误原因（含建议前缀）。
    """
    if not feature_key:
        return "feature key 不能为空"
    for prefix in _FLAG_PREFIXES:
        if feature_key.startswith(prefix):
            return None
    return f"非法 flag 命名前缀: {feature_key!r}；必须以之一开头: {', '.join(_FLAG_PREFIXES)}"


class GrowthBookClient:
    """GrowthBook SDK 单例（进程内），对外仅暴露纯函数式 API。"""

    _instance: Optional["GrowthBookClient"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._gb: Optional[GBGrowthBookClient] = None  # 运行时为 growthbook.GrowthBookClient
        self._enabled: bool = False
        self._initialized: bool = False
        self._degraded: bool = False
        self._gb_reachable: bool = False
        self._circuit_open: bool = False
        self._consecutive_failures: int = 0
        self._last_success_ts: float = 0.0
        self._init_ts: float = 0.0
        self._using_cached_snapshot: bool = False
        self._alert_hook: Optional[Callable[[str], None]] = None
        self._redis_client = None
        self._cache_file: str = getattr(config, "GROWTHBOOK_CACHE_FILE", "/data/growthbook_features.json")

    # ─────────────────────────────────────────────────────────────────────────
    # 单例
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "GrowthBookClient":
        """进程内单例（线程安全，支持多 worker 各自构造后共享）。"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = GrowthBookClient()
            return cls._instance

    def set_alert_hook(self, hook: Callable[[str], None]) -> None:
        """注册告警钩子（钉钉/企微等）；Phase 1 仅 logger.warning，预留扩展。"""
        self._alert_hook = hook

    # ─────────────────────────────────────────────────────────────────────────
    # 生命周期
    # ─────────────────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """启动门禁：阻塞式 loadFeatures（超时 + 指数退避）。

        成功→就绪；失败→若缓存有快照则 stale 快照上岗（degraded，不熔断，仍可服务）；
        若无缓存→安全默认上岗（degraded + circuit_open）。绝不抛异常。
        """
        if self._initialized:
            return
        self._init_ts = time.time()

        # 全量禁用：直接安全默认模式，主流程零影响
        if not getattr(config, "GROWTHBOOK_ENABLED", False):
            self._enabled = False
            self._initialized = True
            logger.info("GrowthBook 禁用（GROWTHBOOK_ENABLED=false），以安全默认模式运行")
            self._init_datasource_best_effort()
            return

        self._enabled = True
        try:
            self._gb = self._build_gb()
        except Exception as e:  # noqa: BLE001
            logger.error(f"GB SDK 构造失败，降级为安全默认模式: {e}")
            self._degraded = True
            self._circuit_open = True
            self._initialized = True
            self._init_datasource_best_effort()
            return

        await self._gate_load_features()
        self._init_datasource_best_effort()
        self._initialized = True

    def _build_gb(self) -> GBGrowthBookClient:
        """构造 growthbook.GrowthBookClient（v3 Options 模型）。"""
        opts = Options(
            api_host=getattr(config, "GROWTHBOOK_API_HOST", "http://growthbook:3100"),
            client_key=getattr(config, "GROWTHBOOK_CLIENT_KEY", "") or None,
            decryption_key=getattr(config, "GROWTHBOOK_DECRYPTION_KEY", "") or None,
            cache_ttl=getattr(config, "GROWTHBOOK_CACHE_TTL", 30),
            enabled=True,
        )
        client: GBGrowthBookClient = GBGrowthBookClient(opts)
        # 曝光回调：experiment_viewed 事件 → Data Source 写入（见 _on_event）
        client.set_event_logger(self._on_event)
        return client

    async def _gate_load_features(self) -> None:
        """阻塞 loadFeatures + 指数退避；全失败 → stale 快照 / 安全默认上岗。"""
        delay = 1.0
        max_delay = 4.0
        for attempt in range(_INIT_MAX_ATTEMPTS):
            try:
                await asyncio.wait_for(self._gb.initialize(), timeout=_LOAD_TIMEOUT)  # type: ignore[union-attr]
                self._last_success_ts = time.time()
                self._degraded = False
                self._gb_reachable = True
                self._consecutive_failures = 0
                self._using_cached_snapshot = False
                logger.info("GrowthBook features 加载成功（就绪）")
                # 层1：成功加载后持久化快照（Redis + 文件），供重启 stale 上岗
                self._persist_snapshot()
                return
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"GB loadFeatures 第 {attempt + 1}/{_INIT_MAX_ATTEMPTS} 次失败: {e}"
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)

        # 全失败：尝试缓存快照上岗
        if self._try_inject_cached_snapshot():
            self._degraded = True
            self._gb_reachable = False
            self._circuit_open = False  # 有缓存可服务 → 不熔断，eval 仍可用
            self._using_cached_snapshot = True
            logger.warning("GB 不可达，使用缓存快照上岗（degraded）")
        else:
            self._degraded = True
            self._gb_reachable = False
            self._circuit_open = True
            self._using_cached_snapshot = False
            logger.error("GB 不可达且无缓存，安全默认模式上岗（degraded）")
        self._alert(
            "GrowthBook init failed, running degraded with safe defaults / cached snapshot"
        )

    async def close(self) -> None:
        """关闭 SDK 客户端与 Data Source（best-effort）。"""
        if self._gb is not None:
            try:
                self._gb.close()
            except Exception:  # noqa: BLE001
                pass
            self._gb = None
        try:
            await GrowthBookDataSource.get_instance().close()
        except Exception:  # noqa: BLE001
            pass
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def health(self) -> Dict[str, Any]:
        """健康状态（供 /health 与 PROBE_PORT readiness 聚合展示）。

        字段：enabled/initialized/degraded/gb_reachable/circuit_open/
        cache_age_seconds/last_success_ts/consecutive_failures/
        using_cached_snapshot/sdk_version。
        """
        now = time.time()
        cache_age = (now - self._last_success_ts) if self._last_success_ts else None
        stale = (
            cache_age is not None
            and self._degraded
            and cache_age > (getattr(config, "GROWTHBOOK_CACHE_TTL", 30) * _STALE_ALERT_MULTIPLIER)
        )
        return {
            "enabled": self._enabled,
            "initialized": self._initialized,
            "degraded": self._degraded,
            "gb_reachable": self._gb_reachable,
            "circuit_open": self._circuit_open,
            "using_cached_snapshot": self._using_cached_snapshot,
            "cache_age_seconds": int(cache_age) if cache_age is not None else None,
            "last_success_ts": self._last_success_ts,
            "consecutive_failures": self._consecutive_failures,
            "stale": bool(stale),
            "sdk_version": _sdk_version(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 核心分配
    # ─────────────────────────────────────────────────────────────────────────

    def eval_variant(
        self,
        feature_key: str,
        user_id: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Assignment:
        """本地 eval（同步、快）。

        失败/未命中 → 返回 control 安全默认 Assignment（exp_mode=None）。
        绝不抛异常、绝不返回 None。
        """
        if not self._enabled or self._gb is None:
            return self._safe_control(user_id)
        if self._circuit_open:
            return self._safe_control(user_id)

        try:
            # v3 SDK：每调用传入 UserContext（并发安全，避免实例级可变属性）
            ctx = UserContext(attributes={"id": user_id, **(attributes or {})})
            res = self._gb.eval_feature(feature_key, ctx)  # type: ignore[union-attr]
            self._consecutive_failures = 0
            if self._circuit_open:  # 自动恢复
                self._circuit_open = False
                logger.info("GB 熔断恢复（连续成功 eval）")
            self._last_success_ts = time.time()
            self._gb_reachable = True
            self._degraded = False
            return self._to_assignment(res, feature_key, user_id)
        except Exception as e:  # noqa: BLE001
            self._consecutive_failures += 1
            if self._consecutive_failures >= _CIRCUIT_THRESHOLD:
                self._circuit_open = True
                self._alert(
                    f"GB circuit open after {self._consecutive_failures} failures: {e}"
                )
            logger.warning(f"GB eval_variant 失败（返回安全默认）: {e}")
            return self._safe_control(user_id)

    def track_exposure(
        self,
        feature_key: str,
        user_id: str,
        variation_key: str,
        domain: str = "ecommerce",
    ) -> None:
        """触发曝光上报（仅 experiment 模式调用；canary 模式不调用）。

        路径：gb.log_event("experiment_viewed", ...) → set_event_logger(_on_event)
        → GrowthBookDataSource.record_exposure（确定性写入 Data Source）。
        best-effort，吞异常。GB 禁用或 SDK 不可用时静默返回。
        """
        if not self._enabled:
            return

        # 1) SDK 侧记录（用于 GB 原生 tracking buffer / analytics，best-effort）
        if self._gb is not None:
            try:
                self._gb.log_event(
                    "experiment_viewed",
                    {
                        "featureKey": feature_key,
                        "variationKey": str(variation_key),
                        "userId": user_id,
                        "domain": domain,
                    },
                    UserContext(attributes={"id": user_id, "domain": domain}),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"GB log_event 失败（忽略）: {e}")

        # 2) Data Source 确定性写入（显著性分析的唯二数据入口，best-effort 吞异常）
        try:
            GrowthBookDataSource.get_instance().record_exposure(
                feature_key=feature_key,
                variation_key=str(variation_key),
                user_id=user_id,
                domain=domain,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"GB 曝光写入 Data Source 失败（忽略）: {e}")

    async def refresh(self) -> None:
        """reload SDK feature 缓存（对接 POST /experiments/refresh）。"""
        if not self._enabled or self._gb is None:
            return
        try:
            await asyncio.wait_for(self._gb.initialize(), timeout=_LOAD_TIMEOUT)  # type: ignore[union-attr]
            self._last_success_ts = time.time()
            self._degraded = False
            self._gb_reachable = True
            self._consecutive_failures = 0
            self._circuit_open = False
            self._persist_snapshot()
            logger.info("GB features 刷新成功")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"GB refresh 失败: {e}")
            self._alert(f"GB refresh failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # REST 代理（建/改/删/查实验）—— best-effort，绝不向上抛（红线：GB 故障不影响主流程）
    # ─────────────────────────────────────────────────────────────────────────

    def create_experiment(self, exp_def: "ExperimentDef", kind: str = "experiment") -> bool:
        """在 GrowthBook 中创建实验（best-effort REST，失败仅告警不抛）。

        - kind="canary"   ：建 Feature（带 variations，每 variation 的 value =
          variant.pipeline_overrides.to_dict()，variation name = variant.name）；
          设 rolloutPercentage（金丝雀渐进开量）。
        - kind="experiment"：建 Feature（同上）+ 建 Experiment（A/B，关联 featureKey，
          control/treatment），写 Data Source 算显著性。

        各 variation 的 value 必须能由 eval_variant 经 PipelineOverrides.from_dict(value)
        还原（eval_variant 已使用该还原逻辑）。

        端点以 GrowthBook 官方 API 为准（design.md §13 缺口③）：先试
        POST /api/features 与 POST /api/experiments；不确定处用 TODO(GB-SDK) 标注。
        任何 REST 失败仅 logger.warning + 返回 False（sidecar 本地仍记录，assign 本地可用）。
        """
        if not self._enabled:
            logger.warning("GB 禁用，create_experiment 跳过 REST 调用（仅本地 sidecar 记录）")
            return False
        api_host = getattr(config, "GROWTHBOOK_API_HOST", "").rstrip("/")
        api_key = getattr(config, "GROWTHBOOK_API_KEY", "")
        if not api_host or not api_key:
            logger.warning("GROWTHBOOK_API_HOST/API_KEY 未配置，create_experiment 跳过 REST 调用")
            return False

        feature_key = exp_def.id
        # ── scope §9.1 命名前缀强制：fail-loud 拒绝非法前缀 ──────────────────
        # sidecar key 与 GB feature key 必须保持一致，故不自动加前缀，而是直接拒绝。
        prefix_err = validate_flag_prefix(feature_key)
        if prefix_err:
            logger.warning(f"create_experiment 拒绝: {prefix_err}（flag 未创建，仅本地 sidecar 记录）")
            return False

        # GB v3：json 类型 feature 的 variation.value / defaultValue 必须是「JSON 字符串」，
        # 不是 dict（前端/SDK 会 JSON.parse 还原）。这里 json.dumps 序列化为字符串。
        variations = []
        for v in exp_def.variants:
            variations.append(
                {
                    "name": v.name,
                    "value": json.dumps(v.pipeline_overrides.to_dict()),
                }
            )

        if kind == "canary":
            # 金丝雀：用非 control 变体的总流量作为整体 rollout 开量
            rollout = 0.0
            for v in exp_def.variants:
                if v.variant_type.value != "control":
                    rollout += float(v.traffic_percent)
            rollout = min(100.0, rollout)
        else:
            # 实验：feature rollout=100 让目标用户全部进入，分流由 Experiment 配置控制
            rollout = 100.0

        # ── scope §9.2 治理 tag：exp_type / expected_end_date（零成本写入）──────
        tags = [f"exp_type:{kind}", f"exp_id:{feature_key}"]
        if exp_def.expected_end and kind in _FLAG_PREFIXES_WITH_END_DATE:
            tags.append(f"expected_end_date:{exp_def.expected_end}")

        feature_body = {
            "key": feature_key,
            "name": exp_def.name,
            "description": exp_def.description,
            "type": "json",
            "defaultValue": "{}",  # json 类型默认值为字符串 "{}"
            "project": "default",
            "tags": tags,
            "environments": {
                "production": {
                    "enabled": True,
                    "rolloutPercentage": rollout,
                    "variations": variations,
                }
            },
        }
        feature_resp = self._rest_call("POST", f"{api_host}/api/features", {"feature": feature_body})
        feature_ok = feature_resp is not None

        if kind == "experiment":
            # GB v3 Experiment 通过 featureId（= feature key）关联 Feature；
            # variations 的 value 同样为 JSON 字符串。
            weights = [max(0.0, float(v.traffic_percent) / 100.0) for v in exp_def.variants]
            exp_body = {
                "name": exp_def.name,
                "featureId": feature_key,
                "type": "code",
                "status": "running",
                "variations": variations,
                "coverage": 1.0,
                "hashAttribute": "id",
                "phases": [{"dateStarted": _now_iso(), "variationWeights": weights}],
            }
            # TODO(GB-SDK): 实验归档/暂停端点（POST /api/experiments/:id/archive 或改 feature
            # rollout=0）以真实实例联调确认；当前 update_experiment_status 用 feature rollout=0 兜底。
            exp_resp = self._rest_call(
                "POST", f"{api_host}/api/experiments", {"experiment": exp_body}
            )
            exp_ok = exp_resp is not None
            return bool(feature_ok and exp_ok)
        return feature_ok

    def update_experiment_status(self, exp_id: str, status: str) -> bool:
        """更新实验状态（best-effort REST，失败仅告警不抛）。

        - paused/stopped/archived → production.rolloutPercentage=0（金丝雀停量）/ 实验归档。
        - running → 恢复 rollout=100。
        TODO(GB-SDK): 暂停/停止的确切端点（PATCH /api/features/:id 的
        environments.production.rolloutPercentage，或 Experiment archive）待对齐官方 API。
        """
        if not self._enabled:
            return False
        api_host = getattr(config, "GROWTHBOOK_API_HOST", "").rstrip("/")
        api_key = getattr(config, "GROWTHBOOK_API_KEY", "")
        if not api_host or not api_key:
            return False

        if status in ("paused", "stopped", "archived"):
            rollout = 0.0
            enabled = status != "stopped"
        else:
            rollout = 100.0
            enabled = True

        body = {
            "feature": {
                "environments": {
                    "production": {"enabled": enabled, "rolloutPercentage": rollout}
                }
            }
        }
        resp = self._rest_call("PUT", f"{api_host}/api/features/{exp_id}", body)
        return resp is not None

    def delete_experiment(self, exp_id: str) -> bool:
        """删除实验（best-effort REST，失败仅告警不抛）。

        TODO(GB-SDK): 删除端点 /api/features/:id 是否软归档待对齐官方 API。
        """
        if not self._enabled:
            return False
        api_host = getattr(config, "GROWTHBOOK_API_HOST", "").rstrip("/")
        api_key = getattr(config, "GROWTHBOOK_API_KEY", "")
        if not api_host or not api_key:
            return False
        resp = self._rest_call("DELETE", f"{api_host}/api/features/{exp_id}")
        return resp is not None

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """获取单个实验（原始 GB feature 字典；best-effort，失败返回 None）。

        TODO(GB-SDK): 与 experiment_service._gb_feature_to_def 协同对齐字段映射。
        """
        if not self._enabled:
            return None
        api_host = getattr(config, "GROWTHBOOK_API_HOST", "").rstrip("/")
        api_key = getattr(config, "GROWTHBOOK_API_KEY", "")
        if not api_host or not api_key:
            return None
        resp = self._rest_call("GET", f"{api_host}/api/features/{exp_id}")
        if resp is None:
            return None
        if isinstance(resp, dict):
            return resp.get("feature", resp)
        return None

    def list_experiments(self) -> List[Dict[str, Any]]:
        """列出实验（原始 GB feature 字典列表；best-effort，失败返回 []）。"""
        if not self._enabled:
            return []
        api_host = getattr(config, "GROWTHBOOK_API_HOST", "").rstrip("/")
        api_key = getattr(config, "GROWTHBOOK_API_KEY", "")
        if not api_host or not api_key:
            return []
        resp = self._rest_call("GET", f"{api_host}/api/features")
        if resp is None:
            return []
        if isinstance(resp, dict):
            feats = resp.get("features", [])
        elif isinstance(resp, list):
            feats = resp
        else:
            feats = []
        # 若每个元素被包裹 {"feature": {...}}，解开
        out: List[Dict[str, Any]] = []
        for f in feats:
            if isinstance(f, dict) and "feature" in f:
                out.append(f["feature"])
            elif isinstance(f, dict):
                out.append(f)
        return out

    def _rest_call(
        self,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """best-effort GB REST 调用（鉴权头 Authorization: Bearer <GROWTHBOOK_API_KEY>）。

        任何异常仅 logger.warning 并返回 None，绝不向上抛（红线：GB 故障不影响主流程）。
        """
        try:
            import urllib.request

            data = json.dumps(payload).encode("utf-8") if payload is not None else None
            req = urllib.request.Request(url, data=data, method=method)
            api_key = getattr(config, "GROWTHBOOK_API_KEY", "")
            req.add_header("Authorization", f"Bearer {api_key}")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                if not raw:
                    return {}
                return json.loads(raw)  # type: ignore[no-any-return]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"GB REST {method} {url} 失败（忽略，不影响主流程）: {e}")
            return None


    # ─────────────────────────────────────────────────────────────────────────
    # 内部：安全默认 / 映射
    # ─────────────────────────────────────────────────────────────────────────

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

    def _to_assignment(self, res: Any, feature_key: str, user_id: str) -> Assignment:
        """FeatureResult → Assignment（design.md §3.3 映射规则）。"""
        if res is None or not getattr(res, "on", False) or getattr(res, "value", None) is None:
            return self._safe_control(user_id)

        value = res.value
        is_experiment = getattr(res, "experiment", None) is not None
        exp_result = getattr(res, "experimentResult", None)

        variation_id = "treatment"
        feature_id = feature_key
        if exp_result is not None:
            vid = getattr(exp_result, "variationId", None)
            if vid is not None:
                variation_id = vid
            fid = getattr(exp_result, "featureId", None)
            if fid:
                feature_id = fid

        exp_mode = "experiment" if is_experiment else "canary"
        variant_type = (
            VariantType.CONTROL
            if (not is_experiment and not getattr(res, "on", False)) or str(variation_id) == "control"
            else VariantType.TREATMENT
        )
        # GB json 类型 value 可能被 SDK 解析为 dict，也可能仍是 JSON 字符串（取决于 SDK 版本/缓存）。
        # 兼容两种形态：字符串则 json.loads，dict 直接用。
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                value = {}
        overrides = PipelineOverrides.from_dict(value if isinstance(value, dict) else {})

        return Assignment(
            user_id=user_id,
            experiment_id=str(feature_id),
            variant_name=str(variation_id),
            variant_type=variant_type,
            pipeline_overrides=overrides,
            traffic_percent=0.0,
            bucket=-1,
            exp_mode=exp_mode,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 内部：曝光事件回调 / 告警
    # ─────────────────────────────────────────────────────────────────────────

    def _on_event(self, event_name: str, data: Dict[str, Any], user_context: Any) -> None:
        """set_event_logger 回调：仅处理 experiment_viewed → 写 Data Source。"""
        if event_name != "experiment_viewed":
            return
        try:
            data = data or {}
            feature_key = data.get("featureKey") or data.get("key")
            if not feature_key:
                return
            user_id = data.get("userId")
            if not user_id and user_context is not None:
                attrs = getattr(user_context, "attributes", None) or {}
                user_id = attrs.get("id")
            domain = data.get("domain") or "ecommerce"
            GrowthBookDataSource.get_instance().record_exposure(
                feature_key=str(feature_key),
                variation_key=str(data.get("variationKey") or "control"),
                user_id=str(user_id or ""),
                domain=str(domain),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"GB 曝光事件处理失败（忽略）: {e}")

    def _alert(self, msg: str) -> None:
        """告警：先 logger.warning（红线要求），再走可选 alert 钩子。"""
        logger.warning(f"[GB-ALERT] {msg}")
        if self._alert_hook is not None:
            try:
                self._alert_hook(msg)
            except Exception:  # noqa: BLE001
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # 内部：DataSource / 缓存快照
    # ─────────────────────────────────────────────────────────────────────────

    def _init_datasource_best_effort(self) -> None:
        """best-effort 初始化 Data Source（GB 不可用时也允许本地 PG 曝光表存在）。"""
        try:
            ds = GrowthBookDataSource.get_instance()
            ds.initialize()
            ds.ensure_schema()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"GrowthBook DataSource 初始化跳过: {e}")

    def _get_redis_client(self):
        """懒加载 Redis 客户端（复用现有 REDIS_HOST/PORT/PASSWORD），失败返回 None。"""
        if self._redis_client is not None:
            return self._redis_client
        try:
            import redis as redis_lib

            self._redis_client = redis_lib.Redis(
                host=getattr(config, "REDIS_HOST", "localhost"),
                port=getattr(config, "REDIS_PORT_NUM", 6379),
                password=getattr(config, "REDIS_PASSWORD", "") or None,
                db=0,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"GB Redis 客户端创建失败（快照缓存不可用）: {e}")
            self._redis_client = None
        return self._redis_client

    def _persist_snapshot(self) -> None:
        """层1 缓存：成功加载后抓取 features 快照 → Redis 字符串 + 本地文件（best-effort）。

        TODO(GB-SDK): v3 GrowthBookClient 未暴露公共 get_features()，此处改用公共
        HTTP 端点 /api/features/{client_key} 抓取（与 SDK 拉取同源）。端点鉴权与
        exp query 参数以 GB 实际版本为准，失败仅告警不阻塞。
        """
        try:
            client_key = getattr(config, "GROWTHBOOK_CLIENT_KEY", "") or ""
            if not client_key:
                return
            api_host = getattr(config, "GROWTHBOOK_API_HOST", "").rstrip("/")
            url = f"{api_host}/api/features/{client_key}"
            import urllib.request

            req = urllib.request.Request(url)
            api_key = getattr(config, "GROWTHBOOK_API_KEY", "")
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            features = payload.get("features", {})
            blob = json.dumps(
                {"features": features, "dateUpdated": payload.get("dateUpdated")}
            ).encode("utf-8")

            # Redis（首选，多实例共享，无撕裂）
            rc = self._get_redis_client()
            if rc is not None:
                try:
                    rc.set("gb:features_snapshot", blob)
                except Exception:  # noqa: BLE001
                    pass
            # 本地文件（回退）
            try:
                os.makedirs(os.path.dirname(self._cache_file) or ".", exist_ok=True)
                with open(self._cache_file, "wb") as f:
                    f.write(blob)
            except Exception:  # noqa: BLE001
                pass
            logger.info(f"GB features 快照持久化完成（{len(features)} 个）")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"GB 快照持久化失败（忽略，不影响服务）: {e}")

    def _try_inject_cached_snapshot(self) -> bool:
        """层1 回放：GB 不可达时，从 Redis/文件快照注入到 SDK（best-effort）。

        成功注入返回 True（stale 快照上岗）；否则 False（安全默认上岗）。
        """
        features: Optional[Dict[str, Any]] = None
        # 先 Redis
        rc = self._get_redis_client()
        if rc is not None:
            try:
                raw = rc.get("gb:features_snapshot")
                if raw:
                    data = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
                    features = data.get("features")
            except Exception:  # noqa: BLE001
                pass
        # 文件回退
        if features is None:
            try:
                if os.path.exists(self._cache_file):
                    with open(self._cache_file, "rb") as f:
                        data = json.loads(f.read().decode("utf-8"))
                    features = data.get("features")
            except Exception:  # noqa: BLE001
                pass
        if features:
            try:
                self._gb.set_features(features)  # type: ignore[union-attr]
                return True
            except Exception as e:  # noqa: BLE001
                logger.warning(f"GB 缓存快照注入失败: {e}")
        return False


def _sdk_version() -> str:
    """返回 growthbook SDK 版本（best-effort）。"""
    try:
        import growthbook

        return getattr(growthbook, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


def _now_iso() -> str:
    """返回当前 ISO 时间戳（best-effort，供 GB Experiment phase 使用）。"""
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    except Exception:  # noqa: BLE001
        return ""
