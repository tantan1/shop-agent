# 架构设计：shop-agent 接入 GrowthBook

> 关联 scope：`.codebuddy/run/growthbook-integration/scope.md`（设计定稿版，已确认用 GrowthBook 自托管）
> 本文档严格遵循 scope 的既有决策，重点覆盖 scope 列出的 10 项诉求，并给出可落地的模块/类/函数签名级设计。
> 行号引用以本仓库当前代码为准（已对照读取）。

---

## 0. 现状对齐（已读取代码，行号校准）

| 文件 | 现状关键点 | 本次处理 |
|---|---|---|
| `src/modules/chat/core/experiment_service.py` | DTO（84-260）、`TrafficRouter`/`_fnv1a`（268-380）、`ExperimentStore` Redis（388-486）、`SafetyGuardEvaluator.evaluate`（494-540）、`ExperimentService`（548-705）、`_ExperimentMetricsCollector`（713-770）、`SampleSizeCalculator`/`StatisticalTest`（785-880）、预置模板（897-973） | 删 268-380/388-486/713-770/785-880；保留 DTO + `SafetyGuardEvaluator.evaluate` + 模板；`ExperimentService` 瘦身为 GB 代理 |
| `src/modules/chat/routers.py` | 分配块 211-221；`create_experiment` 537-635；暂停/校验/列出/详情/删除/刷新 638-764 | 最小化：透传 `attributes={"domain"}`；`force_refresh` 内部改调 GB 缓存 reload；其余保持 |
| `src/modules/chat/agent/orchestrator.py` | `chat_with_agent` 329-331 调 `to_tags()/to_metadata()`；277 `response.experiment_group = variant_name` | **零改动**（数据来源切换为 GB） |
| `src/modules/chat/schemas.py` | `ExperimentCreateRequest` 523-532 / `ExperimentPauseRequest` 535-539 / `ExperimentValidateRequest` 542-548 | **基本不动**；新增可选 `kind` 字段（已拍板，默认 `"experiment"`，见 §5/§11-④） |
| `src/core/config.py` | Redis 配置 24-31；`database_url` property 257-276（→ asyncpg） | 新增 GrowthBook 配置块（§3.2） |
| `docker-compose.yml` | `shop-agent` 493-567（env_file + depends_on redis/pgvector/gateway）；`pgvector`/`postgres` 分离；`.env` 注入 `POSTGRES_PASSWORD`/`REDIS_AUTH` | 新增 `growthbook`+`mongo` 服务（profiles: experiments）；`shop-agent` 增补 GB 环境变量 |
| `requirements.txt` | 无 `growthbook` | 新增 `growthbook` |

**关键现状结论（影响设计）**
- shop-agent **已用 Postgres**（`pgvector` 服务，`shop_agent` 库，asyncpg 读 + psycopg2 写）。→ §4.8 Data Source **复用现有 `pgvector` Postgres**，不新增 PG 实例（解答 scope §8 待核对项）。
- 现有 Redis（实体槽位）保留，与实验解耦；本设计用其作 GB SDK `cacheConnection` 与 SafetyGuard 本地 sidecar。
- `pgvector_service.py` 已用 `psycopg2.pool.SimpleConnectionPool` 做同步写 → 曝光写入复用同一模式。

---

## 1. 架构概述

### 1.1 架构风格与选型理由
- **风格**：模块化单体（进程内） + 外部自托管 GrowthBook（配置/分配后端）。shop-agent 进程内持有 `GrowthBookClient` 单例（SDK），分配为本地内存计算（无每次请求的网络往返），仅配置拉取/刷新与曝光上报走网络。
- **理由**：
  - scope 已锁定 GrowthBook 自托管，数据不出内网 → 必须 mongo（应用存储）+ Data Source（显著性）。
  - 分配走 SDK 本地 eval，避免每次请求打 GB 后端，满足低延迟（对齐 orchestrator 主链路 < 1 次额外同步调用）。
  - 横切关注点（治理/观测/安全护栏调度）作为独立模块/后台任务，不渗入业务逻辑（orchestrator 零改动）。

### 1.2 高层架构图

```
┌───────────────────────────────────────────────────────────────────────────┐
│  shop-agent 进程 (FastAPI)                                                  │
│                                                                            │
│  routers.py ──assign(user_id,domain)──▶ ExperimentService.assign()         │
│     │                                         │ (枚举候选 feature key)      │
│     │                                         ▼                            │
│     │                          GrowthBookClient.eval_variant(key,uid,attr) │
│     │                                         │ 本地 eval (命中 rollout/experiment)
│     │                                         ▼                            │
│     │                          Assignment(exp_mode, variant, overrides)    │
│     │                                         │                            │
│     │                         if exp_mode=="experiment":                   │
│     │                                         ▼                            │
│     │                          GrowthBookClient.track_exposure()           │
│     │                                         │ 曝光回调 (best-effort)      │
│     │                                         ▼                            │
│     │                          GrowthBookDataSource.record_exposure()      │
│     │                                         │ 写 Postgres(gb_exposures)  │
│     └──▶ orchestrator.chat_with_agent() ──▶ Assignment.to_tags()/to_metadata() ─▶ Langfuse │
│                                                                            │
│  ┌── 后台 SafetyGuard 调度 (asyncio loop, ~300s) ──────────────────────┐   │
│  │  list RUNNING(本地 sidecar) → MetricsProvider(Langfuse/Prometheus)   │   │
│  │   → SafetyGuardEvaluator.evaluate() → 越界 → GB REST pause/stop     │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  GrowthBookClient (单例) ──loadFeatures/cache──▶ GrowthBook SDK            │
│        │  SDK cacheConnection(Redis) / cacheFile(磁盘) 持久化 features 快照 │
│        └── HTTP 拉取 features.json ──▶ GrowthBook 自托管 (UI:3100)         │
└───────────────────────────────────────────────────────────────────────────┘
        │                                          │
        │  REST (server API key)                   │  Mongo (应用存储)
        ▼                                          ▼
  GrowthBook REST API  ◀── ExperimentService CRUD ──  mongo (growthbook db)
        │
        │  Data Source 连接 (只读 GB 查 SQL 显著性)
        ▼
  Postgres (pgvector / shop_agent)  ◀── 同一库，gb_exposures / gb_metrics 表
        ▲
        │ 写曝光 (我们的 datasource 模块)
        └── GrowthBookDataSource

关键质量属性优先级：
  可用性 > 一致性（分配可 stale，但绝不能 None / 不能崩主链路）
  韧性 > 实时性（GB 故障时用缓存 + 安全默认继续服务）
  数据不出内网（自托管，mongo + PG 均在内部网络）
```

### 1.3 关键质量属性与首要取舍
- **首要**：**韧性 / 可用性优先**于实时一致性。GB 不可达时，用"上次成功缓存 + 安全默认 control"继续服务（scope §4.9 / §7 红线）。
- **取舍**（显式标注）：
  - 一致性 vs 可用性：`assign` 允许使用 **陈旧缓存**（TTL×3 内仅告警不阻断）→ 选可用性。
  - 多实验合并 vs 单分配：保留原"首个命中即返回"语义（**不**做多 experiment 的 overrides 合并）→ 选简单性；多实验合并列为后续（见 §10 Out of scope）。
  - 曝光丢失 vs 主链路延迟：曝光上报 best-effort、异步、失败不抛 → 选主链路零影响。

---

## 2. 模块划分

| 模块 | 职责 | 边界 | 上游依赖 | 下游依赖 |
|---|---|---|---|---|
| `core/growthbook_client.py` | GB SDK 单例封装：`eval_variant`/`track_exposure`/`refresh`/`is_initialized`/`health`；缓存持久化；启动门禁；熔断 | 进程内单例，只暴露纯函数式 API | `core/config.py`、`core/growthbook_datasource.py`（曝光回调） | `growthbook` SDK；GB 自托管(HTTP)；Redis/磁盘(缓存) |
| `core/growthbook_datasource.py` | Data Source 写入：`record_exposure`/`record_metric`/建表；连接管理 | 仅写 `pgvector` Postgres（`gb_*` 表），best-effort | `core/config.py`、client 曝光回调 | Postgres（psycopg2 池） |
| `modules/chat/core/experiment_service.py` | 对外服务门面：`assign`（委托 client）、CRUD 代理 GB REST、护栏调度编排、SafetyGuard sidecar | 保留全部公开方法签名，内部委托 GB | `growthbook_client`、`growthbook_datasource`、`SafetyGuardEvaluator`、`SafetyGuardRegistry`、`SafetyScheduler`、GB REST client | GB REST API；Redis(sidecar)；Langfuse/Prometheus(指标) |
| `modules/chat/core/safety_guard.py`（**新增**，从 experiment_service 抽出） | `SafetyGuardEvaluator`（保留）、`SafetyGuardRegistry`（Redis sidecar）、`SafetyMetricsProvider` 接口 + Langfuse/Prometheus 实现、`SafetyScheduler` | 指标来源与调度独立于分配后端 | `core/config.py`、GB REST、Langfuse/Prometheus 客户端 | Redis；GB REST |
| `core/config.py` | GrowthBook 配置块 | 配置只读 | `.env` / secret | — |
| `modules/chat/routers.py` | REST 入口，最小化改动 | HTTP 边界 | `experiment_service` | — |
| `modules/chat/agent/orchestrator.py` | 应用 Assignment 到 Langfuse tag / `experiment_group` | **零改动** | `experiment_service.assign` 返回值 | Langfuse |
| `modules/chat/schemas.py` | 请求/响应 schema | **基本不动**（可选 +`kind`） | — | — |
| `docker-compose.yml` | `growthbook` + `mongo` 服务；shop-agent GB 环境变量 | 部署适配 | — | — |

**依赖方向**（稳定→易变）：`config` ← `growthbook_datasource` / `growthbook_client` ← `experiment_service` ← `routers`。`safety_guard` 模块横切，依赖 `experiment_service` 暴露的"RUNNING 实验列表"与 GB REST，但不被业务逻辑反向依赖。

---

## 3. 接口设计（关键契约草案）

### 3.1 `GrowthBookClient`（`core/growthbook_client.py`）— 单例

```python
class GrowthBookClient:
    _instance: Optional["GrowthBookClient"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "GrowthBookClient": ...

    # ── 生命周期 ──
    async def initialize(self) -> None:
        """启动门禁：阻塞式 loadFeatures（超时 + 指数退避）。
        成功→就绪；失败→若缓存有快照则 stale 上岗(degraded)，否则 degraded 无缓存上岗。
        绝不抛异常导致进程起不来。"""

    async def close(self) -> None: ...

    @property
    def is_initialized(self) -> bool: ...

    def health(self) -> dict:
        # {initialized, degraded, gb_reachable, cache_age_seconds,
        #  last_success_ts, consecutive_failures, circuit_open}

    # ── 核心分配 ──
    def eval_variant(self, feature_key: str, user_id: str,
                     attributes: dict | None = None) -> Assignment:
        """本地 eval（同步、快）。失败/未命中→返回 control 安全默认 Assignment。
        绝不抛异常、绝不返回 None。"""

    def track_exposure(self, feature_key: str, user_id: str,
                       variation_key: str, domain: str = "ecommerce") -> None:
        """触发 GB track → 曝光回调 → 写 Data Source。best-effort，吞异常。"""

    async def refresh(self) -> None:
        """reload SDK feature 缓存（对接 POST /experiments/refresh）。"""
```

**SDK 构造（伪代码，字段以 SDK 实际签名校准，见 §11 缺口①）**：
```python
self._gb = GrowthBook(
    api_host=config.GROWTHBOOK_API_HOST,        # http://growthbook:3100
    client_key=config.GROWTHBOOK_CLIENT_KEY,    # 只读评估
    decryption_key=config.GROWTHBOOK_DECRYPTION_KEY,
    cache_ttl=config.GROWTHBOOK_CACHE_TTL,
    cache_connection=self._build_cache(),       # RedisConnection 或 cache_file
    on_exposure=self._on_exposure,              # 曝光回调 → datasource.record_exposure
)
```

**缓存持久化（§4.9 第 1 层）**：
- 首选 `cache_connection`：包一层 `growthbook.RedisConnection(prefix="gb", client=redis_client)`（复用现有 Redis，`REDIS_HOST`/`REDIS_PORT_NUM`/`REDIS_PASSWORD`）。多实例滚动重启共享同一快照，无撕裂。
- 回退 `cache_file=/data/growthbook_features.json`（挂在持久卷）。
- 任一种：重启时 `loadFeatures()` 先读旧快照，GB 暂不可达也用旧值上岗。

**启动门禁（§4.9 第 2 层）**：
```python
async def initialize(self):
    deadline = timeout(GB_INIT_TIMEOUT)   # 如 30s
    delay = 1.0
    for attempt in range(MAX_ATTEMPTS):   # 指数退避 1,2,4,8... 上限
        try:
            await asyncio.wait_for(self._gb.load_features(), timeout=GB_LOAD_TIMEOUT)
            self._last_success_ts = now(); self._degraded = False
            return
        except Exception:
            await asyncio.sleep(delay); delay = min(delay*2, 8)
    # 全部失败：尝试从缓存 stale 上岗
    if self._has_cached_features():
        self._degraded = True; logger.warning("GB 不可达，使用缓存快照上岗")
    else:
        self._degraded = True; logger.error("GB 不可达且无缓存，安全默认模式上岗")
    # 仍标记 is_initialized=True，让进程起来；readiness 探针按 health() 暴露 degraded
```
- K8s：`readinessProbe` 指向 `PROBE_PORT=8001` 的 `/health`，仅在 `is_initialized and not (no_cache_and_degraded)` 时 200。
- Compose：`depends_on: { growthbook: { condition: service_healthy } }` + GB `healthcheck`（见 §8）。

**熔断 + 陈旧告警（§4.9 第 4 层）**：
```python
def eval_variant(...):
    if self._circuit_open:
        return self._safe_control(user_id)          # 直接安全默认，不再打 GB
    try:
        res = self._gb.eval_feature(feature_key, user_id, attributes=attributes)
        self._consecutive_failures = 0
        self._maybe_close_circuit()
        return self._to_assignment(res, feature_key)
    except Exception as e:
        self._consecutive_failures += 1
        if self._consecutive_failures >= CIRCUIT_THRESHOLD:   # 如 5
            self._circuit_open = True
            self._alert("GB circuit open, using cached/safe defaults")
        return self._safe_control(user_id)            # 第 3 层：安全默认
# 陈旧告警：health() 计算 cache_age；cache_age > cache_ttl*3 时打点告警
```

### 3.2 `Assignment` DTO 微调（保留其余字段，`experiment_service.py` 232-260）

```python
@dataclass
class Assignment:
    user_id: str
    experiment_id: str = ""          # feature_key；"" = 降级/无实验
    variant_name: str = "control"
    variant_type: VariantType = VariantType.CONTROL
    pipeline_overrides: PipelineOverrides = field(default_factory=PipelineOverrides)
    traffic_percent: float = 0.0
    bucket: int = -1                 # GB 无桶号概念，留 -1 兼容
    exp_mode: Optional[str] = None   # "experiment" | "canary"；None = control/降级/无实验

    def to_tags(self) -> List[str]:
        if not self.exp_mode:                       # 降级/无实验 → 不写 tag
            return []
        return [
            f"exp:{self.experiment_id}",
            f"variant:{self.variant_name}",
            f"exp_type:{self.exp_mode}",            # experiment / canary 区分
        ]

    def to_metadata(self) -> Dict[str, Any]:
        if not self.exp_mode:
            return {}
        return {"experiment_id": self.experiment_id, "variant": self.variant_name,
                "variant_type": self.variant_type.value, "exp_mode": self.exp_mode,
                "traffic_percent": self.traffic_percent}
```
> 改动极小：`exp_mode` 新增；`to_tags/to_metadata` 加 `exp_mode` 守卫。orchestrator 调用方式不变（329-331 / 277），但降级时自动不写 `exp:/variant:` tag（满足 scope §6 回归项）。

### 3.3 `eval_variant` → `Assignment` 映射规则

```
res = gb.eval_feature(feature_key, user_id, attributes={"domain": domain})
if res is None or not res.on or res.value is None:
    → return _safe_control()                      # 未命中 / 关 / 无值
variation_id = res.experimentResult.variationId if res.experiment else "treatment"
exp_mode = "experiment" if res.experiment else "canary"
overrides = PipelineOverrides.from_dict(res.value or {})
variant_type = CONTROL if (not res.experiment and not res.on) or variation_id=="control" else TREATMENT
return Assignment(user_id, feature_key, variation_id, variant_type, overrides,
                  traffic_percent, bucket=-1, exp_mode=exp_mode)
```
- **安全默认** `_safe_control()`：`Assignment(user_id, experiment_id="", variant_name="control", exp_mode=None, pipeline_overrides=PipelineOverrides())` → 空 overrides = 基线，不写 tag。

### 3.4 `core/growthbook_datasource.py`

```python
class GrowthBookDataSource:
    _instance = None
    def get_instance(cls) -> "GrowthBookDataSource": ...
    def initialize(self) -> None:
        """psycopg2.pool.SimpleConnectionPool（复用 pgvector_service 模式）。
        conn str = config.database_url 改为同步 scheme postgresql://..."""
    def ensure_schema(self) -> None:
        """CREATE TABLE IF NOT EXISTS gb_exposures / gb_metrics（见 §4.8）"""
    def record_exposure(self, feature_key, variation_key, user_id,
                        domain, timestamp=None) -> None:
        """INSERT gb_exposures；best-effort（吞异常、记日志、计数丢失）"""
    def record_metric(self, feature_key, variant_key, metric_name,
                      value, user_id=None, timestamp=None) -> None:
        """INSERT gb_metrics（可选，用于自定义指标上报）"""
```
- 连接串：`config.database_url`（asyncpg）→ 改写为 `postgresql://...` 供 psycopg2。与 `pgvector_service` 同源 `pgvector:5432/shop_agent`。
- GB Data Source（GB 侧只读查显著性）注册连接：**建议只读角色** `gb_ro@pgvector/shop_agent`，与写池分离（见 §7 风险）。

### 3.5 `ExperimentService` 瘦身后公开接口（签名不变，内部委托 GB）

```python
class ExperimentService:
    def assign(self, user_id, domain="ecommerce") -> Optional[Assignment]:
        """枚举候选 feature key（本地 sidecar RUNNING 列表，实验优先于 canary），
        逐个 eval_variant，返回首个 exp_mode 非 None 的 Assignment；
        全部未命中 → None（保持原语义，orchestrator 跳过）；
        GB 整体故障 → 返回 safe control Assignment(exp_mode=None)（绝不 None 于故障路径）。"""

    def create_experiment(self, exp: ExperimentDef) -> bool:
        """翻译为 GB REST：建 feature + (experiment 模式) experiment + variations(value=overrides)
        + targeting(domains/user_id) + rollout；写 SafetyGuard sidecar。
        返回是否成功。"""

    def get_experiment(self, experiment_id) -> Optional[ExperimentDef]: ...
    def list_experiments(self) -> List[ExperimentDef]: ...        # 代理 GB / sidecar
    def delete_experiment(self, experiment_id) -> bool: ...       # 代理 GB API 归档/删
    def pause_experiment / stop_experiment(self, experiment_id) -> bool:
        """代理 GB API：rollout=0 或 archive；同步 sidecar 状态。"""
    def update_status(self, experiment_id, status) -> bool: ...  # 护栏越界调用
    def force_refresh(self) -> None:                             # 内部 reload GB 缓存
        return GrowthBookClient.get_instance().refresh()

    def validate_distribution(self, experiment_id, sample_users) -> dict:
        """透传 GB 自带分布报告（feature 定义的 variations/coverage）；标记 deprecated。"""
    def evaluate_safety(self, experiment_id, metrics) -> List[SafetyGuard]:
        return self._safety_guard.evaluate(exp_from_sidecar, metrics)
```

**删除清单（scope §4.4）**：`ExperimentStore`(388-486)、`TrafficRouter`+`_fnv1a`(268-380)、`_ExperimentMetricsCollector`(713-770)、`SampleSizeCollector`/`StatisticalTest`(785-880)。预置模板(897-973)保留为 `ExperimentDef` 便捷构造器。

### 3.6 `safety_guard.py`（从 experiment_service 抽出，scope §4.4 护栏补调度）

```python
class SafetyGuardEvaluator:                    # 保留原 evaluate (experiment_service.py 494-540)
    def evaluate(self, experiment: ExperimentDef, metrics: dict) -> List[SafetyGuard]: ...
    def set_alert_callback(self, cb): ...

class SafetyGuardRegistry:                     # Redis sidecar（新增）
    def put(self, experiment_id, safety_guards, exp_mode, status): ...
    def get(self, experiment_id) -> Optional[dict]: ...
    def running_keys(self) -> List[str]: ...   # 护栏调度枚举源
    def set_status(self, experiment_id, status): ...

class SafetyMetricsProvider(Protocol):         # 指标来源抽象（解耦 Langfuse/Prometheus）
    async def snapshot(self, experiment_id, variant_name, window_seconds) -> Dict[str, float]: ...

class LangfuseSafetyMetricsProvider(SafetyMetricsProvider): ...   # escalation/sentiment/safety
class PrometheusSafetyMetricsProvider(SafetyMetricsProvider): ... # error_rate/p99_latency

class SafetyScheduler:
    def __init__(self, interval_seconds=300): ...
    async def run_loop(self):                  # asyncio loop
        # 对每个 running key:
        #   exp = registry.get(key); 对每个 variant 取 metrics 快照
        #   triggered = evaluator.evaluate(exp, metrics)
        #   if triggered: exp_service.update_status(pause/stop) + alert_callback
    def start()/stop(): ...
```
- 调度缺失已补：后台 `asyncio` loop（interval≈护栏 `window_seconds`，默认 300s）。
- 指标来源：删除内存聚合，改实时查 Langfuse/Prometheus（接口 `SafetyMetricsProvider`）。
- 越界动作：`set_alert_callback`（钉钉/企微）+ 调 GB API `update_status(pause/stop)`。

### 3.7 版本与兼容策略
- GB feature key = `experiment.id`（含 `exp_`/`canary_` 前缀，见 §9）。
- SDK features 缓存 `cache_ttl=30s`；REST CRUD 即时生效（GB 推/拉，SDK 下次刷新可见）。
- `Assignment`/`ExperimentDef`/`PipelineOverrides` 等 DTO 二进制/JSON 兼容，routers/schemas/orchestrator 零改动。

---

## 4. 两种模式区分设计（scope §2.1）

| 维度 | Feature rollout（金丝雀/开关） | Experiment（A/B 显著性） |
|---|---|---|
| GB 建模 | Feature + `rolloutPercentage`（整体开量）；单 value=overrides | Feature + **GB Experiment 对象**（control/treatment variations，value=overrides）+ Data Source |
| 显著性 | 无（仅 Langfuse/面板手动对比） | 有（GB 原生贝叶斯/频率派，跑 Data Source SQL） |
| `exp_mode` | `"canary"` | `"experiment"` |
| `eval_variant` 路径 | `gb.eval_feature` 命中 rollout → treatment；否则 control | `gb.eval_feature` 命中 experiment → variation；否则 control |
| 曝光上报 | **不写** Data Source（仅 Langfuse tag 供人工对比） | **写** `gb_exposures`（Data Source 显著性前提） |
| 代码分支 | `res.experiment is None` → canary | `res.experiment` 存在 → experiment |

**模式判定（创建时）**：`create_experiment` 依据 `ExperimentCreateRequest.kind`（建议新增可选字段，默认 `"experiment"`；若 `kind=="canary"` 或单 treatment variant 标注 canary → canary 模式）。见 §5 / §10。

**数据流差异图**：
```
canary:  assign → eval_variant(canary_x) → on? treatment : control
                           │ (exp_mode=canary)
                           └─ to_tags 仅 Langfuse 分段，不写 gb_exposures

experiment: assign → eval_variant(exp_x) → variation
                           │ (exp_mode=experiment)
                           └─ track_exposure → gb.track → 回调 → gb_exposures → GB 显著性
```

---

## 5. 关键 REST / 数据流时序

### 5.1 assign 调用链（router → service → client → GB → Langfuse tag）
```
routers.agent_chat (211-221)
  exp_service.assign(user_id, domain)
    └─ ExperimentService.assign
         for key in registry.running_keys():           # 实验优先, 后 canary
           a = client.eval_variant(key, user_id, {domain})
           if a.exp_mode: return a
         return None                                    # 原语义: 无命中
  → chatagent_service.chat_with_agent(request, experiment_assignment=a)
      orchestrator.chat_with_agent (319)
        if a: exp_tags += a.to_tags(); exp_metadata["experiment"]=a.to_metadata()
        create_langfuse_handler(tags=exp_tags, metadata=exp_metadata)
        _inject_response_metadata → response.experiment_group = a.variant_name (277)
  # 注意: 曝光上报在 assign 内部已触发（见 5.2），不阻塞主应答
```
> `assign` 内部在得到 `exp_mode=="experiment"` 的 Assignment 后**立即** `client.track_exposure(...)`（best-effort，异步/线程内快速返回）。

### 5.2 曝光上报链（Experiment 模式）
```
GrowthBookClient.eval_variant → Assignment(exp_mode="experiment")
  → GrowthBookClient.track_exposure(feature_key, user_id, variation_key, domain)
      gb.track(experiment_key, user_id, value, attributes)   # SDK
        └─ on_exposure 回调 (registered at init)
            GrowthBookDataSource.record_exposure(
              feature_key, variation_key, user_id, domain, now())
              └─ INSERT gb_exposures (psycopg2 池, best-effort)
  GB 后台定时在 Data Source 上跑 SQL → 实验结果页显示显著性
```
- canary 模式：`track_exposure` **不调用**（无 experimentResult，且本就无显著性诉求）。

### 5.3 护栏调度链（后台）
```
SafetyScheduler.run_loop (every ~300s):
  for key in registry.running_keys():
    exp = registry.get(key)                       # 含 safety_guards
    for variant in exp.variants:
      metrics = await metrics_provider.snapshot(key, variant.name, window)
        # LangfuseSafetyMetricsProvider / PrometheusSafetyMetricsProvider
      triggered = evaluator.evaluate(exp, metrics)
      if triggered:
        exp_service.update_status(key, guard.action)   # → GB REST pause/stop
        evaluator.alert_callback(triggered)            # 钉钉/企微
```

---

## 6. 失败模式与韧性设计（scope §4.9，4 层防御落地）

| 层 | 机制 | 落地位置 | 行为 |
|---|---|---|---|
| 1 缓存持久化 | `cacheConnection`(Redis) 或 `cache_file`(磁盘) | `GrowthBookClient._build_cache()` | 重启先加载旧快照，GB 不可达也用旧值上岗 |
| 2 启动门禁 | 阻塞 `loadFeatures`（超时+指数退避）；readiness 探针 | `initialize()` + `/health`(PROBE_PORT) + compose `depends_on: service_healthy` | "带 flags 上岗 or 不上岗"，杜绝空缓存上岗 |
| 3 安全默认变体 | `assign`/`eval_variant` 失败→ safe control Assignment | `_safe_control()` | 绝不返回 None；空 overrides=基线；`exp_mode=None`→不写 tag |
| 4 熔断+陈旧告警 | 连续失败→circuit open（停打 GB，用缓存）；cache_age>TTL×3 告警 | `eval_variant` 熔断计数 + `health()` | 静默陈旧被发现；恢复自动退出熔断 |

**场景验证（scope §4.9 问题场景）**：GB 故障期间 shop-agent 重启 → ① 从 Redis/磁盘缓存加载旧 features（层1）→ ② 即便 `loadFeatures` 全失败，`initialize` 仍用缓存 stale 上岗（层2 不阻塞进程，仅 readiness 标记 degraded）→ ③ 请求 `assign` 命中缓存变体或安全默认（层3）→ ④ 熔断打开、陈旧告警触发（层4）。主应答零影响。

---

## 7. Flag 生命周期治理（scope §9，部分实现）

已从"纯规划"推进为**部分实现**（见下方状态标记）：

1. **命名前缀强制**（§9.1）✅ **已实现**：`growthbook_client.validate_flag_prefix` + `create_experiment` 内 fail-loud 校验，非法前缀直接拒绝创建（sidecar key 与 GB feature key 必须一致，故不自动加前缀，而是返回 `False` 仅本地 sidecar 记录）。前缀表 `exp_`/`canary_`/`switch_`/`perm_`。
2. **过期元数据** ✅ **已实现**：`create_experiment` 创建 feature 时写入 `tags`：`exp_type:<kind>`、`exp_id:<key>`，并（仅 `exp_`/`canary_` 且带 `expected_end` 时）写 `expected_end_date:<YYYY-MM-DD>`。`ExperimentDef`/`ExperimentCreateRequest` 新增 `expected_end` 字段；`_gb_feature_to_def` 回读该 tag。
3. **陈旧检测脚本**（§9.3）✅ **已实现**：`scripts/audit_stale_flags.py`（纯标准库，CI 可跑）。`GET /api/features` 后检测三类问题——超 `expected_end_date`(HIGH) / 长期 100% 单有效变体(MEDIUM/Low) / 代码无 `eval_variant("<key>")` 引用的孤儿 flag(MEDIUM)；`--strict` 时 MEDIUM/HIGH 令 exit=1 作为 CI 门禁。
4. **Definition of Done**（§9.2③）⬜ **未实现（后续）**：胜者 overrides 固化进 `AgentConfig` → 删 `if variant==` 分支 → `delete_experiment` 删 GB feature。`delete_experiment` 接口已具备能力，但"固化进 AgentConfig"的链路未接。
5. **CI 兜底**（§9.4）⬜ **未实现（后续）**：删 flag 但代码仍有引用 → lint 报错。目前由审计脚本第 3 项（孤儿检测）覆盖运行时维度，`--strict` 可当 CI 门禁，但编译期 lint 兜底尚未接入流水线。

> 状态锚点（2026-10-09）：§9.1/§9.2/§9.3 已落地；§9.2③ 与 §9.4 留待后续编排（与 scope §9 一致，原定"后续"）。

---

## 8. docker-compose 改动

### 8.1 新增服务（opt-in，`profiles: ["experiments"]`）

```yaml
  # ── GrowthBook 自托管（实验分配后端 + 分析看板）──
  growthbook:
    image: growthbook/growthbook:${GROWTHBOOK_VERSION:-v3.4.0}   # 稳定 tag，实现时取最新 v3 稳定版
    container_name: growthbook
    profiles: ["experiments"]
    ports: ["3100:3100"]
    environment:
      API_HOST: http://growthbook:3100
      GB_ENCRYPTION_KEY: ${GB_ENCRYPTION_KEY}          # secret/env 注入，不落代码
      JWT_SECRET: ${GB_JWT_SECRET}
      MONGODB_URI: mongodb://mongo:27017/growthbook
    depends_on:
      mongo: { condition: service_healthy }
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3100/healthcheck"]
      interval: 30s
      timeout: 10s
      retries: 5
    volumes:
      - growthbook_data:/home/node/app/data

  mongo:
    image: mongo:${MONGO_VERSION:-7}
    container_name: growthbook-mongo
    profiles: ["experiments"]
    environment:
      MONGO_INITDB_DATABASE: growthbook
    volumes:
      - mongo_data:/data/db
    healthcheck:
      test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
      interval: 30s
      timeout: 10s
      retries: 5
```

### 8.2 shop-agent 增补（在现有 `environment` 块后追加；`env_file: .env` 已覆盖）

```yaml
  shop-agent:
    # ... 现有 build/env/depends_on 保持不变 ...
    environment:
      # ── GrowthBook（experiments profile 下启用；GROWTHBOOK_ENABLED=false 可整体关闭）──
      GROWTHBOOK_ENABLED: "${GROWTHBOOK_ENABLED:-true}"
      GROWTHBOOK_API_HOST: "http://growthbook:3100"
      GROWTHBOOK_CLIENT_KEY: "${GROWTHBOOK_CLIENT_KEY}"        # 只读 SDK key
      GROWTHBOOK_DECRYPTION_KEY: "${GROWTHBOOK_DECRYPTION_KEY}"
      GROWTHBOOK_API_KEY: "${GROWTHBOOK_API_KEY}"              # 服务端 REST，绝不进前端
      GROWTHBOOK_CACHE_TTL: "30"
      # Data Source 复用现有 pgvector Postgres（shop_agent 库）
      GROWTHBOOK_DATASOURCE_URL: "postgresql://gb_ro:${POSTGRES_PASSWORD}@pgvector:5432/shop_agent"
    depends_on:
      # 仅在 experiments profile 激活时 growthbook 存在于编排图；
      # 不使用 experiments profile 时本 depends_on 被 compose 忽略（已知限制，见风险）
      growthbook: { condition: service_healthy }
```

> 变量（如 `GB_ENCRYPTION_KEY`/`GROWTHBOOK_CLIENT_KEY`/`GROWTHBOOK_API_KEY`）注入 `.env`（已存在，注释占位），不写进仓库。`.env.example` 增补同名空占位。

### 8.3 volumes
```yaml
volumes:
  growthbook_data:
  mongo_data:
```

### 8.4 已知 compose 限制与建议
- `depends_on` 引用 `profiles:["experiments"]` 的服务：运行时不带该 profile 则 GB 被过滤，`shop-agent` 的 `depends_on` 实际不生效（现代 Compose 忽略被 profile 排除的 target）。若需严格保证"无 experiments 时完全不依赖"，可采用 **override 文件** `docker-compose.experiments.yml` 叠加 GB 依赖与 env。本设计采用 scope §4.1 原意（直接加 depends_on + profile），并依赖 §4.9 启动门禁的 stale 上岗兜底，故即使依赖未满足进程也能起来（degraded）。

### 8.5 启动门禁接线
`apps/shop-agent` 的 FastAPI `lifespan`（或现有启动钩子）调用 `await GrowthBookClient.get_instance().initialize()`；`PROBE_PORT=8001` 的 readiness 端点返回 `health()`。需在 `apps/shop-agent/src/main.py`（或等价入口）增补 lifespan 与探针路由——**此入口文件本次未读取，实现阶段核对**（见 §11 缺口⑥）。

---

## 9. 技术选型

| 维度 | 选型 | 候选 | 放弃理由 |
|---|---|---|---|
| 实验后端 | GrowthBook 自托管 | LaunchDarkly / 自研 | scope 已锁定；LD 需出网/商业；自研已证实难维护（scope 目标） |
| SDK | `growthbook` Python SDK 单例 | 自研 HTTP 拉 features | 复用官方 eval/track/cache，避免重造轮子 |
| 应用存储 | MongoDB（GB 自托管必需） | Postgres（新版支持，但未验证） | scope §7 明确 mongo；降低不确定性 |
| 显著性 Data Source | 复用现有 `pgvector` Postgres | 新增独立 PG | 已用 PG，避免新基础设施；数据不出内网 |
| 缓存持久化 | Redis `cacheConnection`（首选）/ 磁盘 `cache_file`（回退） | 纯内存 | 重启空缓存是 scope §4.9 红线场景，必须持久化 |
| 曝光写入 | psycopg2 `SimpleConnectionPool`（复用 `pgvector_service` 模式） | asyncpg | 与现有同步写路径一致；曝光为 best-effort 快速返回 |
| 指标来源 | Langfuse + Prometheus（`SafetyMetricsProvider` 接口） | 内存聚合（原有） | scope §4.4 明确要求实时查，删除内存聚合 |
| 护栏调度 | asyncio loop（~300s） | APScheduler / celery | 单体进程内足够，零额外依赖；APScheduler 备选 |
| 部署 | docker-compose `profiles:["experiments"]` | 常驻服务 | opt-in，不强制所有环境起 GB/mongo |

---

## 10. Out of scope（明确边界）

- **pipeline_overrides 注入 apply 层**（已知缺口，scope §5）：本任务只做"分配生效 + 上报 + 分析"。`Assignment.pipeline_overrides` 已产出并传到 orchestrator，但**真正喂给 reranker/llm/retrieval/prompt 的 apply 逻辑不在本次**。上线前须告知："实验已分配但参数尚未生效"。
- **多实验 overrides 合并**：原 `assign` 返回首个命中；本次保留该语义（首个非 control Assignment），**不**做多 experiment 的 overrides 合并。多实验并发是后续考量。
- **替换 Langfuse / Prometheus / Grafana**：仅作为分析/指标源保留。
- **现有实体槽位 Redis 逻辑**：与实验无关，不动。
- **§9 Flag 生命周期治理落地**：命名前缀强制 + `expected_end_date` tag 写入 + 陈旧检测脚本 `scripts/audit_stale_flags.py` 已实现（见 §7 状态锚点）。`delete_experiment` 已具备删 GB feature 能力；DoD 固化进 `AgentConfig`（§9.2③）与编译期 CI lint 兜底（§9.4）仍留待后续编排。
- **GB Data Source 的 metric SQL 与注册 UI 操作**：本设计给表结构与连接建议，GB 侧 metric 定义走 UI/REST 人工配置（§11 缺口④）。

---

## 11. 决策 Why（核心取舍记录）

1. **为何复用现有 Postgres 作 Data Source（而非新增 PG）**：shop-agent 已用 `pgvector` Postgres（`shop_agent` 库）。新增 PG 实例违反"最小基础设施"且增加出网/运维面。代价：GB 对 `shop_agent` 库有只读访问（用 `gb_ro` 角色收敛权限缓解）。→ 若未来需强隔离，再迁移独立库。
2. **为何 `assign` 保留"枚举候选 key + 首个命中"语义**：router 契约为 `assign(user_id, domain)` 无 key（scope §4.5 最小化改动），且 orchestrator 单 Assignment 消费。枚举来源从"本地 Redis 实验列表"改为"本地 SafetyGuard sidecar 的 RUNNING key 列表"。代价：O(n) 次本地 eval（n=实验数，量级小，可接受）；多实验合并不做。
3. **为何新增 `exp_mode` 字段而非复用 `variant_type` 区分两种模式**：`variant_type`(control/treatment) 在两种模式都存在，无法表达"这是 canary 还是 experiment"；`exp_mode` 让 `to_tags` 能发 `exp_type:canary|experiment`，且让 `track_exposure` 只在 experiment 模式写 Data Source。代价：Assignment 多一字段（向后兼容，默认 None）。
4. **为何 `create_experiment` 加可选 `kind` 字段（已拍板，scope §4.7 从"可不加"升级为"加"）**：§2.1 明确"两种模式不可混用"，必须有一个信号决定建 GB Feature-only（canary，kind="canary"）还是 Feature+Experiment（ab，kind="experiment" 默认）。已决定采纳：schema 新增可选 `kind`（默认 `"experiment"`），`create_experiment` 据 `kind` 决定建 GB Feature-only（canary，不写 Data Source）还是 Feature+Experiment（ab，写 `gb_exposures`）。零 schema 改动降级方案作废。
5. **为何护栏指标走 `SafetyMetricsProvider` 接口而非直连 Langfuse/Prometheus**：解耦，便于单测与将来换源；首版实现 Langfuse + Prometheus 两个 provider，具体查询 SQL/API 为 Phase-2 细节。
6. **缓存首选 Redis `cacheConnection` 而非 `cache_file`**：多实例滚动重启共享一份快照，无撕裂；`cache_file` 仅作单实例/无 Redis 时的回退。
7. **何时应推翻本决策**：① 若 GB 自托管运维成本过高且团队接受商业 LD → 可整体替换（架构边界已隔离在 `growthbook_client`）；② 若实验规模大到 O(n) eval 不可接受 → 改为 router 显式传 `experiment_id` 精确 eval；③ 若需强数据隔离 → Data Source 迁独立 PG。

---

## 12. 实现建议

### 12.1 分阶段
1. **基础设施**：`docker-compose.yml` 加 `growthbook`+`mongo`；`.env` 加 GB 密钥占位；shop-agent 补 GB env；`requirements.txt` 加 `growthbook`；`config.py` 加配置块。
2. **SDK 单例**：`core/growthbook_client.py`（eval_variant/track_exposure/refresh/health/initialize 门禁/熔断/缓存）。
3. **Data Source**：`core/growthbook_datasource.py` + `gb_exposures`/`gb_metrics` 建表；GB UI 注册 Data Source + metric（手动）。
4. **瘦身 experiment_service**：删 `ExperimentStore`/`TrafficRouter`/`_ExperimentMetricsCollector`/`SampleSize`/`StatisticalTest`；`assign`/`create_experiment`/CRUD 委托 GB；抽 `safety_guard.py`（Registry/Scheduler/Provider）。
5. **接线**：routers 透传 `attributes={"domain"}` + `force_refresh` 改调；orchestrator/schemas **零改动**；main.py lifespan 启动门禁 + 探针 + scheduler。
6. **验证**：scope §6 验收清单逐条（开关/金丝雀/复杂定向/分析/数据源/面板/回归/护栏/韧性）。

### 12.2 关键风险与缓解
| 风险 | 缓解 |
|---|---|
| GB SDK `eval_feature` 的 `attributes` 传参形态（user 对象 vs dict） | 实现阶段核对 SDK 签名（§11 缺口①）；设计已用 dict，必要时包 `User` |
| `cacheConnection` / `cache_file` 在 Python SDK 的确切类与配置 | 首选 `RedisConnection`；回退 `cache_file`；实现校验（缺口②） |
| `create_experiment` → GB REST 字段映射（feature/experiment/variation 创建顺序、value 序列化） | 实现阶段对照 GB OpenAPI（缺口③）；先手工在 UI 建一个 feature 对齐 |
| GB REST pause 接口权限/server key 范围 | `GROWTHBOOK_API_KEY` 仅服务端；缺口④核对 endpoint |
| mongo 数据卷持久化 + GB 初始化 seeding | compose volume + 首次 `docker exec` 建初始 org/api key（缺口⑤） |
| Data Source 凭据安全 | `gb_ro` 只读角色；secret 注入；不落代码 |
| compose `depends_on` + profile 交互 | 启动门禁 stale 上岗兜底；必要时换 override 文件 |
| `main.py` 入口未读取，lifespan/探针接线未知 | 实现阶段核对（缺口⑥） |

### 12.3 与现有系统衔接 / 迁移
- **Redis 实验配置**：删除 `ExperimentStore` 后，原有 Redis 里的 `shop_agent:experiments:*` 配置不再被读；旧实验需在 GB UI 重建（或写一次性迁移脚本：读 Redis → 调 `create_experiment`）。**迁移脚本不在本 scope**，列为后续。
- **Langfuse 打标**：完全保留（orchestrator 零改动），仅 `Assignment` 新增 `exp_mode` 守卫，降级时不写 tag。
- **实体槽位 Redis**：`REDIS_HOST` 等保留（config 24-31），与实验解耦，guard sidecar 复用同一 Redis。

---

## 13. 待澄清 / 待决（实现阶段核对，对应 scope §8 接口缺口）

1. **（§8-①）** GB Python SDK `eval_feature(feature_key, user, attributes)` 的 `attributes` 究竟传 `dict` 还是 `User` 对象；`on_exposure` 回调注册方式（`set_exposure_logging_callback` vs 构造参数）。
2. **（§8-②）** `cacheConnection` 确切类（`growthbook.RedisConnection`?）与 `cache_file` 路径配置；Redis 不可用时回退策略。
3. **（§8-③）** `create_experiment` REST 字段映射：feature / experiment / variation 创建顺序、value(JSON) 序列化、targeting conditions 结构、rollout 字段名。建议先手工在 GB UI 建 1 个 feature 对齐。
4. **（§8-④）** `evaluate_safety` 调 GB 暂停的具体 endpoint（`PATCH /api/features/:id` 的 `environments.production.rollout=0` 或 experiment archive）与 `GROWTHBOOK_API_KEY` 权限范围。
5. **（§8-⑤）** mongo 数据卷持久化 + GB 首次初始化（org/api key seeding）；`GROWTHBOOK_CLIENT_KEY`/`DECRYPTION_KEY` 由 GB 首次启动生成后回填 `.env`。
6. **（§8-⑥，新增）** `apps/shop-agent` 的 FastAPI 入口（`main.py`/`lifespan`/探针路由）未在本轮读取，启动门禁与 `/health` 探针接线需实现阶段核对。
7. **（已拍板）** `ExperimentCreateRequest` 新增可选 `kind` 字段（默认 `"experiment"`，可取值 `experiment`/`canary`），用于 `create_experiment` 显式区分两种模式（见 §5/§11-④）。此为非待决项，已实现决策。
8. **（建议）** `pgvector` Postgres 的 `gb_ro` 只读角色创建方式（compose init sql 或手工），及其与现有 `POSTGRES_PASSWORD` 注入的衔接。
