# Scope: shop-agent 接入 GrowthBook（A/B 实验 / 金丝雀 / 复杂功能开关 / 分析看板）

> 阶段：设计定稿（已确认用 GrowthBook，自托管、数据不出内网）
> 关联：本任务只替换实验「分配后端 + UI + 分析」，与「pipeline_overrides 注入」为独立 task。

## 1. 目标
用 GrowthBook 替换现有自研 `experiment_service` 的分配/存储逻辑，覆盖 5 项诉求：
1. **功能开关**（feature toggle）
2. **金丝雀**（rollout % 渐进 1%→100%，UI 可调、免重启）
3. **复杂功能开关**（按 domain / user_id / 自定义属性的 targeting）
4. **结果分析**（贝叶斯/频率派显著性）
5. **面板显示**（GrowthBook Web UI 管理 flag + 实验结果；Langfuse 做 trace 级下钻）

## 2. 映射模型（每个实验 = 一个 GrowthBook Feature/Experiment）
- `experiment.id` → `featureKey`
- `variants[].pipeline_overrides` → variant 的 JSON `value`
- `variants[].traffic_percent` → 各 variant 在实验内的流量占比
- 金丝雀 → feature **rollout percentage**（整体开量）
- 复杂定向 → GrowthBook **targeting conditions**（`domain==ecommerce`、`user_id` 白名单、自定义属性）
- 运行时分配 → `gb.eval_feature(featureKey, user_id, attributes={domain})` → `{variant_name, pipeline_overrides}`
- 曝光日志 → `track_exposure()`。注意：自托管 GB 的 `gb.track()` 仅是**本地回调**，不会把数据发往 GB 后端；
  真正的显著性来自 **Data Source**（见 §4.8）：`track` 回调需把曝光/指标事件写入 Data Source 表，
  GB 通过在该表上跑 SQL 计算贝叶斯/频率派显著性。
- 下钻 → Langfuse `exp:`/`variant:`/`exp_type:` tag（复用现有，保持不变）

### 2.1 两种接入模式（不可混用）
- **金丝雀 / 功能开关（Feature rollout 模式）**：用 feature 的 `rolloutPercentage` 渐进 1%→100%。
  该模式**无内置 per-variant 显著性**，只能靠 Langfuse/面板手动对比业务指标。
- **A/B 实验（Experiment 模式）**：在 GB 建 `Experiment` 对象（含 control/treatment variant）+ 配置 Data Source，
  才能用 GB 原生显著性分析。需 `track_exposure` 把曝光落 Data Source。
- 建议：纯灰度上线用模式一；需要"哪个变体更好"的量化结论用模式二。

## 3. 现状接入点（改造依据，行号以仓库为准）
- `routers.py:212-221` A/B 分配块：`exp_service.assign(user_id, domain)` → `experiment_assignment`
- `routers.py:537-635` `POST /experiments`：构建 `ExperimentDef/VariantDef/PipelineOverrides` → `exp_service.create_experiment`
- `routers.py:638-764` 暂停/校验分流/列出/详情/删除/刷新 实验 API
- `orchestrator.py:319-385`：`experiment_assignment.to_tags()/to_metadata()` 打 Langfuse tag（**保留**）
- `orchestrator.py:268-278`：`response.experiment_group = variant_name`（**保留**）
- `experiment_service.py:548-` `ExperimentService`：Redis 存配置 + FNV-1a 分桶（**替换为 GB**）
- `schemas.py:507-542`：`ExperimentCreateRequest/PauseRequest/ValidateRequest/Assignment`（**字段基本保留**）
- `config.py:24-31`：现有 Redis 配置保留（实体槽位用，与实验解耦）

## 4. 修改清单
### 4.1 基础设施 `docker-compose.yml`（新增，opt-in）
- 新增 `growthbook` 服务（镜像 `growthbook/growthbook:<稳定版本号，如 v3.x.x>`，UI 端口 `3100`），`profiles:["experiments"]`（不锁 commit hash，用稳定 tag 便于复现与升级）
- 新增 `mongo` 服务（GB 自托管存储，`MONGODB_URI` 注入 GB）
- 注入 GB 环境变量：`GB_ENCRYPTION_KEY`、`API_HOST=http://growthbook:3100`
- `shop-agent` 容器新增：`GROWTHBOOK_API_HOST / GROWTHBOOK_CLIENT_KEY / GROWTHBOOK_DECRYPTION_KEY / GROWTHBOOK_API_KEY / GROWTHBOOK_CACHE_TTL`
- 依赖：`shop-agent` 增加 `depends_on: growthbook`（仅 experiments profile 下）

### 4.2 `core/config.py`
新增配置块（字段名以实现为准）：
```
GROWTHBOOK_API_HOST: str = "http://growthbook:3100"
GROWTHBOOK_CLIENT_KEY: str = ""
GROWTHBOOK_DECRYPTION_KEY: str = ""
GROWTHBOOK_API_KEY: str = ""   # server-side，用于创建/暂停实验（REST 变更）
GROWTHBOOK_CACHE_TTL: int = 30 # SDK feature 缓存刷新秒
```

### 4.3 新增 `core/growthbook_client.py`（单例封装）
- `GrowthBookClient.get_instance()`：基于 `config` 初始化 `growthbook.GrowthBook`
- `eval_variant(feature_key, user_id, attributes) -> Assignment`：调 `gb.eval_feature`，把 `value` 映射为 `PipelineOverrides`，产出 `Assignment`（experiment_id/variant_name/variant_type/pipeline_overrides/tags）
- `track_exposure(feature_key, user_id, variation_key)`：调 `gb.track`，注册回调将曝光事件（feature_key/variation_key/user_id/domain/时间戳）写入 **Data Source 表**（§4.8）。该回调是 GB 显著性分析的**唯一数据入口**，best-effort、失败不影响主流程。
- `refresh()`：reload SDK feature 缓存（对接 `POST /experiments/refresh`）
- `is_initialized` / `health`：健康检查（见 §4.9 启动门禁）
- **缓存持久化**：`gb.loadFeatures()` 使用 `cacheConnection`（Redis）或 `cacheFile`（磁盘）持久化上次成功的 features 快照，供重启时先加载旧值（见 §4.9）
- **启动门禁**：首次初始化时**阻塞等待**首次 `loadFeatures` 成功（带超时 + 指数退避重试），就绪前不对外宣称 ready（见 §4.9）

### 4.4 `experiment_service.py`（改造，保留公开接口）
- `assign(user_id, domain)`：改委托 `growthbook_client.eval_variant(...)`；**失败/无命中时返回安全默认的 control `Assignment`**（含完整 safe overrides），**绝不返回 `None`**（见 §4.9）
- `create_experiment(exp: ExperimentDef)`：翻译为 GrowthBook REST API 调用（建 feature + experiment + variant values=overrides + rollout + targeting）
- `get_experiment / list_experiments / delete_experiment`：代理 GB API
- `update_status`（pause/stop）：代理 GB API 置 rollout=0 或 archive
- **安全护栏（保留，需补调度与指标来源）**：
  - 方法名对齐代码：`SafetyGuardEvaluator.evaluate(experiment, metrics)`（前文 `evaluate_safety` 为别名，落地以 `experiment_service.py` 实际方法名为准）。
  - 指标来源：删除内存聚合的 `_ExperimentMetricsCollector`（`:713-770`），改为从 **Langfuse/Prometheus/Redis** 实时取 `escalation_rate/error_rate/p99_latency_ms/sentiment_negative/safety_failed_rate` 快照，组装为 `evaluate` 所需的 `metrics` dict。
  - **调度缺失（必须补）**：当前无人周期性调用 `evaluate`。新增后台定时任务（APScheduler 或 asyncio loop，间隔约等于护栏 `window_seconds`，默认 300s）轮询各 RUNNING 实验 → 取指标 → `evaluate` → 越界则 `update_status(pause/stop)` 代理 GB API。
  - 越界动作：`set_alert_callback` 发钉钉/企微通知；同时调 GB API 将该 feature rollout 置 0 / archive。
- `validate_distribution`：改透传 GB 自带分布报告（或标注 deprecated）
- `force_refresh`：改调 `growthbook_client.refresh()`
- **删除清单**：
  - `ExperimentStore`（`:388-486`，Redis 配置存储 + 30s 热加载）→ 配置归 GB(mongo)
  - `TrafficRouter` / `_fnv1a_bucket`（`:268-380`，分桶）→ 分配与金丝雀归 GB
  - `_ExperimentMetricsCollector`（`:713-770`，内存聚合）→ 指标改实时查 Langfuse/Prometheus
  - `SampleSizeCalculator` / `StatisticalTest`（`:785-880`，手搓 Z 检验）→ 显著性由 GB 原生承担，可废弃
  - 预置模板（`:897-958`）：迁移为 **GB seed**（启动时调 REST 建 feature/experiment）或保留为 `ExperimentDef` 便捷构造器（二选一，默认保留为构造器）
- **保留 DTO**：`PipelineOverrides / VariantDef / ExperimentDef / Assignment / SafetyGuard` 及其 `from_dict/to_dict/to_tags/to_metadata`，确保 `routers.py`/`schemas.py`/`orchestrator.py` 零改动

### 4.5 `routers.py`（最小化改动）
- `:212-221` 分配块：保持 `exp_service.assign(user_id, domain)`；仅补 `attributes={"domain": request.domain}` 透传给 GB targeting（接口不变）
- `:537-635` `create_experiment`：内部改走 `exp_service.create_experiment` → GB API；请求/响应 schema 不变
- `:755-764` `force_refresh`：改调 `exp_service.force_refresh()`（内部 reload GB 缓存）
- 其余实验 API（暂停/校验/列出/详情/删除）：保持，内部代理 GB

### 4.6 `orchestrator.py`（无改动）
`to_tags()/to_metadata()` 与 `experiment_group` 赋值逻辑完全保留，仅 `experiment_assignment` 数据来源切换为 GB。

### 4.7 `schemas.py`（基本不动）
`ExperimentCreateRequest` 已含 `variants[].pipeline_overrides`，与 GB variant value 直接对应。如需更细 targeting，后续可加 `targeting`/`rollout` 可选字段（本 scope 内不强制）。

### 4.8 新增 `core/growthbook_datasource.py` + 数据源接入（显著性前提）
自托管 GB 的显著性**必须**有 Data Source，否则"结果分析"诉求无法满足：
- **复用现有 Postgres**（shop-agent 若已用 PG）作为 GB Data Source；否则新增 `postgres` 服务（与 mongo 并列）。
- 新增曝光/指标表（如 `gb_exposures(feature_key, variation_key, user_id, domain, timestamp)`、`gb_metrics(...)`），由 `track_exposure` 回调写入。
- `growthbook_datasource.py` 封装：连接管理 + `record_exposure()` + `record_metric()`。
- GB 侧：通过 UI/REST 注册 Data Source（连接串 + 表名）、定义 metric（SQL，对应 escalation_rate/error_rate 等）。
- 风险：Data Source 凭据走 secret，不落代码；表写入失败 best-effort 降级（仅丢失该次曝光，不影响应答）。

### 4.9 失败模式与韧性（针对「GB 故障 + shop-agent 重启」场景）
问题场景：GB 故障期间 shop-agent 重启 → 新实例缓存为空且拉不到 features → 拿不到功能开关，导致行为异常。防御分 4 层：
1. **缓存持久化**：GB SDK 用 `cacheConnection`(Redis，多实例共享) 或 `cacheFile`(磁盘) 持久化上次成功的 features 快照。重启时**先加载旧快照**，GB 暂不可达也用旧值继续服务，避免"空缓存上岗"。多实例滚动重启时配置一致、无撕裂。
2. **启动门禁（readiness gate）**：应用启动**阻塞等待**首次 `loadFeatures` 成功（超时 + 指数退避重试），未就绪前不标 ready；docker-compose 用 `depends_on: { condition: service_healthy }` + GB `healthcheck`，K8s 用 `readinessProbe` 卡流量。效果：新服务"要么带 flags 上岗、要么不上岗"，而非带空缓存上岗。
3. **安全默认变体**：`assign` 失败/无命中时返回**预定义 control `Assignment`**（含完整 safe overrides），代码里每个 flag 声明 `default_variant=control` + `safe_overrides`。绝不让 `None` 泄漏到上层。
4. **熔断 + 陈旧告警**：连续 N 次 GB 调用失败 → 进入 degraded 模式（持续用缓存、停止打 GB、触发告警），恢复后自动退出；监控「GB 不可达 / 缓存年龄超阈值（如 > TTL×3）/ 降级模式触发」并告警，避免静默运行旧配置。

## 5. 范围边界
### In scope
- 替换实验分配后端为 GrowthBook；金丝雀、复杂定向、实验看板、显著性分析上线
- 自托管部署（docker-compose + mongo），数据不出内网
- 保留现有 REST API 形态与 Langfuse 打标（向后兼容）

### Out of scope（独立 task，不在本 scope）
- **pipeline_overrides 注入 apply 层**：把 override 真正喂给 reranker/llm/retrieval/prompt（已知缺口，需单独实现；GB 接入后实验仍"分配了但没生效"）
- 替换 Langfuse / Prometheus / Grafana（仅作为分析/指标源保留）
- 现有实体槽位 Redis 逻辑（与实验无关）

## 6. 验收清单（对应 5 项诉求）
- [ ] **开关**：`POST /experiments` 创建 feature；GB UI 可见开关；`assign` 返回 variant
- [ ] **金丝雀**：GB UI 把某实验 rollout 设 5% → 压测验证仅 ~5% 用户命中 treatment；可调至 100% 免重启
- [ ] **复杂开关**：配置 targeting（`domain==ecommerce` 或 `user_id` 白名单）→ 验证非目标用户走 control
- [ ] **分析**：`track_exposure` 回调把曝光写入 Data Source 表；GB 实验结果页（Experiment 模式）显示贝叶斯/频率派显著性；Langfuse 按 `variant:` 分段可对比
- [ ] **数据源**：GB 成功注册 Postgres Data Source 并读到曝光表；metric（SQL）能查出结果
- [ ] **面板**：浏览器开 GB UI(3100) 管理 flag + 看实验结果；Langfuse UI 下钻 trace
- [ ] **回归**：`routers.py`/`orchestrator.py`/`schemas.py` 公开接口与响应结构不变；主流程在 GB 不可用/无命中时降级为无实验（不写 `exp:/variant:` tag、不报错）
- [ ] **安全护栏**：`SafetyGuardEvaluator.evaluate` 越界仍能暂停实验（走 GB API）
- [ ] **韧性**：GB 故障期间重启 shop-agent，仍能加载上次缓存并正常服务；GB 不可达时 `assign` 返回 control（不返回 None、不报错）；熔断/降级/陈旧缓存触发告警

## 7. 关键约束 / 风险
- **自托管依赖**：GrowthBook 自托管需 MongoDB（或确认新版支持的 Postgres）作应用存储，属新增基础设施；**另需一个 Data Source（建议复用现有 Postgres）** 才能算显著性（见 §4.8）
- **加密 key**：`GROWTHBOOK_DECRYPTION_KEY` 用于解密 feature 定义，须通过 secret/env 注入，不落代码
- **曝光日志**：未 `track_exposure` 则 GB 无法算显著性——`assign` 命中实验后必须触发曝光上报（best-effort，失败不影响主流程）
- **降级**：GB 初始化/SDK 调用失败 → `assign` 返回**安全默认 control**（绝不返回 `None`）→ 主应答零影响（红线）。详见 §4.9
- **API key 分级**：`GROWTHBOOK_CLIENT_KEY`（SDK 只读评估，拉 features.json）进 shop-agent 运行时；`GROWTHBOOK_API_KEY`（REST 变更，建/暂停实验）仅服务端用，绝不可暴露到前端或 client bundle
- **apply 层缺口**：本任务不解决 override 注入，上线前需明确告知"实验分配已生效、参数覆盖待后续 task"

## 8. 接口缺口（实现阶段待核对）
- GrowthBook Python SDK `eval_feature` 在 `attributes` 传参形态（user 对象 vs dict）——以 SDK 实际签名校准
- `create_experiment` 翻译为 GB REST API 的字段映射（feature/experiment/variation 创建顺序、value 序列化）
- `evaluate_safety` 调 GB 暂停接口的具体 endpoint/权限（server API key 范围）
- docker-compose 中 mongo 数据卷持久化与 GB 初始化 seeding
- shop-agent 当前是否已用 Postgres（决定 §4.8 Data Source 是「复用」还是「新增」）
- GB SDK 的 `cacheConnection`/`cacheFile` 持久化在自托管 + Python SDK 下的确切配置方式（§4.9）

## 9. 后续：Flag 生命周期治理规范（规划项，不在本 scope 实现）
借鉴成熟项目，flag 是"出生即债务"，清理在建立时就规划，而非实验结束后临时想起。
GB 自托管**无 LaunchDarkly 式的代码引用扫描 / 陈旧自动检测**，故需用「命名约定 + 过期元数据 + 周期审计」自营补齐。

### 9.1 Flag 类型与命名前缀（强制）
| 前缀 | 类型 | 生命周期 | 处理 |
|---|---|---|---|
| `exp_` | 实验 / A/B（临时） | 有 `expected_end` | 分胜负后**必删**（固化 winner + 删分支 + 删 GB feature） |
| `canary_` | 金丝雀灰度（临时） | 100% 稳定 + soak 后 | 删，或转 `switch_` |
| `switch_` | 长期熔断 / kill switch（永久） | 长期保留 | **保留**，与实验 flag 解耦 |
| `perm_` | 权限 / 租户定向（永久） | 长期保留 | **保留** |

### 9.2 治理机制
1. **过期日内建**：建 `exp_`/`canary_` flag 必须带 `expected_end_date`；到期自动告警催收。
2. **陈旧检测（自营）**：周期审计脚本扫 GB features，命中以下任一标 stale 并告警：
   - 长期处于 100% 单一变体（实验已赢但未清理）；
   - 超 `expected_end` 未删；
   - GB feature 存在但代码仓库无对应 `eval_variant` 引用。
3. **Definition of Done（收尾闭环）**：一个实验 ticket 不算完成，直到：
   ① 胜者 `pipeline_overrides` 固化进默认 `AgentConfig`（基线配置）；
   ② 代码里 `if variant == ...` 分支删除；
   ③ GB 里该 `exp_` feature/experiment 删除。
4. **CI 兜底**：删 flag 但代码仍有引用 → 构建/ lint 报错。
5. **kill switch 分离**：若某能力需紧急熔断，另建 `switch_` 前缀长期开关，不依赖已删的实验 flag。

### 9.3 与现有 scope 的关系
- 本 scope 只实现「接入 GB + 分配/分析」，**不含** §9 的治理落地（命名强制、审计脚本、CI lint）；
- `delete_experiment` 接口（§4.4/§4.5）已为 §9.2③ 的"删 GB feature"提供能力，治理流程在其上编排。
