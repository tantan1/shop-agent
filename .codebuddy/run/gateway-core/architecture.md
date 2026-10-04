# 架构设计 · 批次0：LLM 流量网关核心骨架

> **产出阶段**：阶段1 `architecture-designer`
> **上游依据**：`docs/platform-engineering/01-LLM流量网关设计.md`（目录大纲，章节骨架已定）、`.codebuddy/run/gateway-core/scope.md`、`.codebuddy/run/platform-engineering/scope.md`
> **性质**：**落地骨架，非重推架构**。01 已定的架构前提（独立服务而非内嵌、出口网络策略强制经网关、HA 不破坏无旁路不变量）本文不再论证，仅承接落地。
> **下游消费者**：`python-coder`（批次0 实现）、批次1~2 各子 scope、`code-reviewer`、`test-generator`

---

## 1. 架构目标

> 将 01 的章节骨架落成 `apps/gateway/` 可运行实现：以独立 FastAPI 服务作为所有 LLM 流量的**唯一出口**，提供统一入口 + 多供应商路由（OpenAI 兼容 `/v1/chat/completions`），并为三道检测点位与计量治理预留结构化钩子。（引 `gateway-core/scope.md` §1）

**批次0 的边界**：只做「骨架 + 钩子 + 显式开关」。成本记账（03）、合规护栏（06）、脱敏（07）、语义缓存（08）、注入检测器实现（10）**均不在本批次**，本批次只保证它们**有位置可挂、有契约可依**。

**首要质量属性排序**（本批次冲突时的裁决依据）：

1. **治理完整性**（无旁路）> 2. **可用性**（网关不可用时的处置）> 3. **可维护性/可替换性** > 4. 性能

> 说明：治理完整性排第一是 01 核心论点的直接推论——「网关不是多一层，而是治理完整性的前提」。性能排最后是因为本批次为透明转发层，未引入额外模型调用，延迟增量仅为一跳内网 RTT。

---

## 2. 调用拓扑

承接 01 §2 拓扑图 + §2.1 三道检测点位，标注本批次的落地位置。

```
                    ┌───────────────────────────────────────────────┐
   外部/前端  ─────► │  shop-agent  (apps/shop-agent, :8000)         │
                    │  LLM SDK base_url = env LLM_GATEWAY_URL       │
                    │  ★需改造：当前硬编码直连供应商（见 §5.4）      │
                    └──────────────────────┬────────────────────────┘
                                           │ OpenAI 兼容 HTTP
                                           │ (K8s 内: http://gateway:8001/v1)
                                           ▼
      ┌────────────────────────────────────────────────────────────────────┐
      │  gateway  (apps/gateway, :8001)   —— 无状态 · 可水平扩容            │
      │                                                                    │
      │   ①入向闸  ──► ②治理面 ──► [路由决策] ──► ③出向闸 ──► upstream     │
      │   注入检测     脱敏/合规/     router.py    fail-closed/open         │
      │   (钩子)       语义缓存(钩子)              (本批实现开关)           │
      │                                                                    │
      │   /health  (旁路，不计量)      /metrics  (租户维度计量占位)         │
      └───────────┬───────────┬───────────┬───────────┬────────────────────┘
                  │           │           │           │
        tool_select│    gpt-* │   claude* │     qwen* │   ← 路由键(model)
          /param   │          │           │           │
          local/*  │          │           │           │
                  ▼           ▼           ▼           ▼
            ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
            │ Ollama  │ │  Azure  │ │ Bedrock │ │  百炼   │
            │ (本地)  │ │ OpenAI  │ │  (AWS)  │ │ (阿里)  │
            └─────────┘ └─────────┘ └─────────┘ └─────────┘
             各后端均以 OpenAI 兼容端点接入，env 配置，平等可替换

      ┌──────────────────────┐
      │  monitoring-agent    │──── GET /health ────► gateway:8001/health
      │  (:9091)             │      旁路探活，**不过 /v1**，不污染 LLM 计量
      └──────────────────────┘      (现有 monitoring_agent/main.py 已符合，无需改)
```

**拓扑不变量**（`code-reviewer` 核对项）：

- **唯一出口**：`shop-agent → vendor` 的直连路径在代码层必须不存在（本批次保证）；网络层强制（K8s NetworkPolicy）属部署文档，本文仅文字标注，见 §7。
- **旁路原则**：monitoring-agent 探 `/health` 走独立路径，`/health` 与 `/metrics` **不进入**计量与检测链路。现有 `monitoring_agent/main.py` 已指向 `http://gateway:8001/health`，**本批次无需改动 monitoring-agent**。
- **环境解耦**：K8s 内 `gateway:8001`；裸跑时 `LLM_GATEWAY_URL` 指向真实供应商回退（一套代码三环境，引总 scope §5）。

---

## 3. 模块划分

### 3.1 目录结构（建议）

现状：`apps/gateway/gateway/` 下**仅有 `main.py` 一个文件**（115 行），无 `__init__.py`。批次0 将其拆分为职责清晰的模块，为批次1~2 的并行开发预留互不冲突的落点。

```
apps/gateway/gateway/
├── __init__.py           【新增】包标识（当前缺失）
├── main.py               【改造】纯装配层：仅创建 FastAPI app + include_router，无任何端点逻辑/业务判断
├── controllers/          【新增】HTTP 端点层（controller 模式）
│   ├── __init__.py       【新增】聚合三个 router 供 main.py 挂载
│   ├── proxy.py          【新增】/v1/{path:path} 代理端点（串联 ①→②→路由→③，逻辑调用各模块）
│   ├── health.py         【新增】/health 端点（旁路，不计量）
│   └── metrics_endpoint.py【新增】/metrics 端点（薄壳，调 gateway.metrics.render）
├── config.py             【新增】Settings（pydantic-settings）：路由表、fail 策略、env 配置
├── router.py             【新增】路由决策（承接现有 _choose_backend，扩展为 01 §4 路由表）
├── hooks/
│   ├── __init__.py       【新增】
│   ├── injection.py      【新增】①入向注入第一道闸钩子（批次2/10 落地 RegexInjectionGate；接口预留于本批）
│   ├── governance.py     【新增】②治理面钩子（脱敏/合规/语义缓存/出向合规熔断，批次2 填充）
│   │                     └─ 六个方法全 async（D4）：run / run_stream / guardrails_check / judge_egress / cache_lookup / cache_store
│   └── egress.py         【新增】出向 fail-closed/open 开关（本批实现；**非注入防线，不占三道防线编号**，见 D1）
├── metrics.py            【新增】租户维度计量占位（承接现有 /metrics + _stats）；批次2/2a 重构为可注册 registry（D2）
│   └── 对外：record / tenant_tokens / mark_rate_limited / mark_budget_exceeded / estimate_tokens / render / register_counter / get_metric
├── pii/                  【批次2/07 新增】脱敏引擎包
│   ├── engine.py         【新增】PiiEngine.redact（前置+后置兜底）；确定性层永远在线，语义层留钩子
│   ├── rules.py          【新增】RulesStore 三级来源（基线 YAML / 远程 URL / 本地覆盖）+ 热路径零联网加载
│   └── degrade.py        【新增】引擎健康度→降级决策（healthy/degraded/down；down 按路由敏感度分级，受 GATEWAY_FAIL_MODE 全局压制，D5）
├── cache/                【批次2/08 新增】语义缓存包
│   ├── store.py          【新增】进程内 store + register_counter("gateway_cache_hits_total")（D2，注册写在此模块级）
│   ├── policy.py         【新增】业务维度分片 + TTL + 写操作跳过
│   └── pii_gate.py       【新增】写入门禁（PII 整条不写）
├── rules/                【批次2/07 新增】跨批次规则公共包（guardrails blocklist / 注入正则字典 / 脱敏基线共享）
│   └── *.yaml            【新增】规则即数据（YAML）
└── types.py              【新增】跨批次共享数据结构（RouteDecision / TenantUsage / GatewayMode / Verdict / GovernanceError / GovernanceStage）
```

> 命名约定：业务模块 `metrics.py`（计量逻辑）与端点 `controllers/metrics_endpoint.py`（HTTP 薄壳）刻意不同名，避免批次1~2 并行开发时改同一文件冲突。
> 三道防线编号（D1，唯一权威在 scope-10 §1）：①入向 prompt ②出向 judge_egress ③工具参数检测；`egress.py` 的 fail 开关属 01 §5 降级机制，**不占用三道防线编号**。

### 3.2 各模块职责与 01 对应

| 模块 | 承接 01 | 本批次动作 | 批次1~2 接口约定 |
|---|---|---|---|
| `config.py` | §4 路由表、§5 fail 策略 | 定义 `RouteTable` / `GatewayMode(str, Enum)` | 批次1 扩展限流/计量配置 |
| `router.py` | §2 拓扑、§4 引擎可替换 | 实现 4 类路由键映射 | 暴露 `route(model, tenant) -> RouteDecision` |
| `hooks/injection.py` | §2.1 ① | 定义 `InjectionGate` 协议 + `passthrough` 默认实现 | 批次2 注入 `detect(prompt)->Verdict` |
| `hooks/governance.py` | §2.1 ② | 定义 `GovernanceHooks` 组合调用点 | 批次2 注册脱敏/合规/缓存 |
| `hooks/egress.py` | §5 出向处置 | **实现** `fail_closed`/`fail_open` 开关 | 03/06/07 复用 mode 决策 |
| `limiter.py` | §3 成本治理 | 【批次1 新增】进程内令牌桶按 tenant 隔离 + 全局桶；预算熔断截断 | 批次3 接 redis 分布式桶 + 真实计数 |
| `metrics.py` | §3 可观测 | 升级为按 tenant 的 token 计量占位 | 批次3 接 Prometheus+Langfuse |
| `controllers/proxy.py` | §2 三道点位 | HTTP 入口串联 ①→②→路由→③，并按 02 篇补 `X-Upstream-Model`/`X-Fallback` 响应头 | 批次1 填充 fallback 真实标记；批次2 钩子在此生效 |
| `controllers/health.py` `metrics_endpoint.py` | §2 旁路 | 探活/计量端点薄壳 | 仅 HTTP 装配，逻辑在业务模块 |
| `types.py` | 全篇 | 定义共享类型 | 全局复用，避免批次间漂移 |

---

## 4. 关键决策与 why

### 4.1 路由键选择：按模型名/前缀
- **决策**：路由键 = `model` 字段（含前缀匹配 `gpt-*`/`claude*`/`qwen*`），`tool_select`/`param`/`local/*` → Ollama。
- **why（引 01 §4）**：模型名天然携带"该去哪"的语义，比在业务写 if-else 干净，且利于后续加维度（租户/成本）。
- **可替换性**：后端端点全在 env（`AZURE_OPENAI_BASE_URL` 等），换云厂商只改配置，业务无感。

### 4.2 出向 fail-closed / fail-open（本批实现）
- **决策**：env `GATEWAY_FAIL_MODE` ∈ {closed, open}，默认 `closed`。
- **why（引 01 §5）**：网关整体不可用时，合规护栏优先级高者必须 fail-closed（拒绝全部 LLM 流量），非敏感流量可 fail-open。本批次把"显式决策"落为代码开关，避免静默回退直连破坏无旁路不变量。
- **HA 不变量**：多副本切换目标必须仍是另一实例网关，绝不退直连——开关在副本间一致，网络策略兜底。

### 4.3 计量：估算 → 真实的演进路径
- **决策**：本批次 `metrics.py` 保留 `_estimate_tokens` 估算，但结构升级为按 `tenant` 维度；真实 token 计数（Prometheus+Langfuse）在批次3。
- **why（引 03 §3）**：计量精度是预算熔断地基，但本批次不引入重依赖，留好结构让批次3 平替。

### 4.4 无状态保证
- **决策**：网关不落本地状态；`_stats` 改为进程内计数仅作演示，租户计量走外部（批次3）。多副本水平扩容无共享状态冲突。
- **why（引 01 §5）**：网关无状态转发层可水平扩容，与 shop-agent 扩缩节奏解耦。

---

## 5. 与现有 main.py 的衔接点

> 原则：不破坏现有可运行行为（`scope.md` §8）。现有 `main.py` 可独立 `uvicorn.run(port=8001)` 运行，批次0 改造后必须仍可直接启动。

| 项 | 现状（main.py） | 批次0 动作 | 风险 |
|---|---|---|---|
| `_choose_backend` | 仅分 Ollama / 单 CLOUD | 抽至 `router.py`，扩展 4 类映射 | 低：纯函数迁移 |
| `/v1/{path:path}` 代理 | 直接 httpx 转发 | 逻辑迁至 `controllers/proxy.py`，串联 ①→②→③；main.py 仅 `include_router` | 中：需保证流式/非流式路径不变 |
| `_estimate_tokens` | 全局估算 | 迁至 `metrics.py`，加 tenant 维度 | 低 |
| `/metrics` | 返回 `_stats` | 端点薄壳迁至 `controllers/metrics_endpoint.py`，调 `metrics.py` 租户结构 | 低 |
| `/health` | 返回 ok | 迁至 `controllers/health.py`，标注为旁路（不计量） | 无 |
| 端点层拆分 | 端点与装配混在 main.py | 新增 `controllers/` 包（proxy/health/metrics_endpoint），main.py 纯装配 | 低 |
| 限流/Guardrails TODO | 注释占位 | 改为 `hooks.governance` 调用点（批次2 填充） | 低 |

**明确保留**：`uvicorn.run(host="0.0.0.0", port=8001)` 启动方式、`StreamingResponse` 流式转发、Ollama/云双 base_url 配置名。
**明确新增**：`__init__.py`、`config.py`、`router.py`、`hooks/`、`metrics.py`、`types.py`、`controllers/`（proxy/health/metrics_endpoint）、env `GATEWAY_FAIL_MODE` / `LLM_GATEWAY_URL` 读取。
**明确不碰**：`shop-agent` 业务代码（仅标注 §5.4 需改造点，实际改动在批次0 收尾或单独 PR）。

### 5.4 shop-agent 改造标注（本批次仅标注，实现见编码或单独 PR）
- 现有 `apps/shop-agent` 的 LLM SDK `base_url` 若硬编码直连供应商，需改为 `os.getenv("LLM_GATEWAY_URL", ...)`。
- 此改动属"业务不直连"不变量落地，建议批次0 编码阶段一并提交，但 scope 验收以网关侧无直连钩子为准。

---

## 6. 后续批次接口约定（给批次1~2 预留）

```python
# types.py —— 跨批次共享
@dataclass
class RouteDecision:
    upstream_base_url: str
    model: str
    backend: str          # "ollama" | "azure" | "bedrock" | "bai_lian"
    fallback_allowed: bool

@dataclass
class TenantUsage:
    tenant: str
    requests: int
    est_tokens: int

class GatewayMode(str, Enum):
    FAIL_CLOSED = "closed"
    FAIL_OPEN = "open"
    # 注：全局开关由 settings.fail_mode（GATEWAY_FAIL_MODE）承载，是 07 degrade.py 分级的硬下限（D5）

# 治理链异常载体（D3，已落地 types.py；reason 禁带原文）
class GovernanceStage(str, Enum):
    INGRESS = "ingress"
    EGRESS = "egress"
    STREAM = "stream"

class GovernanceError(Exception):
    def __init__(self, stage: "GovernanceStage | str", cause=None, reason: str = "") -> None: ...
    def log_fields(self) -> dict: ...   # {stage, exc_type, reason}，结构上不含原文

# hooks/injection.py —— 批次2/10 实现
class InjectionGate(Protocol):
    async def check(self, payload: dict) -> "Verdict": ...   # 命中即拦（async，D4）

# hooks/governance.py —— 批次2 注册（六个方法全 async，D4 已落骨架）
# 注意：落地形态为模块级 `hooks` 实例（hooks.run(...)），非 GovernanceHooks().run(...)；
#       若保留类封装，所有方法亦须 async。以下以模块级函数签名表述：
async def run(stage: str, payload: dict) -> dict: ...
    # 非流式：传入完整 body，返回处理后的完整 body（07 脱敏前置+后置兜底）
async def run_stream(stage: str, chunk: dict) -> dict: ...
    # 流式：逐块调用，传入单个 SSE chunk，返回（可能改写后的）chunk
    #   - 入参 chunk 已是解析后的 dict（如 {"choices":[{"delta":{"content":"..."}}]}）
    #   - 返回同结构 dict；脱敏命中时就地掩码，不跨块缓冲
    #   - 若本块无法独立判定（跨块完整 PII），返回原 chunk 并打标，交由调用方按 stage 决定 fail-closed/open
async def guardrails_check(stage: str, text: str) -> "Verdict": ...   # 06 确定性违禁词
async def judge_egress(response: dict) -> "Verdict": ...
    # 出向合规裁决（②出向，属 06/10 第三道闸，scope-10 为唯一权威）：
    #   - 非流式：传入完整响应 body，返回 Verdict；deny 则整块拦截、控制器层返 4xx
    #   - 流式：逐块裁决，deny 则立即停 yield 并补 data: {"finish_reason":"content_policy_violation"} 终止标记（不回滚已发内容）
    #   - 默认 fail-closed（业界通用：可疑即截断），模式由 GATEWAY_FAIL_MODE 统一控制
    #   - 必须基于未脱敏原文判定；异常按 D3 抛 GovernanceError
async def cache_lookup(stage: str, payload: dict) -> "str | None": ...   # 08 ingress 命中短路
async def cache_store(stage: str, payload: dict, answer: str) -> None: ...   # 08 egress 经 PII 门禁写
    # 实现现状：controllers/proxy.py 已按 SSE 行解析 `data:` 负载，仅对合法 JSON 调 run_stream，
    # 非数据行/非 JSON 原样透传——治理钩子在流式下真实生效（修复了初版对原始字节块 json.loads 必抛异常导致钩子被绕过的问题）。
    # async def 内禁止阻塞 IO（规则文件走启动期加载 + 后台刷新，不在热路径同步读盘，D4）。
```

- **批次1（02/03）**：见 `scope-02-routing.md` / `scope-03-cost.md`。
  - 02 路由：`RouteDecision` 增 `fallback_chain: list[str]`；`router.route()` 返回主后端+有序 fallback 链；`controllers/proxy.py` 主后端 429/5xx 沿链重试（仅同模型跨厂商，不自动跨模型降级），全挂返结构化 503；填真实 `X-Upstream-Model`/`X-Fallback`。
  - 03 成本：新增 `limiter.py` 进程内令牌桶（tenant 隔离 + 全局桶），建连前 `check(tenant)` 超限 429+Retry-After；预算熔断超 tenant token 截断 429；`metrics.py` 补 `gateway_rate_limited_total`/`gateway_budget_exceeded_total`；`controllers/proxy.py` 的 `_tenant_of` 扩展 X-Tenant / API Key 解析。
  - 不变量：`limiter`/`egress` 均在「建连前」决策，不破坏无旁路；跨模型自动降级、redis 分布式桶、真实计数不在本批。
- **批次2（06/07/08/10）**：见 `scope-06-compliance.md` / `scope-07-pii.md` / `scope-08-cache.md` / `scope-10-injection.md`。
  - 2a 硬前置（07，须先过 2a 独立验收门 G2 才开 2b）：脱敏引擎 `pii/`（engine 确定性层 + `rules/` 三级来源 + `degrade.py` 降级阶梯）；`metrics.py` 重构为可注册 registry（D2）；`types.py` 增 `GovernanceError`/`GovernanceStage`（D3）；`hooks/governance.py` 六方法 async 骨架（D4）；规则即数据（YAML），热路径零联网，绝不零规则启动。
  - 10 注入三防线（编号体系 D1，唯一权威 scope-10 §1）：①入向 `RegexInjectionGate`（确定性已知模式）+ ③工具参数（ingress 阶段对 `tool_calls.arguments` 检测）；②出向 `judge_egress`（复用 RegexInjectionGate，fail-closed）。
  - 编排顺序（B2，已确认）：**入向 = ①注入检测 → guardrails → ③脱敏 → ④缓存查询(命中短路) → ⑤模型**；**出向 = ②judge_egress → guardrails → 脱敏**（先防攻击、后护隐私；检测器必须基于未脱敏原文；缓存命中不重跑 egress 治理，B3）。
  - 06 合规护栏：复用 07 引擎做前置脱敏 + 后置兜底；`guardrails_check`（确定性违禁词，env `GUARDRAILS_BLOCKLIST`）；模糊边界打 `human_review` 标记不阻断（三通道）；异常按 C1/D3 上抛 `GovernanceError`，禁止钩子内自吞。
  - 08 语义缓存：新增 `cache/`（store 进程内 + policy 业务分片 + pii_gate 写入门禁）；ingress `cache_lookup` 命中短路（不进模型、不 record、`gateway_cache_hits_total`+1，经 D2 registry 注册在 `cache/store.py`），egress `cache_store` 经 PII 门禁 + TTL + 写操作跳过；真实语义 embedding 留钩子。
  - `GATEWAY_FAIL_MODE` 全局硬下限（D5）：三个消费点（`egress.decide` 上游不可达 / C1 治理钩子异常 / 07 `degrade.py` 引擎降级）共用同一开关；closed 下子模块分级放行一律被压制（`degrade.py` 低敏 fail-open 不生效，决策入口先读 `settings.fail_mode`）。
  - 不变量：注入检测严格在脱敏之前（10 §3 纵深）；`GATEWAY_FAIL_MODE` 统一控制 fail-closed/open；语义层（模型判定）均仅留钩子不默认启用（防外部依赖 SPOF）。
- **批次3（04/05/09）**：`metrics.py` 接 Prometheus+Langfuse；`/metrics` 转标准 exposition。

---

## 7. 风险与未决

- **部署层出口网络策略**（K8s NetworkPolicy / service mesh 封死 `shop-agent→vendor`）：属部署文档（`docs/oracle-cloud-deploy.md`），本架构仅文字标注"代码层已无直连，网络层需配策略兜底"，不在此实现。
- **shop-agent 直连改造**依赖业务代码现状，若当前已直连，批次0 收尾需一并改；否则仅留 env 约定。
- **真实 token 计量精度**：本批次为估算占位，批次3 才接真实计数，预算熔断阈值在批次3 才有意义。
- **fail 默认策略**：默认 `closed`（拒绝优先）符合 01 §5 治理完整性优先，但若用户业务对可用性极敏感，可在确认门调整（已写入 scope 确认门）。
