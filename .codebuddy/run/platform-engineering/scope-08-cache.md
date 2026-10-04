# 子 Scope · 批次2-08：语义缓存（网关层 + PII 写入门禁 + 业务维度分片）

> 上游：`docs/platform-engineering/08-语义缓存.md`（核心论点：网关层语义缓存省 token 降延迟）
> 性质：落地 08「embedding + 业务维度分片 + PII 整条不写 + TTL/写操作不命中」；embedding 本批用**本地词频向量 + 余弦相似度**（零依赖、可命中近似 prompt），真实语义 embedding 留钩子
> 下游消费者：`python-coder`、code-reviewer、test-generator、用户验收

---

## 1. 目标（一句话可验证）

在 gateway 层落地语义缓存：相似 prompt 命中即短路返缓存、不进模型，省 token 降延迟。实现 08 的硬门禁（PII 整条不写，调 07 引擎做写入门禁）、业务维度分片（缓存键 = 归一化 prompt + 业务标签，各分片独立阈值/TTL）、写操作永不命中、TTL 失效。embedding 本批用本地**词频向量 + 余弦**（E2 乙，已确认）：`embed(text)` = 归一化后分词词频向量（去除停用词、小写、按业务标签可选自定义词典），`sim = cosine(vec_a, vec_b)`；同义/语序/助词变化可命中，纯 hash 占位下一字之差即不命中的问题被解决。阈值默认 `SIMILARITY_THRESHOLD`（如 0.85，词频向量相似度天然低于语义向量，需留余地）；涉金额/身份业务分片收紧。真实语义 embedding 留 `embed` 钩子不接模型（接口签名不变，未来换真模型只改 `embed` 内部）。

## 2. 设计依据（必读）

- `docs/platform-engineering/08-语义缓存.md` §2（卡网关）、§3（阈值权衡 + PII 整条不写 + 业务维度分片 + 治理归属三层）、§4（TTL + 写操作不命中）、§5（与 03 成本联动）
- `docs/platform-engineering/06-合规护栏.md` §2（PII 整条不写是跨平面硬门禁，调 06 引擎）
- 现有代码：`apps/gateway/gateway/hooks/governance.py`（`run("ingress")` 落点）、`metrics.py`（`record` 计量）、`types.py`、`controllers/proxy.py`

## 3. 子 scope 边界

**做**：
- 新增 `cache/` 包：
  - `store.py`：`SemanticCache`。本批用**进程内**存储（dict：key→(answer, ts)）；`embed(text)` = 归一化后**分词词频向量**（去停用词、小写、按业务标签可选自定义词典），相似度 `sim = cosine(vec_a, vec_b)`（E2 乙，已确认，非 hash）。命中阈值 `SIMILARITY_THRESHOLD`（词频向量相似度天然偏低，默认如 0.85 留余地；涉金额/身份业务分片收紧）。真实语义 embedding 留 `embed` 钩子不接模型（接口签名不变，未来换真模型只改 `embed` 内部）。
  - `policy.py`：业务维度分片（`cache-policy.yml`），映射 业务标签 → {threshold, ttl, enabled, skip_write}。**配置优先级（已确认）：env 作全局默认，yml 分片覆盖**——即 `SIMILARITY_THRESHOLD`/`CACHE_TTL`/`CACHE_ENABLED` 提供全局兜底值，`cache-policy.yml` 中某分片显式声明的同名字段覆盖之；分片未声明的字段继承 env 默认。查找顺序：分片值 → env 默认 → 代码内置常量（三层，绝不出现无值）。硬约束层（PII 整条不写、写操作不命中）**不参与该优先级**，代码强制、任何配置不可覆盖。硬约束层（网关维护者独占）：PII 整条不写、写操作永不命中、调 07 引擎——代码强制，不可业务协商。**调参权限**：业务方仅能在自有分片策略内调整 threshold/ttl/enabled/skip_write，碰不到他人分片与硬约束；硬约束不暴露给业务配置。本批**不做运行期实时调参 API**，自适应阈值留未来。配置加载走**应用内定时轮询 reload（默认 30s，不依赖 inotify 事件，兼容 Docker/K8s ConfigMap 符号链接轮换）**；多节点时各节点独立拉**同一共享配置源**（ConfigMap 挂卷 / S3 / 未来 Nacos），无需节点间协调；**不引入 Nacos/Apollo 配置中心（本批超范围）**。
  - `pii_gate.py`：写入前调 07 `PiiEngine` 检测，命中则整条不写（绕过缓存，不脱敏后存，引 08 §3）。
- 钩子集成：`hooks/governance.py` 增 **`async cache_lookup(stage: str, payload: dict) -> str | None`** 与 **`async cache_store(stage: str, payload: dict, answer: str) -> None`**（D4：钩子统一 async，签名由 2a 骨架定死，本 scope 只填方法体；`async def` 内禁止阻塞 IO——策略文件轮询 reload 走后台任务，不在热路径同步读盘）：ingress 阶段先查缓存（命中即短路，需 proxy 协作返回缓存答案不进模型）；egress 阶段（非流式）写缓存（经 pii_gate 门禁 + 写操作跳过 + TTL）。
  - **写入时机前提（B3 关键前提）**：`cache_store` **仅在 egress 治理链（judge_egress 注入检测 → guardrails → 脱敏兜底）全部通过后调用**。即进缓存的 answer 已是治理后的干净内容——写入时已脱敏、已过注入/guardrails 检测。据此，**命中直接返回，不重跑 egress 治理链**（见 C3）。若未来要防"规则时差"（写入后新增违禁词），归 08 §3「知识版本化联动」钩子，本批不处理。
- **命中返回形态（C3，已确认）：一律返回非流式固定 JSON**，即便原请求 `stream=true` 也返回标准 `chat.completion` 对象（**不模拟 SSE**）。理由：缓存答案是完整文本，逐块伪造 SSE 只为"看起来像流"，徒增复杂度与出错面。**命中不重跑 egress 治理链（B3，已确认）：缓存内容写入时已过 judge_egress + guardrails + 脱敏（见上方写入时机前提），命中即视为合规，直接返回**。约束：
  - 响应体构造为 OpenAI 兼容 `chat.completion`：`{id, object:"chat.completion", created, model, choices:[{index:0,message:{role:"assistant",content:<缓存答案>},finish_reason:"stop"}], usage:{prompt_tokens:0,completion_tokens:0,total_tokens:0}}`（usage 归零，表明零真实消耗）。
  - 证据头：`X-Cache: HIT`、`X-Upstream-Model` 填缓存写入时的模型、`X-Fallback: false`。
  - **已知取舍**：`stream=true` 的调用方会收到非流式响应，需能兼容。此为本批显式约束（流式缓存回放留后续），须在验收与文档中标注。
- 与 03 联动：缓存命中不调 `record`（不计 token），成本账区分真实消耗 vs 缓存短路（08 §5）。

**不做**：
- 真实语义 embedding 模型（如 OpenAI embeddings / 本地 transformers）：留 `embed` 钩子，本批用词频向量占位（**非 hash**，E2 乙），不引入重依赖/外部调用。
- 分布式缓存（redis）：**本批单节点进程内存储，多实例缓存命中不跨节点共享，一致性留批次3**。注意区分两层——① 配置（cache-policy.yml）多节点各自拉同一共享源即可一致（本批支持）；② 缓存数据（prompt→answer）跨节点共享需集中存储（Redis），本批不做。PII 门禁在每个节点本地前置判定（含 PII 整条不写，不依赖集中缓存侧拦）。
- Nacos/Apollo 配置中心：本批不引入（超范围）；配置走文件轮询 + 共享源（CM/S3），未来若上配置中心只替换配置源、策略解析层不变。
- 运行期实时调参 API / 自适应阈值：本批不做，留未来。
- 知识版本化联动（08 §4）：仅 TTL，知识版本维度留钩子。

## 4. 模块落点

| 文件 | 动作 |
|---|---|
| `cache/__init__.py` | 【新增】包标识 |
| `cache/store.py` | 【新增】`SemanticCache`（进程内 + embed 占位 + 阈值命中） |
| `cache/policy.py` | 【新增】业务维度分片策略（读 `cache-policy.yml`，env 兜底） |
| `cache/cache-policy.yml` | 【新增】分片策略文件（`CACHE_POLICY_PATH` 默认指向此） |
| `cache/pii_gate.py` | 【新增】写入前 PII 门禁（调 07 引擎，整条不写） |
| `hooks/governance.py` | 填充 `async cache_lookup` / `async cache_store` 方法体（ingress 查 / egress 写）；**签名与空实现骨架已由 2a 落好（D4），本路只填方法体，不动类头与他人方法** |
| `controllers/proxy.py` | ingress 顺序（B2，已确认）：注入检测 → guardrails → 脱敏 → cache_lookup(命中短路) → 模型；命中即返回非流式缓存答案（不进模型、不 record，且不重跑 egress 治理链）。egress：治理链(judge_egress→guardrails→脱敏)通过后才 cache_store（经 pii_gate 门禁） |
| `config.py` | 增 `CACHE_ENABLED`、`CACHE_TTL`、`SIMILARITY_THRESHOLD`（**均为全局默认，可被 yml 分片覆盖**）、`CACHE_POLICY_PATH`（默认 `cache/cache-policy.yml`） |
| `metrics.py` | **不改 `render()`**——经 2a 已完成的 registry（D2）`register_counter("gateway_cache_hits_total", ..., ("tenant","bucket"))` 注册即可（区分缓存短路 vs 真实消耗，接 03）。注册语句写在 `cache/store.py` 模块级，避免 2b 三路争抢 `metrics.py` |
| `pii/`（07） | 写入门禁调其引擎 |

## 5. 接口契约（验收基准）

- 命中：相同/**近似** prompt（同业务标签，余弦 ≥ 分片阈值）第二次请求 → 返缓存答案，不进模型、`record` 不计数（cache_hits+1）。近似判定由词频向量余弦给出（E2 乙），非 hash 精确匹配；验收用例需含一条"换说法但同义"的命中（如 `退货政策是什么` 与 `怎么退货` 在宽松阈值下命中）。
- PII 门禁：含 PII 的 prompt 写入缓存时被 `pii_gate` 拦截，整条不写。
- 写操作：含写意图（如下单/改地址，按业务标签 skip_write）永不命中缓存。
- TTL：过期缓存不命中。
- 业务分片：不同标签走各自阈值/TTL，互不影响。
- 命中直接返回（B3）：缓存内容写入时已过 egress 治理链（注入+guardrails+脱敏），命中即返、不重跑治理；写入时机前提是 `cache_store` 仅在治理通过后调用。

## 6. 验收清单

- [ ] 词频向量 + 余弦命中短路返答案（含"换说法同义"近似命中用例），不进模型、不计 token（cache_hits+1）
- [ ] PII 整条不写（调 07 引擎门禁）
- [ ] 写操作永不命中
- [ ] TTL 过期不命中
- [ ] 业务维度分片：标签→策略（阈值/TTL/enabled/skip_write）生效
- [ ] 配置优先级：分片值 > env 默认 > 内置常量；硬约束不可被配置覆盖
- [ ] 命中返回非流式固定 `chat.completion`（含 `X-Cache: HIT`、usage 归零）；`stream=true` 亦返非流式（已知取舍）
- [ ] 命中不重跑 egress 治理链（B3 前提：`cache_store` 在治理通过后调用，缓存内容已合规）
- [ ] 真实语义 embedding 仅留钩子，不引入重依赖
- [ ] 进程内实现，多实例一致性留批次3
