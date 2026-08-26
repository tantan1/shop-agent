# monitoring_agent 功能验证计划

> 目标：对三大功能模块做 **demo 页面展示验证** + **端到端（e2e）测试**。
> 配套可执行脚本：`apps/monitoring-agent/scripts/verify_monitoring_e2e.ps1`
> 前置：monitoring-agent 已在 `:9091` 运行（k8s `port-forward svc/monitoring-agent 9091:80` 或本地 `uvicorn`）。

## 状态速览

| 功能模块 | 子项 | 实现状态 | demo 可展示 | e2e 可测 |
|---------|------|---------|------------|---------|
| 一、主动巡检与 RCA | 拓扑健康矩阵 `/status` | ✅ | ✅ | ✅ |
| | 确定性规则归因 | ✅ | ✅ | ✅ |
| | 操作建议 | ✅ | ✅ | ✅ |
| | 周期巡检调度器 | ⚠️ 按需，无后台定时 | ❌ | ⚠️ 见 V-1.4 |
| 二、服务自动发现 | Prometheus 发现 | ✅ | ✅(依赖拓扑) | ✅ |
| | SkyWalking 依赖边 | ✅ | ✅ | ✅ |
| | 新增服务自动纳入 | ✅ | ✅ | ✅ |
| | 级联归因收敛 | ✅ | ✅ | ✅ |
| 三、告警接入 | Alertmanager 通道 | ✅ | ✅ | ✅ |
| | Langfuse 通道 | ✅ | ✅ | ✅ |
| | **HITL 审批流** | ✅ **已实现** | ✅ | ✅ |
| | 沙箱验证预览 | ✅ | ✅ | ✅ |
| | 计划查询 | ✅ | ⚠️ 后台接口 | ✅ |

---

## 一、主动巡检与根因分析

### V-1.1 demo：刷新健康状态（拓扑健康矩阵）
- 操作：打开 `http://localhost:9091/demo` → 点「🔄 刷新健康状态」
- 预期展示：
  - 顶部总览：`✅ 系统整体健康` 或 `❌ 系统存在故障组件`
  - 组件网格：每个组件状态点（绿/红）、延迟 `latency_ms`、详情
  - 卡片徽标 `HEALTHY` / `DEGRADED`
- e2e 断言：`GET /status` → `overall` 为 bool；`components` 含 shop-agent/gateway/redis/postgres 等键；`dependencies` 为非空列表。

### V-1.2 demo：手动触发 RCA（确定性规则）
- 操作：点「🔍 手动触发 RCA 根因分析」
- 预期展示（默认 demo 告警 `{name:"demo-alert"}`）：
  - 根因 `root_cause` 文本
  - 受影响组件 `affected`
  - 处置建议 `recommendations[]`，含「须经审批后执行」字样
  - 证据 `evidence`
- e2e 断言：`POST /rca` → `severity` ∈ {critical,warning,info}；`root_cause` 非空；`recommendations` 非空。

### V-1.3 e2e：网关中断 → GatewayDown 模式（规则主线）
- 操作：构造含故障拓扑的 RCA payload（gateway down，shop-agent 依赖 gateway）
- 预期：`severity=critical`；`root_cause` 指向 gateway；`affected` 含 shop-agent；`used_llm=false`（纯规则，无 LLM 依赖）
- 命令见脚本 `Case-RCA-GatewayDown`

### V-1.4 周期巡检调度器（⚠️ 部分实现，标注预期）
- 现状：`/status` 为按需探测，无后台定时任务。若长时间无人访问，组件故障不会被主动发现（除非 Alertmanager webhook 叫醒）。
- demo/e2e 结论：**此子项不可在 demo 中自动展示**；验证结论记为「按需巡检可用，周期调度待实现」。
- 建议演示话术：「巡检能力已具备，当前以 webhook 叫醒 + 按需探测驱动；后台周期调度为后续迭代项。」

---

## 二、服务自动发现

### V-2.1 demo：依赖拓扑展示（Prometheus + SkyWalking 聚合）
- 操作：`/status` 的 `dependencies` 字段由 `build_dependencies()` 聚合：
  动态（OAP `getGlobalTopology`）+ 静态兜底（`STATIC_DEPENDENCIES`）
- 预期：`dependencies` 含真实调用边（如 `shop-agent→gateway`）；新接入 Prometheus 的服务自动出现
- e2e 断言：`GET /status` → `dependencies` 为列表且至少含 1 条边
- 注：若 Prometheus/SkyWalking 不可达，自动降级为静态清单（不报错）

### V-2.2 e2e：新增服务自动纳入（无需改代码）
- 操作：通过 `STATIC_TARGETS` env 或 Prometheus 注册新服务 → 调 `/status`
- 预期：新服务出现在 `components` 中，无需改监控代码
- 命令见脚本 `Case-Discovery-NewService`（用 `STATIC_TARGETS` 注入 `dummy-svc=http://localhost:1/health`）

### V-2.3 demo + e2e：级联归因收敛根因
- 操作：构造拓扑（gateway down + shop-agent 依赖 gateway）→ `POST /rca`
- 预期：`_cascade_attr()` 按依赖边方向收敛：gateway = 根因候选，shop-agent = 受影响者（`affected` 含 shop-agent，但 `root_cause` 不含 shop-agent）
- 命令见脚本 `Case-Cascade`

---

## 三、告警接入（双通道 + HITL）

### V-3.1 demo：Alertmanager 通道摄入
- 操作：点「🔍 手动触发 RCA」（demo 默认走 alertmanager）；或脚本直接 `POST /ingest/alert`
- 预期：告警入站即脱敏（PII 变 `***`），触发 RCA，返回 `severity/root_cause/recommendations`
- e2e 断言：`POST /ingest/alert` → 200；`root_cause` 非空；回看 `/metrics` 中 `rca_total` 自增

### V-3.2 e2e：Langfuse 通道摄入
- 操作：`POST /ingest/event`（Langfuse 应急口，不经网关 /v1）
- 预期：解析 Langfuse event，触发 RCA
- 命令见脚本 `Case-Langfuse`

### V-3.3 e2e：入站脱敏（PII 不进分析路径）
- 操作：含邮箱/公网 IP 的告警推 `/ingest/alert`
- 预期：`a@b.com`、`203.0.113.5` 脱敏；私网 `10.0.0.1` 保留
- 命令见脚本 `Case-Redact`（复用 `test_rca.py::test_alert_ingest_redacts_pii` 逻辑）

### V-3.4 ✅ HITL 敏感操作审批流（已实现）
- 实现：`POST /remediate/preview` 生成 plan + 证据包 → `POST /remediate/approve` 审批（approved/rejected）→ `POST /remediate/apply` 执行
- 状态机：`plan_ready → pending_approval → approved/rejected → executed/failed`
- 持久化：`remediation_plans / approvals / executions` 四张表落库
- e2e 断言：`POST /remediate/preview` → 200 + `plan_id`；`POST /remediate/approve` → 200；重复审批 → 409

### V-3.5 ✅ 沙箱验证预览（已实现）
- 操作：`POST /remediate/preview` 传入 `{action, target, params}`
- 预期：返回 `plan_id` + `evidence`（含 `current/plan/effective/call_sequence`），默认使用 `FakeK8sClient` 干跑
- e2e 断言：`evidence.call_sequence` 含 `read_current` + `would_apply`；`dry_run=true`

### V-3.6 ✅ Docker 沙箱（已实现，默认关闭）
- 实现：新增 `DockerSandboxBackend`，与 `FakeK8sClient` 接口兼容
- 安全约束：`--network=none` + 资源上限（timeout 30s / Mem 256m / CPU 0.5 / PIDs 64）+ 库白名单（AST 扫描）+ 降权运行（nobody）+ 只读文件系统
- 启用方式：`SANDBOX_ENABLED=1` + 挂载 `/var/run/docker.sock`
- 触发条件：LLM 生成任意脚本 / 自动触碰生产 / 第三方不可信探测插件（设计文档 §10.4）
- e2e 断言：`SANDBOX_ENABLED=1` 时 `evidence.call_sequence` 含 `sandbox=docker` 标记；不可用时自动回退 `FakeK8sClient`

### V-3.7 ✅ 计划查询（已实现）
- 操作：`GET /remediate/plans?status=&limit=`
- 预期：返回计划列表，支持按 `pending_approval/approved/executed/failed/rejected` 过滤
- e2e 断言：`GET /remediate/plans` → 200；`GET /rca/history` → 最近 RCA 记录

---

## e2e 运行

```powershell
# 1. 端口转发（k8s 部署时）
kubectl -n shop-agent port-forward svc/monitoring-agent 9091:80
# 2. 运行 e2e 脚本
pwsh apps/monitoring-agent/scripts/verify_monitoring_e2e.ps1 -BaseUrl http://localhost:9091
```

脚本覆盖：V-1.1 / V-1.2 / V-1.3 / V-2.1 / V-2.2 / V-2.3 / V-3.1 / V-3.2 / V-3.3 / V-3.4 / V-3.5 / V-3.6 / V-3.7。

## 结果记录表（执行后填写）

| 用例 | 结果 | 备注 |
|------|------|------|
| V-1.1 拓扑矩阵 | ☐ | |
| V-1.2 手动 RCA | ☐ | |
| V-1.3 网关中断规则 | ☐ | |
| V-1.4 周期调度 | ☐ 待实现 | |
| V-2.1 依赖拓扑 | ☐ | |
| V-2.2 新服务纳入 | ☐ | |
| V-2.3 级联归因 | ☐ | |
| V-3.1 Alertmanager | ☐ | |
| V-3.2 Langfuse | ☐ | |
| V-3.3 入站脱敏 | ☐ | |
| V-3.4 HITL 审批 | ☐ 已实现 | preview/approve/apply |
| V-3.5 沙箱验证预览 | ☐ 已实现 | evidence 包 |
| V-3.6 Docker 沙箱 | ☐ 已实现 | 默认关闭，SANDBOX_ENABLED=1 启用 |
| V-3.7 计划查询 | ☐ 已实现 | /remediate/plans |
