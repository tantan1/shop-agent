# YAML 编排协议（schema.md）

> 对应 `docs/yaml-orchestration-tasks.md` 阶段 1.1-1.9。
> 代码实现：`src/modules/chat/agent/yaml_flow/`（schema.py / validator.py / loader.py）。
> 示例：`examples/return_flow.yaml`（退货全流程）。

## 一、顶层结构（阶段 1.1）

```yaml
version: "1.0"        # 协议版本
entry: normalize      # 入口节点 id（必须存在于 nodes）
state: {...}          # 显式 State 契约（阶段 1.5）
nodes: [...]          # 节点列表（≥1）
edges: [...]          # 边（固定边 / 条件边）
guards: [...]         # 横切守卫（输入/输出过滤、锁、持久化、埋点）
```

## 二、节点类型（阶段 1.2，共 14 种）

`normalize / truncate / sentiment / intent / rag_pipeline / direct_tool / react / dispute / human_approval / input_filter / output_filter / lock / persist / observe`

- **Skill 引用节点**（仅 `react` / `direct_tool` / `rag_pipeline`）：通过 `config.skill` 从 `SkillRegistry` 取 SOP 与工具（阶段 1.2b）。
- **引擎原语节点**（`normalize` / `input_filter` / `lock` 等）：不指向任何 Skill，是内置原子能力。

## 三、流程与 Skill 正交（阶段 1.2c）

- Skill = 单节点的 SOP 与工具实现；流程 = 节点的连接顺序。
- 一个 Skill 可被多流程节点引用，一个流程可串多个 Skill + 非 Skill 原语。
- **敏感 Skill（request-return / check-balance 等）只能被引用为黑盒**，不允许内联改写其 SOP 或校验规则（阶段 5.5，校验器拦截 `sop_override` / `skip_validation` / `disable_hitl`）。

## 四、条件边（阶段 1.3）

```yaml
edges:
  - from: confirm
    conditions:
      - { op: on_intent==,  value: approve, target: execute_return }
      - { op: on_intent==,  value: reject,  target: reply }
      - { op: on_emergency,            target: escalate }
      - { op: always,                 target: fallback }   # 兜底边，每出口至多一条
```

算子：`on_intent==` / `on_complexity==` / `on_emergency` / `always` / `expr`（自定义表达式）。
固定边用 `to:`；条件边用 `conditions:`；二者互斥（schema 层强制）。

## 五、硬强制声明（阶段 1.4 / 5.1）

```yaml
config:
  skill: request-return
  hardcode:
    - field: order_id
      value: "${params.order_id}"   # 确定性来源（上游抽取/查询），非模型生成
      strip_from_schema: true        # 从工具 schema 切除该字段，剥夺模型写入权限
```

对应 27-参数硬强制注入 的「闭包覆盖 + 字段切除」。高后果字段由引擎在节点执行时定稿注入，业务配流程时改不了。

## 六、节点间参数传递（阶段 1.5，核心）

现状靠三条隐式通道（messages 列表 + `ReActRunContext` 闭包 + system prompt 拼接）。编排图改为**显式 State 契约**：

```yaml
state:
  fields:
    - { name: messages,     type: list }
    - { name: params,       type: dict,  required: true }
    - { name: tool_result,  type: any,   required: true }
    - { name: intent,       type: dict }
    - { name: hitl_pending, type: any }
    - { name: thread_id,    type: string, required: true }

nodes:
  - id: query_order
    type: direct_tool
    config:
      skill: query-order
      read_from:  [{ state: params, required: true }]   # 从 state 取上游参数
      write_to:   [{ state: tool_result, from: result }] # 把产出写回 state
```

- `read_from` / `write_to` 声明式映射，节点函数不感知上下游名字（阶段 1.5b）。
- `read_from.required=true` 时，若上游未写该字段 → 解析/运行期 **fail-closed**（阶段 1.5c）。校验器强制要求该 state 字段本身 `required=true`，否则 fail-closed 无意义。
- 与硬强制关系：`read_from` 的字段若同时被 `hardcode` 覆盖，解析器以硬强制值优先注入闭包（见 5.1），`params` 同名键视为不可信输入剔除（阶段 1.5d）。
- 跨 `human_approval`：`thread_id` + `hitl_pending` 随 checkpointer 持久化，resume 原样取回（阶段 1.5e / 阶段 3）。

## 七、守卫（阶段 1.6 / 5.4）

`input_filter / output_filter / lock / persist / observe`。可声明为图级 `guards`（对所有节点生效）或挂在节点前后置钩子。
**输入/输出安全过滤守卫必须存在**，缺失则编译失败（fail-closed）。

## 八、人在回路（阶段 1.7）

`human_approval` 节点 `config`：
```yaml
config:
  approval_type: refund        # 审批类型
  thread_id_ref: thread_id    # 跨请求恢复用的 thread_id 来源字段
  read_from: [{ state: tool_result, required: true }]
```

## 九、校验器规则（阶段 1.9）

`validate_flow(flow)` 返回非阻断告警列表；非法抛 `FlowValidationError`（fail-closed）：

1. entry 节点存在
2. edge 的 from/to 引用合法
3. condition 的 target 引用合法
4. `read_from.state` 必须存在于 `state.fields`
5. `read_from.required=true` ⇒ 对应 state 字段必须 `required=true`
6. 敏感 Skill 不得内联改写（`sop_override` / `skip_validation` / `disable_hitl` 拦截）
7. input/output 过滤守卫必须存在（阶段 5.4）
8. 仅 `react` / `direct_tool` / `rag_pipeline` 可声明 `skill`

## 十、加载入口

```python
from src.modules.chat.agent.yaml_flow import load_flow_file, FlowValidationError

flow, warnings = load_flow_file("examples/return_flow.yaml")
# flow: FlowFile（可直接进阶段 2 编译器）
```
