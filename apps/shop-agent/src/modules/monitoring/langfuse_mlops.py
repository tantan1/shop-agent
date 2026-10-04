"""工具选择 MLOps 监控（Langfuse 实现）。

取代自研 src/modules/mlops PostgreSQL 模块：捕获/标注/回流统一走 Langfuse
（trace observation + score + Dataset）。主链路已用 create_langfuse_handler 建 trace，
本模块在其基础上新增 tool_select_review 类 trace 承载工具选择复核。

函数：
  - capture_tool_select(user_query, all_tool_names, plan, error, sample_content=None)
      规划期捕获（react_agent._mlops_capture_tool_select 调用）。
  - capture_review(content, category, metadata=None, session_id=None)
      A2A 失败/用户纠正回灌（原 ReviewTask 的等价实现）。
  - label_tool_select(...)
      在 Langfuse 标注台人工纠偏后写 correct_tool / review_label score。
  - export_and_train(...)
      一键导出已标正确样本 → Langfuse Dataset → JSONL → validate → train。
"""

import asyncio
import json
import logging
import structlog
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

from typing import Any, Dict, List, Optional, Tuple

from src.core.config import config
from src.modules.chat.agent.react_agent import ToolPlan  # 仅类型注解
from src.modules.monitoring.langfuse_callback import get_langfuse_client

logger = structlog.get_logger("mlops.langfuse")


# ── 配置常量（同 config.py） ──
MONITOR_ENABLED = "MLOPS_TOOL_SELECT_MONITOR_ENABLED"
LOW_CONF_THRESHOLD = "MLOPS_TOOL_SELECT_LOW_CONF_THRESHOLD"
MONITOR_ALL = "MLOPS_TOOL_SELECT_MONITOR_ALL"

# 设计 3：标注样本导出的 Langfuse Dataset 名称
DATASET_NAME = "tool-select-gold"

# 设计 5（选项 C）：多意图检测。query 命中 ≥2 组不同候选工具且已选未覆盖 → 判定多意图漏选，
# 由 capture_tool_select 自动标出（category=tool_select_multi_intent）。
INTENT_SIGNALS: List[Tuple[List[str], List[str]]] = [
    (["退货", "退款", "破损", "换货", "质量问题", "退换"], ["request-return", "refund-confirm"]),
    (["优惠", "券", "优惠券", "领券", "折扣", "活动", "满减", "红包"], ["coupon-inquiry"]),
    (["订单", "查订单", "我的订单", "下单"], ["query-order"]),
    (["物流", "快递", "发货", "到哪", "收货", "配送"], ["check-shipping"]),
    (["余额", "账户余额", "剩余", "钱"], ["check-balance"]),
]
# 多意图连接词：显式连接两个诉求时即使只命中 2 组也判多意图；
# 无连接词则需命中 ≥3 组（降低“订单+物流”这类单诉求误报）。
_MULTI_INTENT_CONNECTORS = ["另外", "还有", "顺便", "以及", "同时", "也", "和", "与", "再", "一起", "此外", "并且"]


def _client():
    try:
        return get_langfuse_client()
    except Exception:  # noqa: BLE001
        return None


def _detect_multi_intent(user_query, all_tool_names, plan):
    """检测多意图漏选：query 暗示多个工具，但流水线只选了其中一部分。

    返回被暗示的工具集合（已排序 list）；非多意图返回 None。
    仅用于选项 C 的自动标记，不参与选型逻辑。

    决策只看『命中了几组不同诉求』，刻意与工具注册名解耦：
    即使注册名变化 / 当前可用集为空，只要 query 显式表达 ≥2 个独立意图就判多意图，
    避免注册名不匹配导致漏标（这正是线上 fire-and-forget 路径静默返回 None 的根因）。
    """
    if not user_query:
        return None
    q = user_query
    hit_groups = 0
    signal_tools = set()
    for keywords, tools in INTENT_SIGNALS:
        if any(kw in q for kw in keywords):
            hit_groups += 1
            signal_tools.update(tools)
    if hit_groups < 2:
        return None
    has_connector = any(c in q for c in _MULTI_INTENT_CONNECTORS)
    if not has_connector and hit_groups < 3:
        return None
    selected = {a.name for a in (plan.actions or [])} if plan else set()
    # 输出按当前可用集收敛；可用集为空（或不含信号工具）时保留全部信号工具，保证被标记
    out = sorted(t for t in signal_tools if (not all_tool_names or t in all_tool_names))
    if not out:
        out = sorted(signal_tools)
    if selected >= set(out):
        return None
    return out


def _classify(user_query, all_tool_names, plan, error):
    if not getattr(config, MONITOR_ENABLED, True):
        return None
    threshold = float(getattr(config, LOW_CONF_THRESHOLD, 0.85))
    monitor_all = getattr(config, MONITOR_ALL, False)
    selected = [a.name for a in plan.actions] if plan else []
    confs = [a.confidence for a in plan.actions if a.confidence is not None] if plan else []
    confidence = max(confs) if confs else None
    source = plan.source if plan else "none"
    if error is not None or not selected:
        return "tool_select_error"
    # 多意图漏选：query 暗示多个工具，但流水线只选了部分（选项 C 自动标记）
    if _detect_multi_intent(user_query, all_tool_names, plan):
        return "tool_select_multi_intent"
    if plan is not None and plan.stop_condition == "need_llm":
        return "tool_select_uncertain"
    if confidence is not None and confidence < threshold:
        return "tool_select_low_conf"
    if monitor_all:
        return "tool_select"
    return None


def _score_map(plan):
    return {a.name: a.confidence for a in (plan.actions or [])} if plan else {}


def _build_sample_content(user_query, all_tool_names, plan, error):
    selected = [a.name for a in plan.actions] if plan else []
    confs = [a.confidence for a in plan.actions if a.confidence is not None] if plan else []
    confidence = max(confs) if confs else None
    source = plan.source if plan else "none"
    lines = [f"【用户】{user_query}"]
    if error:
        lines.append("【状态】工具选择流水线报错")
        lines.append(f"【错误】{error}")
    else:
        lines.append(f"【实际选出】{', '.join(selected) if selected else '(无)'}")
        lines.append(f"【置信度】{confidence if confidence is not None else '-'}")
        lines.append(f"【由哪层选出】{source}")
        if plan and plan.actions:
            per_src = {a.name: (a.source or source) for a in plan.actions}
            lines.append("【逐工具来源】" + "; ".join(f"{n}:{s}" for n, s in per_src.items()))
        lines.append("【全部可选工具】" + ", ".join(all_tool_names or []))
        detected = _detect_multi_intent(user_query, all_tool_names, plan)
        if detected:
            lines.append(f"【疑似多意图】流水线只选部分，应覆盖工具：{', '.join(detected)}")
    return "\n".join(lines)


def _create_trace(content, category, metadata=None, session_id=None):
    """建一条 Langfuse trace（name=tool_select_review），返回 trace_id 或 None。

    Langfuse v4 采用 OTel 模型：trace 由根 observation（span）隐式创建。
    这里用 ``start_as_current_observation(as_type="span")`` 建根 span，再用
    ``get_current_trace_id()`` 取回真正的 trace id（与 observation id 不同）。

    v4.7.0 不提供 trace 级 session_id/tags 设定 API（``propagate_attributes`` 未实现），
    故将分类信息（含 session_id）统一放进 metadata，供 export_and_train 按 name+metadata 过滤。
    """
    client = _client()
    if client is None:
        return None
    try:
        md = {
            "category": category,
            "sample_content": content,
            "session_id": session_id,
            **(metadata or {}),
        }
        # 除 metadata 外，同时把关键结论写进 input/output：metadata 在 UI 里需要展开查看，
        # 而 input/output 直接展示，排障时能一眼看到"选了什么、由哪层选出、置信度多少"。
        md_non_null = {k: v for k, v in md.items() if v is not None}
        with client.start_as_current_observation(
            as_type="span",
            name="tool_select_review",
            input={"query": content, "available_tools": md.get("available_tools")},
            output={
                k: md_non_null.get(k)
                for k in (
                    "category",
                    "selected_tool",
                    "selection_source",
                    "per_tool_source",
                    "confidence",
                    "stop_condition",
                    "candidate_tools",
                    "original_tool",
                )
                if k in md_non_null
            },
            metadata=md,
        ) as _obs:
            return client.get_current_trace_id()
    except Exception as e:  # noqa: BLE001
        logger.warning("langfuse trace 创建失败（已忽略）", error=str(e))
        return None


def capture_tool_select(user_query, all_tool_names, plan=None, error=None, sample_content=None):
    """规划期捕获：react_agent._mlops_capture_tool_select 调用。"""
    category = _classify(user_query, all_tool_names, plan, error)
    if category is None:
        return None
    content = sample_content or _build_sample_content(user_query, all_tool_names, plan, error)
    meta = {
        "candidate_tools": [a.name for a in (plan.actions or [])] if plan else [],
        "available_tools": list(all_tool_names or []),
        "confidence": (max([a.confidence for a in plan.actions if a.confidence is not None]) if (plan and plan.actions and any(a.confidence is not None for a in plan.actions)) else None),
        "selection_source": (plan.source if plan else "none"),
        "per_tool_source": {a.name: (a.source or (plan.source if plan else "none")) for a in (plan.actions or [])},
    }
    detected = _detect_multi_intent(user_query, all_tool_names, plan)
    if detected:
        meta["detected_intent_tools"] = detected
        meta["is_multi_intent"] = True
    tid = _create_trace(content=content, category=category, metadata=meta)
    # 长驻服务进程必须显式 flush（Langfuse v4 走 OTel BatchSpanProcessor，根 observation 在 with 块退出后即
    # 入队，但后台 exporter 不会立即 export；进程不退出时也不触发 shutdown 强制落盘）。所有捕获路径共用本函数，
    # 故在此统一 flush，确保无论 react / direct / blocked 哪条路径都能把 trace 落到 Langfuse。
    if tid is not None:
        try:
            _c = _client()
            if _c is not None:
                _c.flush()
            logger.info("Langfuse 工具选择复核已捕获", trace_id=tid, category=category)
        except Exception as e:
            logger.warning("Langfuse 工具选择复核 flush 失败（trace 可能稍后落盘）", trace_id=tid, error=str(e))
    return tid


def capture_multi_intent_select(user_query, all_tool_names, selected_tools, source="p0_rule"):
    """简单意图 / 直调路径的多意图盲区捕获。

    ``execute_direct_tool_flow`` 等早停路径只 dispatch 单工具（``ctx.intent_result.action``），
    不会经过四级流水线，故其多意图盲区需单独捕获：把已 dispatch 的单个(或少数)工具名包装成
    ``ToolPlan`` 交给 ``capture_tool_select``，由 ``_detect_multi_intent`` 判定 query 是否暗示多个意图。
    ``all_tool_names`` 可留空——检测决策已与工具注册名解耦，空集时按信号工具集回落。
    """
    from src.modules.chat.schemas import PlannedAction, ToolPlan

    plan = ToolPlan(
        source=source,
        actions=[PlannedAction(name=t, confidence=None) for t in (selected_tools or [])],
        stop_condition="rule_hit",
    )
    return capture_tool_select(user_query=user_query, all_tool_names=all_tool_names, plan=plan)


def capture_review(content, category, metadata=None, session_id=None):
    """A2A 失败/工具选择捕获（原 MLOps 模块立 ReviewTask 的等价实现，改为 Langfuse trace）。

    门控按 category 分支（设计 2 §4/§5，移除原先函数首行的 A2A 早退）：
      - ``a2a_review``（A2A 成功路径采样）：仅由 ``MLOPS_AUTO_CAPTURE_FROM_A2A`` 控制；
      - 其余（``tool_exec_failed`` / ``user_correct`` 等回灌）：由 ``MLOPS_TOOL_SELECT_FEEDBACK_CAPTURE`` 总开关控制。
    """
    if category == "a2a_review":
        if not getattr(config, "MLOPS_AUTO_CAPTURE_FROM_A2A", True):
            return None
    else:
        if not getattr(config, "MLOPS_TOOL_SELECT_FEEDBACK_CAPTURE", True):
            return None
    return _create_trace(
        content=content,
        category=category,
        metadata=metadata,
        session_id=session_id,
    )


def _write_score(trace_id: str, name: str, value: Any) -> None:
    """写一条 score 到指定 trace（best-effort）。

    Langfuse v4：``client.score`` 已移除，改用 ``client.create_score``；
    字符串值用 ``data_type="CATEGORICAL"``。
    """
    client = _client()
    if client is None:
        return
    try:
        client.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            data_type="CATEGORICAL",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("langfuse 评分写入失败（已忽略）", error=str(e))


@dataclass
class UserCorrection:
    """归一化用户纠正事件（设计 4 §2/§3）。四类来源（C1/C2/C3/C4）统一结构。"""

    type: str
    trace_id: Optional[str] = None
    conversation_id: Optional[str] = None
    original_tool: Optional[str] = None
    correct_tool: Optional[str] = None
    rejected_tools: Optional[List[str]] = None
    content: Optional[str] = None
    selection_source: Optional[str] = None


def record_correction(
    type: str,
    trace_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    original_tool: Optional[str] = None,
    correct_tool: Optional[str] = None,
    rejected_tools: Optional[List[str]] = None,
    content: Optional[str] = None,
    selection_source: Optional[str] = None,
) -> Optional[str]:
    """归一化用户纠正回灌（设计 4 §3）。C1/C2/C3/C4 统一入口。

    门控：MLOPS_TOOL_SELECT_FEEDBACK_CAPTURE。
    - 正样本（correct_tool 非空）→ capture_review(category="user_correct") + score correct_tool / review_label=correct；
    - 负样本（rejected_tools 非空、无 correct_tool）→ capture_review(category="user_correct", metadata.rejected_tools) + score rejected_tools。
    带 trace_id（标注台 C4）时直接在该 trace 写 score；无 trace_id（聊天纠正 C1 等）时由
    capture_review 新建复核 trace（设计 4 §3.4：该 trace 无 observation 时补建）。
    幂等：Langfuse score 按 (trace_id, name) upsert，重复调用仅更新。
    返回目标 trace id；被门控关闭或 client 不可用时返回 None。
    """
    if not getattr(config, "MLOPS_TOOL_SELECT_FEEDBACK_CAPTURE", True):
        return None
    positive = bool(correct_tool)
    target_trace = trace_id
    if target_trace is None:
        metadata = {
            "category": "user_correct",
            "correction_type": type,
            "original_tool": original_tool,
            "selection_source": selection_source,
            "content": content,
        }
        if positive:
            metadata["correct_tool"] = correct_tool
        else:
            metadata["rejected_tools"] = rejected_tools or []
        target_trace = capture_review(
            content=content
            or (
                f"纠正：{original_tool} -> {correct_tool}" if positive else f"拒绝：{rejected_tools}"
            ),
            category="user_correct",
            metadata=metadata,
            session_id=conversation_id,
        )
    if not target_trace:
        return None
    try:
        if positive and correct_tool is not None:
            _write_score(target_trace, "correct_tool", str(correct_tool))
            _write_score(target_trace, "review_label", "correct")
        elif rejected_tools:
            _write_score(target_trace, "rejected_tools", str(rejected_tools[0]))
    except Exception:  # noqa: BLE001
        pass
    return target_trace


def label_tool_select(
    trace_id,
    correct_tool,
    review_label="correct",
    conversation_id=None,
    original_tool=None,
    selection_source=None,
):
    """标注员在 Langfuse 标注台人工纠偏（C4）；改为经 record_correction 归一化路径（设计 4 §3.4 / 设计 2 §5）。"""
    return record_correction(
        type="annotator",
        trace_id=trace_id,
        conversation_id=conversation_id,
        original_tool=original_tool,
        correct_tool=correct_tool,
        content=f"标注台纠正：{correct_tool}",
        selection_source=selection_source,
    )


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXPORT_PATH = os.path.join(ROOT, "scripts", "data", "langfuse_tool_select_export.json")

# 设计 5（选项 C）：多意图/已知盲区种子用例。版本化 JSON，登记进 tool-select-gold Dataset，
# 但不进单工具训练集（避免教错），待选项 A/B 支持多工具后再纳入。
SEED_PATH = os.path.join(ROOT, "scripts", "data", "tool_select_gold_seed.json")

# ── 训练数据集版本化存储（可复现性） ──
# 标注数据必须冻结成「版本化目录 + manifest」，不能让训练直接吃标注台的实时快照：
# 今天跑出的指标三个月后要能复现，就必须锁定当时那份数据。
# 默认落在 `scripts/data/tool_select`：容器内 /code/data 由镜像构建期写入、属主 root，
# 运行用户（appuser）无写权限；scripts/data 属主为 appuser，可写。
# 生产部署请通过 MLOPS_DATASET_DIR 指向持久化卷——容器文件系统随重建丢失。
DATA_ROOT = (
    (getattr(config, "MLOPS_DATASET_DIR", "") or "").strip()
    or os.path.join(ROOT, "scripts", "data", "tool_select")
)
DEFAULT_DATASET_VERSION = "v1.0"
# 留出集比例：holdout 永不进训练，且划分必须对同一 query 稳定，
# 否则新增标注会让旧 holdout 样本混入训练集，评估指标虚高。
HOLDOUT_RATIO = 0.1
HOLDOUT_SEED = "tool-select-holdout"


def _dataset_dir(version: str) -> str:
    return os.path.join(DATA_ROOT, version)


def _in_holdout(query: str) -> bool:
    """按 query 的稳定哈希划分留出集。

    用 query 而非随机/顺序划分：保证同一条语料在任何一次导出中都落在同一侧，
    新增标注只会让两侧各自增长，不会污染已冻结的留出集。
    """
    import hashlib  # 局部导入：避免在模块加载期引入额外依赖

    digest = hashlib.sha256(f"{HOLDOUT_SEED}|{query}".encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 100) < int(HOLDOUT_RATIO * 100)


def _write_json(path: str, items: List[Dict[str, Any]]) -> None:
    """写为 JSON 数组。

    下游 validate_tool_select_data.py / train_tool_select_sft.py 都用 ``json.load``
    读取数组（默认文件名亦为 .json），故这里写数组而非 JSONL，保持格式契约一致。
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _sample_query(item: Dict[str, Any]) -> str:
    convs = item.get("conversations") or []
    return convs[1].get("content", "") if len(convs) > 1 else ""


def _maybe_create_dataset(client, dataset_name, items):
    """尽力把标注样本组装进 Langfuse Dataset（设计 3）；失败不影响 JSONL 导出与训练。

    Langfuse v4：``create_dataset_item`` 用 ``source_trace_id`` 关联 trace（旧 SDK 为 trace_id）。
    """
    if client is None:
        return
    try:
        client.create_dataset(name=dataset_name)
    except Exception:  # noqa: BLE001
        pass
    for it in items:
        try:
            convs = it.get("conversations", [])
            q = convs[1].get("content") if len(convs) > 1 else None
            client.create_dataset_item(
                dataset_name=dataset_name,
                input={"query": q},
                expected_output=it.get("correct_tools") or it.get("correct_tool"),
                source_trace_id=it.get("trace_id"),
                metadata={
                    "selection_source": it.get("selection_source"),
                    "multi_intent": it.get("multi_intent", False),
                    "correct_tools": it.get("correct_tools"),
                },
            )
        except Exception:  # noqa: BLE001
            pass


def _score_value(s: Any) -> Any:
    """取一条 score 的真值。

    Langfuse v4 的 CATEGORICAL/TEXT 分数：字符串真值在 ``string_value``
    （服务端 stringValue），而 ``value`` 只是数值类别映射——未关联 score config
    时恒为 0。若只读 value，会把已标注样本误判成 0/未标注而漏导出。
    """
    if isinstance(s, dict):
        sv = s.get("string_value") or s.get("stringValue")
        return sv if sv not in (None, "") else s.get("value")
    sv = getattr(s, "string_value", None)
    return sv if sv not in (None, "") else getattr(s, "value", None)


def _trace_scores(trace) -> Dict[str, Any]:
    """从 trace 对象尽力抽取 scores，兼容 object / str / dict 多种形态。

    Langfuse v4 中 ``trace.list`` 返回的 scores 为字符串列表（名称），
    ``trace.get`` 返回的 scores 为对象（含 name/value/string_value）。一并归一化。
    """
    scores = getattr(trace, "scores", None) or []
    out: Dict[str, Any] = {}
    for s in scores:
        if isinstance(s, str):
            out[s] = s
        elif isinstance(s, dict):
            name = s.get("name")
            if name is not None:
                out[name] = _score_value(s)
        else:
            name = getattr(s, "name", None)
            if name is not None:
                out[name] = _score_value(s)
    return out


def _trace_label(trace):
    scores = _trace_scores(trace)
    if "review_label" in scores:
        return scores["review_label"]
    return (getattr(trace, "metadata", {}) or {}).get("review_label")


def _trace_correct_tool(trace):
    scores = _trace_scores(trace)
    if "correct_tool" in scores:
        return scores["correct_tool"]
    return (getattr(trace, "metadata", {}) or {}).get("correct_tool")


async def export_and_train(
    langfuse_url: str = "http://localhost:3000",
    version: str = DEFAULT_DATASET_VERSION,
):
    """一键触发（设计 3）：拉取已标 correct_tool 的 trace → Dataset → 导出 JSONL → validate → train。

    取代原 `GET /mlops/tasks/export`：数据源从 PostgreSQL mlops_review_tasks 换成 Langfuse trace/score。
    Langfuse v4：``fetch_traces`` 已移除，改用 ``client.api.trace.list(name=...)`` +
    ``client.api.trace.get`` 取完整 details（含 scores 对象）。
    """
    client = _client()
    if client is None:
        return {"status": "error", "reason": "langfuse 未配置"}

    items = []
    page = 1
    while True:
        try:
            resp = client.api.trace.list(name="tool_select_review", page=page, limit=50)
            batch = getattr(resp, "data", []) or []
        except Exception as e:  # noqa: BLE001
            logger.warning("langfuse trace.list 失败", error=str(e))
            batch = []
        if not batch:
            break
        for t in batch:
            tid = getattr(t, "id", None)
            if not tid:
                continue
            # 取完整 trace（含 scores 对象）以拿到 review_label / correct_tool
            full = t
            try:
                full = client.api.trace.get(tid)
            except Exception:  # noqa: BLE001
                full = t
            label = _trace_label(full)
            ct = _trace_correct_tool(full)
            if ct and label in ("correct", "labeled"):
                sample = _trace_to_sample(full, ct)
                sample["trace_id"] = tid
                items.append(sample)
        if len(batch) < 50:
            break
        page += 1

    if not items:
        return {"status": "ok", "exported": 0, "path": None}

    # ── 版本化落盘：train / holdout / manifest ──
    version_dir = _dataset_dir(version)
    os.makedirs(version_dir, exist_ok=True)
    train_items = [it for it in items if not _in_holdout(_sample_query(it))]
    holdout_items = [it for it in items if _in_holdout(_sample_query(it))]
    # 留出集不足时，宁可让训练集少一条，也不把 holdout 放空（评估不能没有基准）
    if items and not holdout_items:
        holdout_items = [items[-1]]
        train_items = items[:-1]
    if not train_items:
        train_items = items

    train_path = os.path.join(version_dir, "train.json")
    holdout_path = os.path.join(version_dir, "holdout.json")
    manifest_path = os.path.join(version_dir, "manifest.json")
    _write_json(train_path, train_items)
    _write_json(holdout_path, holdout_items)

    label_stats: Dict[str, int] = {}
    src_stats: Dict[str, int] = {}
    for it in items:
        ct = str(it.get("correct_tool") or "unknown")
        label_stats[ct] = label_stats.get(ct, 0) + 1
        s = str(it.get("selection_source") or "unknown")
        src_stats[s] = src_stats.get(s, 0) + 1
    manifest = {
        "version": version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "dataset_name": DATASET_NAME,
        "n_total": len(items),
        "n_train": len(train_items),
        "n_holdout": len(holdout_items),
        "holdout_ratio": HOLDOUT_RATIO,
        "label_distribution": label_stats,
        "selection_source_distribution": src_stats,
        "source_trace_ids": [it.get("trace_id") for it in items],
        "files": {"train": "train.json", "holdout": "holdout.json"},
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 兼容旧调用方（scripts 里按原路径取数）：仍写一份聚合 JSON
    os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
    with open(EXPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已导出 {len(items)} 条标注样本（train={len(train_items)} holdout={len(holdout_items)}）：{version_dir}")
    _maybe_create_dataset(_client(), DATASET_NAME, items)

    # 校验（只校验训练集；留出集不参与训练，也不应被校验规则裁剪）
    validate = os.path.join(ROOT, "scripts", "train", "sft", "validate_tool_select_data.py")
    vr = subprocess.run(
        [sys.executable, validate, "--in", train_path],
        capture_output=True, text=True,
    )
    if vr.returncode != 0:
        print("[WARN] 校验未通过，已中止训练：")
        print(vr.stdout)
        print(vr.stderr)
        return {"status": "validation_failed", "exported": len(items), "path": train_path}

    # 训练
    train = os.path.join(ROOT, "scripts", "train", "sft", "train_tool_select_sft.py")
    tr = subprocess.run(
        [sys.executable, train, "--data", train_path],
        capture_output=True, text=True,
    )
    print(tr.stdout)
    print(tr.stderr)
    if tr.returncode != 0:
        # 训练失败不能仍报 ok（如运行时镜像未装 torch / 缺模型）。导出的 JSONL 依然有效，
        # 可在带 requirements-training.txt 的训练环境中复用。
        return {
            "status": "training_failed",
            "exported": len(items),
            "reason": (tr.stderr or tr.stdout or "")[-400:],
            "dataset": langfuse_url,
            "version": version,
            "path": train_path,
            "holdout": holdout_path,
            "manifest": manifest_path,
        }
    return {
        "status": "ok",
        "exported": len(items),
        "dataset": langfuse_url,
        "version": version,
        "path": train_path,
        "holdout": holdout_path,
        "manifest": manifest_path,
    }


def _load_gold_seed() -> List[Dict[str, Any]]:
    """读取多意图/已知盲区种子用例（设计 5 / 选项 C）。

    种子文件为版本化 JSON 数组，每条与 export_and_train 导出的样本同构
    （conversations / correct_tool / correct_tools / multi_intent / selection_source ...）。
    这些用例是『已知应选多个工具却被单意图流水线漏选』的盲区登记，
    进 Langfuse ``tool-select-gold`` Dataset 供复核，但默认**不进训练集**
    （单工具 SFT 目标会教错），待选项 A/B 支持多工具后再纳入训练。
    """
    if not os.path.exists(SEED_PATH):
        return []
    try:
        with open(SEED_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:  # noqa: BLE001
        logger.warning("读取 gold seed 失败（已忽略）", error=str(e))
        return []


async def seed_gold_dataset(version: str = DEFAULT_DATASET_VERSION):
    """把种子盲区用例登记进 ``tool-select-gold`` Dataset（选项 C）。

    与 export_and_train 解耦：只做登记（Langfuse Dataset + 版本化 JSON + 聚合导出），
    不跑训练，可在运行环境直接执行；重复执行幂等。
    """
    client = _client()
    if client is None:
        return {"status": "error", "reason": "langfuse 未配置"}
    seed = _load_gold_seed()
    if not seed:
        return {"status": "ok", "seeded": 0}
    version_dir = _dataset_dir(version)
    os.makedirs(version_dir, exist_ok=True)
    _write_json(os.path.join(version_dir, "gold_seed.json"), seed)
    # 聚合导出（含种子，便于离线查看盲区）
    existing = []
    if os.path.exists(EXPORT_PATH):
        try:
            existing = json.load(open(EXPORT_PATH, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            existing = []
    existing_q = {e.get("query") for e in existing if isinstance(e, dict)}
    merged = existing + [s for s in seed if s.get("query") not in existing_q]
    _write_json(EXPORT_PATH, merged)
    _maybe_create_dataset(client, DATASET_NAME, seed)
    print(f"[INFO] 已登记 {len(seed)} 条盲区种子用例到 Dataset={DATASET_NAME}（version={version}）")
    return {"status": "ok", "seeded": len(seed), "dataset": DATASET_NAME, "version": version}


def _trace_to_sample(trace, correct_tool):
    """把一条 Langfuse trace 转成 ShareGPT 训练样本（含 correct_tool）。"""
    meta = getattr(trace, "metadata", {}) or {}
    content = meta.get("sample_content") or ""
    # 尝试从 sample_content 还原用户 query（【用户】行）
    user_query = ""
    for line in str(content).splitlines():
        if line.startswith("【用户】"):
            user_query = line[len("【用户】") :].strip()
            break
    return {
        "conversations": [
            {"role": "system", "content": "你是工具选择助手。"},
            {"role": "user", "content": user_query or (meta.get("sample_content") or "")},
            {"role": "assistant", "content": json.dumps({"name": correct_tool}, ensure_ascii=False)},
        ],
        "correct_tool": correct_tool,
        "scale": "M",
        "selection_source": meta.get("selection_source"),
        "candidate_tools": meta.get("candidate_tools"),
        "available_tools": meta.get("available_tools"),
    }


async def trigger_training():
    return await export_and_train()
