"""Design 2 回灌（Langfuse 版）单元测试。

不依赖真实 Langfuse 凭证：所有外部客户端调用均被 monkeypatch，
仅验证「门控逻辑 + 调用契约」（设计 2 §4/§5/§6.1）。

v4 SDK 迁移后，本测试同时覆盖：内部逻辑层（monkeypatch 高层函数）+ 真实
客户端调用形状（monkeypatch ``langfuse_mlops._client`` 为 v4 形态假客户端，
断言 ``start_as_current_observation`` / ``create_score`` / ``api.trace.list`` 等
v4 API 被按正确签名调用）。
"""
import types

import pytest

from src.modules.monitoring import langfuse_mlops as m
from src.modules.chat.core.a2a_task_service import A2ATaskService


def _ns(fb: bool, a2a: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        MLOPS_TOOL_SELECT_FEEDBACK_CAPTURE=fb,
        MLOPS_AUTO_CAPTURE_FROM_A2A=a2a,
    )


# ── capture_review 按 category 分支门控（设计 2 §4）──────────────────────────
@pytest.mark.parametrize(
    "fb,a2a,exp_exec_none,exp_a2a_none",
    [
        (False, False, True, True),
        (False, True, True, False),   # a2a_review 不受 FEEDBACK_CAPTURE 影响
        (True, False, False, True),
        (True, True, False, False),
    ],
)
def test_capture_review_gating(monkeypatch, fb, a2a, exp_exec_none, exp_a2a_none):
    monkeypatch.setattr(m, "config", _ns(fb, a2a))
    seen = []

    def fake_create_trace(content, category, metadata, session_id):
        seen.append(category)
        return "TRACE_ID"

    monkeypatch.setattr(m, "_create_trace", fake_create_trace)

    r_exec = m.capture_review(content="x", category="tool_exec_failed", metadata={}, session_id="s")
    r_a2a = m.capture_review(content="x", category="a2a_review", metadata={}, session_id="s")

    assert (r_exec is None) == exp_exec_none
    assert (r_a2a is None) == exp_a2a_none
    # 被门控关掉的 category 绝不能到达 _create_trace
    if exp_exec_none:
        assert "tool_exec_failed" not in seen
    if exp_a2a_none:
        assert "a2a_review" not in seen


# ── A2A 自动捕获：category 选择 + 分支门控（设计 2 a1/a2）─────────────────
@pytest.mark.asyncio
async def test_a2a_auto_capture_review_categories_and_gating(monkeypatch):
    service = types.SimpleNamespace()
    service._format_review_content = lambda message, chat_response, error: (
        f"content:{message}:{error}",
        {"err": error},
    )
    seen = []

    def fake_create_trace(content, category, metadata, session_id):
        seen.append({"category": category, "session_id": session_id})
        return "TRACE"

    monkeypatch.setattr(m, "_create_trace", fake_create_trace)

    # (a1) 执行失败：仅受 FEEDBACK_CAPTURE 门控
    monkeypatch.setattr(m, "config", _ns(True, False))
    await A2ATaskService._auto_capture_review(
        service, task_id="t1", message="hi", chat_response=None, error="boom"
    )
    assert seen[-1]["category"] == "tool_exec_failed"
    assert seen[-1]["session_id"] == "t1"

    # (a2) A2A 成功路径：仅受 A2A 标志门控（FEEDBACK 关也不影响）
    seen.clear()
    monkeypatch.setattr(m, "config", _ns(False, True))
    await A2ATaskService._auto_capture_review(
        service, task_id="t2", message="ok", chat_response=None, error=None
    )
    assert seen[-1]["category"] == "a2a_review"

    # (a2) A2A 标志关：成功路径不捕获
    seen.clear()
    monkeypatch.setattr(m, "config", _ns(True, False))
    await A2ATaskService._auto_capture_review(
        service, task_id="t3", message="ok", chat_response=None, error=None
    )
    assert seen == []


# ── 四层流水线本地执行失败回灌钩子（设计 2 a2）────────────────────────────
@pytest.mark.asyncio
async def test_exec_failed_hook_calls_capture_review(monkeypatch):
    from src.modules.chat.agent.react_agent import _mlops_capture_exec_failed

    calls = []

    def fake_capture_review(**kwargs):
        calls.append(kwargs)
        return "TRACE"

    monkeypatch.setattr(m, "capture_review", fake_capture_review)

    await _mlops_capture_exec_failed(
        user_query="重置密码",
        tool_name="refund_tool",
        selection_source="rule",
        error="timeout",
        conversation_id="conv1",
    )

    assert len(calls) == 1
    kw = calls[0]
    assert kw["category"] == "tool_exec_failed"
    assert kw["metadata"]["selected"] == "refund_tool"
    assert kw["metadata"]["selection_source"] == "rule"
    assert kw["metadata"]["error"] == "timeout"
    assert kw["session_id"] == "conv1"


# ── record_correction 归一化（设计 4 §3）──────────────────────────────────────
@pytest.mark.asyncio
async def test_record_correction_positive_attaches_to_existing_trace(monkeypatch):
    calls = []

    def fake_capture_review(**kwargs):
        calls.append(("capture", kwargs))
        return "EXISTING_TRACE"

    def fake_write_score(trace_id, name, value):
        calls.append(("score", trace_id, name, value))

    monkeypatch.setattr(m, "capture_review", fake_capture_review)
    monkeypatch.setattr(m, "_write_score", fake_write_score)
    monkeypatch.setattr(m, "config", _ns(True, False))

    tid = m.record_correction(
        type="annotator",
        trace_id="EXISTING_TRACE",
        correct_tool="reset_pw",
        original_tool="refund",
        selection_source="rule",
    )
    # 带 trace_id：不新建复核 trace，直接在该 trace 写 score
    assert tid == "EXISTING_TRACE"
    assert [c for c in calls if c[0] == "capture"] == []
    scores = [c for c in calls if c[0] == "score"]
    assert ("score", "EXISTING_TRACE", "correct_tool", "reset_pw") in scores
    assert ("score", "EXISTING_TRACE", "review_label", "correct") in scores


@pytest.mark.asyncio
async def test_record_correction_negative_creates_review_trace(monkeypatch):
    calls = []

    def fake_capture_review(**kwargs):
        calls.append(("capture", kwargs))
        return "NEW_TRACE"

    def fake_write_score(trace_id, name, value):
        calls.append(("score", trace_id, name, value))

    monkeypatch.setattr(m, "capture_review", fake_capture_review)
    monkeypatch.setattr(m, "_write_score", fake_write_score)
    monkeypatch.setattr(m, "config", _ns(True, False))

    tid = m.record_correction(
        type="chat_correction",
        conversation_id="conv9",
        rejected_tools=["refund"],
        original_tool="refund",
    )
    # 负样本无 correct_tool：新建复核 trace + 写 rejected_tools score（非 correct_tool）
    assert tid == "NEW_TRACE"
    cap = [c for c in calls if c[0] == "capture"][0][1]
    assert cap["category"] == "user_correct"
    assert cap["metadata"]["rejected_tools"] == ["refund"]
    scores = [c for c in calls if c[0] == "score"]
    assert ("score", "NEW_TRACE", "rejected_tools", "refund") in scores
    assert all(s[2] != "correct_tool" for s in scores)


@pytest.mark.asyncio
async def test_record_correction_gated_off(monkeypatch):
    calls = []

    def fake_capture_review(**kwargs):
        calls.append(kwargs)
        return "T"

    monkeypatch.setattr(m, "capture_review", fake_capture_review)
    monkeypatch.setattr(m, "config", _ns(False, True))  # FEEDBACK_CAPTURE 关

    tid = m.record_correction(type="annotator", correct_tool="x")
    assert tid is None
    assert calls == []


# ── v4 客户端调用形状（防回退到 v2/v3 API）──────────────────────────────────
class _FakeObs:
    id = "OBS_ID"


class _FakeCtx:
    def __init__(self, obs):
        self._obs = obs

    def __enter__(self):
        return self._obs

    def __exit__(self, *a):
        return False


class _FakeApiTrace:
    def __init__(self):
        self.list_calls = []
        self.get_calls = []
        self._traces = {}
        self._list_resp = None

    def list(self, **kw):
        self.list_calls.append(kw)
        return self._list_resp

    def get(self, trace_id, **kw):
        self.get_calls.append((trace_id, kw))
        return self._traces.get(trace_id)


class _FakeClient:
    def __init__(self):
        self.start_calls = []
        self.score_calls = []
        self.ds_calls = []
        self.dsi_calls = []
        self.obs = _FakeObs()
        self.api = types.SimpleNamespace(trace=_FakeApiTrace())

    def start_as_current_observation(self, **kw):
        self.start_calls.append(kw)
        return _FakeCtx(self.obs)

    def get_current_trace_id(self):
        return "TRACE_ID"

    def create_score(self, **kw):
        self.score_calls.append(kw)

    def create_dataset(self, **kw):
        self.ds_calls.append(kw)

    def create_dataset_item(self, **kw):
        self.dsi_calls.append(kw)

    def flush(self):
        pass


def test_create_trace_v4_api_shape(monkeypatch):
    fc = _FakeClient()
    monkeypatch.setattr(m, "_client", lambda: fc)
    tid = m._create_trace(content="c", category="tool_exec_failed", metadata={"x": 1}, session_id="s")
    assert tid == "TRACE_ID"
    assert fc.start_calls, "应调用 start_as_current_observation 建根 span"
    kw = fc.start_calls[0]
    assert kw["as_type"] == "span"
    assert kw["name"] == "tool_select_review"
    assert kw["metadata"]["category"] == "tool_exec_failed"
    assert kw["metadata"]["session_id"] == "s"


def test_write_score_v4_api_shape(monkeypatch):
    fc = _FakeClient()
    monkeypatch.setattr(m, "_client", lambda: fc)
    m._write_score("T1", "review_label", "correct")
    assert fc.score_calls, "应调用 create_score（v4）而非 client.score"
    assert fc.score_calls[0] == {
        "trace_id": "T1",
        "name": "review_label",
        "value": "correct",
        "data_type": "CATEGORICAL",
    }


@pytest.mark.asyncio
async def test_export_and_train_v4_api_shape(monkeypatch):
    import subprocess

    fc = _FakeClient()
    list_item = types.SimpleNamespace(
        id="T1", metadata={"sample_content": "【用户】hi\n【实际选出】search"}, scores=[]
    )
    full = types.SimpleNamespace(
        id="T1",
        metadata={
            "sample_content": "【用户】hi\n【实际选出】search",
            "correct_tool": "search",
            "review_label": "correct",
        },
        scores=[
            types.SimpleNamespace(name="review_label", value="correct"),
            types.SimpleNamespace(name="correct_tool", value="search"),
        ],
    )
    fc.api.trace._traces = {"T1": full}
    fc.api.trace._list_resp = types.SimpleNamespace(data=[list_item])

    monkeypatch.setattr(m, "_client", lambda: fc)
    # 避免真正执行 validate/train 子进程
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""))

    res = await m.export_and_train()
    assert res["status"] == "ok"
    assert fc.api.trace.list_calls, "应调用 api.trace.list（v4）而非 fetch_traces"
    assert fc.api.trace.list_calls[0]["name"] == "tool_select_review"
    assert fc.api.trace.get_calls and fc.api.trace.get_calls[0][0] == "T1"
    assert res["exported"] == 1

