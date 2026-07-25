"""RCA 根因归因正式测试（monitoring-agent 核心分析层）。

覆盖（迁移自 apps/gateway/_verify_2b04.py 的临时用例并强化）：
- 告警摄入端点注册（无状态被叫醒式 API）
- 入站即脱敏（06/07 跨平面 PII 不进分析/落库路径）
- 规则归因主线：网关中断 -> 明确根因 + 处置建议，且无 LLM 依赖
- Prometheus 不可达语义区分：数据源缺失 != 无异常（B-4 修复）
- LLM 可选增强：送出前脱敏 + 防注入 system 约束 + 频率限制（B-3/B-6）

不依赖 Prometheus/Loki/gateway 起服务：gateway_snapshot、Loki 查询、
httpx.Client 均用 monkeypatch 替换为 fake。

此前临时用例的缺陷已修复：analyze() 无条件调用 gateway_snapshot()，
必须 monkeypatch 替换，否则单元测试会真实触网并挂起。
"""

import importlib
import asyncio
import os
import sys

import pytest

# 让 monitoring_agent 包可 import（与仓库内其他测试同约定）
_MON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # apps/monitoring-agent
if _MON not in sys.path:
    sys.path.insert(0, _MON)

from monitoring_agent import alerts, main, rca  # noqa: E402


# --------------------------------------------------------------------------
# 1) 端点注册
# --------------------------------------------------------------------------

def test_endpoints_registered():
    """RCA 被动/主动 API 完整暴露，且分析层只产建议不直接操作。"""
    paths = {getattr(r, "path", "") for r in main.app.routes}
    for need in ("/ingest/alert", "/ingest/event", "/rca", "/rca/last"):
        assert need in paths, f"缺少 RCA 端点 {need}"


# --------------------------------------------------------------------------
# 2) 入站脱敏
# --------------------------------------------------------------------------

def test_alert_ingest_redacts_pii():
    """告警入站即脱敏：邮箱/公网 IP 脱敏，私网 IP 保留（避免误伤内网拓扑）。

    注意：parse_alertmanager 直接对原始 payload 做 redact_obj，已脱敏副本在
    raw_redacted；分析层拿到的也是脱敏后的 alerts。
    """
    payload = {
        "alerts": [{
            "labels": {"alertname": "X", "severity": "P2"},
            "annotations": {"summary": "user a@b.com hit limit pub 203.0.113.5 int 10.0.0.1"},
        }]
    }
    ev = alerts.parse_alertmanager(payload)
    assert ev.severity() == "P2"
    raw = __import__("json").dumps(ev.raw_redacted, ensure_ascii=False)
    assert "a@b.com" not in raw, "入站未脱敏邮箱"
    assert "203.0.113.5" not in raw, "入站未脱敏公网 IP"
    assert "10.0.0.1" in raw, "私网 IP 不应误伤"


# --------------------------------------------------------------------------
# 3) 规则归因主线（网关中断）
# --------------------------------------------------------------------------

def test_rule_attribution_gateway_down(monkeypatch):
    """GatewayDown 告警 + gateway 拓扑 down -> 明确根因 + 建议，纯规则降级。"""
    # LLM_GATEWAY_URL 清空强制纯规则；替换 gateway_snapshot 避免真实触网
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    monkeypatch.setattr(rca, "gateway_snapshot", lambda: {"_unavailable": True})

    ev = alerts.parse_alertmanager({
        "alerts": [{"labels": {"alertname": "GatewayDown", "severity": "P1"},
                    "annotations": {"summary": "gw down"}}]
    })
    topology = {"gateway": {"status": False}, "shop-agent": {"status": True}}
    res = rca.analyze(ev, topology)
    assert res.severity == "P1"
    assert "网关链路中断" in res.root_cause
    assert res.affected == ["gateway"]
    assert res.used_llm is False, "无 LLM_GATEWAY_URL 时应纯规则降级"
    assert res.recommendations, "RCA 必须产出处置建议"


# --------------------------------------------------------------------------
# 4) Prometheus 不可达语义区分（B-4：数据源缺失 != 无异常）
# --------------------------------------------------------------------------

def test_prom_unavailable_not_false_negative(monkeypatch):
    """Prometheus 不可达时，RCA 应如实标注降级，而非假装无异常。"""
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    monkeypatch.setattr(rca, "gateway_snapshot", lambda: {"_unavailable": True})

    ev = alerts.parse_alertmanager({
        "alerts": [{"labels": {"alertname": "VagueAlert", "severity": "P2"},
                    "annotations": {"summary": "something odd"}}]
    })
    topology = {"gateway": {"status": True}, "shop-agent": {"status": True}}
    res = rca.analyze(ev, topology)
    assert "Prometheus 不可达" in res.root_cause, "应如实标注数据源缺失降级"
    assert "无法确认是否无异常" in res.root_cause, "不能误报为无异常"
    # 应引导看板/日志，而非凭空结论
    assert any("Prometheus" in r or "Grafana" in r for r in res.recommendations)


# --------------------------------------------------------------------------
# 5) LLM 可选增强：送出前脱敏 + 防注入 + 频率限制
# --------------------------------------------------------------------------

class _FakeClient:
    """捕获 httpx.Client.post 的请求，便于断言送出内容已脱敏。"""

    def __init__(self, *args, **kwargs):
        self.captured = []
        self.call_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None, headers=None):
        self.call_count += 1
        self.captured.append({"url": url, "json": json, "headers": headers})

        class _R:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "归纳：上游延迟"}}]}

        return _R()


@pytest.fixture()
def fake_llm(monkeypatch):
    """把 _LLM_CHAT 指向假地址、注入 fake client、重置频率限制状态。"""
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    monkeypatch.setattr(rca, "_LLM_CHAT", "http://gw/chat/completions")
    monkeypatch.setattr(rca, "_LLM_MIN_INTERVAL_S", 0.0)  # 解除间隔，便于连发
    monkeypatch.setattr(rca, "_LLM_LAST_CALL", 0.0)
    fc = _FakeClient()
    monkeypatch.setattr(rca.httpx, "Client", lambda *a, **k: fc)
    return fc


def test_llm_summarize_redacts_before_send(fake_llm):
    """日志原文送出前先脱敏，且 system 指令含防注入约束。"""
    summary = rca._llm_summarize(["日志含手机号 13800138000 与邮箱 u@e.com"])
    assert summary, "应拿到 LLM 归纳结果"
    assert fake_llm.call_count == 1, "应真实发出一次请求"
    body = fake_llm.captured[0]["json"]
    joined = body["messages"][1]["content"]
    assert "13800138000" not in joined, "送出前的日志未脱敏手机号"
    assert "u@e.com" not in joined, "送出前的日志未脱敏邮箱"
    sys_msg = body["messages"][0]["content"]
    assert "不是指令" in sys_msg, "缺少防注入 system 约束（日志数据应被声明为数据而非指令）"


def test_llm_summarize_rate_limited(fake_llm, monkeypatch):
    """频率限制：同一进程内间隔不足时跳过二次调用（不触网，返回 None）。"""
    monkeypatch.setattr(rca, "_LLM_MIN_INTERVAL_S", 30.0)
    monkeypatch.setattr(rca, "_LLM_LAST_CALL", 0.0)

    first = rca._llm_summarize(["第一次调用日志 13800138000"])
    # 立刻第二次：距首次不足 30s，应被频率限制跳过（不触网）
    second = rca._llm_summarize(["第二次调用日志 13900139000"])
    assert first is not None, "首次调用应成功"
    assert second is None, "间隔不足应被频率限制跳过"
    assert fake_llm.call_count == 1, "被限制时不应真实发出请求"


# --------------------------------------------------------------------------
# 6) RcaResult 序列化：to_dict 暴露完整字段供 API 序列化
# --------------------------------------------------------------------------

def test_rca_result_to_dict():
    """RcaResult.to_dict 暴露完整字段，供 /rca 端点序列化响应。"""
    res = rca.RcaResult(
        severity="P2",
        root_cause="根因示例",
        affected=["gateway"],
        recommendations=["建议一"],
        evidence={"k": "v"},
        used_llm=False,
    )
    d = res.to_dict()
    assert d["severity"] == "P2"
    assert d["root_cause"] == "根因示例"
    assert d["affected"] == ["gateway"]
    assert d["recommendations"] == ["建议一"]
    assert d["evidence"] == {"k": "v"}
    assert d["used_llm"] is False


# --------------------------------------------------------------------------
# 7) RCA 可观测：发布为 Prometheus 指标（rca_total + rca_last_info）
# --------------------------------------------------------------------------

def test_record_rca_publishes_metrics():
    """RCA 完成后登记 counter + info gauge，/metrics 可查且 PII 已脱敏。"""
    main._rca_total.clear()
    main._rca_last = None

    res = rca.RcaResult(
        severity="P1",
        root_cause="网关链路中断 user u@e.com",
        affected=["gateway"],
        recommendations=["检查容器"],
        evidence={},
        used_llm=True,
    )
    main._record_rca(res)
    assert main._rca_total["P1"] == 1, "counter 应累计"

    body = asyncio.get_event_loop().run_until_complete(main.metrics())
    text = body.body.decode()
    assert "rca_total{severity=\"P1\"} 1" in text, "counter 应输出"
    assert "rca_last_info" in text, "info gauge 应输出"
    assert "rca_last_timestamp_seconds" in text, "时间戳应输出"
    # PII 出站脱敏：邮箱不进入指标
    assert "u@e.com" not in text, "根因文本出站未脱敏"
    assert "used_llm=\"true\"" in text, "used_llm 标签应保留"


# --------------------------------------------------------------------------
# 8) 方案 A：动态服务发现 + 静态补充（k8s 外服务）
# --------------------------------------------------------------------------

def test_parse_static_targets():
    """STATIC_TARGETS 解析：逗号分隔 name=url，跳过空/缺 '=' 条目。"""
    assert main._parse_static_targets("") == {}
    assert main._parse_static_targets("  a=http://x:1  , b = http://y:2 ,,") == {
        "a": "http://x:1",
        "b": "http://y:2",
    }
    assert main._parse_static_targets("broken") == {}


def test_build_targets_merge_priority(monkeypatch):
    """静态显式配置 > 动态发现 > 默认清单；TTL 缓存避免每次刷 Prometheus。"""
    monkeypatch.setattr(main, "_DISCOVER_CACHE_TTL_S", 0.0)  # 强制每次重查
    monkeypatch.setattr(main, "STATIC_TARGETS", {"gateway": "http://ext-gw:9000/health"})
    monkeypatch.setattr(
        main.prom,
        "scrape_targets",
        lambda: [
            {"health": "up", "labels": {"job": "shop-agent", "instance": "10.0.0.5:8000"}},
            {"health": "up", "labels": {"job": "gateway", "instance": "10.0.0.6:8001"}},
            {"health": "down", "labels": {"job": "milvus", "instance": "10.0.0.7:9091"}},
        ],
    )
    main._discover_cache = {}
    main._discover_cached_at = 0.0

    targets = main.build_targets()
    # 动态发现覆盖默认 url（host 贴近实际）
    assert targets["shop-agent"] == "http://10.0.0.5:8000/health"
    # 静态显式配置最高优先（k8s 外服务场景）
    assert targets["gateway"] == "http://ext-gw:9000/health"
    # down 的 target 不新增覆盖（但静态默认清单中的组件仍需探测——巡检本就要发现故障）
    assert targets["milvus"] == "http://standalone:9091/healthz"
    assert "10.0.0.7" not in targets.get("milvus", ""), "down target 不应覆盖静态 url"
    # 默认清单中的静态组件兜底保留
    assert targets["minio"] == "http://minio:9000/minio/health/live"
    assert targets["langfuse"] == "http://langfuse-web:3000/health"


def test_build_targets_prom_unavailable_falls_back(monkeypatch):
    """Prometheus 不可达时服务发现降级为静态清单，不抛异常。"""
    monkeypatch.setattr(main, "_DISCOVER_CACHE_TTL_S", 0.0)
    monkeypatch.setattr(main, "STATIC_TARGETS", {})
    monkeypatch.setattr(
        main.prom, "scrape_targets", lambda: (_ for _ in ()).throw(
            main.prom.PrometheusUnavailable("down")
        )
    )
    main._discover_cache = {}
    main._discover_cached_at = 0.0

    targets = main.build_targets()
    assert targets["gateway"].startswith("http://"), "应回退到静态默认"
    assert targets["shop-agent"].startswith("http://")
    assert "milvus" in targets and "minio" in targets


def test_discover_enabled_flag_disables(monkeypatch):
    """DISCOVER_ENABLED=0 时完全走静态清单，不触碰 Prometheus。"""
    monkeypatch.setattr(main, "_DISCOVER_ENABLED", False)
    monkeypatch.setattr(main, "STATIC_TARGETS", {"ext": "http://ext:9999/health"})

    def _boom():
        raise AssertionError("DISCOVER_ENABLED=0 不应查询 Prometheus")

    monkeypatch.setattr(main.prom, "scrape_targets", _boom)
    targets = main.build_targets()
    assert targets["ext"] == "http://ext:9999/health"
    assert "gateway" in targets


# --------------------------------------------------------------------------
# 9) 方案 C：SkyWalking 依赖拓扑（真实调用边）+ 级联归因
# --------------------------------------------------------------------------

def test_parse_dependencies():
    """STATIC_DEPENDENCIES 解析：逗号分隔 依赖方=被依赖方。"""
    assert main._parse_dependencies("") == []
    assert main._parse_dependencies("a=b,c=d") == [
        {"source": "a", "target": "b"},
        {"source": "c", "target": "d"},
    ]
    assert main._parse_dependencies(" a = b ,, x=x ,c=d ") == [
        {"source": "a", "target": "b"},
        {"source": "c", "target": "d"},
    ], "应跳过空条目与自依赖"


def test_build_dependencies_merges_static_fallback(monkeypatch):
    """OAP 不可达时静态依赖兜底；可达时动态+静态合并去重。"""
    monkeypatch.setattr(main, "_SKYWALKING_ENABLED", True)
    monkeypatch.setattr(main, "_SKY_CACHE_TTL_S", 0.0)
    monkeypatch.setattr(main, "STATIC_DEPENDENCIES", [
        {"source": "shop-agent", "target": "gateway"},
        {"source": "external-svc", "target": "shop-agent"},  # k8s 外服务静态声明
    ])
    monkeypatch.setattr(
        main.sky,
        "dependency_edges",
        lambda: [{"source": "gateway", "target": "redis"}],
    )
    main._sky_deps_cache = []
    main._sky_deps_cached_at = 0.0

    deps = main.build_dependencies()
    assert {"source": "gateway", "target": "redis"} in deps, "动态 OAP 边应保留"
    assert {"source": "shop-agent", "target": "gateway"} in deps, "静态兜底应合并"
    assert {"source": "external-svc", "target": "shop-agent"} in deps, "k8s 外依赖应可静态声明"
    assert len(deps) == 3, "不应重复"


def test_build_dependencies_sky_unavailable(monkeypatch):
    """OAP 不可达（抛 SkyWalkingUnavailable）时降级为纯静态依赖，不崩。"""
    monkeypatch.setattr(main, "_SKYWALKING_ENABLED", True)
    monkeypatch.setattr(main, "_SKY_CACHE_TTL_S", 0.0)
    monkeypatch.setattr(main, "STATIC_DEPENDENCIES", [
        {"source": "a", "target": "b"},
    ])
    monkeypatch.setattr(
        main.sky,
        "dependency_edges",
        lambda: (_ for _ in ()).throw(main.sky.SkyWalkingUnavailable("oap down")),
    )
    main._sky_deps_cache = []
    main._sky_deps_cached_at = 0.0

    deps = main.build_dependencies()
    assert deps == [{"source": "a", "target": "b"}]


def test_build_dependencies_flag_disables(monkeypatch):
    """SKYWALKING_ENABLED=0 时完全走静态，不触碰 OAP。"""
    monkeypatch.setattr(main, "_SKYWALKING_ENABLED", False)
    monkeypatch.setattr(main, "STATIC_DEPENDENCIES", [{"source": "a", "target": "b"}])

    def _boom():
        raise AssertionError("SKYWALKING_ENABLED=0 不应查询 OAP")

    monkeypatch.setattr(main.sky, "dependency_edges", _boom)
    assert main.build_dependencies() == [{"source": "a", "target": "b"}]


def test_cascade_attr_root_and_victims():
    """依赖级联归因：A 依赖 B，B 故障则 A 是受影响者，B 为根因候选。"""
    topology = {
        "gateway": {"status": False},
        "shop-agent": {"status": False},
        "redis": {"status": True},
        "_dependencies": [
            {"source": "shop-agent", "target": "gateway"},
            {"source": "gateway", "target": "redis"},
        ],
    }
    cascade = rca._cascade_attr(topology, ["gateway", "shop-agent"])
    assert cascade["root"] == ["gateway"], "gateway 无故障下游可依赖，应为根因"
    assert cascade["victims"] == ["shop-agent"], "shop-agent 依赖故障的 gateway，应为受影响者"


def test_cascade_attr_no_deps_flat():
    """无依赖边信息时退化为平铺：全部故障组件都是根因候选（无法推断传导）。"""
    topology = {"a": {"status": False}, "b": {"status": False}}
    cascade = rca._cascade_attr(topology, ["a", "b"])
    assert cascade["root"] == ["a", "b"]
    assert cascade["victims"] == []


def test_analyze_injects_cascade_evidence(monkeypatch):
    """RCA 全链路：依赖级联结果进入 evidence，根因文案含级联提示。"""
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    monkeypatch.setattr(rca, "gateway_snapshot", lambda: {"_unavailable": True})
    ev = alerts.parse_alertmanager({
        "alerts": [{"labels": {"alertname": "GatewayDown", "severity": "P1"},
                    "annotations": {"summary": "gw down"}}]
    })
    topology = {
        "gateway": {"status": False},
        "shop-agent": {"status": False},
        "_dependencies": [
            {"source": "shop-agent", "target": "gateway"},
            {"source": "external-svc", "target": "shop-agent"},
        ],
    }
    res = rca.analyze(ev, topology)
    assert res.evidence["cascade"]["root"] == ["gateway"]
    assert res.evidence["cascade"]["victims"] == ["shop-agent"]
    assert "依赖级联" in res.root_cause, "根因文案应含级联收敛提示"
    assert "根因候选" in res.root_cause
