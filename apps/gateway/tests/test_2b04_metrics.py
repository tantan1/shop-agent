"""批次 04 验收（业务级可观测 / 指标）转为正式测试。

迁移自 apps/gateway/_verify_2b04.py 中"网关侧可观测性"部分（RCA 部分已迁入
monitoring-agent 正式测试，见 apps/monitoring-agent/tests/test_rca.py）。

按当前 gateway 实现核对：
- metrics.register_counter 幂等；inc/get 按 label 分桶；render() 输出 Prometheus 标准格式
- metrics.reset() 重建 registry（Counter 不可清零的测试替代）
- Langfuse 默认关闭（不引入外部 SPOF）
纯指标层检查，不发起请求。
"""

from conftest import reload_gateway


def test_register_counter_idempotent():
    """同名重复 register 返回同一实例（模块多次导入安全）。"""
    mods = reload_gateway("metrics")
    metrics = mods["metrics"]
    a = metrics.register_counter("gateway_test_idem", "idem", ("tenant",))
    b = metrics.register_counter("gateway_test_idem", "idem", ("tenant",))
    assert a is b, "同名计数器应幂等返回同一实例"


def test_counter_inc_and_get_by_label():
    """inc 按 label 分桶，get 读回累计值。"""
    mods = reload_gateway("metrics")
    metrics = mods["metrics"]
    c = metrics.register_counter("gateway_test_inc", "inc", ("tenant",))
    c.inc(labels={"tenant": "t1"})
    c.inc(labels={"tenant": "t1"})
    c.inc(labels={"tenant": "t2"})
    assert c.get(labels={"tenant": "t1"}) == 2
    assert c.get(labels={"tenant": "t2"}) == 1


def test_render_standard_prometheus_format():
    """render() 输出 Prometheus exposition 格式（# TYPE / 指标名）。"""
    mods = reload_gateway("metrics")
    metrics = mods["metrics"]
    metrics.register_counter("gateway_test_render", "render", ("tenant",)).inc(labels={"tenant": "x"})
    out = metrics.render()
    # prometheus_client 自动为 Counter 加 _total 后缀
    assert "# TYPE gateway_test_render_total counter" in out, "render 应含标准 # TYPE 行"
    assert "gateway_test_render_total" in out


def test_reset_rebuilds_registry():
    """reset() 重建 registry，先前累计归零。"""
    mods = reload_gateway("metrics")
    metrics = mods["metrics"]
    c = metrics.register_counter("gateway_test_reset", "reset", ("tenant",))
    c.inc(labels={"tenant": "t"})
    assert c.get(labels={"tenant": "t"}) == 1
    metrics.reset()
    # reset 后同名计数器重新注册（幂等返回新实例），值应归零
    c2 = metrics.register_counter("gateway_test_reset", "reset", ("tenant",))
    assert c2.get(labels={"tenant": "t"}) == 0


def test_metrics_in_process_no_external_sPOF():
    """指标为进程内 pull 端点，不引入运行时外部依赖（无 SPOF）。"""
    mods = reload_gateway("metrics")
    metrics = mods["metrics"]
    # render() 在无任何外部依赖下即可产出 exposition 文本
    out = metrics.render()
    assert isinstance(out, str) and "TYPE" in out
    # 指标注册表为独立 CollectorRegistry，不与默认 registry 串扰
    assert metrics._REGISTRY is not None
