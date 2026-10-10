#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ci_failure_triage 的离线单测：构造假 CI 产物目录验证分诊结论。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ci_failure_triage import (  # noqa: E402
    COVERAGE_GATE,
    MAX_FIX_TARGETS,
    parse_junit_failures,
    triage,
)

JUNIT_PASS = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="2" failures="0" errors="0" skipped="0">
<testcase classname="tests.test_a.TestA" name="test_ok" time="0.01"/>
<testcase classname="tests.test_a.TestA" name="test_ok2" time="0.01"/>
</testsuite></testsuites>
"""

JUNIT_FAIL = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="2" failures="1" errors="1" skipped="0">
<testcase classname="tests.test_b.TestB" name="test_bad" time="0.01">
<failure message="assert 1 == 2">E  assert 1 == 2</failure>
</testcase>
<testcase classname="tests.test_b.TestB" name="test_boom" time="0.01">
<error message="ZeroDivisionError: division by zero">traceback...</error>
</testcase>
</testsuite></testsuites>
"""


def _cov(pct: float, covered: int, stmts: int, files: dict | None = None) -> str:
    return json.dumps({
        "totals": {"covered_lines": covered, "num_statements": stmts,
                   "percent_covered": pct},
        "files": files or {},
    })


def _mk(tmp: Path, app: str, junit: str | None, cov: str | None) -> None:
    d = tmp / f"pytest-results-{app}"
    d.mkdir(parents=True, exist_ok=True)
    if junit is not None:
        (d / f"pytest-results-{app}.xml").write_text(junit, encoding="utf-8")
    if cov is not None:
        (d / "coverage.json").write_text(cov, encoding="utf-8")


def test_parse_junit_extracts_failures(tmp_path: Path) -> None:
    p = tmp_path / "j.xml"
    p.write_text(JUNIT_FAIL, encoding="utf-8")
    targets = parse_junit_failures(p, "shop-agent")
    assert len(targets) == 2
    ids = {t.test_id for t in targets}
    assert "tests/test_b.py::TestB::test_bad" in ids
    assert all(t.kind == "test_failure" for t in targets)
    assert any("assert 1 == 2" in t.message for t in targets)


def test_parse_junit_ignores_passing(tmp_path: Path) -> None:
    p = tmp_path / "j.xml"
    p.write_text(JUNIT_PASS, encoding="utf-8")
    assert parse_junit_failures(p, "gateway") == []


def test_all_green_no_fix_needed(tmp_path: Path) -> None:
    for app in ("gateway", "monitoring-agent", "shop-agent"):
        _mk(tmp_path, app, JUNIT_PASS, _cov(85.0, 85, 100))
    res = triage(tmp_path)
    assert res.total_failures == 0
    assert res.overall_coverage == 85.0
    assert res.auto_fixable is False
    assert res.needs_human is False
    assert res.targets == []


def test_test_failure_dispatches_coder(tmp_path: Path) -> None:
    _mk(tmp_path, "gateway", JUNIT_PASS, _cov(90.0, 90, 100))
    _mk(tmp_path, "monitoring-agent", JUNIT_PASS, _cov(90.0, 90, 100))
    _mk(tmp_path, "shop-agent", JUNIT_FAIL, _cov(90.0, 90, 100))
    res = triage(tmp_path)
    assert res.total_failures == 2
    assert res.auto_fixable is True
    assert all(t.suggested_agent == "python-coder"
               for t in res.targets if t.kind == "test_failure")


def test_coverage_gate_dispatches_test_generator(tmp_path: Path) -> None:
    low_files = {
        "src/big_uncovered.py": {
            "summary": {"num_statements": 200, "percent_covered": 12.0}},
        "src/tiny.py": {
            "summary": {"num_statements": 3, "percent_covered": 0.0}},
    }
    _mk(tmp_path, "gateway", JUNIT_PASS, _cov(30.0, 30, 100, low_files))
    _mk(tmp_path, "monitoring-agent", JUNIT_PASS, _cov(30.0, 30, 100))
    _mk(tmp_path, "shop-agent", JUNIT_PASS, _cov(30.0, 30, 100))
    res = triage(tmp_path)
    assert res.total_failures == 0
    assert res.overall_coverage is not None and res.overall_coverage < COVERAGE_GATE
    assert res.auto_fixable is True
    cov_targets = [t for t in res.targets if t.kind == "coverage_gate"]
    assert cov_targets, "覆盖率不达标应产生补测试目标"
    assert all(t.suggested_agent == "test-generator" for t in cov_targets)
    # 语句数 <10 的小文件不应被选中
    assert not any("tiny.py" in t.detail for t in cov_targets)


def test_missing_report_needs_human(tmp_path: Path) -> None:
    _mk(tmp_path, "gateway", JUNIT_PASS, _cov(90.0, 90, 100))
    _mk(tmp_path, "monitoring-agent", None, None)   # 报告缺失
    _mk(tmp_path, "shop-agent", JUNIT_PASS, _cov(90.0, 90, 100))
    res = triage(tmp_path)
    assert res.needs_human is True
    assert any(t.kind == "missing_report" for t in res.targets)


def test_no_reports_at_all(tmp_path: Path) -> None:
    res = triage(tmp_path)
    assert res.needs_human is True
    assert res.auto_fixable is False


def test_targets_capped(tmp_path: Path) -> None:
    many = "".join(
        f'<testcase classname="tests.test_m.TestM" name="t{i}">'
        f'<failure message="boom{i}">x</failure></testcase>'
        for i in range(20)
    )
    xml = f'<testsuites><testsuite name="pytest">{many}</testsuite></testsuites>'
    _mk(tmp_path, "gateway", xml, _cov(90.0, 90, 100))
    _mk(tmp_path, "monitoring-agent", JUNIT_PASS, _cov(90.0, 90, 100))
    _mk(tmp_path, "shop-agent", JUNIT_PASS, _cov(90.0, 90, 100))
    res = triage(tmp_path)
    assert len(res.targets) == MAX_FIX_TARGETS
