#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agent_orchestrator.py — 多 Agent 工作流的领域层（纯规则引擎）。

设计依据：.codebuddy/agents/multi-agent-workflow.md

职责边界（关键，勿越界）：
  本模块是**领域规则引擎**，只回答"下一阶段是谁、要不要跳过、要不要回退、
  产物怎么落盘、怎么判定阻塞/伪绿"。它不读 stdin、不写 stdout、不感知 hook
  事件名、不碰 state.json 的持久化（那是 orchestrator_state 的事）。

  这样做的唯一理由：hooks 路径（跨进程，load→advance→save）与 eval 路径
  （纯内存，run_*→advance）必须汇聚到同一个 advance()。否则会重现
  "两套阶段规则各写一遍"的漂移，且 eval 绿灯无法证明 hooks 路径正确。

产物落盘（_save）为什么留在领域层：
  文件名映射（code→code.diff、fix→code.fix{N}.diff）是阶段机语义；
  且 test 分支必须当场对刚落盘的 tests.py 跑伪绿扫描才能决定回退。
  三条硬约束见 _save 的 docstring（C1/C2/C3）。

stdout 纪律：一切日志走 stderr。hook 协议中 stdout 是给 Agent 的消息通道
（优先级最高），领域层的 print 一旦进 stdout 就会被当成 hook 的返回内容。
"""
from __future__ import annotations

import argparse
import json
import re as _re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

# 阶段提示词集中管理（版本化 + 回归基线，见 prompts.py / PROMPTS_VERSION）
from prompts import (  # noqa: E402
    SCOPE_PROMPT, DESIGN_PROMPT, CODE_PROMPT, FIX_PROMPT,
    REVIEW_PROMPT, TEST_PROMPT, CONTRACT_FIRST_HINT,
)

# ── 路径约定（与 workflow.md / 项目结构对齐）──
REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / ".codebuddy" / "run"
ARCH_DIR = REPO_ROOT / "docs" / "architecture"

# 自动回退上限（对应 workflow.md "快速修复：重试 2-3 次后转人工"）
MAX_AUTO_RETRIES = 2

# 硬性质量门（方案②）：test 之后、merge 之前的覆盖率 / SAST 硬门禁。
# 低于覆盖率下限或命中高危 SAST 即标 [QUALITY] BLOCKING，最终在合并门禁处转人工。
QUALITY_COVERAGE_GATE = 80          # 覆盖率下限（%）
_QUALITY_SAST_SEVERITY = {"HIGH", "MEDIUM"}  # SAST 命中即阻塞的严重级别

# state.json 的 schema 版本。领域层新增字段时 +1，由 state_from_dict 做兼容迁移
SCHEMA_VERSION = 1

# ── 阶段 → subagent 映射（模块级常量，供 check_workflow_drift.py D1 校验）──
# 原为 Orchestrator 类属性；该类是"主 Agent 手动驱动"的适配壳，已随 hooks 改造删除。
STAGE_AGENT = {
    "scope": "scope",
    "design": "architecture-designer",
    "contract": "contract-gen",
    "code": "python-coder",
    "review": "code-reviewer",
    "fix": "python-coder",
    "test": "test-generator",
    "redteam": "red-team-reviewer",
}

# ── phase 常量 ──
PHASE_INIT = "init"
PHASE_SCOPE = "scope"
PHASE_DESIGN = "design"
PHASE_CONTRACT = "contract"
PHASE_CODE = "code"
PHASE_REVIEW = "review"
PHASE_FIX = "fix"
PHASE_TEST = "test"
PHASE_REDTEAM = "redteam"
PHASE_MERGE = "merge"   # 合并门禁（方案④）：test 完成后、done 前的人工放行闸
PHASE_DONE = "done"

# 产物文件名映射（领域规则：fix 的序号来自重试次数）
_ARTIFACT_FILENAME = {
    PHASE_SCOPE: "scope.md",
    PHASE_DESIGN: "design.md",
    PHASE_CONTRACT: "contract/",
    PHASE_CODE: "code.diff",
    PHASE_FIX: "code.fix{retries}.diff",
    PHASE_REVIEW: "review.md",
    PHASE_REDTEAM: "redteam.md",
    PHASE_TEST: "tests.py",
}


# ── 阶段推进结果 ────────────────────────────────────────────────────────────
@dataclass
class AdvanceResult:
    """advance() 的完整决策结果。

    为什么要返回结构化结果而不是只返回 phase 字符串：
    hook 层还需要"要不要阻塞、原因是什么、下一阶段的 prompt 是什么"。
    若让它拿字符串再 if-else 判断一遍，就会在 hook 层长出第二套状态机——
    而 eval 不走 hook 层，那套状态机将永远不被测试覆盖。
    """
    phase: str                          # 归档后当前所处的 phase
    next_stage: Optional[str] = None    # 下一待派发 stage；None = 不再派发
    next_agent: Optional[str] = None    # 对应的 subagent 名
    next_prompt: Optional[str] = None   # 对应的 prompt
    blocked: bool = False               # 是否存在阻塞问题
    blocking_reasons: List[str] = field(default_factory=list)
    needs_human: bool = False           # 是否需转人工（重试超限 / 伪绿）
    finished: bool = False              # 全部阶段完成

    @property
    def instruction(self) -> Optional[str]:
        """生成给主 Agent 的下一步指令文案；无后续动作时返回 None。"""
        if self.finished and not self.needs_human:
            return None
        if self.needs_human and self.next_stage is None:
            reasons = "；".join(self.blocking_reasons) or "原因见产物"
            return (
                f"【编排已停止，需人工介入】未解决的阻塞问题：{reasons}。"
                f"产物见 .codebuddy/run/ 下对应目录。"
            )
        if self.next_stage is None:
            return None
        head = "【编排未完成】" if self.blocked else ""
        return (
            f"{head}请用 Task 工具派发 subagent `{self.next_agent}`，"
            f"执行阶段 `{self.next_stage}`。prompt：{self.next_prompt}"
        )


@dataclass
class Artifact:
    """阶段产物：在阶段间传递，避免上下文丢失（防遗忘）。"""
    task: str
    stage: str
    content: str
    path: Optional[Path] = None


@dataclass
class OrchestrationState:
    """贯穿全阶段的共享状态（对应 workflow.md 的 State 契约思想）。

    phase 字段：原实现把 phase 存在 Orchestrator._phase 里，导致纯函数化后
    状态机无处安身。现下沉到本 dataclass，使 advance() 能纯内存地推进。
    """
    task: str
    user_request: str
    phase: str = PHASE_INIT
    scope_ref: Optional[str] = None          # scope.md 锚定的设计文档路径
    design_doc: Optional[Artifact] = None
    code: Optional[Artifact] = None
    review: Optional[Artifact] = None
    tests: Optional[Artifact] = None
    artifacts: dict = field(default_factory=dict)   # stage -> Artifact 索引
    blocking_issues: list = field(default_factory=list)
    retries: int = 0
    needs_human: bool = False
    dispatched: list = field(default_factory=list)  # 已派发 stage，用于幂等去重
    log: list = field(default_factory=list)

    def record(self, msg: str) -> None:
        """记录日志。一律走 stderr（stdout 是 hook 的消息通道，不可污染）。"""
        self.log.append(msg)
        print(f"[orchestrate] {msg}", file=sys.stderr)


# ── subagent 调用钩子：eval/离线模式注入真实或假的 LLM 调用 ──
# 签名：dispatch_stage(stage_name: str, prompt: str) -> str
DispatchFn = Callable[[str, str], str]
_dispatch: Optional[DispatchFn] = None


def set_dispatcher(fn: DispatchFn) -> None:
    """注入 subagent 调用（benchmark/eval 用假 LLM 注入）。未注入时走 dry-run。"""
    global _dispatch
    _dispatch = fn


def _call(stage: str, prompt: str) -> str:
    if _dispatch is not None:
        return _dispatch(stage, prompt)
    # dry-run：仅记录，便于先验证编排结构而不真正消耗 token
    print(f"  [dry-run] → {stage}: {prompt[:120]}...", file=sys.stderr)
    return f"<dry-run output for {stage}>"


# ── 产物落盘（领域层职责，见模块 docstring）──
def _save(task: str, stage: str, filename: str, content: str) -> Path:
    """落盘阶段产物到 .codebuddy/run/{task}/（workflow.md 可追溯约定）。

    三条硬约束（违反即破坏分层）：
      C1 领域层只允许写 RUN_DIR/{task}/ 下的**产物文件**；禁止读写
         state.json / .current / *.lock —— 那些是 orchestrator_state 独占。
      C2 顺序必须是「先落盘产物 → 后提交 state.json」（WAL 思想）。崩溃后
         最坏结果是重跑一个阶段，不会出现"state 说完成但产物不存在"。
      C3 领域层的 FS 读取只限两处：find_design_doc() 探测外部既有资产，
         以及读自己刚落盘的产物做校验（粒度/伪绿）。新增 FS 读取需回评审。
    """
    out = RUN_DIR / task
    out.mkdir(parents=True, exist_ok=True)
    p = out / filename
    p.write_text(content, encoding="utf-8")
    return p


def find_design_doc(task: str) -> Optional[Path]:
    """查找 docs/architecture/ 中是否已有对应设计文档（阶段1 跳过条件）。

    这是领域层允许的 FS 读取之一（C3-①）：探测外部既有资产。
    """
    if not ARCH_DIR.exists():
        return None
    keywords = task.lower().split()
    for f in ARCH_DIR.glob("*.md"):
        name = f.stem.lower()
        if any(k in name for k in keywords if len(k) > 3):
            return f
    return None


# 兼容旧名（内部与历史脚本引用）
_find_design_doc = find_design_doc


# ── 粒度自检（阶段0 拦截超大任务）──
# 原则：粒度 = 一次人工 review 能看完的量。
_GRAN_MAX_APPS = 2           # 跨越 >2 个 app 即拆分
_GRAN_CONJUNCTION_LIMIT = 2  # 并列连词出现 >=2 次即疑多目标


def check_granularity(state: OrchestrationState) -> list:
    """解析 scope.md，返回粒度预警列表（空 = 粒度合适）。

    FS 读取属 C3-②：读领域层自己刚落盘的产物做校验。
    """
    warns: list = []
    ref = state.scope_ref
    if not ref or not Path(ref).exists():
        return warns

    text = Path(ref).read_text(encoding="utf-8")

    # 1) 跨模块过多
    apps = set(_re.findall(r"apps/([\w-]+)", text))
    if len(apps) > _GRAN_MAX_APPS:
        warns.append(
            f"scope 跨越 {len(apps)} 个 app（{', '.join(sorted(apps))}），"
            f"建议按 app 拆成多个垂直切片"
        )

    # 2) 并列连词信号（以及/顺便/同时/另外 出现多次 → 多目标）
    for w in ("以及", "顺便", "同时", "另外"):
        n = text.count(w)
        if n >= _GRAN_CONJUNCTION_LIMIT:
            warns.append(
                f"scope 含多个并列目标（'{w}' 出现 {n} 次），"
                f"建议拆分为独立可验收任务"
            )

    # 3) 动作动词过多 → 疑似多任务合并
    if len(_re.findall(r"(新增|修改|重构|实现|接入|迁移)\s", text)) > 3:
        warns.append(
            "scope 列出 >3 个动作动词（新增/修改/重构/实现/接入/迁移），"
            "疑似多任务合并，建议按动词拆分"
        )

    return warns


# ── 阶段5 防假绿：机器验实现（文章 23 第2层）──
_FAKEGREEN_PATTERNS = [
    (r"assert\s+True", "恒真断言 assert True"),
    (r"assert\s+\w+\s*is\s*not\s*None\s*$", "弱断言：只判非 None，未验实际行为"),
    (r"^\s*pass\s*$", "pass 占位：测试体为空"),
    (r"#\s*assert", "断言被注释掉"),
    (r"assert\s+\w+\s*==\s*\w+\s*#\s*TODO", "断言被标记为 TODO"),
]


def check_test_not_fakegreen(tests_path: Path) -> list:
    """扫描测试文件，返回伪绿问题列表（空 = 干净）。对应 workflow.md 铁律。

    FS 读取属 C3-②：读领域层自己刚落盘的产物做校验。
    """
    if not tests_path.exists():
        return []
    text = tests_path.read_text(encoding="utf-8")
    problems: list = []
    for i, line in enumerate(text.splitlines(), 1):
        for pat, desc in _FAKEGREEN_PATTERNS:
            if _re.search(pat, line):
                problems.append(f"{tests_path.name}:{i} {desc}")
    return problems


# ── 阶段5 硬性质量门（方案②）：覆盖率 + SAST ──
def check_quality_gate(state: OrchestrationState) -> list:
    """test 后的硬性质量门：覆盖率下限 + SAST 高危。返回问题列表（空=通过）。

    产物由 test-generator 或 CI 生成在 .codebuddy/run/{task}/ 下：
       - coverage.json：pytest-cov 的 `--cov-report=json` 输出，取 totals.percent_covered；
       - sast.json：bandit `-f json` 输出，取 results[].issue_severity。
     两者任一缺失 → 视为该闸门未启用（软跳过），不报错——这样离线 eval / 缺工具
     环境不会被质量门卡死，只在真实 CI（产出上述文件）时生效。

     FS 读取属 C3-②：读领域层自己刚落盘的产物做校验。
     """
    run = RUN_DIR / state.task
    problems: list = []

    cov = run / "coverage.json"
    if cov.exists():
        try:
            data = json.loads(cov.read_text(encoding="utf-8"))
            tot = float(data.get("totals", {}).get("percent_covered", 100.0))
        except (OSError, ValueError):
            tot = 100.0
        if tot < QUALITY_COVERAGE_GATE:
            problems.append(
                f"覆盖率 {tot:.1f}% < 门禁 {QUALITY_COVERAGE_GATE}%"
            )

    sast = run / "sast.json"
    if sast.exists():
        try:
            data = json.loads(sast.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        for r in data.get("results", []):
            sev = str(r.get("issue_severity") or "").upper()
            if sev in _QUALITY_SAST_SEVERITY:
                problems.append(
                    f"SAST[{sev}] {r.get('filename', '')}:"
                    f"{r.get('line_number', '')} {r.get('issue_text', '')}"
                )
    return problems


# ── 阶段1.5 契约门禁（方案① T11）：spec 校验 + 生成物 import 检查 ──
def check_contract(state: OrchestrationState) -> list:
    """contract 阶段的硬性门禁：spec 合法性 + 生成物可 import。

    产物在 .codebuddy/run/{task}/contract/ 下：
       - spec.yaml / openapi.json：原始 spec
       - models.py：pydantic v2 模型
       - client.py：typed client
       - test_contract_schemathesis.py：schemathesis 属性测试
    产物缺失 → 视为该闸门未启用（软跳过），不报错。
    """
    run = RUN_DIR / state.task
    contract_dir = run / "contract"
    if not contract_dir.exists():
        return []

    problems: list = []

    # 1. spec 校验
    spec = None
    for candidate in ["openapi.json", "spec.yaml", "openapi.yaml"]:
        p = contract_dir / candidate
        if p.exists():
            spec = p
            break
    if spec:
        try:
            import yaml
            text = spec.read_text(encoding="utf-8")
            data = yaml.safe_load(text) if spec.suffix in (".yaml", ".yml") else json.loads(text)
            if not isinstance(data, dict):
                problems.append("[CONTRACT] spec 不是合法 JSON/YAML 对象")
            if "openapi" not in data and "swagger" not in data:
                problems.append("[CONTRACT] spec 缺少 openapi/swagger 版本声明")
        except Exception as exc:
            problems.append(f"[CONTRACT] spec 校验异常：{exc}")

    # 2. 生成物 import 干净检查
    for mod in ["models.py", "client.py"]:
        p = contract_dir / mod
        if p.exists():
            try:
                code = p.read_text(encoding="utf-8")
                # 简单语法检查
                compile(code, str(p), "exec")
            except SyntaxError as exc:
                problems.append(f"[CONTRACT] {mod} 语法错误：{exc}")

    return problems


# ── 阶段5 形式化门禁（方案④ T13）：关键路径 mypy + 不变式 ──
def check_critical_guarantees(state: OrchestrationState) -> list:
    """形式化门禁：对关键路径跑 mypy --strict + 不变式测试。

    产物由 critical_guarantees.py 生成在 .codebuddy/run/{task}/ 下：
       - type_check.json：mypy --strict 输出
       - invariant_test.json：不变式测试结果
    产物缺失 → 软跳过。
    """
    run = RUN_DIR / state.task
    problems: list = []

    tc = run / "type_check.json"
    if tc.exists():
        try:
            data = json.loads(tc.read_text(encoding="utf-8"))
            errors = data.get("errors", [])
            if errors:
                problems.append(f"[FORMAL] mypy 错误：{errors[0]}")
        except Exception as exc:
            problems.append(f"[FORMAL] type_check 解析异常：{exc}")

    inv = run / "invariant_test.json"
    if inv.exists():
        try:
            data = json.loads(inv.read_text(encoding="utf-8"))
            if not data.get("passed", True):
                problems.append(f"[FORMAL] 不变式测试失败：{data.get('failure', '')}")
        except Exception as exc:
            problems.append(f"[FORMAL] invariant_test 解析异常：{exc}")

    return problems


# ── 阶段5 沙箱门禁（方案② T2）：隔离执行证据 ──
def check_sandbox(state: OrchestrationState) -> list:
    """test 后的沙箱门禁：读取 sandbox.json，返回问题列表（空=通过/未启用）。

    产物由 sandbox_run.py 生成在 .codebuddy/run/{task}/sandbox.json：
       - ran: 是否实际执行
       - exit_code: pytest 退出码
       - runtime_errors: 运行时错误文本
       - wall_time: 耗时秒
       - killed_by: null | "timeout" | "oom"
     文件缺失 → 视为该闸门未启用（软跳过），不报错。
     """
    run = RUN_DIR / state.task
    sb = run / "sandbox.json"
    if not sb.exists():
        return []

    problems: list = []
    try:
        data = json.loads(sb.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    exit_code = data.get("exit_code", 0)
    killed_by = data.get("killed_by")
    runtime_errors = data.get("runtime_errors", "")

    if exit_code != 0:
        problems.append(f"[SANDBOX] exit_code={exit_code}")
    if killed_by in ("timeout", "oom"):
        problems.append(f"[SANDBOX] killed_by={killed_by}")
    if runtime_errors:
        problems.append(f"[SANDBOX] runtime_errors={runtime_errors[:200]}")
    return problems


def _maybe_contract_first(state: OrchestrationState) -> Optional[str]:
    """文章 23 第1层：若 scope/design 含 OpenAPI 契约，提示优先走确定性生成路径。"""
    refs = [state.scope_ref]
    if state.design_doc and state.design_doc.path:
        refs.append(str(state.design_doc.path))
    for ref in refs:
        if not ref or not Path(ref).exists():
            continue
        if _re.search(
            r"openapi|swagger|\.yaml|\.json",
            Path(ref).read_text(encoding="utf-8", errors="ignore"),
            _re.I,
        ):
            return CONTRACT_FIRST_HINT
    return None


def _scope_has_contract(state: OrchestrationState) -> bool:
    """检查 scope.md 是否声明了 contract（契约驱动）。"""
    ref = state.scope_ref
    if not ref or not Path(ref).exists():
        return False
    text = Path(ref).read_text(encoding="utf-8", errors="ignore")
    return bool(_re.search(r"contract", text, _re.I))


# ── 状态序列化（schema 归领域层，orchestrator_state 只做字节 IO）──
def state_to_dict(state: OrchestrationState) -> dict:
    """序列化为 state.json 的 state 段。

    只存 Artifact 的 path，不存 content：产物全文可达数百 KB，而 hooks
    每次事件都会读写一次，存内容会把写放大两个数量级。需要内容时由
    hydrate_state() 按 path 回填。
    """
    def _art(a: Optional[Artifact]) -> Optional[dict]:
        if a is None:
            return None
        return {"stage": a.stage, "path": str(a.path) if a.path else None}

    return {
        "schema": SCHEMA_VERSION,
        "task": state.task,
        "user_request": state.user_request,
        "phase": state.phase,
        "scope_ref": state.scope_ref,
        "artifacts": {k: _art(v) for k, v in state.artifacts.items() if v is not None},
        "blocking_issues": list(state.blocking_issues),
        "retries": state.retries,
        "needs_human": state.needs_human,
        "dispatched": list(state.dispatched),
        "log": list(state.log),
    }


def state_from_dict(d: dict) -> OrchestrationState:
    """从 state.json 的 state 段恢复 OrchestrationState（不含产物正文）。"""
    if not isinstance(d, dict):
        raise ValueError("state 段必须是 dict")

    def _art(spec) -> Optional[Artifact]:
        if not isinstance(spec, dict):
            return None
        p = spec.get("path")
        return Artifact(
            task=d.get("task", ""),
            stage=spec.get("stage", ""),
            content="",
            path=Path(p) if p else None,
        )

    arts = {
        k: _art(v)
        for k, v in (d.get("artifacts") or {}).items()
        if isinstance(v, dict)
    }
    st = OrchestrationState(
        task=d.get("task", ""),
        user_request=d.get("user_request", ""),
        phase=d.get("phase", PHASE_INIT),
        scope_ref=d.get("scope_ref"),
        blocking_issues=list(d.get("blocking_issues") or []),
        retries=int(d.get("retries", 0)),
        needs_human=bool(d.get("needs_human", False)),
        dispatched=list(d.get("dispatched") or []),
        log=list(d.get("log") or []),
    )
    st.artifacts = {k: v for k, v in arts.items() if v is not None}
    # 具名字段：benchmark/eval 直接读 state.code.content / state.tests.content
    st.design_doc = st.artifacts.get(PHASE_DESIGN)
    st.code = st.artifacts.get(PHASE_CODE)
    st.review = st.artifacts.get(PHASE_REVIEW)
    st.tests = st.artifacts.get(PHASE_TEST)
    return st


def hydrate_state(state: OrchestrationState) -> OrchestrationState:
    """按 path 把产物正文回填进 Artifact.content（供拼 prompt 用）。

    state.json 只存路径，读回后 content 为空，需要内容时调本函数。
    文件缺失时静默留空——产物丢失属异常，由调用方按业务语义处理。
    """
    for a in state.artifacts.values():
        if a is None or a.path is None or a.content:
            continue
        try:
            a.content = Path(a.path).read_text(encoding="utf-8")
        except OSError:
            a.content = ""
    if state.design_doc:
        state.design_doc = state.artifacts.get(PHASE_DESIGN)
    if state.code:
        state.code = state.artifacts.get(PHASE_CODE)
    return state


def new_state(task: str, user_request: str) -> OrchestrationState:
    """创建新编排的初始状态。

    phase 的语义是「当前待执行阶段」，故初值为 scope 而非 init：
    hooks 的 PostToolUse 直接拿 state.phase 作为待归档阶段，若初值是 init
    会对着一个不存在的阶段归档。init 仅作兜底兼容（旧状态文件无 phase 字段时）。
    """
    return OrchestrationState(task=task, user_request=user_request, phase=PHASE_SCOPE)


# ── 阶段 prompt 构造（供 run_* 与 hooks 共用，避免两处各拼一遍）──
def stage_prompt(state: OrchestrationState, stage: str) -> str:
    """构造某阶段的派发 prompt。"""
    if stage == PHASE_SCOPE:
        return SCOPE_PROMPT.format(user_request=state.user_request)
    if stage == PHASE_DESIGN:
        return DESIGN_PROMPT.format(scope_ref=state.scope_ref)
    if stage == PHASE_CODE:
        design = state.design_doc.content if state.design_doc else ""
        return CODE_PROMPT.format(
            scope_ref=state.scope_ref or "",
            design=design[:4000],
            user_request=state.user_request,
        )
    if stage == PHASE_REVIEW:
        code = state.code.content if state.code else ""
        return REVIEW_PROMPT.format(code=code[:6000])
    if stage == PHASE_TEST:
        code = state.code.content if state.code else ""
        return TEST_PROMPT.format(code=code[:6000])
    if stage == PHASE_FIX:
        code = state.code.content if state.code else ""
        return FIX_PROMPT.format(
            code=code[:4000],
            issues=chr(10).join(state.blocking_issues),
        )
    if stage == PHASE_REDTEAM:
        code = state.code.content if state.code else ""
        return f"红队审查：请对以下代码做对抗式审查，只输出有可复现失败用例的 [BLOCKING] 问题。\n\n{code[:6000]}"
    raise ValueError(f"未知阶段：{stage}")


# ── 核心：阶段推进（纯领域逻辑）──
def advance(
    state: OrchestrationState,
    stage: str,
    output: str,
    design_exists: Optional[bool] = None,
) -> AdvanceResult:
    """归档一个阶段的产出并推进状态机，返回完整决策结果。

    纯领域逻辑：只读写内存中的 OrchestrationState + 落盘产物文件，
    不读 stdin、不写 stdout、不感知 hook 事件名、不碰 state.json 持久化。

    design_exists 用于注入"是否已存在设计文档"的探测结果：
      - None（默认）→ 调 find_design_doc() 真实探测文件系统；
      - True/False  → 直接使用，便于测试零 mock 覆盖两条分支。
    把 FS 探测收拢到这一个可注入参数上，是为了让纯函数保持可被测。

    幂等：同一 stage 在同一重试轮次下重复推进不会重复归档。
    幂等键取 `stage@retries` 而非 stage：因为 review/fix 会随回退循环
    被合法地多次执行，若只按 stage 去重，第二次 review 会被误判为重复
    而直接跳过，回退链路当场卡死。
    """
    if stage not in STAGE_AGENT:
        raise ValueError(f"未知阶段：{stage}")

    idem_key = f"{stage}@{state.retries}"
    if idem_key in state.dispatched:
        state.record(f"阶段 {idem_key} 已归档，跳过重复推进（幂等）")
        return next_instruction(state)

    state.record(f"归档阶段 {stage}")

    # ── 1. 落盘产物（C2：先产物后状态）──
    if stage == PHASE_CONTRACT:
        # contract 产物是目录（含 spec/models/client/tests/mock），跳过单文件落盘
        contract_dir = RUN_DIR / state.task / "contract"
        contract_dir.mkdir(parents=True, exist_ok=True)
        path = contract_dir
        art = Artifact(state.task, stage, output, path)
        state.artifacts[stage] = art
        state.dispatched.append(idem_key)
    else:
        filename = _ARTIFACT_FILENAME[stage]
        if stage == PHASE_FIX:
            filename = filename.format(retries=state.retries)
        path = _save(state.task, stage, filename, output)
        art = Artifact(state.task, stage, output, path)
        state.artifacts[stage] = art
        state.dispatched.append(idem_key)

    # 具名字段同步（benchmark/eval 直接读 state.code / state.tests 等）
    if stage == PHASE_DESIGN:
        state.design_doc = art
    elif stage in (PHASE_CODE, PHASE_FIX):
        state.code = art
    elif stage == PHASE_REVIEW:
        state.review = art
    elif stage == PHASE_TEST:
        state.tests = art
    elif stage == PHASE_CONTRACT:
        state.artifacts[PHASE_CONTRACT] = art

    # ── 2. 阶段专属领域判定 ──
    if stage == PHASE_SCOPE:
        state.scope_ref = str(path)
        for w in check_granularity(state):
            state.record(f"[粒度预警] {w}")
        # 既有设计文档 → 作为 design 阶段的权威依据。
        # 此时 design 阶段会被 _next_phase 跳过，故必须在此把既有文档载入
        # state.design_doc，否则后续 code 阶段拿不到设计上下文。
        existing = find_design_doc(state.task)
        if existing is not None:
            state.record(f"检测到既有设计文档：{existing}")
            state.scope_ref = str(existing)
            try:
                content = existing.read_text(encoding="utf-8")
            except OSError:
                content = ""
            state.design_doc = Artifact(state.task, PHASE_DESIGN, content, existing)
            state.artifacts[PHASE_DESIGN] = state.design_doc

    if stage == PHASE_CONTRACT:
        cg = check_contract(state)
        if cg:
            state.blocking_issues.extend(cg)
            state.record(f"契约门禁命中 {len(cg)} 处：{cg[0]}")

    if stage == PHASE_REVIEW:
        issues = [l for l in output.splitlines() if "[BLOCKING]" in l]
        if issues:
            state.blocking_issues.extend(issues)
            state.record(f"审查发现阻塞问题 {len(issues)} 项")
        else:
            state.record("审查通过（无 [BLOCKING]）")

    if stage == PHASE_TEST:
        hint = _maybe_contract_first(state)
        if hint:
            state.record(f"[第1层提示] {hint}")
        fg = check_test_not_fakegreen(path)
        if fg:
            state.blocking_issues.extend(f"[FAKEGREEN] {p}" for p in fg)
            state.record(f"测试伪绿检测命中 {len(fg)} 处：{fg[0]}")
        # 硬性质量门（方案②）：覆盖率 / SAST。缺产物文件视为未启用（软跳过），
        # 故离线 eval / 缺工具环境不报错；test-generator 或 CI 产出
        # coverage.json / sast.json 后才真正生效。
        qg = check_quality_gate(state)
        if qg:
            state.blocking_issues.extend(f"[QUALITY] {p}" for p in qg)
            state.record(f"质量门命中 {len(qg)} 处：{qg[0]}")
        # 沙箱门禁（方案② T3）：读取 sandbox.json。缺产物视为未启用（软跳过）。
        sb = check_sandbox(state)
        if sb:
            state.blocking_issues.extend(sb)
            state.record(f"沙箱门禁命中 {len(sb)} 处：{sb[0]}")
        # 形式化门禁（方案④ T14）：关键路径 mypy + 不变式。缺产物视为未启用（软跳过）。
        fg = check_critical_guarantees(state)
        if fg:
            state.blocking_issues.extend(fg)
            state.record(f"形式化门禁命中 {len(fg)} 处：{fg[0]}")

    # ── 3. 推进 phase ──
    state.phase = stage
    state.phase = _next_phase(state, design_exists=design_exists)

    return next_instruction(state)


def _next_phase(state: OrchestrationState, design_exists: Optional[bool] = None) -> str:
    """根据当前状态决定下一 phase（纯决策，无副作用）。"""
    cur = state.phase

    if cur == PHASE_INIT:
        return PHASE_SCOPE
    if cur == PHASE_SCOPE:
        # 已有设计文档 → 跳过 design 直接进 code
        exists = design_exists
        if exists is None:
            exists = find_design_doc(state.task) is not None
        if exists:
            return PHASE_CODE
        # scope.md 声明 contract → 进契约阶段（确定性生成）
        if _scope_has_contract(state):
            return PHASE_CONTRACT
        return PHASE_DESIGN
    if cur == PHASE_DESIGN:
        # design 完成后进 contract（若 scope 声明）
        if _scope_has_contract(state):
            return PHASE_CONTRACT
        return PHASE_CODE
    if cur == PHASE_CONTRACT:
        return PHASE_CODE
    if cur == PHASE_CODE:
        return PHASE_REVIEW
    if cur == PHASE_FIX:
        return PHASE_REVIEW
    if cur == PHASE_REVIEW:
        if state.blocking_issues and state.retries < MAX_AUTO_RETRIES:
            state.retries += 1
            state.record(f"阻塞未解决，自动回退重做代码（第 {state.retries} 次）")
            return PHASE_FIX
        if state.blocking_issues:
            state.needs_human = True
            state.record(f"已达自动重试上限({MAX_AUTO_RETRIES})，转人工")
        # 审查通过后进红队审查（可选）
        return PHASE_REDTEAM
    if cur == PHASE_REDTEAM:
        return PHASE_TEST
    if cur == PHASE_TEST:
        if state.blocking_issues:
            state.needs_human = True
            state.record("测试阶段存在未解决阻塞（含伪绿），转人工")
        return PHASE_MERGE
    if cur == PHASE_MERGE:
        # 合并门禁已人工放行 → 收尾
        return PHASE_DONE
    return PHASE_DONE


def next_instruction(state: OrchestrationState) -> AdvanceResult:
    """根据当前状态生成下一步决策（不推进，只读）。

    hook 层的 Stop 事件直接调本函数，拿到"要不要继续 + 继续干什么"。
    """
    phase = state.phase
    blocked = bool(state.blocking_issues)

    if phase == PHASE_DONE:
        return AdvanceResult(
            phase=phase, blocked=blocked,
            blocking_reasons=list(state.blocking_issues),
            needs_human=state.needs_human, finished=True,
        )

    if phase == PHASE_MERGE:
        # 合并门禁：test 完成、待人工 merge 放行。领域层不强制 needs_human，
        # 由 hook 层 await_merge_approval 闸住（与 design 门禁同构），
        # 故 eval/run_workflow 不受影响（离线自动批准）。
        return AdvanceResult(
            phase=phase, blocked=blocked,
            blocking_reasons=list(state.blocking_issues),
            needs_human=state.needs_human, finished=False,
            next_stage=None, next_agent=None, next_prompt=None,
        )

    # init 是"尚未开始"的占位 phase，真正要派发的是第一个阶段 scope
    stage = PHASE_SCOPE if phase == PHASE_INIT else phase
    if stage == PHASE_CONTRACT:
        # 契约阶段是确定性 runner，不走 LLM subagent，但需要推进
        return AdvanceResult(
            phase=phase,
            next_stage=PHASE_CONTRACT,
            next_agent=None,
            next_prompt=None,
            blocked=blocked,
            blocking_reasons=list(state.blocking_issues),
            needs_human=state.needs_human,
            finished=False,
        )
    agent = STAGE_AGENT.get(stage, stage)
    try:
        prompt = stage_prompt(state, stage)
    except ValueError:
        prompt = ""

    return AdvanceResult(
        phase=phase,
        next_stage=stage,
        next_agent=agent,
        next_prompt=prompt,
        blocked=blocked,
        blocking_reasons=list(state.blocking_issues),
        needs_human=state.needs_human,
        finished=False,
    )


# ── 函数式阶段入口（薄包装：只负责派发，推进统一交给 advance）──
# 保留这些符号与签名，供 benchmark/eval 直接 import（shop_agent_driver.py）。
# 它们与 hooks 路径共用同一个 advance()，因此 eval 绿灯能证明 hooks 路径正确。
def run_scope(state: OrchestrationState) -> None:
    state.record("阶段0：目标澄清 + scope")
    advance(state, PHASE_SCOPE, _call(PHASE_SCOPE, stage_prompt(state, PHASE_SCOPE)))


def run_design(state: OrchestrationState) -> None:
    existing = find_design_doc(state.task)
    if existing is not None and state.design_doc is None:
        state.record("阶段1：跳过（依据既有设计文档落地）")
        advance(state, PHASE_DESIGN, existing.read_text(encoding="utf-8"),
                design_exists=True)
        return
    state.record("阶段1：架构设计（触发：无现成设计文档）")
    advance(state, PHASE_DESIGN,
            _call(STAGE_AGENT[PHASE_DESIGN], stage_prompt(state, PHASE_DESIGN)),
            design_exists=False)


def run_code(state: OrchestrationState) -> None:
    state.record("阶段2/3：编码实现")
    advance(state, PHASE_CODE,
            _call(STAGE_AGENT[PHASE_CODE], stage_prompt(state, PHASE_CODE)))


def run_contract(state: OrchestrationState) -> None:
    """阶段1.5：确定性契约生成（不走 LLM subagent，由 contract_gen.py 执行）。"""
    state.record("阶段1.5：契约生成（确定性）")
    # 契约阶段由确定性 runner 执行，不派发 subagent
    # 产物由 contract_gen.py 生成并落盘
    advance(state, PHASE_CONTRACT, "<contract-gen deterministic output>")


def run_review(state: OrchestrationState) -> None:
    state.record("阶段4：代码审查")
    advance(state, PHASE_REVIEW,
            _call(STAGE_AGENT[PHASE_REVIEW], stage_prompt(state, PHASE_REVIEW)))


def run_test(state: OrchestrationState) -> None:
    state.record("阶段5：测试生成")
    advance(state, PHASE_TEST,
            _call(STAGE_AGENT[PHASE_TEST], stage_prompt(state, PHASE_TEST)))


def maybe_retry_code(state: OrchestrationState) -> bool:
    """审查阻塞且未超重试上限 → 回退编码重做（workflow.md 快速修复机制）。

    注意：重试计数由 _next_phase 在 review→fix 迁移时递增，此处**不再自增**，
    否则一次回退会自增两次，使 MAX_AUTO_RETRIES 的实际允许次数减半。
    """
    if not state.blocking_issues:
        return False
    if state.retries >= MAX_AUTO_RETRIES:
        state.record(f"已达自动重试上限({MAX_AUTO_RETRIES})，转人工")
        return False
    state.record(f"自动回退重做代码（第 {state.retries} 次）")
    advance(state, PHASE_FIX,
            _call(STAGE_AGENT[PHASE_FIX], stage_prompt(state, PHASE_FIX)))
    state.blocking_issues = []  # 重置，下一轮 review 再判
    return True


def run_workflow(
    task: str, user_request: str, dispatcher: Optional[DispatchFn] = None
) -> OrchestrationState:
    """按 workflow.md 阶段链自动调度（内存模式，供 eval / dry-run 使用）。

    驱动方式刻意与 hooks 路径一致：都由 `next_instruction()` 决定下一阶段、
    由 `advance()` 推进。这样本函数在 CI 里跑通，等价于验证了 hooks 在会话里
    跑的那套状态机（eval 绿灯才有意义）。
    """
    if dispatcher:
        set_dispatcher(dispatcher)
    state = new_state(task, user_request)
    print(f"=== 编排启动: {task} ===", file=sys.stderr)

    result = next_instruction(state)
    while result.next_stage is not None:
        stage = result.next_stage
        if stage == PHASE_CONTRACT:
            output = "<contract-gen deterministic output>"
        else:
            output = _call(STAGE_AGENT[stage], result.next_prompt or "")
        result = advance(state, stage, output)

    # 离线模式自动批准合并门禁（人工 merge 仅在 hook 路径需要，见 orchestrator_hook.cmd_merge）
    if state.phase == PHASE_MERGE:
        state.phase = PHASE_DONE

    _save(task, "summary", "summary.json", json.dumps({
        "task": task,
        "retries": state.retries,
        "needs_human": state.needs_human,
        "blocking_issues": state.blocking_issues,
        "stages": ["scope", "design(skip?)", "code", "review", "fix?", "test"],
        "artifacts": {
            "scope": str(state.scope_ref),
            "design": str(state.design_doc.path) if state.design_doc else None,
            "code": str(state.code.path) if state.code else None,
            "review": str(state.review.path) if state.review else None,
            "tests": str(state.tests.path) if state.tests else None,
        },
    }, ensure_ascii=False, indent=2))
    return state


def main() -> int:
    ap = argparse.ArgumentParser(description="多 Agent 工作流编排器（dry-run 默认）")
    ap.add_argument("--task", required=True, help="任务名，用作 .codebuddy/run/{task}/ 目录")
    ap.add_argument("--request", required=True, help="用户需求描述")
    args = ap.parse_args()
    # 未注入 dispatcher → dry-run，仅验证编排结构
    state = run_workflow(args.task, args.request)
    # CLI 的最终报告进 stdout（这是本命令的产品输出）；过程日志已全部走 stderr，
    # 因此本模块被 hook 层 import 时不会污染 stdout 消息通道。
    print(json.dumps({
        "task": state.task,
        "phase": state.phase,
        "retries": state.retries,
        "needs_human": state.needs_human,
        "blocking_issues": state.blocking_issues,
        "artifacts": {k: str(v.path) for k, v in state.artifacts.items() if v},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
