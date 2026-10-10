#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多 Agent 编排的 git 集成（方案① PR 模型，防御式、默认关闭）。

把"编排产出"从"落在工作区"升级为"一个 branch + 提交 + PR"，对齐成熟公司的
agentic SDLC：Agent 产物作为 PR 进入真实研发管线，由人类在 merge 点把关，
而非 session 内直接落地。

设计约束（务必遵守）：
  - 默认不生效：仅当环境变量 ORCH_GIT=1 才真正执行 git/gh 操作；
    否则所有函数安全 no-op 返回，绝不触碰用户仓库。
  - 全程防御：单次 git 调用失败只记日志、不抛异常、不阻断编排主流程
    （编排的零干预/不崩溃纪律高于 git 集成）。
  - 只动 .codebuddy/run/{task}/ 下的产物（用 --force 绕过 .gitignore），
    不 git add 整个仓库，避免把无关改动卷进 PR。
  - 本模块只做字节级 git/gh 操作；分支/提交/PR 的"语义时机"由 orchestrator_hook 决定。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / ".codebuddy" / "run"


def _log(msg: str) -> None:
    print(f"[orch-git] {msg}", file=sys.stderr)


def _run(args: list, timeout: int = 60):
    """跑一次子进程，返回 (returncode, combined_output)。任何异常都归一为失败。"""
    try:
        r = subprocess.run(
            args, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as exc:  # 超时 / 命令不存在 / 权限等
        return 1, str(exc)


def git_available() -> bool:
    rc, _ = _run(["git", "--version"])
    return rc == 0


def in_git_repo() -> bool:
    rc, _ = _run(["git", "rev-parse", "--is-inside-work-tree"])
    return rc == 0


def enabled() -> bool:
    """是否真正启用 git 集成（默认关闭，需显式 ORCH_GIT=1）。"""
    return os.environ.get("ORCH_GIT") == "1"


def branch_name(task: str) -> str:
    """每任务专属分支名（方案①）。"""
    return f"orch/{task}"


def ensure_branch(task: str) -> str | None:
    """开启任务时建/切专属分支；失败返回 None（不阻断编排）。"""
    if not (enabled() and git_available() and in_git_repo()):
        return None
    br = branch_name(task)
    rc, out = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if rc == 0 and out.strip() == br:
        return br
    rc, err = _run(["git", "checkout", "-b", br])
    if rc != 0:
        # 分支已存在则直接切换
        rc2, _ = _run(["git", "checkout", br])
        if rc2 != 0:
            _log(f"ensure_branch 失败（{br}）：{err}")
            return None
    _log(f"已切到分支 {br}")
    return br


def commit_task(task: str, message: str) -> bool:
    """归档产物后把 .codebuddy/run/{task}/ 提交进专属分支。"""
    if not (enabled() and git_available() and in_git_repo()):
        return False
    run_dir = RUN_DIR / task
    rc, err = _run(["git", "add", "--force", str(run_dir)])
    if rc != 0:
        _log(f"commit_task add 失败：{err}")
        return False
    rc, err = _run(["git", "commit", "-m", message])
    # rc==1 通常代表"无新改动"，视为成功（避免误报失败）
    if rc not in (0, 1):
        _log(f"commit_task commit 失败：{err}")
        return False
    return True


def open_pr(task: str) -> str | None:
    """合并门禁放行后：推送分支并开 PR，返回 PR URL 或 None。"""
    if not (enabled() and git_available() and in_git_repo()):
        return None
    br = branch_name(task)
    rc, err = _run(["git", "push", "-u", "origin", br])
    if rc != 0:
        _log(f"open_pr push 失败：{err}")
        return None
    rc, out = _run([
        "gh", "pr", "create",
        "--title", f"orch: {task}",
        "--body", "由多 Agent 编排自动生成，请人工 review 后在合并门禁处合并。",
        "--head", br,
    ])
    if rc == 0:
        _log(f"已开 PR：{out.strip()}")
        return out.strip()
    _log(f"open_pr 失败（gh 未安装或鉴权失败）：{out or err}")
    return None
