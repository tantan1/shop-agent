#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""编排状态持久化层：state.json 的字节 IO、原子写、跨进程锁、当前任务指针、路径安全校验。

职责边界（关键，勿越界）：
  本层**不认识任何业务字段**。state.json 的 schema 由领域层
  （agent_orchestrator.state_to_dict / state_from_dict / SCHEMA_VERSION）定义，
  本层只把它们当作一个 opaque dict 做序列化与落盘。这样领域层新增字段时
  本层零改动，避免"schema 在两处各写一遍"的漂移。

为什么需要本层（hooks 改造的命门）：
  CodeBuddy hooks 每次事件都是**独立的新进程**，编排状态无法留在内存里，
  必须落盘后由下一次事件读回。因此本层提供：
    - 读-改-写的原子性（临时文件 + os.replace）
    - 跨进程互斥（POSIX flock / Windows msvcrt.locking）
    - 当前活跃任务指针（.current），让独立进程知道"现在在跑哪个 task"
    - task 名与路径安全校验（防路径穿越）

只依赖标准库，可被 benchmark/eval 直接 import。
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / ".codebuddy" / "run"

# task 名白名单：只允许这些字符，杜绝 "../" 之类的路径穿越
_TASK_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# 当前任务指针文件名（位于 RUN_DIR 根，不是某个 task 子目录）
_CURRENT_FILE = ".current"

# 锁等待上限（秒）。hooks 有 timeout 约束，不能无限等
_LOCK_TIMEOUT_SEC = 5.0

_IS_WINDOWS = os.name == "nt"
if not _IS_WINDOWS:
    import fcntl
else:
    import msvcrt


class StateError(Exception):
    """状态层异常（task 名非法 / 状态文件损坏 / 拿不到锁）。"""


def _log(msg: str) -> None:
    """调试日志一律走 stderr。

    原因：hook 协议中 stdout 是给 Agent 的消息通道（优先级最高），
    任何调试输出写进 stdout 都会被当成 hook 的返回内容。
    """
    print(f"[orchestrator_state] {msg}", file=sys.stderr)


def validate_task(task: str) -> str:
    """校验 task 名，非法即抛 StateError。防路径穿越（`..`、分隔符）。"""
    if not isinstance(task, str) or not task:
        raise StateError("task 名不能为空")
    if not _TASK_RE.match(task):
        raise StateError(
            f"task 名只允许 [A-Za-z0-9_-]，收到：{task!r}"
        )
    return task


def task_dir(task: str) -> Path:
    """某 task 的产物目录 `.codebuddy/run/{task}/`（不自动创建）。"""
    return RUN_DIR / validate_task(task)


def state_path(task: str) -> Path:
    return task_dir(task) / "state.json"


def _lock_path(task: str) -> Path:
    return task_dir(task) / "state.json.lock"


# ── 跨进程文件锁 ────────────────────────────────────────────────────────────
@contextmanager
def _file_lock(path: Path, timeout: float = _LOCK_TIMEOUT_SEC) -> Iterator[None]:
    """跨进程排他锁。POSIX 用 flock，Windows 用 msvcrt.locking。

    官方明确"匹配的 hooks 并行执行"，多个事件同时触发时会并发改写 state.json，
    必须互斥。锁失败（超时）抛 StateError，由调用方决定降级策略，绝不静默丢更新。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        # Windows 下 msvcrt.locking 要求被锁区间真实存在，空文件需先填 1 字节
        if os.fstat(fd).st_size == 0:
            os.write(fd, b"\0")
        os.lseek(fd, 0, os.SEEK_SET)

        deadline = time.monotonic() + timeout
        while True:
            try:
                if _IS_WINDOWS:
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                else:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise StateError(f"获取状态锁超时（{timeout}s）：{path}")
                time.sleep(0.05)
        yield
    finally:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            if _IS_WINDOWS:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass  # 解锁失败不掩盖主流程结果
        finally:
            os.close(fd)


# ── 原子写 ──────────────────────────────────────────────────────────────────
def _atomic_write(path: Path, text: str) -> None:
    """写文件：先写同目录临时文件，再 os.replace 原子替换。

    保证任何时刻读到的 state.json 都是完整内容，不会因进程被杀而留下半个 JSON。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ── 状态读写（对 dict 内容不做任何假设）────────────────────────────────────
def load(task: str) -> Optional[dict]:
    """读取 state.json 并解析为 dict；不存在返回 None，损坏抛 StateError。"""
    validate_task(task)
    p = state_path(task)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise StateError(f"state.json 损坏或不可读：{p}（{exc}）") from exc


def update(task: str, mutator: Callable[[dict], dict]) -> dict:
    """加锁读-改-写，返回写入后的完整 dict。

    mutator 收到当前 dict（若文件不存在则收到 None，需自行处理），返回新 dict。
    整个过程持排他锁，保证并发下不丢更新。
    """
    validate_task(task)
    with _file_lock(_lock_path(task)):
        current = None
        p = state_path(task)
        if p.exists():
            try:
                current = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise StateError(f"state.json 损坏或不可读：{p}（{exc}）") from exc
        new = mutator(current)
        if not isinstance(new, dict):
            raise StateError("mutator 必须返回 dict")
        _atomic_write(p, json.dumps(new, ensure_ascii=False, indent=2))
        return new


def write_new(task: str, envelope: dict) -> dict:
    """创建 state.json（已存在则抛 StateError，避免误覆盖在跑的编排）。"""
    validate_task(task)
    with _file_lock(_lock_path(task)):
        p = state_path(task)
        if p.exists():
            raise StateError(f"编排已存在，拒绝覆盖：{p}（如需重来请先 reset）")
        _atomic_write(p, json.dumps(envelope, ensure_ascii=False, indent=2))
        return envelope


def reset(task: str) -> None:
    """中止并清理某 task 的状态（保留产物文件，便于事后复盘）。"""
    validate_task(task)
    with _file_lock(_lock_path(task)):
        for name in ("state.json",):
            try:
                (task_dir(task) / name).unlink()
            except FileNotFoundError:
                pass
    if current_task() == task:
        clear_current()
    _log(f"reset: {task}")


# ── 当前活跃任务指针 ────────────────────────────────────────────────────────
def current_task() -> Optional[str]:
    """当前活跃 task：优先读 ORCH_TASK 环境变量，否则读 `.codebuddy/run/.current`。

    hook 进程无法从事件 JSON 得知"现在在跑哪个 task"，靠这个指针续接上下文。
    """
    env = (os.environ.get("ORCH_TASK") or "").strip()
    if env:
        return env if _TASK_RE.match(env) else None
    p = RUN_DIR / _CURRENT_FILE
    if not p.exists():
        return None
    try:
        name = p.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return name if _TASK_RE.match(name) else None


def set_current(task: str) -> None:
    """写入当前任务指针。"""
    validate_task(task)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write(RUN_DIR / _CURRENT_FILE, task)


def clear_current() -> None:
    """清除当前任务指针（编排完成或中止时调用）。"""
    try:
        (RUN_DIR / _CURRENT_FILE).unlink()
    except FileNotFoundError:
        pass
