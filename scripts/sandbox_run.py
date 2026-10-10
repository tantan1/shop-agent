#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""沙箱运行器（方案② T1）：在隔离环境真实执行生成代码 + 测试，采集运行证据。

优先 Docker 容器（--network=none、内存/CPU 上限、超时强制 kill）；
无容器环境降级为子进程 + timeout + ulimit（不抛异常）。
输出 JSON：{ran, exit_code, runtime_errors, wall_time, killed_by: null|timeout|oom}
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _run_docker(
    work_dir: Path,
    test_path: str,
    requirements: str = "",
    timeout: int = 120,
    memory_mb: int = 512,
    cpu_quota: int = 50000,
) -> dict:
    """Docker 容器内运行 pytest。"""
    import hashlib
    container_name = f"sandbox_{hashlib.md5(str(work_dir).encode()).hexdigest()[:8]}"
    image = "python:3.11-slim"
    # 构建 Dockerfile + context
    build_dir = work_dir / ".sandbox_build"
    build_dir.mkdir(exist_ok=True)
    df = build_dir / "Dockerfile"
    df.write_text(
        f"FROM {image}\n"
        "WORKDIR /workspace\n"
        "COPY . /workspace\n"
        "RUN pip install --no-cache-dir -q pytest\n",
        encoding="utf-8",
    )
    if requirements and (work_dir / requirements).exists():
        df.write_text(
            df.read_text(encoding="utf-8") +
            f"RUN pip install --no-cache-dir -q -r {requirements}\n",
            encoding="utf-8",
        )

    # 构建镜像
    build_cmd = [
        "docker", "build", "-t", f"sandbox_img_{container_name}", str(build_dir)
    ]
    subprocess.run(build_cmd, cwd=str(work_dir), capture_output=True, check=False)

    # 运行容器
    run_cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "--network=none",
        "-m", f"{memory_mb}m",
        "--cpus", str(cpu_quota / 100000),
        "-v", f"{work_dir}:/workspace",
        f"sandbox_img_{container_name}",
        "python", "-m", "pytest", test_path, "-q", "--no-header", "-p", "no:cacheprovider",
    ]
    wall_start = time.perf_counter()
    try:
        proc = subprocess.run(
            run_cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        wall_time = time.perf_counter() - wall_start
        result = {
            "ran": True,
            "exit_code": proc.returncode,
            "runtime_errors": proc.stderr.strip() if proc.stderr else "",
            "wall_time": round(wall_time, 3),
            "killed_by": None,
        }
    except subprocess.TimeoutExpired:
        wall_time = time.perf_counter() - wall_start
        # 尝试杀掉容器
        subprocess.run(["docker", "kill", container_name], capture_output=True)
        result = {
            "ran": False,
            "exit_code": -1,
            "runtime_errors": "container timeout",
            "wall_time": round(wall_time, 3),
            "killed_by": "timeout",
        }
    except MemoryError:
        result = {
            "ran": False,
            "exit_code": -1,
            "runtime_errors": "oom",
            "wall_time": 0,
            "killed_by": "oom",
        }
    except Exception as exc:
        result = {
            "ran": False,
            "exit_code": -1,
            "runtime_errors": str(exc),
            "wall_time": 0,
            "killed_by": None,
        }
    finally:
        # 确保 runtime_errors 在非零退出时有内容
        if result.get("exit_code") not in (0, None) and not result.get("runtime_errors"):
            result["runtime_errors"] = f"docker exit_code={result['exit_code']}"
        # 清理镜像
        subprocess.run(["docker", "rmi", "-f", f"sandbox_img_{container_name}"], capture_output=True)
        # 清理 build dir
        shutil.rmtree(build_dir, ignore_errors=True)
    return result


def _run_subprocess(
    work_dir: Path,
    test_path: str,
    requirements: str = "",
    timeout: int = 120,
) -> dict:
    """降级路径：子进程 + timeout（Windows 无 ulimit 等价物，仅 timeout）。"""
    # 安装依赖
    if requirements and (work_dir / requirements).exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(work_dir / requirements)],
            capture_output=True,
            check=False,
        )

    cmd = [
        sys.executable, "-m", "pytest",
        str(work_dir / test_path),
        "-q", "--no-header", "-p", "no:cacheprovider",
    ]
    wall_start = time.perf_counter()
    result = {
        "ran": False,
        "exit_code": -1,
        "runtime_errors": "",
        "wall_time": 0,
        "killed_by": None,
    }
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(work_dir),
        )
        wall_time = time.perf_counter() - wall_start
        result = {
            "ran": True,
            "exit_code": proc.returncode,
            "runtime_errors": proc.stderr.strip() if proc.stderr else "",
            "wall_time": round(wall_time, 3),
            "killed_by": None,
        }
    except subprocess.TimeoutExpired:
        wall_time = time.perf_counter() - wall_start
        result = {
            "ran": False,
            "exit_code": -1,
            "runtime_errors": "process timeout",
            "wall_time": round(wall_time, 3),
            "killed_by": "timeout",
        }
    except MemoryError:
        result = {
            "ran": False,
            "exit_code": -1,
            "runtime_errors": "oom",
            "wall_time": 0,
            "killed_by": "oom",
        }
    except Exception as exc:
        result = {
            "ran": False,
            "exit_code": -1,
            "runtime_errors": str(exc),
            "wall_time": 0,
            "killed_by": None,
        }
    # 确保 runtime_errors 在非零退出时有内容
    if result["exit_code"] not in (0, None) and not result["runtime_errors"]:
        result["runtime_errors"] = f"pytest exit_code={result['exit_code']}"
    return result


def run_sandbox(
    work_dir: Path,
    test_path: str,
    requirements: str = "",
    timeout: int = 120,
    memory_mb: int = 512,
) -> dict:
    """统一入口：Docker 可用则容器化，否则降级子进程。"""
    work_dir = work_dir.resolve()
    if _docker_available():
        return _run_docker(work_dir, test_path, requirements, timeout, memory_mb)
    return _run_subprocess(work_dir, test_path, requirements, timeout)


def main() -> int:
    ap = argparse.ArgumentParser(description="沙箱运行器：隔离执行代码 + 测试")
    ap.add_argument("--work-dir", required=True, help="工作目录（含代码/测试）")
    ap.add_argument("--test-path", required=True, help="测试文件相对 work-dir 的路径")
    ap.add_argument("--requirements", default="", help="requirements 文件名（相对 work-dir）")
    ap.add_argument("--timeout", type=int, default=120, help="超时秒数")
    ap.add_argument("--memory-mb", type=int, default=512, help="内存上限（MB，仅 Docker）")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    if not work_dir.exists():
        print(json.dumps({"ran": False, "error": f"work_dir not found: {work_dir}"}, ensure_ascii=False))
        return 2

    result = run_sandbox(work_dir, args.test_path, args.requirements, args.timeout, args.memory_mb)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ran") and result.get("exit_code") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
