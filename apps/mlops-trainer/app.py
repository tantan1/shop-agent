"""mlops-trainer：独立的训练 / 评测 worker 服务。

设计目标（训练端解耦）：
- 训练与评测不再由 shop-agent 进程内 subprocess 执行，而是提交到本服务，
  在独立 GPU 容器中运行训练/评测脚本，进程与故障域与在线 serving 完全隔离。
- shop-agent 通过 httpx 提交任务并轮询 /jobs/{id} 拿结果，状态机仍在 shop-agent 侧
  （单一事实源），本服务只负责"把脚本跑完 + 回报状态"。
- 训练/评测脚本路径、模型、数据、产物目录均通过同名挂载在 shop-agent 与 trainer 间共享：
    /code/scripts/...        训练&评测脚本（构建时拷入）
    /code/models/Qwen3-1.7B  基座模型（ro 挂载）
    /code/data/llamafactory  训练&评测数据集（ro 挂载）
    /code/mlops_artifacts    训练产物与评测 JSON（rw 共享卷）
"""
from __future__ import annotations

import asyncio
import subprocess
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="mlops-trainer")

# 内存态任务表（MVP：单实例足够；多副本需换外部存储）
JOBS: Dict[str, Dict[str, Any]] = {}

TRAIN_SCRIPT = "/code/scripts/train/sft/train_unified_sft.py"
EVAL_SCRIPT = "/code/scripts/eval/param/eval_sft_before_after.py"


class TrainReq(BaseModel):
    model: str
    data: str
    output_dir: str
    smoke: bool = False
    max_samples: Optional[int] = None
    merge: bool = False
    merge_out: Optional[str] = None


class EvalReq(BaseModel):
    base: str
    sft: str
    data: str
    device: str = "cuda"
    max_samples: Optional[int] = None
    out: str


async def _run(cmd: list):
    """执行脚本，stdout/stderr 合并捕获，返回 (returncode, stdout_bytes)。"""
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd="/code",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    return proc.returncode, stdout


def _tail(out: bytes, n: int = 80) -> str:
    return "\n".join((out or b"").decode(errors="replace").strip().splitlines()[-n:])


async def _train_job(job_id: str, req: TrainReq):
    try:
        cmd = ["python", TRAIN_SCRIPT, "--model", req.model, "--data", req.data,
               "--output-dir", req.output_dir]
        if req.smoke:
            cmd += ["--smoke", "--max-samples", str(req.max_samples or 40)]
        if req.merge:
            cmd += ["--merge", "--merge-out", req.merge_out]
        rc, out = await _run(cmd)
        if rc != 0:
            JOBS[job_id].update(status="failed", error=_tail(out))
        else:
            JOBS[job_id].update(status="success", result={"output_dir": req.output_dir})
    except Exception as e:  # noqa: BLE001
        JOBS[job_id].update(status="failed", error=repr(e))


async def _eval_job(job_id: str, req: EvalReq):
    try:
        cmd = ["python", EVAL_SCRIPT, "--base", req.base, "--sft", req.sft,
               "--data", req.data, "--device", req.device, "--out", req.out]
        if req.max_samples:
            cmd += ["--max-samples", str(req.max_samples)]
        rc, out = await _run(cmd)
        if rc != 0:
            JOBS[job_id].update(status="failed", error=_tail(out))
        else:
            JOBS[job_id].update(status="success", result={"out": req.out})
    except Exception as e:  # noqa: BLE001
        JOBS[job_id].update(status="failed", error=repr(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/jobs/train")
async def train(req: TrainReq):
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "running"}
    asyncio.create_task(_train_job(job_id, req))
    return {"job_id": job_id}


@app.post("/jobs/eval")
async def eval(req: EvalReq):
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "running"}
    asyncio.create_task(_eval_job(job_id, req))
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "job not found")
    return JOBS[job_id]
