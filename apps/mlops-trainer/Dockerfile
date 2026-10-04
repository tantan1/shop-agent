# syntax=docker/dockerfile:1
# mlops-trainer：独立训练/评测 worker 镜像。
#
# 设计：复用已验证可用的训练栈基础镜像 mlops-trainer-base（含 torch==2.5.1+cu121 +
# transformers/peft/bitsandbytes/datasets/accelerate + 训练/评测脚本 + fastapi/uvicorn），
# 仅覆盖最新脚本与 trainer 服务进程，避免重新解析训练依赖引发 torch 版本冲突。
# 训练/评测完全在独立 GPU 容器中执行，与在线 serving 进程/故障域隔离。
#
# 注：mlops-trainer-base 由当前 shop-agent 镜像 tag 而来（含完整训练栈）。
# Phase-2 可改为纯训练基础镜像（python:3.11-slim + 仅训练依赖）进一步瘦身。
FROM mlops-trainer-base:latest

# 用最新脚本（含 PEFT 适配器加载修复等）覆盖基础镜像内版本
COPY apps/shop-agent/scripts/train/sft/train_unified_sft.py /code/scripts/train/sft/train_unified_sft.py
COPY apps/shop-agent/scripts/eval/param/eval_sft_before_after.py /code/scripts/eval/param/eval_sft_before_after.py
COPY apps/mlops-trainer/app.py /code/mlops_trainer_app.py

# 基础镜像以 appuser(uid 1000)运行；训练产物需写入共享卷 mlops_artifacts，
# 故 trainer 以 root 运行（serving 侧对该卷只读，不受影响）。
USER root

EXPOSE 8002
CMD ["sh", "-c", "cd /code && python -m uvicorn mlops_trainer_app:app --host 0.0.0.0 --port 8002"]
