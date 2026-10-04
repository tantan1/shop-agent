# syntax=docker/dockerfile:1
# 多阶段构建 + BuildKit
# DEBUG=1 启用 debugpy 远程调试端口 5678
#
# P2 线性头权重与元信息已随源码打包在 src/modules/chat/agent/assets/
# （p2_linear_head.onnx + head_meta.json），由下方 COPY ./ /code 一并固化进镜像，
# 构建时无需从外部下载任何 embedding 模型或 torch 权重；embedding 推理统一走 vLLM 端点。

# 训练栈(torch/peft/bitsandbytes/datasets/accelerate)已解耦到独立 mlops-trainer 镜像，
# 本 serving 镜像保持精简，只装在线推理所需依赖；embedding 推理统一走 vLLM 端点。
# 注意：若启用 LOCAL_MODEL_BACKEND=local（本地 torch 推理），需改用 mlops-trainer 基础镜像或另行安装 torch。
FROM python:3.11-slim AS builder
WORKDIR /build

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

COPY ./requirements.txt /build/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --timeout 600 --retries 20 --prefix=/install \
    -r /build/requirements.txt -i ${PIP_INDEX_URL}

FROM python:3.11-slim AS runtime
WORKDIR /code

COPY --from=builder /install /usr/local
COPY ./ /code

# K 维度安全基线：以非 root 用户运行，避免容器以 root 权限运行
# 预建数据集挂载点：镜像内先建好并随 chown 归 appuser，卷初始化时才能继承正确属主
# （若挂载点在镜像中不存在，Docker 会以 root 创建，运行用户将无写权限）。
RUN groupadd --gid 1000 appgroup \
 && useradd --uid 1000 --gid appgroup --home-dir /code appuser \
 && mkdir -p /code/mlops_datasets \
 && chown -R appuser:appgroup /code \
 && chmod +x /code/start.sh
USER 1000

EXPOSE 8000
# DEBUG=1 时启动 debugpy，监听 0.0.0.0:5678（VSCode 远程 attach）
ARG DEBUG=0
ENV DEBUG=${DEBUG}
ARG DEBUG_WAIT_FOR_CLIENT=0
ENV DEBUG_WAIT_FOR_CLIENT=${DEBUG_WAIT_FOR_CLIENT}
# start.sh 封装 DEBUG/DEBUG_WAIT_FOR_CLIENT 分支逻辑；JSON 表单 CMD 有利于信号传递
CMD ["sh", "/code/start.sh"]
