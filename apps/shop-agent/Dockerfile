# syntax=docker/dockerfile:1
# 多阶段构建 + BuildKit
# 默认 LITE=1 用轻量依赖（k8s 部署），LITE=0 用全量依赖（本地训练/调试）
# DEBUG=1 启用 debugpy 远程调试端口 5678

FROM python:3.11-slim AS builder
WORKDIR /build

ARG LITE=1
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

COPY ./requirements-lite.txt /build/requirements-lite.txt
COPY ./requirements.txt /build/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$LITE" = "1" ]; then \
      pip install --no-cache-dir --prefix=/install -r /build/requirements-lite.txt -i ${PIP_INDEX_URL}; \
    else \
      pip install --no-cache-dir --prefix=/install -r /build/requirements.txt -i ${PIP_INDEX_URL}; \
    fi

FROM python:3.11-slim AS runtime
WORKDIR /code

COPY --from=builder /install /usr/local
COPY ./ /code

EXPOSE 8000
# DEBUG=1 时启动 debugpy，监听 0.0.0.0:5678（VSCode 远程 attach）
ARG DEBUG=0
ENV DEBUG=${DEBUG}
ARG DEBUG_WAIT_FOR_CLIENT=0
ENV DEBUG_WAIT_FOR_CLIENT=${DEBUG_WAIT_FOR_CLIENT}
# start.sh 封装 DEBUG/DEBUG_WAIT_FOR_CLIENT 分支逻辑；JSON 表单 CMD 有利于信号传递
CMD ["sh", "/code/start.sh"]
