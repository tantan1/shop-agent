# syntax=docker/dockerfile:1
# 多阶段构建 + BuildKit（[cache mount] 跨构建复用 pip 下载缓存）
# 阶段1 builder：安装依赖到独立前缀目录（含编译产物）
# 阶段2 runtime：精简运行镜像，仅复制已安装依赖与源码

FROM python:3.11-slim AS builder
WORKDIR /build

COPY ./requirements.txt /build/requirements.txt

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
# BuildKit cache mount：pip 下载缓存跨构建复用，不与镜像层绑定
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --prefix=/install -r /build/requirements.txt -i ${PIP_INDEX_URL}

FROM python:3.11-slim AS runtime
WORKDIR /code

# 从 builder 复制已安装依赖（含编译产物；runtime 阶段无需编译工具链）
COPY --from=builder /install /usr/local

COPY ./ /code

EXPOSE 9091
CMD ["uvicorn", "monitoring_agent.main:app", "--host", "0.0.0.0", "--port", "9091"]
