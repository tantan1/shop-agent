# syntax=docker/dockerfile:1
# vLLM qwen3-unified 统一小模型服务（参数抽取 + 工具选择）
# 使用官方 vLLM 镜像，挂载本地模型

FROM vllm/vllm-openai:v0.21.0-cu129-ubuntu2404

WORKDIR /app

ENV MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
ENV MAX_MODEL_LEN=2048
ENV TENSOR_PARALLEL_SIZE=1
ENV GPU_MEMORY_UTILIZATION=0.8

# 支持本地模型挂载：/models/qwen3-unified
# 若本地存在则优先使用，否则自动从 HF 下载
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]