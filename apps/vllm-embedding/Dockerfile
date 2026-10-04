# syntax=docker/dockerfile:1
# vLLM bge-small-zh-v1.5 嵌入服务
# 使用官方 vLLM 镜像，挂载本地模型

FROM vllm/vllm-openai:v0.21.0-cu129-ubuntu2404

WORKDIR /app

ENV MODEL_NAME=BAAI/bge-small-zh-v1.5
ENV MAX_MODEL_LEN=512
ENV TENSOR_PARALLEL_SIZE=1
ENV GPU_MEMORY_UTILIZATION=0.3

# 支持本地模型挂载：/models/bge-small-zh-v1.5
# 若本地存在则优先使用，否则自动从 HF 下载
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]