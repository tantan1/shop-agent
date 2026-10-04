# syntax=docker/dockerfile:1
# vLLM bge-reranker 重排序服务
# 使用官方 vLLM 镜像，挂载本地模型

FROM vllm/vllm-openai:v0.21.0-cu129-ubuntu2404

WORKDIR /app

ENV MODEL_NAME=BAAI/bge-reranker-base
ENV MAX_MODEL_LEN=512
ENV TENSOR_PARALLEL_SIZE=1
ENV GPU_MEMORY_UTILIZATION=0.8

# 支持本地模型挂载：/models/bge-reranker-base
# 若本地存在则优先使用，否则自动从 HF 下载
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]