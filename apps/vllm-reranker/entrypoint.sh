#!/bin/bash
set -e

if [ -d /models/bge-reranker-base ] && [ -f /models/bge-reranker-base/config.json ]; then
    echo "使用本地模型 /models/bge-reranker-base"
    MODEL_PATH=/models/bge-reranker-base
else
    echo "本地模型不存在，从 HF 下载: ${MODEL_NAME}"
    MODEL_PATH=${MODEL_NAME}
fi

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --host 0.0.0.0 \
    --port 8000