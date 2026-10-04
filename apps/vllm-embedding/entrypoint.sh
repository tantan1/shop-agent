#!/bin/bash
set -e

if [ -d /models/bge-small-zh-v1.5 ] && [ -f /models/bge-small-zh-v1.5/config.json ]; then
    echo "使用本地模型 /models/bge-small-zh-v1.5"
    MODEL_PATH=/models/bge-small-zh-v1.5
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
    --port 8000 \
    --enforce-eager \
    --disable-custom-all-reduce