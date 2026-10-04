#!/bin/bash
set -e

# 使用本地已下载的模型
MODEL_PATH=/models/Qwen2.5-1.5B-Instruct

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name qwen3-unified \
    --max-model-len 2048 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml