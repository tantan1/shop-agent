#!/usr/bin/env bash
# MLOps 发布钩子（1 周 MVP 简化版：换路径 + 重启 serving）。
# 由 MLOps 模块在确认发布后调用，传入新模型路径（第 1 个参数）。
# 环境变量 MODEL_PATH 也已注入，二者等价，脚本任选其一。
set -euo pipefail

NEW_MODEL="${1:-${MODEL_PATH}}"
if [ -z "${NEW_MODEL}" ]; then
  echo "[publish] 未提供模型路径，退出" >&2
  exit 1
fi

echo "[publish] 新模型: ${NEW_MODEL}"

# 将新模型路径写入 serving 读取的 active 文件（与 config.MLOPS_ACTIVE_MODEL_FILE 对应）
ACTIVE_FILE="models/active_model.txt"
echo "${NEW_MODEL}" > "${ACTIVE_FILE}"
echo "[publish] 已写入 ${ACTIVE_FILE}"

# 重启 vLLM serving（短停机可接受；生产应替换为滚动/零停机方案）
if command -v docker >/dev/null 2>&1; then
  echo "[publish] 重启 vllm-qwen3 容器"
  docker restart vllm-qwen3 || echo "[publish] docker restart 失败（请手动重启）" >&2
else
  echo "[publish] 未检测到 docker，请手动以 ${NEW_MODEL} 重启 serving" >&2
fi

echo "[publish] 完成"
