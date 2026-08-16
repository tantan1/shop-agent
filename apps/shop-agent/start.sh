#!/bin/sh
# shop-agent 启动脚本
# DEBUG=1    → 启动 debugpy 监听 0.0.0.0:5678（VSCode 远程 attach）
# DEBUG_WAIT_FOR_CLIENT=1 → 额外等待 VSCode 客户端连接才放行 uvicorn
# 否则正常以生产模式启动 uvicorn。
set -e

if [ "${DEBUG:-0}" = "1" ] && [ "${DEBUG_WAIT_FOR_CLIENT:-0}" = "1" ]; then
  exec python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m uvicorn src.main:app --host 0.0.0.0 --port 8000
elif [ "${DEBUG:-0}" = "1" ]; then
  exec python -m debugpy --listen 0.0.0.0:5678 -m uvicorn src.main:app --host 0.0.0.0 --port 8000
else
  exec uvicorn src.main:app --host 0.0.0.0 --port 8000
fi
