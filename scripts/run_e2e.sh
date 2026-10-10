#!/usr/bin/env bash
# shop-agent e2e 冒烟一键脚本（Linux/macOS）
#
# 动作：拉起 docker-compose 中 shop-agent 及其依赖 -> 等 /health 就绪
#       -> 运行 tests/e2e/smoke_e2e.py（覆盖 #2/#4/#7/#8）-> 默认回收容器
#
# 用法：
#   ./run_e2e.sh            # 跑完自动 down
#   ./run_e2e.sh -k         # 保留容器（-Keep）
#   ./run_e2e.sh -n         # 假定服务已在跑，只执行冒烟（-NoUp）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE="$REPO_ROOT/apps/shop-agent/tests/e2e/smoke_e2e.py"
COMPOSE="$REPO_ROOT/docker-compose.yml"

KEEP=0
NOUP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k|--keep) KEEP=1 ;;
    -n|--noup) NOUP=1 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
  shift
done

# FIXED_API_KEY：优先环境变量，否则从 .env 读取，否则默认
if [[ -z "${FIXED_API_KEY:-}" && -f "$REPO_ROOT/.env" ]]; then
  FIXED_API_KEY="$(grep -E '^[[:space:]]*FIXED_API_KEY[[:space:]]*=' "$REPO_ROOT/.env" | head -n1 | sed -E 's/.*=[[:space:]]*//' | tr -d '\"'"'"'')"
fi
export FIXED_API_KEY="${FIXED_API_KEY:-test-key-for-pytest}"
export REDIS_HOST="${REDIS_HOST:-localhost}"
export REDIS_PORT="${REDIS_PORT:-6379}"

EXIT_CODE=0
cleanup() {
  if [[ "$NOUP" -eq 0 && "$KEEP" -eq 0 ]]; then
    echo "[run_e2e] 回收 compose 服务..."
    docker compose -f "$COMPOSE" down
  elif [[ "$NOUP" -eq 0 && "$KEEP" -eq 1 ]]; then
    echo "[run_e2e] 已保留容器（-k）。手动回收: docker compose down"
  fi
}
trap cleanup EXIT

if [[ "$NOUP" -eq 0 ]]; then
  echo "[run_e2e] 拉起 compose 服务（redis / shop-agent / gateway / order-service 及依赖）..."
  docker compose -f "$COMPOSE" up -d redis shop-agent gateway order-service
fi

echo "[run_e2e] 等待 shop-agent /health 就绪..."
HEALTH="http://localhost:8000/health"
READY=0
for i in $(seq 1 60); do
  if curl -fsS --max-time 3 "$HEALTH" >/dev/null 2>&1; then READY=1; break; fi
  sleep 3
done
if [[ "$READY" -ne 1 ]]; then
  echo "[run_e2e] shop-agent 超时未就绪，请检查: docker compose logs shop-agent" >&2
  exit 1
fi
echo "[run_e2e] shop-agent 已就绪。"

echo "[run_e2e] 运行 e2e 冒烟脚本..."
python "$SMOKE"
EXIT_CODE=$?
echo "[run_e2e] 冒烟脚本退出码: $EXIT_CODE"

exit $EXIT_CODE
