#!/usr/bin/env bash
# port-forward-k8s.sh - 将 k8s shop-agent 命名空间的服务映射到本地 localhost
# 用法: ./port-forward-k8s.sh [start|stop|status]
# 默认无参数时执行 start

set -euo pipefail

NAMESPACE="${K8S_NAMESPACE:-shop-agent}"
PIDFILE="$(cd "$(dirname "$0")" && pwd)/.port-forward-pids"

declare -A SERVICES=(
  [redis]="6379:6379"
  [postgres]="5432:5432"
  [gateway]="8001:80"
  [order-service]="8080:80"
  [milvus]="19530:19530"
  [langfuse-web]="3000:3000"
  [ollama]="11434:11434"
  [skywalking-oap]="11800:11800"
  [otel-gateway]="4317:4317"
)

start() {
  echo "==> [port-forward] 启动 k8s 服务映射 (namespace=${NAMESPACE})"
  mkdir -p "$(dirname "$PIDFILE")"
  > "$PIDFILE"

  for svc in "${!SERVICES[@]}"; do
    ports="${SERVICES[$svc]}"
    local_port="${ports%%:*}"
    remote_port="${ports##*:}"
    if lsof -i :"$local_port" >/dev/null 2>&1; then
      echo "    ⚠️  本地端口 $local_port 已被占用，跳过 $svc"
      continue
    fi
    kubectl -n "$NAMESPACE" port-forward "svc/$svc" "$local_port:$remote_port" >/dev/null 2>&1 &
    pid=$!
    if kill -0 "$pid" >/dev/null 2>&1; then
      echo "    ✅ $svc -> localhost:$local_port (pid=$pid)"
      echo "$svc|$local_port|$remote_port|$pid" >> "$PIDFILE"
    else
      echo "    ❌ $svc 启动失败"
    fi
  done

  echo ""
  echo "    查看状态: $0 status"
  echo "    停止所有: $0 stop"
}

stop() {
  echo "==> [port-forward] 停止所有映射"
  if [[ ! -f "$PIDFILE" ]]; then
    echo "    未找到 PID 文件 ($PIDFILE)"
    return 0
  fi
  while IFS='|' read -r svc local_port remote_port pid; do
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" 2>/dev/null || true
      echo "    🛑 $svc (localhost:$local_port) 已停止"
    fi
  done < "$PIDFILE"
  rm -f "$PIDFILE"
}

status() {
  echo "==> [port-forward] 当前映射状态"
  if [[ ! -f "$PIDFILE" ]]; then
    echo "    无活跃映射（PID 文件不存在）"
    return 0
  fi
  printf "    %-20s %-15s %-15s %s\n" "SERVICE" "LOCAL" "REMOTE" "PID"
  printf "    %-20s %-15s %-15s %s\n" "-------" "-----" "------" "---"
  while IFS='|' read -r svc local_port remote_port pid; do
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      state="running"
    else
      state="stopped"
    fi
    printf "    %-20s %-15s %-15s %s (%s)\n" "$svc" "$local_port" "$remote_port" "$pid" "$state"
  done < "$PIDFILE"
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *)
    echo "用法: $0 [start|stop|status]"
    exit 1
    ;;
esac
