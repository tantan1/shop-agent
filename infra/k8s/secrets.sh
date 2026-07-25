#!/usr/bin/env bash
# 读取 secrets.env 并创建 Kubernetes Secrets。
# 用法：./secrets.sh [secrets.env]
#   secrets.env 格式（每行 KEY=VALUE，# 开头为注释）：
#     REDIS_AUTH=xxx
#     POSTGRES_PASSWORD=xxx
#     OPENAI_API_KEY=sk-xxx
#     GRAFANA_API_KEY=xxx
#     ...
#   未指定文件时默认读取同目录下的 secrets.env。
#   文件不存在时使用环境变量或内置默认值。
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-shop-agent}"

# ── 读取 secrets.env ─────────────────────────────────────────────
SECRETS_FILE="${1:-$DIR/secrets.env}"
declare -A S=()

if [[ -f "$SECRETS_FILE" ]]; then
  echo "==> 读取密钥文件: $SECRETS_FILE"
  while IFS= read -r line; do
    line="${line%%$'\r'}"          # 去掉 Windows 换行符
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    key="${line%%=*}"
    val="${line#*=}"
    S["$key"]="$val"
  done < "$SECRETS_FILE"
else
  echo "⚠️  未找到 $SECRETS_FILE，使用环境变量或默认值"
fi

# 辅助：取 S[key] → env[key] → default
get() {
  local key="$1" default="${2:-}"
  echo "${S[$key]:-${!key:-$default}}"
fi

# ── 确保 namespace 存在 ──────────────────────────────────────────
kubectl get ns "$NAMESPACE" &>/dev/null || kubectl create ns "$NAMESPACE"

# ── 创建 app-secrets ─────────────────────────────────────────────
echo "==> 创建 secret: app-secrets"
kubectl create secret generic app-secrets -n "$NAMESPACE" \
  --from-literal=REDIS_AUTH="$(get REDIS_AUTH local-redis-password)" \
  --from-literal=POSTGRES_PASSWORD="$(get POSTGRES_PASSWORD local-postgres-password)" \
  --from-literal=OPENAI_API_KEY="$(get OPENAI_API_KEY "")" \
  --from-literal=AZURE_OPENAI_API_KEY="$(get AZURE_OPENAI_API_KEY "")" \
  --from-literal=AZURE_OPENAI_ENDPOINT="$(get AZURE_OPENAI_ENDPOINT "")" \
  --from-literal=FIXED_API_KEY="$(get FIXED_API_KEY local-fixed-key)" \
  --from-literal=AWS_ACCESS_KEY_ID="$(get AWS_ACCESS_KEY_ID local)" \
  --from-literal=AWS_SECRET_ACCESS_KEY="$(get AWS_SECRET_ACCESS_KEY local)" \
  --from-literal=S3_ENDPOINT="$(get S3_ENDPOINT "")" \
  --from-literal=LANGFUSE_INIT_USER_PASSWORD="$(get LANGFUSE_INIT_USER_PASSWORD local-langfuse-password)" \
  --from-literal=ORDER_DATABASE_URL="$(get ORDER_DATABASE_URL "postgres://postgres:$(get POSTGRES_PASSWORD local-postgres-password)@postgres:5432/order_service")" \
  --from-literal=LANGFUSE_DATABASE_URL="$(get LANGFUSE_DATABASE_URL "postgres://postgres:$(get POSTGRES_PASSWORD local-postgres-password)@postgres:5432/langfuse")" \
  --dry-run=client -o yaml | kubectl apply -f -

# ── 创建 grafana-api-token ───────────────────────────────────────
echo "==> 创建 secret: grafana-api-token"
kubectl create secret generic grafana-api-token -n "$NAMESPACE" \
  --from-literal=GRAFANA_API_KEY="$(get GRAFANA_API_KEY local-grafana-token)" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "✅ Secrets 已就绪 ($NAMESPACE)"
kubectl -n "$NAMESPACE" get secrets
