#!/usr/bin/env bash
# 一键部署完整系统到 OKE（含托管层之外的全部自建中间件 + 业务服务）。
# 用法：
#   ./infra/k8s/apply-all.sh            # 默认：核心链路 + 业务服务
#   FULL=1 ./infra/k8s/apply-all.sh      # 含重组件（milvus/clickhouse/nebula）
#
# 前置：已配置 kubectl 指向 OKE 集群，且已 export 以下变量（CI 自动注入）：
#   IMAGE_PREFIX  IMAGE_TAG  IMAGE_PULL_POLICY  STORAGE_CLASS  INGRESS_HOST
#   + versions.env 中所有镜像版本变量（脚本自动 source）
#   + app-secrets 各密钥（由 CI 的 deploy.yml 注入；本地需先 kubectl create secret）
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 统一版本常量（versions.env 须存在）
set -a; source "$DIR/versions.env"; set +a

# 部署参数兜底
export IMAGE_PREFIX="${IMAGE_PREFIX:-}"
export IMAGE_TAG="${IMAGE_TAG:-local}"
export IMAGE_PULL_POLICY="${IMAGE_PULL_POLICY:-IfNotPresent}"
export STORAGE_CLASS="${STORAGE_CLASS:-oci-bv}"
export INGRESS_HOST="${INGRESS_HOST:-api.local}"
export S3_ENDPOINT="${S3_ENDPOINT:-}"
export OCI_REGION="${OCI_REGION:-}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"

apply() { envsubst < "$1" | kubectl apply -f -; }

# 1) 命名空间 + 密钥声明（app-secrets 需先存在）
apply "$DIR/00-namespace.yaml"
apply "$DIR/01-secrets.yaml"            || true   # 占位，真实值由 CI 注入
apply "$DIR/middleware/object-storage-secret.yaml"

# 2) 核心链路中间件（默认开启）
for f in redis postgres-pgvector prometheus grafana skywalking langfuse loki; do
  apply "$DIR/middleware/$f.yaml"
done
# Grafana 观测配置（datasource + 仪表盘模板 + JSON）
for f in grafana-datasource grafana-dashboards-provider grafana-dashboard-shop-agent grafana-dashboard-loki-logs; do
  apply "$DIR/middleware/$f.yaml"
done

# 2b) 本地小模型（LOCAL_MODEL=1 时开启；Ollama + unified 1.7B，模型权重走 PVC）
if [[ "${LOCAL_MODEL:-0}" == "1" ]]; then
  apply "$DIR/middleware/ollama.yaml"
fi

# 3) 业务服务
for f in shop-agent gateway monitoring-agent order-service; do
  apply "$DIR/$f.yaml"
done

# 4) 重组件（FULL=1 时开启；Always Free 默认关）
if [[ "${FULL:-0}" == "1" ]]; then
  for f in milvus clickhouse nebula; do
    apply "$DIR/middleware/$f.yaml"
  done
fi

# 5) 入口
apply "$DIR/ingress.yaml"

echo "✅ 完整系统已 apply。核心链路 + 业务服务已部署；重组件状态: ${FULL:+已开启}${FULL:-已跳过(设 FULL=1 开启)}；本地小模型(Ollama): ${LOCAL_MODEL:+已开启}${LOCAL_MODEL:-已跳过(设 LOCAL_MODEL=1 开启)}"
