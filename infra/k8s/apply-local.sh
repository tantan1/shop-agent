#!/usr/bin/env bash
# 本机一键部署（方案 A）：docker build 业务镜像 → 加载进本地集群 → 建占位 secret → apply 中间件+业务。
# 适配 docker-desktop / minikube / kind。重组件（milvus/clickhouse/nebula）默认关闭（FULL=1 开启）。
#
# 前置：
#   1) 复制并编辑变量：cp local.env.example local.env
#   2) 本机已装 docker + kubectl，且 kubectl 指向本地集群（docker-desktop/minikube/kind）
#   3) 若用 minikube/kind，脚本会自动把镜像 load 进集群；docker-desktop 直接用本地 docker 镜像。
#
# 用法：./apply-local.sh [STEP ...]
#   无参数：输出帮助并退出。
#   有参数：仅执行指定步骤（按顺序）。
#   示例：./apply-local.sh ns secrets middleware apps ingress expose
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"

# 加载统一镜像版本（versions.env 始终存在）
set -a; source "$DIR/versions.env"; set +a

# 加载本机变量（存在才加载，覆盖 versions.env 中的部署变量如 IMAGE_PREFIX/IMAGE_TAG）
if [[ -f "$DIR/local.env" ]]; then
  set -a; source "$DIR/local.env"; set +a
else
  echo "⚠️ 未找到 local.env，使用示例默认值（请先 cp local.env.example local.env 并按需修改）"
  set -a; source "$DIR/local.env.example"; set +a
fi

# 默认兜底（export 以确保 envsubst 能替换 YAML 中的 ${VAR}）
export IMAGE_PREFIX="${IMAGE_PREFIX:-}"
export IMAGE_TAG="${IMAGE_TAG:-local}"
# 默认 Always：保证 rollout restart 时 kubelet 始终重新拉取，避免「同 tag(local)
# 不重拉 → Pod 跑旧镜像」。若要节省带宽可显式 export IMAGE_PULL_POLICY=IfNotPresent。
export IMAGE_PULL_POLICY="${IMAGE_PULL_POLICY:-Always}"
export STORAGE_CLASS="${STORAGE_CLASS:-standard}"
export INGRESS_HOST="${INGRESS_HOST:-localhost}"
export FULL="${FULL:-0}"
export LOCAL_MODEL="${LOCAL_MODEL:-0}"   # 1=部署 Ollama（本地小模型 unified 1.7B）
export CLUSTER_TYPE="${CLUSTER_TYPE:-docker-desktop}"
# ---- 推理后端切换：ollama（k8s 内）| vllm（宿主机 Docker Desktop 的 vLLM）----
export LLM_BACKEND="${LLM_BACKEND:-ollama}"
if [[ "$LLM_BACKEND" == "vllm" ]]; then
  # 宿主机 Docker Desktop 中的 vLLM（端口来自 docker-compose.vllm.yml）
  # 地址可通过 local.env 的 VLLM_HOST / 端口变量覆盖（host.docker.internal 不通时改节点 IP）
  _vllm_host="${VLLM_HOST:-host.docker.internal}"
  export GATEWAY_VLLM_BASE_URL="http://${_vllm_host}:${VLLM_QWEN3_PORT:-8003}/v1"
  export SHOP_EMBEDDING_PROVIDER="vllm"
  export SHOP_VLLM_EMBEDDING_BASE_URL="http://${_vllm_host}:${VLLM_BGE_M3_PORT:-8101}"
  export SHOP_VLLM_EMBEDDING_MODEL="bge-m3"
  export SHOP_RERANKER_PROVIDER="vllm"
  export SHOP_VLLM_RERANK_BASE_URL="http://${_vllm_host}:${VLLM_RERANKER_PORT:-8102}"
  export SHOP_VLLM_RERANK_MODEL="bge-reranker-base"
  export SHOP_LOCAL_MODEL_BACKEND="vllm"
  export SHOP_VLLM_BASE_URL="http://${_vllm_host}:${VLLM_QWEN3_PORT:-8003}/v1"
  export SHOP_VLLM_PARAM_MODEL="qwen3-unified"
  export SHOP_VLLM_TOOL_SELECTOR_MODEL="qwen3-unified"
  # shop-agent 模型配置（与 shop-agent.yaml 引用对齐）
  export LOCAL_MODEL_BACKEND="vllm"
  export VLLM_TOOL_SELECTOR_MODEL="qwen3-unified"
  export CHAT_MODEL="${CHAT_MODEL:-qwen3.7-plus-2026-05-26}"
  export TOOL_SELECTOR_MODEL="${TOOL_SELECTOR_MODEL:-qwen3-unified}"
else
  # 默认：k8s 内的 Ollama
  export GATEWAY_VLLM_BASE_URL="http://vllm-qwen3:8000/v1"
  export SHOP_EMBEDDING_PROVIDER="ollama"
  export SHOP_VLLM_EMBEDDING_BASE_URL=""
  export SHOP_VLLM_EMBEDDING_MODEL=""
  export SHOP_RERANKER_PROVIDER="local"
  export SHOP_VLLM_RERANK_BASE_URL=""
  export SHOP_VLLM_RERANK_MODEL=""
  export SHOP_LOCAL_MODEL_BACKEND="ollama"
  export SHOP_VLLM_BASE_URL=""
  export SHOP_VLLM_PARAM_MODEL=""
  export SHOP_VLLM_TOOL_SELECTOR_MODEL=""
  # shop-agent 模型配置（ollama 模式：本地小模型走 ollama）
  export LOCAL_MODEL_BACKEND="ollama"
  export VLLM_TOOL_SELECTOR_MODEL=""
  export CHAT_MODEL="${CHAT_MODEL:-qwen3.7-plus-2026-05-26}"
  export TOOL_SELECTOR_MODEL="${TOOL_SELECTOR_MODEL:-qwen3-unified}"
fi
# 注意：LLM API Key（TONGYI_API_KEY）统一由 Gateway 持有，
# 经 app-secrets 注入 Gateway，shop-agent 不再设置/注入 LLM key。
# Mock LLM + 限流（默认关闭/默认值；压测时 export 覆盖：LLM_ADAPTER_TYPE=mock CHAT_RATE_LIMIT=5000 等）
export LLM_ADAPTER_TYPE="${LLM_ADAPTER_TYPE:-langchain}"
export MOCK_LLM_LATENCY_MIN="${MOCK_LLM_LATENCY_MIN:-500}"
export MOCK_LLM_LATENCY_MAX="${MOCK_LLM_LATENCY_MAX:-800}"
export MOCK_LLM_ERROR_RATE="${MOCK_LLM_ERROR_RATE:-0.01}"
export MOCK_LLM_OUTPUT_TOKENS="${MOCK_LLM_OUTPUT_TOKENS:-200}"
# 网关压测开关：1=云端 API(gpt*/claude*/qwen*) 全部改路由到 mock-llm 上游（免真实云调用）
# 0=正常路由云端。本地 vLLM 始终不受影响。
export MOCK_CLOUD_OVERRIDE="${MOCK_CLOUD_OVERRIDE:-0}"
# mock-llm 上游服务（Go，apps/mock-llm）配置：延迟区间 → (基准延迟, 抖动)
export MOCK_LATENCY_MS="${MOCK_LATENCY_MS:-$MOCK_LLM_LATENCY_MIN}"
export MOCK_LATENCY_JITTER_MS="${MOCK_LATENCY_JITTER_MS:-$(( MOCK_LLM_LATENCY_MAX - MOCK_LLM_LATENCY_MIN ))}"
export MOCK_STREAM_CHUNK_MS="${MOCK_STREAM_CHUNK_MS:-0}"
export MOCK_ERROR_RATE="${MOCK_ERROR_RATE:-$MOCK_LLM_ERROR_RATE}"
export MOCK_ERROR_STATUS="${MOCK_ERROR_STATUS:-500}"
export MOCK_OUTPUT_TOKENS="${MOCK_OUTPUT_TOKENS:-$MOCK_LLM_OUTPUT_TOKENS}"
export CHAT_RATE_LIMIT="${CHAT_RATE_LIMIT:-15}"
export GLOBAL_RATE_LIMIT="${GLOBAL_RATE_LIMIT:-30}"
# 本地私有仓库（方案 A）：docker-desktop 默认开启，镜像 push 进集群内 registry:2
export REGISTRY_ENABLED="${REGISTRY_ENABLED:-1}"
# 宿主 push / 节点 pull 地址。注意：必须用 host 可解析的 localhost:5000 形式——
# Docker Desktop 的 kubelet 镜像拉取会被 registry-mirror:1273 转发到宿主 docker 引擎，
# 只有 localhost:5000（经 port-forward 指向集群内 registry）才能被正确代理；
# 若用集群内 DNS registry:5000 会被 mirror 当成外网仓库代理而失败（500）。
export REGISTRY_HOST_ADDR="${REGISTRY_HOST_ADDR:-localhost:5000}"   # push 地址（kubectl port-forward svc/registry）
export REGISTRY_PORT_FORWARD="${REGISTRY_PORT_FORWARD:-1}"          # 1=自动起 port-forward 暴露 registry

# 启用本地仓库时，业务清单改为从 registry 拉取 + Always 必拉新；
# 关闭 registry（REGISTRY_ENABLED=0）或显式设了 IMAGE_PREFIX（如 CI/OKE）则保持原样。
if [[ "$REGISTRY_ENABLED" == "1" ]] && [[ -z "${IMAGE_PREFIX:-}" ]]; then
  export IMAGE_PREFIX="${REGISTRY_HOST_ADDR}/"
  export IMAGE_PULL_POLICY="Always"
fi

# 业务镜像列表：app目录 -> 镜像名
declare -A APPS=( [shop-agent]=apps/shop-agent [gateway]=apps/gateway [monitoring-agent]=apps/monitoring-agent [order-service]=apps/order-service )

# ── help ──────────────────────────────────────────────────────────
usage() {
  cat <<'EOF'
用法: ./apply-local.sh [STEP ...]

可选步骤（按依赖顺序排列）：
  build       构建并加载业务镜像（docker build + load）
  ns          创建命名空间
  secrets     创建占位密钥（app-secrets / grafana-api-token）
  middleware  部署中间件（redis/postgres/etcd + prometheus/grafana/skywalking/langfuse + clickhouse/model-data；FULL=1 含 milvus/nebula/banyandb；LOCAL_MODEL=1 含 ollama）
  ollama      部署 Ollama 本地小模型（需 LOCAL_MODEL=1）
  apps        部署业务服务（shop-agent / gateway / monitoring-agent / order-service）
  debugapp    构建 DEBUG=1 镜像并部署 shop-agent-debug（带 debugpy 5678 端口）
  mockapps    独立构建 + 部署 mock-llm 与 order-service（压测链路，不触碰其余业务）
  ingress     部署 Ingress 入口
  expose      暴露 LoadBalancer 端口（docker-desktop 免 port-forward）
  down        停止所有服务（scale replicas=0，保留配置/数据，可随时恢复）

示例：
  ./apply-local.sh ns secrets middleware apps ingress   # 标准部署
  ./apply-local.sh secrets                              # 仅重建密钥
  ./apply-local.sh ollama expose                        # 仅部署 Ollama 并暴露
  ./apply-local.sh apps                                 # 仅重新部署业务服务
  ./apply-local.sh mockapps                             # 仅部署压测链路（mock-llm + order-service）
  ./apply-local.sh ns secrets middleware build apps ingress expose ollama # 全量部署（含重组件加 FULL=1）
  ./apply-local.sh build debugapp                                    # 构建 debug 镜像并部署 debug pod
  ./apply-local.sh debugapp                                          # 仅部署/更新 debug pod（跳过构建）
  ./apply-local.sh down                                               # 停止所有服务（保留数据）
  ./apply-local.sh middleware apps                                    # 从 down 状态恢复中间件+业务

环境变量（可在 local.env 中覆盖）：
  LOCAL_MODEL=1              启用 Ollama 本地小模型
  FULL=1                     启用重组件（milvus/clickhouse/nebula）
  SKIP_BUILD=1               跳过镜像构建（仅验证 apply 链路）
  CLUSTER_TYPE               docker-desktop / minikube / kind（默认 docker-desktop）
  MOCK_CLOUD_OVERRIDE=1      网关云端 API(gpt*/claude*/qwen*) 全部改路由 mock-llm（压测）
  MOCK_LLM_LATENCY_MIN/MAX   mock-llm 延迟区间 ms（默认 500/800）
  MOCK_LLM_ERROR_RATE        mock-llm 错误注入概率（默认 0.01）
  MOCK_LLM_OUTPUT_TOKENS     mock-llm 回复 token 数（默认 200）
EOF
  exit 0
}

# ── 工具函数 ──────────────────────────────────────────────────────
apply() { envsubst < "$1" | kubectl apply -f -; }

# ── 步骤函数 ──────────────────────────────────────────────────────

# 单个镜像构建 + 载入集群（registry 模式 push，否则 kind/minikube load；docker-desktop 直接用本地镜像）
build_img() {
  local name="$1" appdir="$2"
  local img
  # 仅 shop-agent 的 Dockerfile 定义了 ARG LITE（轻量依赖开关），其余服务
  # （gateway/monitoring-agent/order-service/mock-llm）无此参数，传了也被忽略，故按需加。
  local lite_arg=""
  if [[ "$name" == "shop-agent" ]]; then
    lite_arg="--build-arg LITE=1"
  fi
  if [[ "$REGISTRY_ENABLED" == "1" ]]; then
    ensure_registry
    img="${REGISTRY_HOST_ADDR}/$name:$IMAGE_TAG"
    echo "    build & push $img  (from $appdir)"
    docker build $lite_arg -t "$img" "$appdir"
    docker push "$img"
  else
    img="${IMAGE_PREFIX}$name:$IMAGE_TAG"
    echo "    build $img  (from $appdir)"
    docker build $lite_arg -t "$img" "$appdir"
    case "$CLUSTER_TYPE" in
      kind)      kind load docker-image "$img" --name "${KIND_CLUSTER:-kind}" ;;
      minikube)  minikube image load "$img" ;;
      *)         echo "    docker-desktop: 直接用本地 docker 镜像（registry 关闭时可能踩缓存坑）" ;;
    esac
  fi
}

step_build() {
  echo "==> [build] 构建业务镜像"
  if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    echo "    SKIP_BUILD=1：跳过构建（仅验证 apply 链路；业务 Pod 会因缺镜像 Pending/失败，属预期）"
    return 0
  fi

  if [[ "$REGISTRY_ENABLED" == "1" ]]; then
    echo "    REGISTRY_ENABLED=1：镜像 push 进本地仓库 ${REGISTRY_HOST_ADDR}（先确保 registry + port-forward）"
  else
    echo "    REGISTRY_ENABLED=0：走 load 进集群（kind/minikube/docker-desktop 旧模式）"
  fi
  for name in "${!APPS[@]}"; do
    build_img "$name" "$ROOT/${APPS[$name]}"
  done
}

# 确保 registry 就绪 + host 侧可达（幂等）：
# 1) 拉取 registry:2 镜像（Registry 自身也要先入集群）
# 2) apply registry Deployment + Service
# 3) 若 REGISTRY_PORT_FORWARD=1 且 localhost:5000 未通，起一个后台 port-forward 到 svc/registry
ensure_registry() {
  if [[ -z "${REGISTRY_READY:-}" ]]; then
    docker pull "$(grep '^REGISTRY_IMAGE=' "$DIR/versions.env" | cut -d= -f2)" >/dev/null 2>&1 || true
    apply "$DIR/middleware/registry.yaml"
    kubectl -n shop-agent rollout status deployment/registry --timeout=90s 2>/dev/null || true
    if [[ "$REGISTRY_PORT_FORWARD" == "1" ]]; then
      port="${REGISTRY_HOST_ADDR##*:}"   # 5000
      if ! curl -s --noproxy "*" --max-time 3 "http://${REGISTRY_HOST_ADDR}/v2/" >/dev/null 2>&1; then
        echo "    port-forward ${REGISTRY_HOST_ADDR} → svc/registry（后台）"
        nohup kubectl -n shop-agent port-forward svc/registry "${port}:5000" >/dev/null 2>&1 &
        sleep 3
      fi
    fi
    REGISTRY_READY=1
  fi
}

step_ns() {
  echo "==> [ns] 创建命名空间"
  apply "$DIR/00-namespace.yaml"
}

step_secrets() {
  echo "==> [secrets] 创建占位密钥"
  # 01-secrets.yaml 占位（REDIS_AUTH/POSTGRES_PASSWORD/OPENAI_* 等）
  apply "$DIR/01-secrets.yaml" || true

  # 补充 CI 才注入、但本机必需的密钥（order-service / langfuse 的数据库连接串）
  kubectl create secret generic app-secrets -n shop-agent \
    --from-literal=REDIS_AUTH="${REDIS_AUTH:-local-redis-password}" \
    --from-literal=POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-local-postgres-password}" \
    --from-literal=OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
    --from-literal=AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-}" \
    --from-literal=AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-}" \
    --from-literal=FIXED_API_KEY="${FIXED_API_KEY:-local-fixed-key}" \
    --from-literal=MONITORING_WEBHOOK_TOKEN="${MONITORING_WEBHOOK_TOKEN:-local-webhook-token}" \
    --from-literal=AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-local}" \
    --from-literal=AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-local}" \
    --from-literal=S3_ENDPOINT="${S3_ENDPOINT:-}" \
    --from-literal=LANGFUSE_INIT_USER_PASSWORD="${LANGFUSE_INIT_USER_PASSWORD:-local-langfuse-password}" \
    --from-literal=ORDER_DATABASE_URL="${ORDER_DATABASE_URL:-postgres://postgres:local-postgres-password@postgres:5432/order_service}" \
    --from-literal=LANGFUSE_DATABASE_URL="${LANGFUSE_DATABASE_URL:-postgres://postgres:local-postgres-password@postgres:5432/langfuse}" \
    --from-literal=TONGYI_API_KEY="${TONGYI_API_KEY:-}" \
    --dry-run=client -o yaml | kubectl apply -f -

  # grafana-api-token 占位
  kubectl create secret generic grafana-api-token -n shop-agent \
    --from-literal=GRAFANA_API_KEY="${GRAFANA_API_KEY:-local-grafana-token}" \
    --dry-run=client -o yaml | kubectl apply -f -

  # object-storage 占位（S3_ENDPOINT 留空时应用降级）
  apply "$DIR/middleware/object-storage-secret.yaml"
}

step_middleware() {
  echo "==> [middleware] 中间件（基础设施 + 可观测 + 数据存储）"
  # 基础设施
  for f in registry redis postgres-pgvector etcd; do
    apply "$DIR/middleware/$f.yaml"
  done
  # 数据存储（banyandb 为 skywalking 的存储，须先于 skywalking）
  for f in banyandb clickhouse model-data; do
    apply "$DIR/middleware/$f.yaml"
  done
   # 可观测（minio 为 langfuse 的 S3 暂存存储）
   for f in minio prometheus grafana alertmanager skywalking langfuse loki otel-agent otel-gateway; do
     apply "$DIR/middleware/$f.yaml"
   done
   # Grafana 观测配置（datasource + 仪表盘模板 + JSON）
   for f in grafana-datasource grafana-dashboards-provider grafana-dashboard-shop-agent grafana-dashboard-loki-logs; do
     apply "$DIR/middleware/$f.yaml"
   done
  # 数据存储
  for f in clickhouse model-data; do
    apply "$DIR/middleware/$f.yaml"
  done
  # 重组件（milvus / nebula，需 FULL=1）
  if [[ "$FULL" == "1" ]]; then
    echo "    FULL=1：部署重组件（milvus / nebula）"
    for f in milvus nebula; do
      apply "$DIR/middleware/$f.yaml"
    done
  fi
  # 注：Ollama 已从 middleware 步骤移除，如需本地小模型请单独调用 step_ollama（LOCAL_MODEL=1）。
}

step_ollama() {
  echo "==> [ollama] 本地小模型（需 LOCAL_MODEL=1）"
  if [[ "$LOCAL_MODEL" != "1" ]]; then
    echo "    跳过：LOCAL_MODEL 未设为 1（当前值=$LOCAL_MODEL）"
    return 0
  fi
  apply "$DIR/middleware/ollama.yaml"
}

step_apps() {
  echo "==> [apps] 业务服务"
  for f in shop-agent gateway monitoring-agent order-service; do
    apply "$DIR/$f.yaml"
  done
  # 强制重建业务 Pod：docker-desktop + 同 tag(local) 下，kubectl apply 在镜像引用
  # 未变时不会触发滚动更新，导致「build 过但 Pod 仍跑旧镜像」。rollout restart 让
  # kubelet（IMAGE_PULL_POLICY=Always）重新拉取 registry 新镜像。
  # 首次部署时资源可能尚未 Ready，restart 失败用 || true 兜底（不影响新建）。
  # 幂等：重复执行本步骤（如仅 ./apply-local.sh apps）也会重新拉起新镜像。
  echo "    rollout restart 业务 Pod（确保新镜像生效）"
  for dep in shop-agent gateway monitoring-agent order-service; do
    kubectl -n shop-agent rollout restart "deployment/$dep" 2>/dev/null || true
    kubectl -n shop-agent rollout status "deployment/$dep" --timeout=120s 2>/dev/null || true
  done
  kubectl -n shop-agent rollout restart statefulset/shop-agent 2>/dev/null || true
}

step_debugapp() {
  echo "==> [debugapp] 构建 DEBUG=1 镜像并部署 shop-agent-debug（debugpy 5678）"
  if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    echo "    SKIP_BUILD=1：跳过构建"
  else
    if [[ "$REGISTRY_ENABLED" == "1" ]]; then
      ensure_registry
      img="${REGISTRY_HOST_ADDR}/shop-agent:$IMAGE_TAG"
      echo "    build & push $img (DEBUG=1, from apps/shop-agent)"
      docker build --build-arg LITE=1 --build-arg DEBUG=1 -t "$img" "$ROOT/apps/shop-agent"
      docker push "$img"
    else
      img="${IMAGE_PREFIX}shop-agent:$IMAGE_TAG"
      echo "    build $img (DEBUG=1, from apps/shop-agent)"
      docker build --build-arg LITE=1 --build-arg DEBUG=1 -t "$img" "$ROOT/apps/shop-agent"
      case "$CLUSTER_TYPE" in
        kind)      kind load docker-image "$img" --name "${KIND_CLUSTER:-kind}" ;;
        minikube)  minikube image load "$img" ;;
        *)         echo "    docker-desktop: 直接用本地 docker 镜像" ;;
      esac
    fi
  fi
  apply "$DIR/shop-agent-debug.yaml"
  echo "    rollout restart shop-agent-debug（确保新镜像生效）"
  kubectl -n shop-agent rollout restart deployment/shop-agent-debug 2>/dev/null || true
  kubectl -n shop-agent rollout status deployment/shop-agent-debug --timeout=120s 2>/dev/null || true
}

step_mockapps() {
  echo "==> [mockapps] mock-llm + order-service（压测用独立构建/部署）"
  if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    echo "    SKIP_BUILD=1：跳过构建（仅 apply）"
  else
    build_img mock-llm "$ROOT/apps/mock-llm"
    build_img order-service "$ROOT/apps/order-service"
  fi
  for f in mock-llm order-service; do
    apply "$DIR/$f.yaml"
  done
  # 强制重建 Pod：同 tag(local) 下 kubectl apply 在镜像引用未变时不会触发滚动更新，
  # 导致「build 过但 Pod 仍跑旧镜像」。rollout restart 让 kubelet（IMAGE_PULL_POLICY=Always）
  # 重新拉取 registry 新镜像。首次部署资源未 Ready 时 restart 失败用 || true 兜底。
  echo "    rollout restart mock Pod（确保新镜像生效）"
  for dep in mock-llm order-service; do
    kubectl -n shop-agent rollout restart "deployment/$dep" 2>/dev/null || true
  done
}

step_ingress() {
  echo "==> [ingress] 入口（host=$INGRESS_HOST）"
  apply "$DIR/ingress.yaml"
}

step_down() {
  echo "==> [down] 停止所有业务服务（scale replicas=0，保留配置/数据）"
  NS="shop-agent"
  # 按步骤逆序停止：先 apps/mockapps，再 middleware，保留 ns/secrets 不删
  for dep in $(kubectl -n "$NS" get deployments -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
    echo "    scale down deployment/$dep"
    kubectl -n "$NS" scale "deployment/$dep" --replicas=0 2>/dev/null || true
  done
  for sts in $(kubectl -n "$NS" get statefulsets -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
    echo "    scale down statefulset/$sts"
    kubectl -n "$NS" scale "statefulset/$sts" --replicas=0 2>/dev/null || true
  done
  echo "    ✅ 已停止（可随时 ./apply-local.sh middleware apps 恢复）"
}

step_expose() {
  echo "==> [expose] 暴露外部访问（LoadBalancer）"
  if [[ "$CLUSTER_TYPE" != "docker-desktop" ]]; then
    echo "    跳过：非 docker-desktop（当前=$CLUSTER_TYPE），请用 port-forward"
    return 0
  fi
  if [[ "${EXPOSE_LOCAL:-1}" != "1" ]]; then
    echo "    跳过：EXPOSE_LOCAL!=1"
    return 0
  fi
  expose_lb() {
    kubectl -n shop-agent patch svc "$1" --type=merge -p \
      "{\"spec\":{\"type\":\"LoadBalancer\",\"ports\":[{\"name\":\"http\",\"port\":$3,\"nodePort\":$2,\"targetPort\":$4,\"protocol\":\"TCP\"}]}}" \
      && echo "    $1 -> http://localhost:$2"
  }
  # 业务/可观测入口
  expose_lb shop-agent        30080  80    8000
  expose_lb gateway           30081  80    8001
  expose_lb order-service     30088  80    8080
  expose_lb monitoring-agent  30091  80    9091
  expose_lb grafana           32100  3000  3000
  expose_lb prometheus        32090  9090  9090
  if [[ "$LOCAL_MODEL" == "1" ]]; then
    expose_lb ollama          31143  11434 11434
  fi
  # 中间件（对齐 port-forward-k8s.sh 的映射，便于 VSCode/本机工具直连调试）。
  # 注：patch 按 port 键做 strategic merge，仅给已有端口补 nodePort，不会删除多端口服务的其他端口。
  expose_lb redis             30079  6379  6379
  expose_lb postgres          30082  5432  5432
  expose_lb langfuse-web      30000  3000  3000
  expose_lb skywalking-oap    31180  11800 11800
  expose_lb otel-gateway      30431  4317  4317
  if [[ "$FULL" == "1" ]]; then
    expose_lb milvus          31953  19530 19530
  fi
}

# ── 步骤注册（按依赖顺序） ──────────────────────────────────────
ALL_STEPS=(build ns secrets middleware ollama apps debugapp mockapps ingress expose down)

# ── 入口 ──────────────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
  usage
fi

# 校验参数
for arg in "$@"; do
  found=0
  for s in "${ALL_STEPS[@]}"; do
    [[ "$arg" == "$s" ]] && found=1 && break
  done
  if [[ "$found" == "0" ]]; then
    echo "❌ 未知步骤: $arg"
    echo "   可用步骤: ${ALL_STEPS[*]}"
    echo "   运行 ./apply-local.sh 查看完整帮助"
    exit 1
  fi
done

# 执行指定步骤
for step in "$@"; do
  "step_$step"
done

echo ""
echo "✅ 已执行步骤: $*"
echo "   查看状态：kubectl -n shop-agent get pods"
if [[ "$CLUSTER_TYPE" == "docker-desktop" ]]; then
  echo "   docker-desktop：已用 LoadBalancer 暴露，直接访问"
else
  echo "   minikube/kind 端口转发示例："
  echo "     kubectl -n shop-agent port-forward svc/shop-agent 8000:80"
  echo "     kubectl -n shop-agent port-forward svc/gateway 8001:80"
  echo "     kubectl -n shop-agent port-forward svc/grafana 3000:3000"
fi
echo "   重组件: $([[ "$FULL" == "1" ]] && echo 已开启 || echo 已跳过（需 FULL=1）)"
echo "   Ollama: $([[ "$LOCAL_MODEL" == "1" ]] && echo 已开启 || echo 已跳过（需 LOCAL_MODEL=1）)"
echo "   Debug pod: 运行 ./apply-local.sh debugapp 部署；port-forward: kubectl -n shop-agent port-forward svc/shop-agent-debug 5678:5678"
