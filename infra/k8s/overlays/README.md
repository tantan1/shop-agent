# overlays/ — k8s 配置覆盖层

本目录放「叠加在基础清单之上的临时/环境特定配置」，用 `kubectl apply -f` 叠加，
不修改基础 yaml（gateway.yaml / shop-agent.yaml 等），可 Git 跟踪、可一键还原。

> 注意：现在**首选** `local.env` 里的 `LLM_BACKEND` 开关切换推理后端（由
> `apply-local.sh` 自动注入 env，见下）。本目录的 `local-vllm.yaml` 是
> 「不依赖 apply-local.sh、纯 kubectl 手动叠加」的备选方案。

## 方式 A（推荐）：local.env + apply-local.sh 切换

`infra/k8s/local.env` 增加：

```bash
LLM_BACKEND=ollama   # ollama = k8s 内 Ollama（默认）；vllm = 宿主机 Docker Desktop 的 vLLM
```

- 切到本地 vLLM：把 `LLM_BACKEND` 改为 `vllm`，然后重新 apply：
  ```bash
  docker compose -f docker-compose.vllm.yml up -d        # 先起 Docker 里的 vLLM
  ./infra/k8s/apply-local.sh shop-agent gateway          # 按开关注入 env
  ```
- 切回 Ollama：把 `LLM_BACKEND` 改回 `ollama`，重新 apply，再 `docker compose -f docker-compose.vllm.yml down` 释放 GPU。

`apply-local.sh` 会根据 `LLM_BACKEND` 自动导出 `GATEWAY_VLLM_BASE_URL` /
`SHOP_EMBEDDING_PROVIDER` / `SHOP_VLLM_*` 等变量并 envsubst 进 yaml。
不依赖任何命令行临时补丁。

vLLM 宿主机地址/端口在 `local.env` 中可配（默认 host.docker.internal / 8003 / 8101 / 8102）：
```bash
VLLM_HOST=host.docker.internal
VLLM_QWEN3_PORT=8003
VLLM_BGE_M3_PORT=8101
VLLM_RERANKER_PORT=8102
```
若 `host.docker.internal` 在 Pod 内不通，把 `VLLM_HOST` 改成
`kubectl get nodes -o wide` 的 INTERNAL-IP 即可，无需改任何脚本或 yaml。

## 方式 B（备选，纯 kubectl）：local-vllm.yaml overlay

不碰 apply 流程，直接用 kubectl 叠加：

```bash
docker compose -f docker-compose.vllm.yml up -d
kubectl apply -f infra/k8s/overlays/local-vllm.yaml
kubectl -n shop-agent rollout status deployment/gateway --timeout=120s
kubectl -n shop-agent rollout status deployment/shop-agent --timeout=120s
```

还原：
```bash
kubectl apply -f infra/k8s/gateway.yaml infra/k8s/shop-agent.yaml
docker compose -f docker-compose.vllm.yml down
```

## 验证链路（两种方式通用）

```bash
kubectl -n shop-agent port-forward svc/shop-agent 8000:80
# 另开终端
curl http://localhost:8000/health
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-unified","messages":[{"role":"user","content":"你好"}]}'
```

## 端口映射（来自 docker-compose.vllm.yml）

| 模型 | 端口 | served-model-name |
|------|------|-------------------|
| qwen3 (unified) | 8003 | qwen3-unified |
| bge-m3 | 8101 | bge-m3 |
| bge-reranker | 8102 | bge-reranker-base |

> 若 `host.docker.internal` 在 Pod 内不通，改为 `kubectl get nodes -o wide` 的
> INTERNAL-IP（两种方式里的地址都要改）。
