# 一键部署完整系统（shop-agent on OKE）

策略：**尽量用 OCI 托管服务，托管没有的才用 K8s 自建**。

## 托管 vs 自建对照

| 组件 | 方式 |
|---|---|
| OKE 控制面 | OCI 托管 |
| 对象存储（MinIO→S3） | **OCI Object Storage**（S3 兼容，Secret `object-storage`） |
| Load Balancer（Ingress） | OCI 灵活 LB |
| 监控/日志 | OCI Monitoring/Logging（可叠加自建 Prometheus/Grafana） |
| Redis | K8s 自建 |
| PostgreSQL + pgvector | K8s 自建（`ankane/pgvector`） |
| Milvus + etcd | K8s 自建（重，FULL=1） |
| ClickHouse | K8s 自建（重，FULL=1） |
| NebulaGraph | K8s 自建（重，FULL=1） |
| SkyWalking + BanyanDB | K8s 自建 |
| Prometheus + Grafana | K8s 自建 |
| Langfuse（web+worker） | K8s 自建（连托管 Object Storage） |
| 业务服务（gateway/monitoring-agent/shop-agent/order-service） | K8s 自建（镜像来自 OCIR） |

## 一键部署

**方式一：GitHub Actions（推荐）**
推送 `main` → `deploy.yml` 自动 build+push OCIR，再 `apply` 业务服务 + 调用 `apply-all.sh` 部署中间件。
- 重组件默认跳过；设 GitHub Secret `DEPLOY_FULL_STACK=1` 开启。

**方式二：本地/手动**
```bash
export OCIR_REGISTRY=nrt.ocir.io/xxxx  IMAGE_TAG=latest  INGRESS_HOST=api.yourdomain.com
export S3_ENDPOINT=https://xxxxx.compat.objectstorage.ap-tokyo-1.oraclecloud.com  OCI_REGION=ap-tokyo-1
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
# 先注入 app-secrets（含 POSTGRES_PASSWORD / ORDER_DATABASE_URL / LANGFUSE_DATABASE_URL 等）
./infra/k8s/apply-all.sh            # 核心链路 + 业务
FULL=1 ./infra/k8s/apply-all.sh     # 含 milvus/clickhouse/nebula
```

## 资源提示
Always Free（2×1/8 OCPU、24GB）跑不满重组件。默认关重组件留出余量；需完整链路请升规格或开 `FULL=1`。

## 依赖顺序
业务服务依赖 Postgres/Redis 就绪。当前靠应用自身重连；如需严格顺序，可加 initContainer 探活（见部署文档）。
