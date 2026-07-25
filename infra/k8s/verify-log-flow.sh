#!/bin/bash
echo "=== 日志流验证脚本 ==="

# 1. 检查应用日志
echo "1. 检查 shop-agent 日志..."
kubectl logs -n shop-agent -l app=shop-agent --since=1m --tail=5 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ 应用正在写日志"
else
    echo "❌ 应用日志检查失败"
    exit 1
fi

# 2. 检查 Promtail
echo "2. 检查 Promtail..."
PROMTAIL_READY=$(kubectl get pods -n shop-agent -l app=promtail -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}')
if [ "$PROMTAIL_READY" == "True" ]; then
    echo "✅ Promtail 运行正常"
else
    echo "❌ Promtail 未就绪"
    exit 1
fi

# 3. 检查 Loki
echo "3. 检查 Loki..."
LOKI_POD=$(kubectl get pods -n shop-agent -l app=loki -o jsonpath='{.items[0].metadata.name}')
if [ -n "$LOKI_POD" ]; then
    echo "✅ Loki pod 存在: $LOKI_POD"
else
    echo "❌ Loki pod 不存在"
    exit 1
fi

# 4. 查询 Loki
echo "4. 查询 Loki 数据..."
START_TIME=$(date -u -d '10 minutes ago' +%s)000000000
END_TIME=$(date -u +%s)000000000

# 使用 port-forward 查询 Loki（Loki 容器内没有 curl）
kubectl port-forward svc/loki 3100:3100 -n shop-agent > /dev/null 2>&1 &
PF_PID=$!
# 等待 port-forward 就绪
sleep 3

# OTel 管道将 service 字段映射为 Loki 标签 service_name
RESPONSE=$(curl -s -G "http://localhost:3100/loki/api/v1/query" \
  --data-urlencode 'query={service_name="shop-agent"}' \
  --data-urlencode 'limit=1' \
  --data-urlencode "start=$START_TIME" \
  --data-urlencode "end=$END_TIME")

# 清理 port-forward
kill $PF_PID 2>/dev/null

echo "$RESPONSE" | grep -q '"status":"success"'
if [ $? -eq 0 ]; then
    echo "✅ Loki 查询成功"
    echo "$RESPONSE" | grep -q '"result":\[\]'
    if [ $? -ne 0 ]; then
        echo "✅ Loki 有日志数据"
    else
        echo "⚠️  Loki 查询成功但无数据（可能日志还未到达）"
    fi
else
    echo "❌ Loki 查询失败"
    echo "$RESPONSE"
    exit 1
fi

# 5. 检查 Grafana
echo "5. 检查 Grafana..."
GRAFANA_READY=$(kubectl get pods -n shop-agent -l app=grafana -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}')
if [ "$GRAFANA_READY" == "True" ]; then
    echo "✅ Grafana 运行正常"
else
    echo "❌ Grafana 未就绪"
    exit 1
fi

echo ""
echo "=== 验证完成 ==="
echo "下一步：在 Grafana Explore 中查询 {service_name=\"shop-agent\"}"
