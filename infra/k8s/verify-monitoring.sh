#!/usr/bin/env bash
# 验证监控栈（Grafana + SkyWalking + Prometheus + Loki）连接
# 无依赖 jq：使用 python3 解析 JSON，kubectl jsonpath 提取字段
set -euo pipefail

# python3 可用性检查
if ! command -v python3 &>/dev/null; then
  echo "❌ python3 未找到，请安装 python3 后重试"
  exit 1
fi

echo "==> [1/8] 检查 Pod 状态"
kubectl -n shop-agent get pods -l app=shop-agent
kubectl -n shop-agent get pods -l app=skywalking-oap
kubectl -n shop-agent get pods -l app=grafana
kubectl -n shop-agent get pods -l app=prometheus
kubectl -n shop-agent get pods -l app=loki
kubectl -n shop-agent get pods -l app=otel-agent
kubectl -n shop-agent get pods -l app=otel-gateway

echo ""
echo "==> [2/8] 检查 shop-agent SkyWalking 环境变量"
kubectl -n shop-agent get deployment shop-agent -o json | \
  python3 -c "
import sys, json
data = json.load(sys.stdin)
envs = data['spec']['template']['spec']['containers'][0]['env']
for e in envs:
    if e['name'].startswith('SW_'):
        v = e.get('value', 'N/A')
        print('    ' + e['name'] + '=' + v)
"

echo ""
echo "==> [3/8] 检查 shop-agent 日志（SkyWalking 初始化）"
kubectl -n shop-agent logs deployment/shop-agent 2>/dev/null | grep -i skywalking | tail -5 || echo "  (无 SkyWalking 日志，可能未启用)"

echo ""
echo "==> [4/8] 检查 Prometheus 抓取目标"
kubectl -n shop-agent port-forward svc/prometheus 9090:9090 &>/dev/null &
PF_PROM=$!
sleep 2
echo "  shop-agent 指标抓取状态："
curl -s http://localhost:9090/api/v1/targets | python3 -c "
import sys, json
data = json.load(sys.stdin)
for t in data['data']['activeTargets']:
    if t['labels'].get('job') == 'shop-agent':
        print('    job=' + t['labels']['job'] + ', health=' + t['health'] + ', lastScrape=' + t.get('lastScrape', 'N/A'))
" 2>/dev/null || echo "  (无法获取抓取状态，检查 Prometheus 是否就绪)"
kill $PF_PROM 2>/dev/null || true

echo ""
echo "==> [5/8] 检查 Grafana 数据源"
kubectl -n shop-agent port-forward svc/grafana 3000:3000 &>/dev/null &
PF_GRAFANA=$!
sleep 2
echo "  Grafana 数据源："
curl -s -u admin:local-langfuse-password http://localhost:3000/api/datasources | python3 -c "
import sys, json
data = json.load(sys.stdin)
for ds in data:
    print('    ' + ds['name'] + ': ' + ds['type'] + ' (' + ds['url'] + ')')
" 2>/dev/null || echo "  (无法获取数据源列表，检查 Grafana 是否就绪)"
kill $PF_GRAFANA 2>/dev/null || true

echo ""
echo "==> [6/8] 检查 Loki"
kubectl -n shop-agent port-forward svc/loki 3100:3100 &>/dev/null &
PF_LOKI=$!
sleep 2
echo "  Loki 健康状态："
curl -s http://localhost:3100/ready && echo "  (Loki ready)" || echo "  (Loki not ready)"
echo "  Loki 标签："
curl -s http://localhost:3100/loki/api/v1/labels | python3 -c "
import sys, json
data = json.load(sys.stdin)
labels = data.get('data', [])
if labels:
    for l in labels:
        print('    ' + l)
else:
    print('  (无标签，可能尚无日志流入)')
" 2>/dev/null || echo "  (无标签，可能尚无日志流入)"
echo "  Loki 最近日志 (service=shop-agent):"
curl -s 'http://localhost:3100/loki/api/v1/query' --data-urlencode '{service="shop-agent"}' | python3 -c "
import sys, json
data = json.load(sys.stdin)
results = data.get('data', {}).get('result', [])
if results:
    for r in results[:3]:
        vals = r.get('values', [])
        for v in vals[:2]:
            print('    ' + v[1][:200])
else:
    print('  (无日志)')
" 2>/dev/null || echo "  (无日志)"
kill $PF_LOKI 2>/dev/null || true

echo ""
echo "==> [7/8] 检查 Shop-Agent 分布式追踪"
kubectl -n shop-agent get pods -l app=skywalking-oap 2>/dev/null | tail -n +2 | head -3 || echo "  (SkyWalking OAP 未部署)"

echo ""
echo "==> [8/8] 检查 Grafana 仪表盘"
kubectl -n shop-agent port-forward svc/grafana 3000:3000 &>/dev/null &
PF_GRAFANA=$!
sleep 2
echo "  Grafana 仪表盘："
curl -s -u admin:local-langfuse-password http://localhost:3000/api/search?query=shop-agent | python3 -c "
import sys, json
data = json.load(sys.stdin)
for d in data:
    print('    ' + d['title'])
" 2>/dev/null || echo "  (无法获取仪表盘列表)"
kill $PF_GRAFANA 2>/dev/null || true

echo ""
echo "✅ 验证完成"
echo ""
echo "访问地址："
echo "  Grafana:    http://localhost:32100 (admin / local-langfuse-password)"
echo "  SkyWalking: http://localhost:8080 (port-forward: kubectl -n shop-agent port-forward svc/skywalking-ui 8080:8080)"
echo "  Prometheus: http://localhost:32090 (port-forward: kubectl -n shop-agent port-forward svc/prometheus 9090:9090)"
echo "  Loki:       http://localhost:3100 (port-forward: kubectl -n shop-agent port-forward svc/loki 3100:3100)"
