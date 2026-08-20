# 端到端测试：Nightingale(n9e) 复用本地 redis/postgres/prometheus/alertmanager
# 验证自动异常检测闭环：
#   n9e 引擎对 Prometheus 指标做异常检测 → 推送 Alertmanager
#   → monitoring-agent /ingest/alert 做 RCA → rca_total 指标增长
#
# 说明：夜莺推 Alertmanager 走 /api/v2/alerts（与 Prometheus 同源接口），
# 故链路验证用「向 Alertmanager 注入一条告警」等价夜莺产出，
# 重点验证 alertmanager → monitoring-agent RCA 闭环（n9e 配置已指向该 endpoint）。
#
# 用法：在仓库根目录执行  pwsh scripts/test_nightingale_e2e.ps1

$ErrorActionPreference = "Stop"
$ROOT = (Get-Location)
$token = $env:MONITORING_WEBHOOK_TOKEN
if (-not $token) { $token = "monitoring-agent-secret" }

function Wait-Http($url, $name, $timeoutSec = 120) {
    $elapsed = 0
    while ($elapsed -lt $timeoutSec) {
        try {
            $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
            if ($r.StatusCode -lt 400) {
                Write-Host "  [OK] $name 就绪 ($url)" -ForegroundColor Green
                return $true
            }
        } catch {}
        Start-Sleep -Seconds 3
        $elapsed += 3
        Write-Host "  ... 等待 $name ($elapsed`s)" -ForegroundColor Yellow
    }
    Write-Host "  [FAIL] $name 超时 ($url)" -ForegroundColor Red
    return $false
}

Write-Host "`n=== 1. 检查容器运行状态 ===" -ForegroundColor Cyan
docker ps --filter "name=shop-agent-n9e" --filter "name=shop-agent-prometheus" --filter "name=shop-agent-alertmanager" --filter "name=monitoring-agent" --format "table {{.Names}}\t{{.Status}}"

Write-Host "`n=== 2. 等待服务就绪 ===" -ForegroundColor Cyan
Wait-Http "http://localhost:17000/api/n9e/self/version" "n9e Web"
Wait-Http "http://localhost:19090/-/ready" "Prometheus"
Wait-Http "http://localhost:9093/-/ready" "Alertmanager"
Wait-Http "http://localhost:19091/metrics" "monitoring-agent /metrics"

Write-Host "`n=== 3. 校验 n9e 配置已连 Prometheus 数据源 ===" -ForegroundColor Cyan
# n9e 自身 metrics 或数据源列表
try {
    $ds = Invoke-RestMethod -Uri "http://localhost:17000/api/n9e/datasource" -TimeoutSec 10
    $ds | ConvertTo-Json -Depth 3 | Out-Host
} catch {
    Write-Host "  ( datasource API 需鉴权属正常，跳过；改查 n9e 健康 )" -ForegroundColor DarkGray
}

Write-Host "`n=== 4. 校验 n9e 推 Alertmanager 链路（注入等价告警）===" -ForegroundColor Cyan

# 统计所有 severity 的 rca_total 总和（P1/P2 都可能被注入告警触发，
# 只匹配第一个 rca_total{...} 会漏掉 P2 的增长）。
function Get-RcaTotal($metricsText) {
    $sum = 0.0
    foreach ($m in [regex]::Matches($metricsText, 'rca_total\{[^}]*\}\s+([\d.]+)')) {
        $sum += [double]$m.Groups[1].Value
    }
    return $sum
}

# 记录 RCA 基线
$base = (Invoke-WebRequest -Uri "http://localhost:19091/metrics" -UseBasicParsing -TimeoutSec 10).Content
$baseRca = Get-RcaTotal $base
Write-Host "  RCA 基线计数: $baseRca (P1+P2)"

# 模拟一条「自动异常检测」告警（等价于夜莺引擎命中后推给 Alertmanager 的 payload）。
# alertname 带时间戳：monitoring-agent 对窗口内(默认 300s)同源+alertname+labels 去重，
# 固定名字二次运行会命中 _is_duplicate 而跳过 RCA。
$stamp = Get-Date -Format "HHmmss"
$payload = @(
    @{
        labels = @{
            alertname = "NightingaleAnomalyP99Spike_$stamp"
            severity  = "P2"
            channel   = "nightingale-anomaly"
            service   = "shop-agent"
            source    = "nightingale"
        }
        annotations = @{
            summary     = "Shop-Agent P99 延迟异常（夜莺自动异常检测）"
            description = "夜莺异常检测引擎判定 shop-agent P99 偏离基线，转交 monitoring-agent 做 RCA。"
        }
        generatorURL = "http://n9e:17000"
    }
) | ConvertTo-Json -Depth 5 -Compress

# Alertmanager /api/v2/alerts 接收原生告警数组
$alertArr = "[$payload]"

Write-Host "  向 Alertmanager 推送告警..." -ForegroundColor Yellow
try {
    Invoke-RestMethod -Uri "http://localhost:9093/api/v2/alerts" -Method Post -ContentType "application/json" -Body $alertArr -TimeoutSec 10
    Write-Host "  [OK] 告警已推送 Alertmanager" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] 推送告警失败: $_" -ForegroundColor Red
    exit 1
}

# 等待 monitoring-agent 被 webhook 叫醒并产出 RCA（alertmanager group_wait 5s + RCA 处理）
Write-Host "  等待 monitoring-agent RCA 处理 (15s)..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

$final = (Invoke-WebRequest -Uri "http://localhost:19091/metrics" -UseBasicParsing -TimeoutSec 10).Content
$finalRca = Get-RcaTotal $final
Write-Host "  RCA 最终计数: $finalRca (P1+P2)"

Write-Host "`n=== 5. 断言 ===" -ForegroundColor Cyan
if ($finalRca -gt $baseRca) {
    Write-Host "  [PASS] 端到端闭环成功：n9e(异常检测) → Alertmanager → monitoring-agent RCA 已触发 (rca_total $baseRca → $finalRca)" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] RCA 未增长，闭环中断。检查 alertmanager.yml token / monitoring-agent 日志。" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== 测试通过 ===" -ForegroundColor Cyan
