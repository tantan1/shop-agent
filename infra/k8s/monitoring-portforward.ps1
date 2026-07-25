# 持久化 port-forward：将 monitoring-agent 的 80 端口常驻映射到本机 9091。
# 用法：
#   启动（后台常驻）：  pwsh infra/k8s/monitoring-portforward.ps1
#   停止：              Stop-Process -Name kubectl -ErrorAction SilentlyContinue
# 该端口转发会在当前会话后台运行；若需开机自启，可将其加入 Windows 计划任务。
$ErrorActionPreference = "Stop"

$ns = "shop-agent"
$svc = "monitoring-agent"
$localPort = 9091
$svcPort = 80

# 若已有同名转发在跑则先退出，避免重复
$existing = Get-Process -Name kubectl -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "port-forward svc/$svc $localPort" }
if ($existing) {
    Write-Host "port-forward already running (pid $($existing.Id)), skip."
    exit 0
}

Write-Host "Starting port-forward svc/$svc $localPort->$svcPort (background)..."
Start-Process -FilePath "kubectl.exe" `
    -ArgumentList "-n $ns port-forward svc/$svc $localPort`:$svcPort" `
    -WindowStyle Hidden `
    -RedirectStandardOutput "$env:TEMP\monitoring-pf.out" `
    -RedirectStandardError "$env:TEMP\monitoring-pf.err"

Start-Sleep -Seconds 4

try {
    $r = Invoke-WebRequest -Uri "http://localhost:$localPort/status" -UseBasicParsing -TimeoutSec 8 -ErrorAction Stop
    Write-Host "OK: http://localhost:$localPort/status -> $($r.StatusCode)"
}
catch {
    Write-Warning "port-forward started but health check failed: $_"
}
