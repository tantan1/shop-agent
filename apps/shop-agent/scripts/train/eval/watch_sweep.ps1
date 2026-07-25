# 监控规模扫描评测：M=20 完成后自动停止 PID 26560 并输出汇总
# 用法：在 e:/workspace/shop-agent 下另开一个 PowerShell 终端运行
#       .\scripts\watch_sweep.ps1
# 行为：每 30 秒检查 outputs/sweep.log，出现 [M=20] free_acc= 且进程存活则 Stop-Process，
#       打印 M=5/10/15/20 四档汇总 + 已知 M=40 对照，然后退出。

$ErrorActionPreference = 'SilentlyContinue'
$log = "outputs/sweep.log"
$pid_target = 26560
$interval = 30   # 秒

Write-Host "[watch] 开始监控 PID=$pid_target，日志=$log（每 $interval`s 轮询）"

while ($true) {
    $summaries = Get-Content -Encoding UTF8 $log -ErrorAction SilentlyContinue | Select-String 'free_acc='
    $m20done   = $summaries | Where-Object { $_ -match '\[M=20\]' }
    $procAlive = Get-Process -Id $pid_target -ErrorAction SilentlyContinue

    if ($m20done) {
        if ($procAlive) {
            Write-Host "[watch] 检测到 [M=20] 汇总，停止进程 PID=$pid_target ..."
            Stop-Process -Id $pid_target -Force
            Start-Sleep -Seconds 1
        }
        Write-Host ""
        Write-Host "==== 规模扫描最终结果（free/con/ambiguous/p95）===="
        $summaries | ForEach-Object { Write-Host $_.ToString().Trim() }
        Write-Host ""
        Write-Host "[对照] M=40 base = 44.4% (ambiguous 30.6%)，p95 更高"
        Write-Host ""
        Write-Host "[结论] 规模扫描已按计划停止；训练目标锁定 M=8（top-k 软预过滤）。"
        break
    }

    # 未完成：简短报告
    $tail = Get-Content -Encoding UTF8 $log -Tail 1
    $m20prog = Get-Content -Encoding UTF8 $log | Select-String '\[M=20\] \[(\d+)/480\]' |
               Select-Object -Last 1
    $prog = if ($m20prog) { $m20prog.Matches.Groups[1].Value } else { '尚未开始' }
    Write-Host "[watch] $(Get-Date -Format 'HH:mm:ss') M=20 进度: $prog/480 | 尾部: $tail"
    Start-Sleep -Seconds $interval
}
