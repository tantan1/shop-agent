#!/usr/bin/env pwsh
# shop-agent 性能压测一键脚本（PowerShell）
#
# 动作：
#   1. 创建 results/<timestamp>/ 输出文件夹
#   2. 【默认】不启动服务，复用已运行的服务；指定 -Up 时拉起 docker-compose（MockLLM 模式）
#   3. 等待 shop-agent /health 就绪
#   4. 【默认】k6 全链路冒烟测试（1 次请求）
#   5. 【可选】完整压测：Locust 多角色对话模拟 + k6 编排层压测
#   6. 并行采集 shop-agent 与宿主的 CPU/内存/IO
#   7. 保存所有测试结果 + 生成性能分析报告到 results/<timestamp>/
#
# 用法：
#   .\run_stress.ps1                 # 默认：复用已运行服务，仅全链路冒烟测试（1 次请求）
#   .\run_stress.ps1 -Up             # 冒烟测试前拉起服务（配 -Keep 保留容器）
#   .\run_stress.ps1 -FullStress     # 完整压测（Locust + k6，默认不拉起服务）
#   .\run_stress.ps1 -FullStress -Up -Keep   # 完整压测并启动、保留容器
#   .\run_stress.ps1 -LocustOnly     # 完整压测中仅 Locust
#   .\run_stress.ps1 -K6Only         # 完整压测中仅 k6
#   .\run_stress.ps1 -QuickTest      # 快速完整压测（10 并发，30s，50-100ms 延迟）
#   .\run_stress.ps1 -MonitorInterval 5  # 资源采样间隔（秒），采集 shop-agent 与宿主的 CPU/内存/IO 并写入报告
#   .\run_stress.ps1 -Concurrency 500 -ChatRateLimit 50000   # 并发 500，后端单 Key 限流 50000 次/分钟（用于吞吐瓶颈判定）

param(
    [switch]$Keep,          # 保留容器（配合 -Up 使用）
    [switch]$Up,            # 启动服务（默认不启动，直接使用已运行服务）
    [switch]$NoUp,          # 兼容旧参数：强制不启动服务
    [switch]$SmokeOnly,     # 仅全链路冒烟测试（默认）
    [switch]$FullStress,    # 完整压测（Locust + k6）
    [switch]$LocustOnly,    # 完整压测中仅 Locust
    [switch]$K6Only,        # 完整压测中仅 k6
    [switch]$QuickTest,     # 快速完整压测（10 并发，30s，50-100ms 延迟）
    [int]$Concurrency = 100,
    [string]$MockLatencyMin = "500",
    [string]$MockLatencyMax = "800",
    [string]$MockErrorRate = "0.01",
    [int]$MonitorInterval = 5,
    [string]$Namespace = "shop-agent",
    [int]$ChatRateLimit = 50000
)

# 兼容 `-Name=value` 形式（部分 PowerShell 版本不解析等号语法，参数会落入 $args）
foreach ($arg in $args) {
    $m = [regex]::Match($arg, '^-(\w+)=(.+)$')
    if ($m.Success) {
        $name = $m.Groups[1].Value
        $value = $m.Groups[2].Value
        switch ($name) {
            'Concurrency'    { $Concurrency = [int]$value }
            'MockLatencyMin' { $MockLatencyMin = $value }
            'MockLatencyMax' { $MockLatencyMax = $value }
            'MockErrorRate'  { $MockErrorRate = $value }
            'MonitorInterval'{ $MonitorInterval = [int]$value }
            'Namespace'      { $Namespace = $value }
            'ChatRateLimit'  { $ChatRateLimit = [int]$value }
        }
    }
}

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
$Compose = Join-Path $RepoRoot "docker-compose.yml"
$StressDir = Join-Path $RepoRoot "tests/stress"
$ResultsRoot = Join-Path $RepoRoot "tests/stress/results"

# ── 创建按时间排序的结果文件夹 ──
$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$ResultDir = Join-Path $ResultsRoot $Timestamp
New-Item -ItemType Directory -Path $ResultDir -Force | Out-Null
Write-Host "[run_stress] 结果文件夹: $ResultDir" -ForegroundColor Cyan

# ── 资源采集输出目录 ──
$ResourceDir = Join-Path $ResultDir "resources"
New-Item -ItemType Directory -Path $ResourceDir -Force | Out-Null
$resourceJob = $null

# ── 后台资源采集脚本：kubectl top pod/node（CPU/内存）+ cAdvisor（块 IO/网络）+ Get-Counter（宿主磁盘 IO 速率） ──
$ResourceMonitorScript = {
    param($resourceDir, $intervalSec, $namespace)

    $podCsv = Join-Path $resourceDir "pod-stats.csv"
    $hostCsv = Join-Path $resourceDir "host-stats.csv"
    $culture = [System.Globalization.CultureInfo]::InvariantCulture

    if (-not (Test-Path $podCsv)) {
        Set-Content -Path $podCsv -Value "timestamp,name,cpu_percent,mem_usage,mem_limit,mem_percent,block_io_read,block_io_write,net_io_read,net_io_write,pids" -Encoding UTF8
    }
    if (-not (Test-Path $hostCsv)) {
        Set-Content -Path $hostCsv -Value "timestamp,cpu_total_percent,mem_total_percent,mem_available_mb,disk_read_bytes_sec,disk_write_bytes_sec" -Encoding UTF8
    }

    function Parse-Size([string]$s) {
        $s = $s.Trim()
        $m = [regex]::Match($s, '^([\d]+(?:\.[\d]+)?)\s*([KMGT]?i?B?)$', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
        if (-not $m.Success) { return $null }
        $val = [double]::Parse($m.Groups[1].Value, $culture)
        $u = $m.Groups[2].Value.ToUpperInvariant().TrimEnd('B')
        switch ($u) {
            ''    { return $val }
            'K'   { return $val * 1000 }
            'M'   { return $val * 1000 * 1000 }
            'G'   { return $val * 1000 * 1000 * 1000 }
            'T'   { return $val * 1000 * 1000 * 1000 * 1000 }
            'KI'  { return $val * 1024 }
            'MI'  { return $val * 1024 * 1024 }
            'GI'  { return $val * 1024 * 1024 * 1024 }
            'TI'  { return $val * 1024 * 1024 * 1024 * 1024 }
            default { return $null }
        }
    }

    # ── 启动时一次性解析静态信息 ──
    $podName = $null
    $nodeName = $null
    $containerName = "shop-agent"
    $cpuLimitCores = 0.0
    $memLimitBytes = 0.0
    $nodeCpuCores = 0.0
    $nodeMemBytes = 0.0
    try {
        $podName = (kubectl get pod -n $namespace -l app=shop-agent --no-headers 2>$null |
            Select-Object -First 1).Split()[0]
        if (-not $podName) {
            $podName = (kubectl get pod -n $namespace --no-headers 2>$null |
                Where-Object { $_ -match '^\S*shop-agent\S*\s' } |
                Select-Object -First 1).Split()[0]
        }
        if ($podName) {
            $nodeName = (kubectl get pod $podName -n $namespace -o jsonpath="{.spec.nodeName}" 2>$null).Trim()
            $containerName = (kubectl get pod $podName -n $namespace -o jsonpath="{.spec.containers[0].name}" 2>$null).Trim()
            if (-not $containerName) { $containerName = "shop-agent" }
            $cpuLimitStr = (kubectl get pod $podName -n $namespace -o jsonpath="{.spec.containers[0].resources.limits.cpu}" 2>$null).Trim()
            $memLimitStr = (kubectl get pod $podName -n $namespace -o jsonpath="{.spec.containers[0].resources.limits.memory}" 2>$null).Trim()
            if ($cpuLimitStr -match '^([\d.]+)m$') {
                $cpuLimitCores = [double]$Matches[1] / 1000
            } elseif ($cpuLimitStr -match '^([\d.]+)$') {
                $cpuLimitCores = [double]$Matches[1]
            }
            $memLimitBytes = Parse-Size $memLimitStr
            if ($null -eq $memLimitBytes) { $memLimitBytes = 0.0 }
            if ($nodeName) {
                $nodeCpuStr = (kubectl get node $nodeName -o jsonpath="{.status.allocatable.cpu}" 2>$null).Trim()
                $nodeMemStr = (kubectl get node $nodeName -o jsonpath="{.status.allocatable.memory}" 2>$null).Trim()
                if ($nodeCpuStr -match '^([\d.]+)$') { $nodeCpuCores = [double]$Matches[1] }
                $nodeMemBytes = Parse-Size $nodeMemStr
                if ($null -eq $nodeMemBytes) { $nodeMemBytes = 0.0 }
            }
        }
    } catch { }

    # 宿主磁盘 IO 速率（kubectl 无等价指标，保留 Get-Counter）
    $counterPaths = @(
        '\PhysicalDisk(_Total)\Disk Read Bytes/sec',
        '\PhysicalDisk(_Total)\Disk Write Bytes/sec'
    )

    while ($true) {
        $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

        # Pod 级：kubectl top pod + cAdvisor
        try {
            if ($podName) {
                $topLine = kubectl top pod $podName -n $namespace --no-headers 2>$null
                $fields = @($topLine -split '\s+' | Where-Object { $_ })
                $cpuCores = 0.0
                $memBytes = 0.0
                if ($fields.Count -ge 3) {
                    $cpuStr = $fields[1]
                    if ($cpuStr -match '^([\d.]+)m$') { $cpuCores = [double]$Matches[1] / 1000 }
                    elseif ($cpuStr -match '^([\d.]+)$') { $cpuCores = [double]$Matches[1] }
                    $memBytes = Parse-Size $fields[2]
                    if ($null -eq $memBytes) { $memBytes = 0.0 }
                }
                $cpuPerc = if ($cpuLimitCores -gt 0) { [math]::Round($cpuCores / $cpuLimitCores * 100, 2) } elseif ($nodeCpuCores -gt 0) { [math]::Round($cpuCores / $nodeCpuCores * 100, 2) } else { 0 }
                $memPerc = if ($memLimitBytes -gt 0) { [math]::Round($memBytes / $memLimitBytes * 100, 2) } elseif ($nodeMemBytes -gt 0) { [math]::Round($memBytes / $nodeMemBytes * 100, 2) } else { 0 }

                # cAdvisor：块 IO 读/写累计字节、网络收/发累计字节、进程数
                $ioR = 0.0; $ioW = 0.0; $netR = 0.0; $netW = 0.0; $pids = 0
                if ($nodeName) {
                    $raw = kubectl get --raw "/api/v1/nodes/$nodeName/proxy/metrics/cadvisor" 2>$null
                    $podPat = [regex]::Escape('pod="' + $podName + '"')
                    $containerPat = [regex]::Escape('container="' + $containerName + '"')
                    foreach ($line in $raw) {
                        $line = [string]$line
                        if ($line -notmatch $podPat) { continue }
                        $m = [regex]::Match($line, '^(\S+)\s+([0-9.eE+-]+)\s+\d+\s*$')
                        if (-not $m.Success) { continue }
                        $metric = $m.Groups[1].Value
                        $val = [double]$m.Groups[2].Value
                        if ($metric -match '^container_blkio_device_usage_total\{' -and $line -match 'operation="Read"') { $ioR += $val }
                        elseif ($metric -match '^container_blkio_device_usage_total\{' -and $line -match 'operation="Write"') { $ioW += $val }
                        elseif ($metric -match '^container_network_receive_bytes_total\{') { $netR += $val }
                        elseif ($metric -match '^container_network_transmit_bytes_total\{') { $netW += $val }
                        elseif ($metric -match '^container_processes\{' -and $line -match $containerPat) { $pids = [math]::Max($pids, [int]$val) }
                    }
                }

                $csvLine = "$ts,$podName,$cpuPerc,$([math]::Round($memBytes, 0)),$([math]::Round($memLimitBytes, 0)),$memPerc,$([math]::Round($ioR, 0)),$([math]::Round($ioW, 0)),$([math]::Round($netR, 0)),$([math]::Round($netW, 0)),$pids"
                Add-Content -Path $podCsv -Value $csvLine -Encoding UTF8
            }
        } catch { }

        # Node 级：kubectl top node（CPU/内存%）+ Get-Counter（宿主磁盘 IO 速率）
        try {
            $cpuTotal = 0.0
            $memTotalPerc = 0.0
            $memAvailMB = 0
            $nodeTop = kubectl top node --no-headers 2>$null
            foreach ($nl in $nodeTop) {
                if ([string]::IsNullOrWhiteSpace($nl)) { continue }
                $nf = @($nl -split '\s+' | Where-Object { $_ })
                if ($nf.Count -lt 5) { continue }
                if ($nodeName -and $nf[0] -ne $nodeName) { continue }
                if ($nf[2] -match '^([\d.]+)%$') { $cpuTotal = [double]$Matches[1] }
                if ($nf[4] -match '^([\d.]+)%$') { $memTotalPerc = [double]$Matches[1] }
                $usedBytes = Parse-Size $nf[3]
                if ($null -ne $usedBytes -and $nodeMemBytes -gt 0) {
                    $memAvailMB = [math]::Round(($nodeMemBytes - $usedBytes) / 1MB, 0)
                }
            }
            $readRate = 0.0
            $writeRate = 0.0
            try {
                $samples = (Get-Counter -Counter $counterPaths -SampleInterval 1 -MaxSamples 1).CounterSamples
                $readRate = $samples[0].CookedValue
                $writeRate = $samples[1].CookedValue
            } catch { }
            Add-Content -Path $hostCsv -Value "$ts,$([math]::Round($cpuTotal, 2)),$([math]::Round($memTotalPerc, 2)),$([math]::Round($memAvailMB, 0)),$([math]::Round($readRate, 0)),$([math]::Round($writeRate, 0))" -Encoding UTF8
        } catch { }

        Start-Sleep -Seconds $intervalSec
    }
}

# ── 启动后台资源采集 ──
$resourceJob = Start-Job -ScriptBlock $ResourceMonitorScript -ArgumentList $ResourceDir, $MonitorInterval, $Namespace
Write-Host "[run_stress] 后台资源采集已启动（采样间隔 ${MonitorInterval}s）: $ResourceDir" -ForegroundColor Cyan

# ── 解析 .env 里的 FIXED_API_KEY ──
if (-not $env:FIXED_API_KEY) {
    $envFile = Join-Path $RepoRoot ".env"
    if (Test-Path $envFile) {
        foreach ($line in (Get-Content $envFile)) {
            if ($line -match '^\s*FIXED_API_KEY\s*=\s*(.+)\s*$') {
                $env:FIXED_API_KEY = $Matches[1].Trim('"').Trim("'")
                break
            }
        }
    }
}
if (-not $env:FIXED_API_KEY) { $env:FIXED_API_KEY = "test-key-for-pytest" }

# ── MockLLM 环境变量 ──
$env:MOCK_LLM_ENABLED = "true"
$env:MOCK_LLM_LATENCY_MIN = $MockLatencyMin
$env:MOCK_LLM_LATENCY_MAX = $MockLatencyMax
$env:MOCK_LLM_ERROR_RATE = $MockErrorRate

# ── 模式解析 ──
# 默认：不启动服务 + 仅冒烟测试。指定 -FullStress/-QuickTest/-LocustOnly/-K6Only 时为完整压测。
$smokeMode = -not ($FullStress -or $QuickTest -or $LocustOnly -or $K6Only)
$upNeeded = $Up -and -not $NoUp

if ($smokeMode) {
    Write-Host "[run_stress] 默认模式：全链路冒烟测试（1 次请求），不启动服务，复用已运行服务。" -ForegroundColor Yellow
}

$locustScript = Join-Path $StressDir "locust/locustfile.py"
$k6Script = Join-Path $StressDir "k6/chat-stress.js"

# ── QuickTest 模式：降低并发和时间 ──
if ($QuickTest) {
    $Concurrency = 10
    $MockLatencyMin = "50"
    $MockLatencyMax = "100"
    Write-Host "[run_stress] ⚡ QuickTest 模式：并发=$Concurrency, 延迟=50-100ms" -ForegroundColor Yellow
}

# 后端单 Key 速率限制（次/分钟）→ 每秒上限，用于吞吐瓶颈判定
$rateLimitRps = [math]::Round($ChatRateLimit / 60, 1)

try {
    if ($upNeeded) {
        Write-Host "[run_stress] 拉起 compose 服务（MockLLM 模式）..." -ForegroundColor Cyan
        docker compose -f $Compose up -d redis shop-agent gateway order-service
        if ($LASTEXITCODE -ne 0) { throw "docker compose up 失败" }
    }

    # ── 等待 shop-agent 就绪 ──
    Write-Host "[run_stress] 等待 shop-agent /health 就绪..." -ForegroundColor Cyan
    $health = "http://localhost/agent/health"
    $ready = $false
    $maxAttempts = 60
    for ($i = 0; $i -lt $maxAttempts; $i++) {
        try {
            $resp = Invoke-WebRequest -Uri $health -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue
            if ($resp.StatusCode -eq 200) { $ready = $true; break }
        } catch { }
        if ($i % 10 -eq 0 -and $i -gt 0) {
            Write-Host "[run_stress] 已等待 $($i * 3) 秒..." -ForegroundColor Yellow
        }
        Start-Sleep -Seconds 3
    }
    if (-not $ready) {
        throw "shop-agent 在超时内未就绪（$($maxAttempts * 3) 秒）。请检查：`n1. shop-agent 容器是否运行：docker ps`n2. 容器日志：docker compose logs shop-agent`n3. 端口是否正确：netstat -ano | findstr 8000"
    }
    Write-Host "[run_stress] shop-agent 已就绪。" -ForegroundColor Green

    # ── 运行 Locust 多角色对话模拟（完整压测，冒烟模式跳过） ──
    $locustOutput = $null
    if (-not $K6Only -and -not $smokeMode) {
        Write-Host "[run_stress] 运行 Locust 多角色对话模拟..." -ForegroundColor Cyan
        $locustOutput = Join-Path $ResultDir "locust-results"
        New-Item -ItemType Directory -Path $locustOutput -Force | Out-Null

        # Locust headless 模式，输出 JSON + CSV 到 results 文件夹
        $locustRunTime = if ($QuickTest) { "30s" } else { "5m" }
        locust -f $locustScript `
            --headless `
            -u $Concurrency `
            -r 10 `
            --run-time $locustRunTime `
            --host http://localhost:8000 `
            --csv $locustOutput/locust-report `
            --json-file $locustOutput/locust-report.json `
            --loglevel INFO `
            2>&1 | Tee-Object -FilePath (Join-Path $ResultDir "locust.log")

        Write-Host "[run_stress] Locust 结果保存到: $locustOutput" -ForegroundColor Green
    }

    # ── 运行 k6 全链路测试（冒烟模式=1 次请求；完整压测按并发选择阶段模板，精确并发由 CONCURRENCY 覆盖） ──
    $k6Output = $null
    if (-not $LocustOnly) {
        Write-Host "[run_stress] 运行 k6 $(if ($smokeMode) { "全链路冒烟测试" } else { "编排层压测" })..." -ForegroundColor Cyan
        $k6Output = Join-Path $ResultDir "k6-results"
        New-Item -ItemType Directory -Path $k6Output -Force | Out-Null

        # k6 输出 JSON 到 results 文件夹
        $env:K6_OUT = "json"
        $env:K6_JSON_OUTPUT = (Join-Path $k6Output "k6-results.json")
        $k6SummaryFile = Join-Path $k6Output "k6-summary.json"
        
        $k6Stage = if ($smokeMode) { "smoke" } elseif ($Concurrency -le 100) { "baseline" } elseif ($Concurrency -le 500) { "peak" } else { "extreme" }

        $k6Start = Get-Date
        k6 run `
            --env CONCURRENCY=$Concurrency `
            --env STAGE=$k6Stage `
            --env MOCK_LLM_ENABLED=true `
            --env MOCK_LLM_LATENCY_MIN=$MockLatencyMin `
            --env MOCK_LLM_LATENCY_MAX=$MockLatencyMax `
            --env MOCK_LLM_ERROR_RATE=$MockErrorRate `
            --out json=$env:K6_JSON_OUTPUT `
            --summary-export="$k6SummaryFile" `
            $k6Script `
            2>&1 | Tee-Object -FilePath (Join-Path $ResultDir "k6.log")
        $k6ElapsedSec = [math]::Round(((Get-Date) - $k6Start).TotalSeconds, 1)

        Write-Host "[run_stress] k6 结果保存到: $k6Output" -ForegroundColor Green
    }

    # ── 保存环境信息 ──
    $envInfo = @{
        timestamp = $Timestamp
        test_mode = $(if ($smokeMode) { "smoke" } else { "full-stress" })
        services_started = $upNeeded
        concurrency = $Concurrency
        mock_llm_enabled = $true
        mock_llm_latency_min = $MockLatencyMin
        mock_llm_latency_max = $MockLatencyMax
        mock_llm_error_rate = $MockErrorRate
        chat_rate_limit = $ChatRateLimit
        locust_only = $LocustOnly.IsPresent
        k6_only = $K6Only.IsPresent
    } | ConvertTo-Json -Depth 3
    Set-Content -Path (Join-Path $ResultDir "test-config.json") -Value $envInfo

    # ── 停止后台资源采集 ──
    if ($resourceJob) {
        Stop-Job $resourceJob -ErrorAction SilentlyContinue | Out-Null
        Wait-Job $resourceJob -Timeout 10 -ErrorAction SilentlyContinue | Out-Null
        Remove-Job $resourceJob -Force -ErrorAction SilentlyContinue | Out-Null
        $resourceJob = $null
        Write-Host "[run_stress] 资源采集已停止，原始数据: $ResourceDir" -ForegroundColor Green
    }

    # ── 分析资源监控数据（shop-agent 容器 + 宿主） ──
    $shopCpuAvg = 0.0; $shopCpuMax = 0.0
    $shopMemAvgMB = 0.0; $shopMemMaxMB = 0.0; $shopMemLimitMB = 0.0
    $shopMemPercAvg = 0.0; $shopMemPercMax = 0.0
    $shopIoReadMB = 0.0; $shopIoWriteMB = 0.0
    $shopIoReadRate = 0.0; $shopIoWriteRate = 0.0
    $hostCpuAvg = 0.0; $hostCpuMax = 0.0
    $hostMemAvg = 0.0; $hostMemMax = 0.0
    $hostDiskReadRate = 0.0; $hostDiskWriteRate = 0.0
    $resourceSamples = 0
    $shopSampleCount = 0
    $ioDuration = 0

    $containerCsv = Join-Path $ResourceDir "pod-stats.csv"
    $hostCsv = Join-Path $ResourceDir "host-stats.csv"

    if (Test-Path $containerCsv) {
        $shopRows = @(Import-Csv $containerCsv | Where-Object { $_.name -like 'shop-agent*' })
        $n = $shopRows.Count
        if ($n -gt 0) {
            $shopSampleCount = $n
            $resourceSamples = $n
            $shopSamples = @($shopRows | ForEach-Object {
                [pscustomobject]@{
                    cpu      = [double]$_.cpu_percent
                    memPerc  = [double]$_.mem_percent
                    memUsage = [double]$_.mem_usage
                    memLimit = [double]$_.mem_limit
                    ioR      = [double]$_.block_io_read
                    ioW      = [double]$_.block_io_write
                }
            })
            $shopCpuAvg     = [math]::Round(($shopSamples | Measure-Object cpu -Average).Average, 2)
            $shopCpuMax     = [math]::Round(($shopSamples | Measure-Object cpu -Maximum).Maximum, 2)
            $shopMemAvgMB   = [math]::Round(($shopSamples | Measure-Object memUsage -Average).Average / 1MB, 1)
            $shopMemMaxMB   = [math]::Round(($shopSamples | Measure-Object memUsage -Maximum).Maximum / 1MB, 1)
            $shopMemLimitMB = [math]::Round($shopSamples[0].memLimit / 1MB, 1)
            $shopMemPercAvg = [math]::Round(($shopSamples | Measure-Object memPerc -Average).Average, 2)
            $shopMemPercMax = [math]::Round(($shopSamples | Measure-Object memPerc -Maximum).Maximum, 2)
            $lastIoR = ($shopSamples | Measure-Object ioR -Maximum).Maximum
            $lastIoW = ($shopSamples | Measure-Object ioW -Maximum).Maximum
            $shopIoReadMB  = [math]::Round($lastIoR / 1MB, 1)
            $shopIoWriteMB = [math]::Round($lastIoW / 1MB, 1)
            $ioDuration = if ($n -gt 1) { ($n - 1) * $MonitorInterval } else { 0 }
            if ($ioDuration -gt 0) {
                $shopIoReadRate  = [math]::Round($lastIoR / 1MB / $ioDuration, 3)
                $shopIoWriteRate = [math]::Round($lastIoW / 1MB / $ioDuration, 3)
            }
        }
    }

    if (Test-Path $hostCsv) {
        $hrows = @(Import-Csv $hostCsv)
        if ($hrows.Count -gt 0) {
            $hostCpuAvg = [math]::Round(($hrows | Measure-Object cpu_total_percent -Average).Average, 2)
            $hostCpuMax = [math]::Round(($hrows | Measure-Object cpu_total_percent -Maximum).Maximum, 2)
            $hostMemAvg = [math]::Round(($hrows | Measure-Object mem_total_percent -Average).Average, 2)
            $hostMemMax = [math]::Round(($hrows | Measure-Object mem_total_percent -Maximum).Maximum, 2)
            $hostDiskReadRate  = [math]::Round(($hrows | Measure-Object disk_read_bytes_sec -Average).Average / 1MB, 2)
            $hostDiskWriteRate = [math]::Round(($hrows | Measure-Object disk_write_bytes_sec -Average).Average / 1MB, 2)
            if ($hrows.Count -gt $resourceSamples) { $resourceSamples = $hrows.Count }
            if ($ioDuration -eq 0 -and $hrows.Count -gt 1) { $ioDuration = ($hrows.Count - 1) * $MonitorInterval }
        }
    }

    $hasShopData = $shopSampleCount -gt 0
    $resourceDataStatus = if ($hasShopData) { "" } else { "（未采集到 shop-agent Pod 采样，请确认 k8s 中存在该 Pod）" }
    $bottleneckNote = if ($shopCpuMax -ge 70) { "CPU 使用率峰值超过 70%，可能成为瓶颈，建议关注 HPA 扩容策略。" } elseif ($shopMemPercMax -ge 80) { "内存占比峰值超过 80%，接近配额上限，存在 OOM 风险。" } else { "CPU 与内存余量充足，瓶颈不在本机资源，建议结合真实 LLM 延迟进一步分析。" }
    $ioNote = if ($shopIoWriteRate -gt 5) { "写 IO 偏高（$shopIoWriteRate MB/s），关注日志/持久化写入。" } else { "磁盘 IO 压力较低，未成为瓶颈。" }

    # ── 生成性能测试报告 ──
    $reportPath = Join-Path $ResultDir "性能测试报告.md"
    
    # ── 分析 Locust 结果 ──
    $locustStats = @{}
    $locustTotalRequests = 0
    $locustTotalFailures = 0
    $locustMaxP95 = 0
    $locustMaxP99 = 0
    $locustMaxAvg = 0
    
    if (-not $K6Only -and -not $smokeMode) {
        $locustCsv = Join-Path $locustOutput "locust-report_stats.csv"
        if (Test-Path $locustCsv) {
            $lines = Get-Content $locustCsv | Select-Object -Skip 1
            foreach ($line in $lines) {
                # 跳过空行
                if ([string]::IsNullOrWhiteSpace($line)) { continue }
                
                $parts = $line -split ','
                if ($parts.Length -ge 8) {
                    $name = $parts[0].Trim()
                    
                    # 跳过表头或无效行
                    if ($name -eq "Name" -or $name -eq "Aggregated") { continue }
                    
                    # 尝试解析数字，跳过非数字行（使用 [int]::TryParse 返回值）
                    $parsedCount = 0; $parsedFail = 0; $parsedP50 = 0; $parsedP95 = 0; $parsedP99 = 0
                    $okCount = [long]::TryParse($parts[1].Trim(), [ref]$parsedCount)
                    $okFail = [long]::TryParse($parts[2].Trim(), [ref]$parsedFail)
                    $okP50 = [double]::TryParse($parts[3].Trim(), [ref]$parsedP50)
                    $okP95 = [double]::TryParse($parts[4].Trim(), [ref]$parsedP95)
                    $okP99 = [double]::TryParse($parts[5].Trim(), [ref]$parsedP99)
                    
                    if (-not ($okCount -and $okFail -and $okP50 -and $okP95 -and $okP99)) { continue }
                    
                    $locustStats[$name] = @{
                        requests = $parsedCount
                        failures = $parsedFail
                        p50 = $parsedP50
                        p95 = $parsedP95
                        p99 = $parsedP99
                    }
                    
                    $locustTotalRequests += $parsedCount
                    $locustTotalFailures += $parsedFail
                    if ($parsedP95 -gt $locustMaxP95) { $locustMaxP95 = $parsedP95 }
                    if ($parsedP99 -gt $locustMaxP99) { $locustMaxP99 = $parsedP99 }
                    if ($parsedP50 -gt $locustMaxAvg) { $locustMaxAvg = $parsedP50 }
                }
            }
        }
    }
    
    $locustFailureRate = if ($locustTotalRequests -gt 0) { 
        [math]::Round(($locustTotalFailures / $locustTotalRequests) * 100, 2) 
    } else { 
        0 
    }
    
    # ── 分析 k6 结果（优先读 k6 --summary-export 小文件；无汇总时对 NDJSON 流式解析，避免逐行 ConvertFrom-Json 在百万行级文件上过慢） ──
    $k6Stats = @{
        totalRequests = 0
        failures = 0
        avgLatency = 0
        p95Latency = 0
        throughput = 0
        err429  = 0
        err4xx  = 0
        err5xx  = 0
        errConn = 0
    }
    
    if (-not $LocustOnly) {
        $k6Json = Join-Path $k6Output "k6-results.json"
        $k6Summary = Join-Path $k6Output "k6-summary.json"
        $summaryLoaded = $false

        if (Test-Path $k6Summary) {
            try {
                $sum = Get-Content $k6Summary -Raw | ConvertFrom-Json
                $m = $sum.metrics
                $sumReqs = [long]$m.'http_reqs'.count
                if ($sumReqs -gt 0) {
                    $k6Stats.totalRequests = $sumReqs
                    # http_req_failed 的 value 字段是失败率（0~1），fails 字段不可靠（与 k6 控制台不一致）
                    $k6Stats.failures = [math]::Round([double]$m.'http_req_failed'.value * $sumReqs)
                    $k6Stats.avgLatency = [math]::Round([double]$m.'http_req_duration'.avg, 1)
                    $k6Stats.p95Latency = [math]::Round([double]$m.'http_req_duration'.'p(95)', 1)
                    $k6Stats.err429  = if ($m.'http_status_429')  { [long]$m.'http_status_429'.count } else { 0 }
                    $k6Stats.err4xx  = if ($m.'http_status_4xx')  { [long]$m.'http_status_4xx'.count } else { 0 }
                    $k6Stats.err5xx  = if ($m.'http_status_5xx')  { [long]$m.'http_status_5xx'.count } else { 0 }
                    $k6Stats.errConn = if ($m.'http_status_conn') { [long]$m.'http_status_conn'.count } else { 0 }
                    $summaryLoaded = $true
                }
            } catch {
                Write-Host "[run_stress] 警告: 无法解析 k6 summary，回退到 NDJSON 解析" -ForegroundColor Yellow
            }
        }

        if (-not $summaryLoaded -and (Test-Path $k6Json)) {
            try {
                $k6Reqs = [long]0
                $k6Fails = 0
                $err429 = 0; $err4xx = 0; $err5xx = 0; $errConn = 0
                $k6Durations = New-Object System.Collections.Generic.List[double]
                $durSum = 0.0
                $valRe = [regex]'"value":\s*([0-9.eE+-]+)'
                $statusRe = [regex]'"status":"([0-9]+)"'
                foreach ($line in [System.IO.File]::ReadLines($k6Json)) {
                    if ($line.IndexOf('http_req') -lt 0) { continue }
                    if ($line.IndexOf('"http_req_duration"') -ge 0) {
                        $m = $valRe.Match($line)
                        if ($m.Success) {
                            $v = [double]$m.Groups[1].Value
                            $k6Durations.Add($v)
                            $durSum += $v
                        }
                        $sm = $statusRe.Match($line)
                        if ($sm.Success) {
                            $st = [int]$sm.Groups[1].Value
                            if ($st -eq 429) { $err429++ }
                            elseif ($st -eq 0) { $errConn++ }
                            elseif ($st -ge 500) { $err5xx++ }
                            elseif ($st -ge 400) { $err4xx++ }
                        }
                    } elseif ($line.IndexOf('"http_reqs"') -ge 0) {
                        $m = $valRe.Match($line)
                        if ($m.Success) { $k6Reqs += [long]$m.Groups[1].Value }
                    } elseif ($line.IndexOf('"http_req_failed"') -ge 0) {
                        $m = $valRe.Match($line)
                        if ($m.Success -and [double]$m.Groups[1].Value -gt 0) { $k6Fails++ }
                    }
                }
                $k6Stats.totalRequests = $k6Reqs
                $k6Stats.failures = $k6Fails
                $k6Stats.err429  = $err429
                $k6Stats.err4xx  = $err4xx
                $k6Stats.err5xx  = $err5xx
                $k6Stats.errConn = $errConn
                if ($k6Durations.Count -gt 0) {
                    $k6Durations.Sort()
                    $k6Stats.avgLatency = [math]::Round($durSum / $k6Durations.Count, 1)
                    $p95Idx = [int][math]::Floor($k6Durations.Count * 0.95)
                    if ($p95Idx -ge $k6Durations.Count) { $p95Idx = $k6Durations.Count - 1 }
                    $k6Stats.p95Latency = [math]::Round($k6Durations[$p95Idx], 1)
                }
            } catch {
                Write-Host "[run_stress] 警告: 无法解析 k6 JSON" -ForegroundColor Yellow
            }
        }
        if ($k6ElapsedSec -gt 0) {
            $k6Stats.throughput = [math]::Round($k6Stats.totalRequests / $k6ElapsedSec, 2)
        }
    }
    
    $k6FailureRate = if ($k6Stats.totalRequests -gt 0) {
        [math]::Round(($k6Stats.failures / $k6Stats.totalRequests) * 100, 2)
    } else {
        0
    }
    
    # ── 检查数据有效性 ──
    if ($locustTotalRequests -eq 0 -and $k6Stats.totalRequests -eq 0) {
        Write-Host "[run_stress] 警告: 未收集到任何测试数据，请检查 Locust 和 k6 是否正常运行" -ForegroundColor Yellow
    }

    # ── 瓶颈分析（规则驱动，每项附带数据依据与建议） ──
    $p95Val = if ($smokeMode) { [double]$k6Stats.p95Latency } else { [double]$locustMaxP95 }
    $failRate = if ($smokeMode) { $k6FailureRate } else { [math]::Max($locustFailureRate, $k6FailureRate) }

    $bottlenecks = New-Object System.Collections.Generic.List[object]
    $bn = 0

    # 规则 1：错误率 > 5%
    if ($failRate -gt 5) {
        $bn++
        $bottlenecks.Add([pscustomobject]@{
            Num = $bn
            Title = "错误率超标（可靠性瓶颈）"
            Data = "错误率 $failRate%（阈值 < 5%）"
            Verdict = "未达到 < 5% 目标"
            Suggestion = "查看 k6.log / locust.log 中非 2xx 与超时分布；检查依赖服务（LLM/Redis/Postgres/Milvus）可用性"
        })
    }

    # 规则 2：P95 延迟 > 2000ms
    if ($p95Val -gt 2000) {
        $bn++
        $bottlenecks.Add([pscustomobject]@{
            Num = $bn
            Title = "延迟超标（P95 延迟瓶颈）"
            Data = "P95 = ${p95Val}ms（阈值 < 2000ms）"
            Verdict = "未达到 P95 < 2s SLA"
            Suggestion = "语义缓存命中率目标 > 30%；高并发下检查 Ingress/限流排队"
        })
    }

    # 规则 3：服务端 5xx / 连接失败
    $serverErrTotal = $k6Stats.err5xx + $k6Stats.errConn
    if ($serverErrTotal -gt 0) {
        $bn++
        $bottlenecks.Add([pscustomobject]@{
            Num = $bn
            Title = "服务端错误（5xx/连接失败）"
            Data = "5xx 响应 $($k6Stats.err5xx) 个 + 连接失败 $($k6Stats.errConn) 个，占请求 $([math]::Round($serverErrTotal / [math]::Max($k6Stats.totalRequests, 1) * 100, 1))%（错误明细见测试结果）"
            Verdict = "后端/网关返回 5xx 或连接失败"
            Suggestion = "检查 Ingress 与后端连接池/worker 上限、Pod 就绪与资源限制；查看 5xx 来源日志（nginx / 应用）；必要时 HPA 扩容"
        })
    }

    # 规则 4：429 限流（以实际 429 响应为准，避免高吞吐但 5xx 时误判为限流）
    if ($k6Stats.err429 -gt 0) {
        $bn++
        $bottlenecks.Add([pscustomobject]@{
            Num = $bn
            Title = "触发速率限制（429）"
            Data = "429 响应 $($k6Stats.err429) 个，吞吐 $([math]::Round($k6Stats.throughput, 2)) req/s（单 Key 上限 ≈ $rateLimitRps req/s，CHAT_RATE_LIMIT=$ChatRateLimit/min）"
            Verdict = "请求超过限流阈值"
            Suggestion = "生产环境按租户分发独立 API Key；调高 CHAT_RATE_LIMIT"
        })
    }

    # 规则 5：Pod CPU > 70%（仅 Pod 有采样时）
    if ($shopSampleCount -gt 0 -and $shopCpuMax -gt 70) {
        $bn++
        $bottlenecks.Add([pscustomobject]@{
            Num = $bn
            Title = "CPU 使用率过高"
            Data = "shop-agent CPU 峰值 $shopCpuMax%（阈值 70%）"
            Verdict = "CPU 接近/超过警戒线"
            Suggestion = "部署 HPA（CPU > 70% 自动扩容）；优化 FAISS 检索/Embedding 热点"
        })
    }

    # 规则 6：Pod 内存 > 80%
    if ($shopSampleCount -gt 0 -and $shopMemPercMax -gt 80) {
        $bn++
        $bottlenecks.Add([pscustomobject]@{
            Num = $bn
            Title = "内存使用率过高"
            Data = "shop-agent 内存峰值 $shopMemMaxMB MB（占比 $shopMemPercMax%，阈值 80%）"
            Verdict = "接近配额上限，存在 OOM 风险"
            Suggestion = "检查内存泄漏；调大容器内存配额"
        })
    }

    # 规则 7：Pod 写 IO > 5 MB/s
    if ($shopSampleCount -gt 0 -and $shopIoWriteRate -gt 5) {
        $bn++
        $bottlenecks.Add([pscustomobject]@{
            Num = $bn
            Title = "磁盘写 IO 偏高"
            Data = "写 IO $shopIoWriteRate MB/s（阈值 5 MB/s）"
            Verdict = "写 IO 偏高"
            Suggestion = "关注日志/持久化写入频率；考虑异步批量写入"
        })
    }

    if ($bottlenecks.Count -eq 0) {
        $bottleneckBlocks = @"
### 未发现明确瓶颈

- **数据**：错误率 $failRate%（< 5%）、P95 $([math]::Round($p95Val, 0))ms（< 2000ms）、吞吐 $([math]::Round($k6Stats.throughput, 2)) req/s
- **判定**：✅ 各项关键指标均在阈值内，当前负载下资源余量充足
- **建议**：用真实 LLM 替换 MockLLM 做端到端验证，再评估是否需要扩容
"@.Trim()
    } else {
        $bottleneckBlocks = ($bottlenecks | ForEach-Object {
@"
### 瓶颈 $($_.Num)：$($_.Title)

- **数据**：$($_.Data)
- **判定**：$($_.Verdict)
- **建议**：$($_.Suggestion)
"@.Trim()
        }) -join "`n`n"
    }

    $report = @"
# Shop-Agent 性能测试报告

> 测试时间：$($Timestamp -replace '_', ' ')
> 测试环境：Docker Desktop K8s（本地），MockLLM 模式
> 测试工具：$(if ($smokeMode) { "k6（冒烟测试）" } else { "k6 + Locust" })

---

## 一、测试环境

| 组件 | 配置 |
|------|------|
| K8s 集群 | Docker Desktop（Windows），单节点 |
| shop-agent | MockLLM 模式（延迟 $MockLatencyMin-$MockLatencyMax ms，错误率 $MockErrorRate） |
| 限流配置 | CHAT_RATE_LIMIT=$ChatRateLimit/min，GLOBAL_RATE_LIMIT=$ChatRateLimit/min |
| API 路径 | `http://localhost/agent/api/v1/chatagent/agent/chat`（Ingress） |
| 认证 | `Authorization: Bearer local-fixed-key` |
| 服务启动方式 | $(if ($upNeeded) { "本次由脚本拉起" } else { "复用已运行服务（未启动）" }) |

---

## 二、测试配置

| 参数 | 值 |
|------|-----|
| 测试模式 | $(if ($smokeMode) { "全链路冒烟（k6 1 次请求）" } elseif ($LocustOnly) { "完整压测：Locust 仅" } elseif ($K6Only) { "完整压测：k6 仅" } else { "完整压测：Locust + k6" }) |
| 并发用户数 | $Concurrency |
| MockLLM 最小延迟 | ${MockLatencyMin}ms |
| MockLLM 最大延迟 | ${MockLatencyMax}ms |
| MockLLM 错误率 | $MockErrorRate |
| $(if ($smokeMode) { "冒烟请求数" } else { "Locust 运行时间" }) | $(if ($smokeMode) { "1 次全链路请求" } else { "5 分钟" }) |

---

## 三、测试结果汇总

$(if ($smokeMode) { "### 1. k6 全链路冒烟测试（1 次请求）" } else { "### 1. Locust 多角色对话模拟" })
"@

    if (-not $smokeMode) {
        $report += @"


| 角色 | 请求数 | 失败率 | P50 | P95 | P99 |
|------|--------|--------|-----|-----|-----|
"@
        # 添加 Locust 角色数据
        if ($locustStats.Count -gt 0) {
            foreach ($role in $locustStats.Keys) {
                $s = $locustStats[$role]
                $report += "| $role | $($s.requests) | $($s.failures) | $([math]::Round($s.p50, 0))ms | $([math]::Round($s.p95, 0))ms | $([math]::Round($s.p99, 0))ms |`n"
            }
        } else {
            $report += "| 无数据 | - | - | - | - | - |`n"
        }
        $report += @"
| **总计** | **$locustTotalRequests** | **$locustTotalFailures ($locustFailureRate%)** | **$([math]::Round($locustMaxAvg, 0))ms** | **$([math]::Round($locustMaxP95, 0))ms** | **$([math]::Round($locustMaxP99, 0))ms** |

**关键发现**：
- 总请求数：$locustTotalRequests
- 失败率：$locustFailureRate%
- P95 延迟：$([math]::Round($locustMaxP95, 0))ms
- P99 延迟：$([math]::Round($locustMaxP99, 0))ms
"@
    }

    # ── 错误明细表（k6：429 限流 / 其他 4xx / 5xx / 连接错误）──
    $errTotal = $k6Stats.err429 + $k6Stats.err4xx + $k6Stats.err5xx + $k6Stats.errConn
    $errPct = { param($n) if ($errTotal -gt 0) { "$([math]::Round($n / $errTotal * 100, 1))%" } else { "0%" } }
    $errorDetailTable = @"

**错误明细**（按 HTTP 状态 / 连接分类）：

| 错误类型 | 数量 | 占失败比 |
|------|------|------|
| 429 限流 | $($k6Stats.err429) | $(& $errPct $k6Stats.err429) |
| 其他 4xx | $($k6Stats.err4xx) | $(& $errPct $k6Stats.err4xx) |
| 5xx 服务端错误 | $($k6Stats.err5xx) | $(& $errPct $k6Stats.err5xx) |
| 连接错误（status=0） | $($k6Stats.errConn) | $(& $errPct $k6Stats.errConn) |

"@

    if ($smokeMode) {
        $report += @"


| 指标 | 值 |
|------|-----|
| 测试阶段 | smoke（1 并发 / 1 次） |
| 总请求数 | $($k6Stats.totalRequests) |
| 失败数 | $($k6Stats.failures) |
| 失败率 | $k6FailureRate% |
| 平均延迟 | $([math]::Round($k6Stats.avgLatency, 0))ms |
| P95 延迟 | $([math]::Round($k6Stats.p95Latency, 0))ms |
| 吞吐量 | $([math]::Round($k6Stats.throughput, 2)) req/s |

**关键发现**：
- 吞吐量：$([math]::Round($k6Stats.throughput, 2)) req/s
- P95 延迟：$([math]::Round($k6Stats.p95Latency, 0))ms
- 错误率：$k6FailureRate%
$errorDetailTable
"@
    } else {
        $report += @"


### 2. k6 编排层压测

| 指标 | 值 |
|------|-----|
| 测试阶段 | $k6Stage$(if ($k6Stage -ne "smoke") { "（$Concurrency 并发）" }) |
| 总请求数 | $($k6Stats.totalRequests) |
| 失败数 | $($k6Stats.failures) |
| 失败率 | $k6FailureRate% |
| 平均延迟 | $([math]::Round($k6Stats.avgLatency, 0))ms |
| P95 延迟 | $([math]::Round($k6Stats.p95Latency, 0))ms |
| 吞吐量 | $([math]::Round($k6Stats.throughput, 2)) req/s |

**关键发现**：
- 吞吐量：$([math]::Round($k6Stats.throughput, 2)) req/s
- P95 延迟：$([math]::Round($k6Stats.p95Latency, 0))ms
- 错误率：$k6FailureRate%
$errorDetailTable
"@
    }

    $report += @"


---

## 四、资源使用情况（CPU / 内存 / IO）

> 采样间隔 ${MonitorInterval}s，随压测并行采集（kubectl top + cAdvisor + 宿主磁盘 IO 计数器），共 $resourceSamples 次采样，约 $ioDuration 秒

### 1. shop-agent Pod 资源 $resourceDataStatus

| 指标 | 平均 | 峰值 | 说明 |
|------|------|------|------|
| CPU 使用率 | $shopCpuAvg% | $shopCpuMax% | 相对 CPU limit（500m） |
| 内存使用量 | $shopMemAvgMB MB | $shopMemMaxMB MB | Pod 实际占用 |
| 内存占比 | $shopMemPercAvg% | $shopMemPercMax% | 相对内存 limit（1Gi） |
| 容器内存配额 | $shopMemLimitMB MB | - | k8s 内存 limit |
| 磁盘读（Block I/O） | $shopIoReadMB MB（$shopIoReadRate MB/s） | - | 压测期间累计读（cAdvisor） |
| 磁盘写（Block I/O） | $shopIoWriteMB MB（$shopIoWriteRate MB/s） | - | 压测期间累计写（cAdvisor） |

### 2. 宿主（Node）整体资源

| 指标 | 平均 | 峰值 |
|------|------|------|
| CPU 使用率 | $hostCpuAvg% | $hostCpuMax% |
| 内存使用率 | $hostMemAvg% | $hostMemMax% |
| 磁盘读速率 | $hostDiskReadRate MB/s | - |
| 磁盘写速率 | $hostDiskWriteRate MB/s | - |

### 3. 资源与性能关联分析

- **CPU**：压测吞吐 $([math]::Round($k6Stats.throughput, 2)) req/s 时，shop-agent CPU 均值 $shopCpuAvg%、峰值 $shopCpuMax%。$(if ($shopCpuMax -ge 70) { "⚠️ 已接近/超过 70% 警戒线。" } else { "余量充足，未成为瓶颈。" })
- **内存**：shop-agent 内存峰值 $shopMemMaxMB MB（占比 $shopMemPercMax%），$(if ($shopMemPercMax -ge 80) { "⚠️ 接近配额上限。" } else { "余量充足，无 OOM 风险。" })
- **磁盘 IO**：压测期间 shop-agent 累计读 $shopIoReadMB MB / 写 $shopIoWriteMB MB（读 $shopIoReadRate MB/s / 写 $shopIoWriteRate MB/s）。$ioNote
- **结论**：$bottleneckNote

---

## 五、瓶颈分析

$bottleneckBlocks

---

## 六、SLA 达标情况

| SLA 指标 | 目标 | $(if ($smokeMode) { "k6 冒烟结果" } else { "Locust 结果 / k6 结果" }) | 达标 |
|----------|------|-------------|------|
| P95 延迟 | < 2000ms | $([math]::Round($(if ($smokeMode) { $k6Stats.p95Latency } else { $locustMaxP95 }), 0))ms | $(if ($(if ($smokeMode) { $k6Stats.p95Latency } else { $locustMaxP95 }) -lt 2000) { "✅" } else { "❌" }) |
| 错误率 | < 5% | $(if ($smokeMode) { "$k6FailureRate%" } else { "$locustFailureRate% / $k6FailureRate%" }) | $(if ($(if ($smokeMode) { $k6FailureRate } else { [math]::Max($locustFailureRate, $k6FailureRate) }) -lt 5) { "✅" } else { "❌" }) |
| 可用性 | > 99% | $(if ($smokeMode) { if ($k6Stats.totalRequests -gt 0) { "$([math]::Round((1 - $k6Stats.failures / $k6Stats.totalRequests) * 100, 2))%" } else { "N/A" } } else { "$(if ($locustTotalRequests -gt 0) { "$([math]::Round((1 - $locustTotalFailures / $locustTotalRequests) * 100, 2))%" } else { "N/A" }) / $(if ($k6Stats.totalRequests -gt 0) { "$([math]::Round((1 - $k6Stats.failures / $k6Stats.totalRequests) * 100, 2))%" } else { "N/A" })" }) | $(if ($(if ($smokeMode) { $k6Stats.totalRequests } else { $locustTotalRequests }) -gt 0 -and (1 - $(if ($smokeMode) { $k6Stats.failures } else { $locustTotalFailures }) / $(if ($smokeMode) { $k6Stats.totalRequests } else { $locustTotalRequests })) * 100 -gt 99) { "✅" } else { "❌" }) |

---

## 七、结论与建议

### 结论

$(if ($smokeMode) {
@"
1. **系统可用性**：$(if ($k6FailureRate -lt 5) { "✅ 通过" } else { "❌ 未通过" }) k6 错误率 $k6FailureRate%（$(if ($k6FailureRate -lt 5) { "< 5% 目标" } else { "> 5% 目标" })）。
2. **延迟表现**：$(if ($k6Stats.p95Latency -lt 2000) { "✅ 通过" } else { "❌ 未通过" }) P95=$([math]::Round($k6Stats.p95Latency, 0))ms（$(if ($k6Stats.p95Latency -lt 2000) { "< 2s SLA" } else { "> 2s SLA" })）。
3. **全链路连通性**：冒烟请求 $($k6Stats.totalRequests) 次，成功 $($k6Stats.totalRequests - $k6Stats.failures) 次（失败 $($k6Stats.failures) 次）。
4. **资源占用**：shop-agent CPU 均值 $shopCpuAvg%（峰值 $shopCpuMax%），内存均值 $shopMemAvgMB MB（峰值 $shopMemMaxMB MB / 占比 $shopMemPercMax%），磁盘写 $shopIoWriteRate MB/s。
5. **安全防护**：恶意注入请求 $(if ($locustStats.ContainsKey("malicious-injection")) { "已拦截" } else { "冒烟模式未测试" })。
"@
} else {
@"
1. **系统可用性**：$(if ($locustFailureRate -lt 5) { "✅ 通过" } else { "❌ 未通过" }) Locust 错误率 $locustFailureRate%（$(if ($locustFailureRate -lt 5) { "< 5% 目标" } else { "> 5% 目标" })）。
2. **延迟表现**：$(if ($locustMaxP95 -lt 2000) { "✅ 通过" } else { "❌ 未通过" }) P95=$([math]::Round($locustMaxP95, 0))ms（$(if ($locustMaxP95 -lt 2000) { "< 2s SLA" } else { "> 2s SLA" })）。
3. **吞吐量**：$([math]::Round($k6Stats.throughput, 2)) req/s（$(if ($k6Stats.err429 -gt 0) { "触发限流 429" } elseif ($k6Stats.err5xx -gt 0) { "伴随 5xx 服务端错误" } elseif ($k6Stats.throughput -gt $rateLimitRps) { "超过速率限制 $rateLimitRps req/s" } elseif ($k6Stats.throughput -gt ($rateLimitRps * 0.8)) { "接近速率限制 $rateLimitRps req/s" } else { "在速率限制内" })）。
4. **多角色支持**：$($locustStats.Count) 个角色（$(if ($locustStats.Count -ge 5) { "✅ 达标" } else { "❌ 未达标" })）。
5. **安全防护**：恶意注入请求 $(if ($locustStats.ContainsKey("malicious-injection")) { "已拦截" } else { "未测试" })。
"@
})

### 生产化建议

| 优先级 | 建议 | 预期效果 |
|--------|------|---------|
| P0 | $(if ($k6Stats.err5xx -gt 0 -or $k6Stats.errConn -gt 0) { "排查 5xx/连接失败根因并扩容（Ingress/连接池/HPA）" } elseif ($k6Stats.err429 -gt 0) { "按租户分发 API Key，提升速率限制" } else { "当前速率限制足够" }) | $(if ($k6Stats.err5xx -gt 0 -or $k6Stats.errConn -gt 0 -or $k6Stats.err429 -gt 0) { "支持 $Concurrency+ 并发" } else { "维持现状" }) |
| P1 | 部署 HPA（CPU > 70% 自动扩容） | 应对流量洪峰 |
| P1 | 开启语义缓存（命中率目标 > 30%） | P95 降至 < 500ms |
| P2 | Ingress Controller 多副本 | 消除网络瓶颈 |
| P2 | 真实 LLM 压测（替换 MockLLM） | 验证端到端延迟 |

---

## 八、原始数据

| 文件 | 路径 |
|------|------|
$(if (-not $locustOutput) { "" } else { "`n| Locust CSV 报告 | $locustOutput/locust-report_stats.csv |" +
"`n| Locust JSON 报告 | $locustOutput/locust-report.json |" +
"`n| Locust 日志 | $ResultDir/locust.log |" })$(if (-not $k6Output) { "" } else { "`n| k6 JSON 结果 | $k6Output/k6-results.json |" +
"`n| k6 汇总 | $k6Output/k6-summary.json |" +
"`n| k6 日志 | $ResultDir/k6.log |" })`n| Pod 资源采样 | $containerCsv |
| 宿主资源采样 | $hostCsv |
| 测试配置 | $ResultDir/test-config.json |

---

> 生成时间：$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
"@

    Set-Content -Path $reportPath -Value $report -Encoding UTF8
    Write-Host "[run_stress] 性能测试报告已生成: $reportPath" -ForegroundColor Green

    Write-Host "[run_stress] 所有结果已保存到: $ResultDir" -ForegroundColor Green
    Write-Host "[run_stress] 完成！" -ForegroundColor Green

} finally {
    if ($resourceJob) {
        Stop-Job $resourceJob -ErrorAction SilentlyContinue | Out-Null
        Remove-Job $resourceJob -Force -ErrorAction SilentlyContinue | Out-Null
        $resourceJob = $null
    }
    if ($upNeeded -and -not $Keep) {
        Write-Host "[run_stress] 回收 compose 服务..." -ForegroundColor Cyan
        docker compose -f $Compose down
    } elseif ($upNeeded -and $Keep) {
        Write-Host "[run_stress] 已保留容器（-Keep）。手动回收: docker compose down" -ForegroundColor DarkGray
    }
}
