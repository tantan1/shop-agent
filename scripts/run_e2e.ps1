#!/usr/bin/env pwsh
# shop-agent e2e 冒烟一键脚本（PowerShell）
#
# 动作：
#   1. 拉起 docker-compose 中 shop-agent 及其依赖（redis / gateway / order-service / milvus / postgres ...）
#   2. 等待 shop-agent /health 就绪
#   3. 运行 tests/e2e/smoke_e2e.py（覆盖 #2/#4/#7/#8）
#   4. 默认跑完回收容器；加 -Keep 参数可保留现场
#
# 用法：
#   .\run_e2e.ps1                 # 跑完自动 down
#   .\run_e2e.ps1 -Keep           # 保留容器
#   .\run_e2e.ps1 -NoUp           # 假定服务已在跑，只执行冒烟
#   .\run_e2e.ps1 -WithVLLM       # 额外拉起 docker-compose.vllm.yml 中的真实 LLM 模型
#                                 # （bge-reranker / bge-small-zh-v1.5 / qwen3，需本机有 NVIDIA GPU）。
#                                 # 默认不启：e2e 冒烟走 mock 降级，不需要 GPU；
#                                 # 仅端到端验证真实 LLM（参数抽取/嵌入/重排）时才加此开关。

param(
    [switch]$Keep,
    [switch]$NoUp,
    [switch]$WithVLLM
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
$Smoke = Join-Path $RepoRoot "apps/shop-agent/tests/e2e/smoke_e2e.py"
$Compose = Join-Path $RepoRoot "docker-compose.yml"
$VllmCompose = Join-Path $RepoRoot "docker-compose.vllm.yml"

# ── 解析 .env 里的 FIXED_API_KEY（与 shop-agent 启动一致）──
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

# Redis 连接参数（与 compose redis 服务一致；宿主侧映射为 6390，见 docker-compose.yml）
if (-not $env:REDIS_HOST) { $env:REDIS_HOST = "localhost" }
if (-not $env:REDIS_PORT) { $env:REDIS_PORT = "6390" }
# 若 .env 为 redis 配置了密码，需透传给冒烟脚本（否则连不上报 Authentication required）
if (-not $env:REDIS_AUTH) {
    $envFile = Join-Path $RepoRoot ".env"
    if (Test-Path $envFile) {
        foreach ($line in (Get-Content $envFile)) {
            if ($line -match '^\s*REDIS_AUTH\s*=\s*(.+)\s*$') {
                $env:REDIS_AUTH = $Matches[1].Trim('"').Trim("'")
                break
            }
        }
    }
}

$upNeeded = -not $NoUp

try {
    if ($upNeeded) {
        Write-Host "[run_e2e] 拉起 compose 服务（redis / shop-agent / gateway / order-service 及依赖）..." -ForegroundColor Cyan
        docker compose -f $Compose up -d --remove-orphans redis shop-agent gateway order-service
        if ($LASTEXITCODE -ne 0) { throw "docker compose up 失败" }

        # 真实 LLM 端到端场景：显式 -WithVLLM 才拉起本地 vLLM 模型服务（需本机 GPU）。
        # 默认跳过——冒烟走 mock/降级，不要求 GPU。vLLM 与主栈同处默认网络
        # （shop-agent_default），容器名 vllm-bge-small-zh / vllm-bge-reranker / vllm-qwen3 可被直接解析。
        if ($WithVLLM) {
            if (-not (Test-Path $VllmCompose)) {
                throw "未找到 vLLM compose 文件: $VllmCompose"
            }
            Write-Host "[run_e2e] -WithVLLM：拉起本地 vLLM 模型（bge-reranker / bge-small-zh-v1.5 / qwen3）..." -ForegroundColor Cyan
            docker compose -f $VllmCompose up -d --remove-orphans
            if ($LASTEXITCODE -ne 0) { throw "docker compose -f docker-compose.vllm.yml up 失败（确认本机有可用 NVIDIA GPU）" }
        }
    }

    # ── 等待 shop-agent 就绪（最多 ~180s）──
    Write-Host "[run_e2e] 等待 shop-agent /health 就绪..." -ForegroundColor Cyan
    $health = "http://localhost:8000/health"
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $resp = Invoke-WebRequest -Uri $health -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue
            if ($resp.StatusCode -eq 200) { $ready = $true; break }
        } catch { }
        Start-Sleep -Seconds 3
    }
    if (-not $ready) { throw "shop-agent 在超时内未就绪（请检查容器日志：docker compose logs shop-agent）" }
    Write-Host "[run_e2e] shop-agent 已就绪。" -ForegroundColor Green

    # ── 运行冒烟脚本 ──
    Write-Host "[run_e2e] 运行 e2e 冒烟脚本..." -ForegroundColor Cyan
    python $Smoke
    $exitCode = $LASTEXITCODE
    Write-Host "[run_e2e] 冒烟脚本退出码: $exitCode" -ForegroundColor $(if ($exitCode -eq 0) { "Green" } else { "Yellow" })
}
finally {
    if ($upNeeded -and -not $Keep) {
        Write-Host "[run_e2e] 回收 compose 服务..." -ForegroundColor Cyan
        docker compose -f $Compose down
        if ($WithVLLM) {
            Write-Host "[run_e2e] 回收 vLLM 模型服务..." -ForegroundColor Cyan
            docker compose -f $VllmCompose down
        }
    } elseif ($upNeeded -and $Keep) {
        $keepMsg = "[run_e2e] 已保留容器（-Keep）。手动回收: docker compose down"
        if ($WithVLLM) { $keepMsg += "`n            vLLM 模型: docker compose -f docker-compose.vllm.yml down" }
        Write-Host $keepMsg -ForegroundColor DarkGray
    }
}

exit $exitCode
