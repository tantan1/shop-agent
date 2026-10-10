#!/usr/bin/env pwsh
# run_tests.ps1 — 一键运行所有 app 单元测试
#
# 用法：
#   .\run_tests.ps1                  # 全部 app
#   .\run_tests.ps1 -App shop        # 只跑 shop-agent
#   .\run_tests.ps1 -App gateway     # 只跑 gateway
#   .\run_tests.ps1 -App monitor     # 只跑 monitoring-agent
#   .\run_tests.ps1 -Filter "stress" # 追加 -k 过滤

param(
    [ValidateSet("all","shop","gateway","monitor")]
    [string]$App = "all",
    [string]$Filter = ""
)

$ErrorActionPreference = "Continue"
$Root = $PSScriptRoot

# 已知失败（历史遗留 / 缺 opentelemetry-exporter-otlp，不计入 fail）
$knownFails = @(
    "test_mockapi.py",            # shop-agent: 依赖外部 mock
    "test_mcp_server.py",         # shop-agent: 依赖外部 mock
    "test_content_compression.py",# shop-agent: 依赖外部 mock
    "test_2b01_structure.py",     # gateway: 缺 otel grpc exporter
    "test_2b02_routing.py",
    "test_2b03_cost.py",
    "test_2b05_loopguard.py",
    "test_2b06_compliance.py",
    "test_2b09_metrics_business.py",
    "test_2b10_injection.py",
    "test_2b11_e2e_mock_server.py"
)

# 构建待跑 suites：(@{N=name; D=dir})
$suites = @()
if ($App -eq "all" -or $App -eq "shop")    { $suites += @{N="shop-agent";       D="$Root/apps/shop-agent"} }
if ($App -eq "all" -or $App -eq "gateway") { $suites += @{N="gateway";          D="$Root/apps/gateway"} }
if ($App -eq "all" -or $App -eq "monitor") { $suites += @{N="monitoring-agent"; D="$Root/apps/monitoring-agent"} }

$pass=0; $fail=0; $empty=0

foreach ($s in $suites) {
    Write-Host "`n=== $($s.N) ===" -ForegroundColor Cyan
    Push-Location $s.D
    try {
        $pyArgs = @("-m","pytest","tests/","-v","--tb=short","--no-header","--color=yes")
        foreach ($f in $knownFails) { $pyArgs += "--ignore=tests/$f" }
        if ($Filter) { $pyArgs += "-k"; $pyArgs += $Filter }
        & python @pyArgs
        switch ($LASTEXITCODE) {
            0       { $pass++ }
            5       { $empty++ }
            default { $fail++ }
        }
    } finally { Pop-Location }
}

Write-Host "`n===== RESULT =====" -ForegroundColor $(if($fail -eq 0){"Green"}else{"Red"})
Write-Host "Suites: $($suites.Count) | Pass: $pass | Fail: $fail | Empty: $empty"
exit $fail
