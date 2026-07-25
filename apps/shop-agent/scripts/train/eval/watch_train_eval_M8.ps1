$ErrorActionPreference = 'Continue'
$merged = "models/Qwen2.5-1.5B-Instruct-tool-select-M8"
$log = "outputs/watch_eval_M8.log"
function Ts { return (Get-Date -Format "yyyy-MM-dd HH:mm:ss") }
Add-Content -Encoding utf8 $log "[$(Ts)] watcher: waiting for merged model $merged ..."

# 轮询等待导出完成（最多 3 小时）
$ok = $false
for ($i = 0; $i -lt 360; $i++) {
    if (Test-Path $merged) { $ok = $true; break }
    Start-Sleep -Seconds 30
}
if (-not $ok) {
    Add-Content -Encoding utf8 $log "[$(Ts)] watcher TIMEOUT: merged model not found."
    exit 1
}
Add-Content -Encoding utf8 $log "[$(Ts)] watcher: merged model found. starting M=8 evaluations..."

# —— base @ M=8 ——
Add-Content -Encoding utf8 $log "[$(Ts)] >>> eval base @ M=8"
& .\venv_cuda\Scripts\python.exe scripts/eval_tool_select_sft.py --model ./models/Qwen2.5-1.5B-Instruct --M 8 --runs 3 --out outputs/eval_M8_base.json 2>&1 | Out-File -Encoding utf8 -Append $log

# —— SFT-M8 @ M=8 ——
Add-Content -Encoding utf8 $log "[$(Ts)] >>> eval SFT-M8 @ M=8"
& .\venv_cuda\Scripts\python.exe scripts/eval_tool_select_sft.py --model ./models/Qwen2.5-1.5B-Instruct-tool-select-M8 --M 8 --runs 3 --out outputs/eval_M8_sft.json 2>&1 | Out-File -Encoding utf8 -Append $log

Add-Content -Encoding utf8 $log "[$(Ts)] watcher DONE. results: outputs/eval_M8_base.json / outputs/eval_M8_sft.json"
Write-Host "[watcher] M=8 评测完成，详见 $log"
