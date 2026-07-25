$ErrorActionPreference = 'Continue'
$pidTrain = 33540
$log = "outputs/export_tool_select.log"
$err = "outputs/export_tool_select.err"

function Ts { return (Get-Date -Format "yyyy-MM-dd HH:mm:ss") }
Add-Content -Encoding utf8 $log "[$(Ts)] waiting for training PID $pidTrain to finish..."
try {
    $p = Get-Process -Id $pidTrain -ErrorAction SilentlyContinue
    if ($p) { $p.WaitForExit(3600000) }   # 最多等 1 小时
} catch { Add-Content -Encoding utf8 $log "[$(Ts)] wait error: $_" }

# 确认训练进程已退出
Start-Sleep -Seconds 5
if (Get-Process -Id $pidTrain -ErrorAction SilentlyContinue) {
    Add-Content -Encoding utf8 $log "[$(Ts)] training still alive, aborting export."
    exit 1
}
Add-Content -Encoding utf8 $log "[$(Ts)] training finished. launching export (LoRA -> merged model)..."

# 导出 merge
& .\venv_cuda\Scripts\llamafactory-cli.exe export export_qwen_tool_select.yaml dataset_dir=data/llamafactory 2>$err | Out-File -Encoding utf8 -Append $log
Add-Content -Encoding utf8 $log "[$(Ts)] EXPORT_EXIT=$LASTEXITCODE"

# 校验产出
if (Test-Path "models/Qwen2.5-1.5B-Instruct-tool-select") {
    Add-Content -Encoding utf8 $log "[$(Ts)] OK: models/Qwen2.5-1.5B-Instruct-tool-select created."
} else {
    Add-Content -Encoding utf8 $log "[$(Ts)] WARN: export dir not found, check $err"
}
