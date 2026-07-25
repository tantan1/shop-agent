$ErrorActionPreference = 'Continue'
$pidTrain = (Get-Content -Encoding utf8 outputs/train_M8.pid).Trim()
$log = "outputs/export_M8.log"
$err = "outputs/export_M8.err"

function Ts { return (Get-Date -Format "yyyy-MM-dd HH:mm:ss") }
Add-Content -Encoding utf8 $log "[$(Ts)] waiting for training PID $pidTrain to finish..."

try {
    $p = Get-Process -Id $pidTrain -ErrorAction SilentlyContinue
    if ($p) { $p.WaitForExit(7200000) }   # 最多等 2 小时
} catch { Add-Content -Encoding utf8 $log "[$(Ts)] wait error: $_" }

# 确认训练进程已退出
Start-Sleep -Seconds 5
if (Get-Process -Id $pidTrain -ErrorAction SilentlyContinue) {
    Add-Content -Encoding utf8 $log "[$(Ts)] training still alive, aborting export."
    exit 1
}

# 关键：确认 adapter 真正写出（崩溃退出时进程也会结束，但无 adapter_config.json）
$adapterCfg = "outputs/qwen15b-tool-select-lora-M8/adapter_config.json"
if (-not (Test-Path $adapterCfg)) {
    Add-Content -Encoding utf8 $log "[$(Ts)] WARN: $adapterCfg not found -> training likely crashed (no adapter). SKIP export."
    exit 2
}
Add-Content -Encoding utf8 $log "[$(Ts)] training finished & adapter verified. launching export (LoRA -> merged model)..."

# 导出 merge
& .\venv_cuda\Scripts\llamafactory-cli.exe export export_qwen_tool_select_M8.yaml dataset_dir=data/llamafactory 2>$err | Out-File -Encoding utf8 -Append $log
Add-Content -Encoding utf8 $log "[$(Ts)] EXPORT_EXIT=$LASTEXITCODE"

# 校验产出
if (Test-Path "models/Qwen2.5-1.5B-Instruct-tool-select-M8") {
    Add-Content -Encoding utf8 $log "[$(Ts)] OK: models/Qwen2.5-1.5B-Instruct-tool-select-M8 created."
} else {
    Add-Content -Encoding utf8 $log "[$(Ts)] WARN: export dir not found, check $err"
}
