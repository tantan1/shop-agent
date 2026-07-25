$ErrorActionPreference = 'Continue'
$log = "outputs/train_tool_select.log"
$err = "outputs/train_tool_select.err"
$expLog = "outputs/export_tool_select.log"
$expErr = "outputs/export_tool_select.err"
function Ts { return (Get-Date -Format "yyyy-MM-dd HH:mm:ss") }

Add-Content -Encoding utf8 $log "[$(Ts)] === START S-scale fast train (cutoff 1024, batch 8) ==="
& .\venv_cuda\Scripts\llamafactory-cli.exe train train_qwen_tool_select.yaml dataset_dir=data/llamafactory 2>$err | Out-File -Encoding utf8 -Append $log
Add-Content -Encoding utf8 $log "[$(Ts)] TRAIN_EXIT=$LASTEXITCODE"

if ($LASTEXITCODE -ne 0) {
    Add-Content -Encoding utf8 $expLog "[$(Ts)] TRAIN failed, skip export. see $err"
    exit 1
}

Add-Content -Encoding utf8 $expLog "[$(Ts)] === EXPORT (LoRA -> merged model) ==="
& .\venv_cuda\Scripts\llamafactory-cli.exe export export_qwen_tool_select.yaml dataset_dir=data/llamafactory 2>$expErr | Out-File -Encoding utf8 -Append $expLog
Add-Content -Encoding utf8 $expLog "[$(Ts)] EXPORT_EXIT=$LASTEXITCODE"

if (Test-Path "models/Qwen2.5-1.5B-Instruct-tool-select") {
    Add-Content -Encoding utf8 $expLog "[$(Ts)] OK: models/Qwen2.5-1.5B-Instruct-tool-select created."
} else {
    Add-Content -Encoding utf8 $expLog "[$(Ts)] WARN: export dir not found, check $expErr"
}
