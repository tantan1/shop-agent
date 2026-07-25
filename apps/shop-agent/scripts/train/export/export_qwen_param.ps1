# 导出脚本：把全量训练产出的 LoRA adapter merge 成独立 sft 模型目录
# 环境：venv_cuda (CUDA torch 2.6.0+cu124)
# 说明：
#  - 全程用 "python -m llamafactory.cli export" 绕过损坏的 pip.exe 启动器。
#  - 配置见项目根 export_qwen_param.yaml。
#  - 产物默认写到 ./models/Qwen2.5-1.5B-Instruct-sft（eval 脚本的 --sft 指向它）。
# 用法：
#   .\scripts\export_qwen_param.ps1
#   .\scripts\export_qwen_param.ps1 -Config export_qwen_param.yaml

param(
    [string]$Config = "export_qwen_param.yaml"
)

$ErrorActionPreference = "Continue"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPy  = Join-Path $ProjectRoot "venv_cuda\Scripts\python.exe"
$Cfg     = Join-Path $ProjectRoot $Config

# 0) 确认 llamafactory 已装入 venv_cuda
& $VenvPy -c "import llamafactory" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] venv_cuda 未安装 llamafactory，请先执行：" -ForegroundColor Red
    Write-Host "  $VenvPy -m pip install llamafactory"
    exit 1
}

# 1) 确认 adapter 目录存在（全量训练需先跑完）
$AdapterDir = (& $VenvPy -c "import yaml,sys; print(yaml.safe_load(open(r'$Cfg',encoding='utf-8'))['adapter_name_or_path'])").Trim()
if (-not (Test-Path $AdapterDir)) {
    Write-Host "[ERROR] 未找到 adapter 目录: $AdapterDir" -ForegroundColor Red
    Write-Host "        请先运行全量训练： .\scripts\train_qwen_param.ps1" -ForegroundColor Yellow
    exit 1
}

# 2) 执行导出（merge）
Write-Host "[INFO] 开始导出： $Cfg"
Push-Location $ProjectRoot
try {
    & $VenvPy -m llamafactory.cli export $Cfg
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] 导出失败，llamafactory-cli 退出码 $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}

Write-Host "[INFO] 导出完成，独立 sft 模型目录已生成（见 yaml 中 export_dir）。" -ForegroundColor Green
