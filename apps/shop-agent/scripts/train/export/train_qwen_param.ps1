# 微调启动脚本：Qwen2.5-1.5B 参数抽取 (LLaMA-Factory SFT)
# 环境：venv_cuda (CUDA torch 2.6.0+cu124)
# 说明：
#  - venv_cuda 的 pip.exe 启动器路径已损坏（指向旧的 venv_gpu），
#    故全程用 "python -m llamafactory.cli" 绕过启动器。
#  - 训练配置见项目根目录 train_qwen_param.yaml（模型已指向本地 ./models/Qwen2.5-1.5B-Instruct）。
#  - 数据集 data/llamafactory/ 下的 dataset_info.json 与 shop_param_v1.json
#    会被拷进 LLaMA-Factory 包内 data 目录，确保框架能找到。
# 用法：
#   全量训练： .\scripts\train_qwen_param.ps1
#   冒烟测试： .\scripts\train_qwen_param.ps1 -Config train_qwen_param_smoke.yaml

param(
    [string]$Config = "train_qwen_param.yaml"
)

$ErrorActionPreference = "Continue"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPy  = Join-Path $ProjectRoot "venv_cuda\Scripts\python.exe"
$Cfg     = Join-Path $ProjectRoot $Config
$SrcData = Join-Path $ProjectRoot "data\llamafactory"

# 0) 确认 llamafactory 已装入 venv_cuda
& $VenvPy -c "import llamafactory" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] venv_cuda 未安装 llamafactory，请先执行：" -ForegroundColor Red
    Write-Host "  $VenvPy -m pip install llamafactory"
    exit 1
}

# 1) 注册数据集：LLaMA-Factory 默认从 当前工作目录的 data/ 读取 dataset_info.json
#    （不读包内 data 目录），故拷到项目根 data/，并用 --dataset_dir 显式指定。
$ProjData = Join-Path $ProjectRoot "data"
if (-not (Test-Path $ProjData)) { New-Item -ItemType Directory -Path $ProjData | Out-Null }
Copy-Item (Join-Path $SrcData "dataset_info.json") (Join-Path $ProjData "dataset_info.json") -Force
Copy-Item (Join-Path $SrcData "shop_param_v1_aug.json") (Join-Path $ProjData "shop_param_v1_aug.json") -Force
Write-Host "[INFO] 数据集已注册到: $ProjData"

# 2) 启动训练（CWD 切到项目根；dataset_dir 不通过命令行传，
#    框架默认读 CWD 下的 data/，脚本已把数据拷到 $ProjData）
Write-Host "[INFO] 开始训练： $Cfg"
Push-Location $ProjectRoot
try {
    & $VenvPy -m llamafactory.cli train $Cfg
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] 训练失败，llamafactory-cli 退出码 $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}
