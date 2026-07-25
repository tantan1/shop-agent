#  微调 + 自动合并脚本：训练 Qwen2.5-0.5B 参数抽取 LoRA，完成后自动 merge 到
# ./models/Qwen2.5-0.5B-Instruct-sft（eval 脚本 --sft 指向它）。
#
# 环境：venv_cuda (CUDA torch 2.6.0+cu124)
# 与 1.5B 版本完全相同的流程，仅模型从 1.5B 换为 0.5B。
# 项目根 = workspace 根（e:\workspace\shop-agent）
#
# 用法：
#   标准训练并合并（用 shop_param_v1）：
#     .\apps\shop-agent\scripts\train_and_merge_05b.ps1
#
#   合并后自动跑一遍评测验证（base 0.5B vs sft 0.5B）：
#     .\apps\shop-agent\scripts\train_and_merge_05b.ps1 -Eval

param(
    [string]$Config = "benchmark_results/configs/train_qwen_param_05b.yaml",  # 训练配置（相对 workspace 根）
    [string]$ExportConfig = "benchmark_results/configs/export_qwen_param_05b.yaml",  # 导出/合并配置
    [switch]$Augment,                                         # 训练前先跑误差驱动增强
    [string]$Dataset = "",                                    # 显式指定数据集名；默认按 -Augment 推断
    [string]$EvalJson = "eval_sft_before_after.json",        # 增强依赖的评测结果
    [int]$AugMultiplier = 3,
    [int]$AugMinPerIntent = 30,
    [int]$AugMaxPerIntent = 120,
    [switch]$SkipTrain,                                       # 跳过训练，只合并（已有 adapter）
    [switch]$NoMerge,                                         # 只训练，不合并
    [switch]$Resume,                                          # 从已有 checkpoint 续训（默认：覆盖重训）
    [switch]$Eval,                                             # 合并后自动评测对比 base vs sft
    [int]$BatchSize = 8                                        # 评测批量大小
)

$ErrorActionPreference = "Stop"

# ── 计时 ──────────────────────────────────────────────────────────
$Timings = @{}
$swTotal = [System.Diagnostics.Stopwatch]::StartNew()
function Format-Span($ts) {
    $h = [int]$ts.TotalHours
    $m = $ts.Minutes
    $s = $ts.Seconds
    $d1 = [int]($ts.Milliseconds / 100)
    if ($h -gt 0) { return ("{0}h {1}m {2}s" -f $h, $m, $s) }
    if ($m -gt 0) { return ("{0}m {1}.{2:D1}s" -f $m, $s, $d1) }
    return ("{0}.{1:D1}s" -f $s, $d1)
}

# 项目根 = workspace 根（models/venv_cuda/data 都在这里）
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$VenvPy  = Join-Path $ProjectRoot "venv_cuda\Scripts\python.exe"
$Cfg     = Join-Path $ProjectRoot $Config
$ExpCfg  = Join-Path $ProjectRoot $ExportConfig
$LLFData = Join-Path $ProjectRoot "data\llamafactory"
$ProjData = Join-Path $ProjectRoot "data"
$Stamp   = (Get-Date -Format "yyyyMMddHHmmss")

function Write-Step($msg) { Write-Host "`n[STEP] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "[ OK ] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[ERR ] $msg" -ForegroundColor Red }

# 0) 确认 llamafactory 已装入 venv_cuda
& $VenvPy -c "import llamafactory" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Err "venv_cuda 未安装 llamafactory，请先执行： $VenvPy -m pip install llamafactory"
    exit 1
}

# 1) 决定使用的数据集名
$DatasetName = $Dataset
if (-not $DatasetName) {
    $DatasetName = if ($Augment) { "shop_param_v1_aug" } else { "shop_param_v1" }
}
Write-Host "[INFO] 使用数据集: $DatasetName"

# ── 可选：误差驱动增强 ────────────────────────────────────────────
if ($Augment) {
    $swAug = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Step "误差驱动增强（读 $EvalJson，补数据 -> $DatasetName）"
    $EvalPath = Join-Path $ProjectRoot $EvalJson
    if (-not (Test-Path $EvalPath)) {
        Write-Err "未找到评测结果 $EvalPath，无法增强。请先跑： scripts/eval_sft_before_after.py --base ./models/Qwen2.5-0.5B-Instruct --sft ./models/Qwen2.5-0.5B-Instruct-sft --data data/llamafactory/shop_param_test.json"
        exit 1
    }
    & $VenvPy apps/shop-agent/scripts/augment_by_errors.py `
        --eval $EvalPath `
        --multiplier $AugMultiplier `
        --min-per-intent $AugMinPerIntent `
        --max-per-intent $AugMaxPerIntent
    if ($LASTEXITCODE -ne 0) {
        Write-Err "增强失败（退出码 $LASTEXITCODE），中止以防用脏数据训练。"
        exit $LASTEXITCODE
    }
    Write-Ok "增强完成 -> data/llamafactory/shop_param_v1_aug.json"
    $Timings["augment"] = $swAug.Elapsed
    $DatasetName = "shop_param_v1_aug"
}

# 2) 确保 dataset_info.json 注册了要用到的数据集
$swReg = [System.Diagnostics.Stopwatch]::StartNew()
Write-Step "校验数据集注册（data/llamafactory/dataset_info.json）"
$diPath = Join-Path $LLFData "dataset_info.json"
$dataInfo = Get-Content $diPath -Raw -Encoding UTF8 | ConvertFrom-Json -Depth 5
$needRegister = @()
if ($DatasetName -eq "shop_param_v1_aug" -and -not $dataInfo.PSObject.Properties["shop_param_v1_aug"]) {
    $needRegister += "shop_param_v1_aug"
}
if ($needRegister.Count -gt 0) {
    $regJson = @{
        "file_name"  = "shop_param_v1_aug.json"
        "formatting" = "sharegpt"
        "tags" = @{
            "role_tag"      = "role"
            "content_tag"   = "content"
            "user_tag"      = "user"
            "assistant_tag" = "assistant"
            "system_tag"    = "system"
            "messages"      = "conversations"
        }
    } | ConvertTo-Json -Depth 5 -Compress
    & $VenvPy -c @"
import json,sys
p=r'$diPath'
d=json.load(open(p,encoding='utf-8'))
d.setdefault('shop_param_v1_aug', $regJson)
json.dump(d, open(p,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
print('registered shop_param_v1_aug')
"@
    Write-Ok "已注册 shop_param_v1_aug"
} else {
    Write-Ok "数据集 $DatasetName 已注册"
}
$Timings["register"] = $swReg.Elapsed

# 3) 准备训练配置副本，并把 dataset 字段改成目标数据集
Write-Step "准备训练配置副本（dataset -> $DatasetName）"
if (-not (Test-Path $Cfg)) {
    Write-Err "训练配置不存在: $Cfg"
    exit 1
}
$RunCfg = Join-Path $ProjectRoot "outputs/_train_merge_05b_$Stamp.yaml"
Copy-Item $Cfg $RunCfg -Force
(Get-Content $RunCfg) -replace '^dataset:\s*[A-Za-z0-9_]+.*$', "dataset: $DatasetName" |
    Set-Content $RunCfg

$OutputDir = (& $VenvPy -c "import yaml; print(yaml.safe_load(open(r'$RunCfg',encoding='utf-8'))['output_dir'])").Trim()
$OutputDirAbs = Join-Path $ProjectRoot ($OutputDir -replace '^\./','')

if (-not $Resume) {
    Add-Content $RunCfg "`noverwrite_output_dir: true"
    if (Test-Path $OutputDirAbs) {
        Write-Warn "覆盖重训：清理旧输出目录 $OutputDirAbs"
        Remove-Item $OutputDirAbs -Recurse -Force
    }
    Write-Ok "训练副本: $RunCfg （overwrite_output_dir=true，从头训练）"
} else {
    Write-Ok "训练副本: $RunCfg （Resume 模式）"
}

# 4) 训练
if (-not $SkipTrain) {
    $swTrain = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Step "训练 LoRA（配置: $Config，数据集: $DatasetName）"
    Push-Location $ProjectRoot
    try {
        if (-not (Test-Path $ProjData)) { New-Item -ItemType Directory -Path $ProjData | Out-Null }
        Copy-Item (Join-Path $LLFData "dataset_info.json") (Join-Path $ProjData "dataset_info.json") -Force
        if ($DatasetName -eq "shop_param_v1_aug") {
            Copy-Item (Join-Path $LLFData "shop_param_v1_aug.json") (Join-Path $ProjData "shop_param_v1_aug.json") -Force
        } else {
            Copy-Item (Join-Path $LLFData "shop_param_v1.json") (Join-Path $ProjData "shop_param_v1.json") -Force
        }
        & $VenvPy -m llamafactory.cli train $RunCfg
        if ($LASTEXITCODE -ne 0) {
            Write-Err "训练失败（退出码 $LASTEXITCODE），不执行合并。"
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
    Write-Ok "训练完成，adapter -> ./outputs/qwen05b-param-lora"
    $Timings["train"] = $swTrain.Elapsed
} else {
    Write-Warn "跳过训练（SkipTrain），直接合并已有 adapter。"
}

# 5) 合并（导出）到 ./models/Qwen2.5-0.5B-Instruct-sft
if (-not $NoMerge) {
    $swMerge = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Step "合并 LoRA -> ./models/Qwen2.5-0.5B-Instruct-sft"
    if (-not (Test-Path $ExpCfg)) {
        Write-Err "导出配置不存在: $ExpCfg"
        exit 1
    }
    $adapter = (& $VenvPy -c "import yaml; print(yaml.safe_load(open(r'$ExpCfg',encoding='utf-8'))['adapter_name_or_path'])").Trim()
    if (-not (Test-Path $adapter)) {
        Write-Err "未找到 adapter 目录: $adapter 。请先跑训练。"
        exit 1
    }
    Push-Location $ProjectRoot
    try {
        & $VenvPy -m llamafactory.cli export $ExpCfg
        if ($LASTEXITCODE -ne 0) {
            Write-Err "合并失败（退出码 $LASTEXITCODE）。"
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
    $sftDir = (& $VenvPy -c "import yaml; print(yaml.safe_load(open(r'$ExpCfg',encoding='utf-8'))['export_dir'])").Trim()
    Write-Ok "合并完成 -> $sftDir"
    $Timings["merge"] = $swMerge.Elapsed
} else {
    Write-Warn "跳过合并（NoMerge），仅产出 adapter。"
}

# 6) 可选：合并后自动评测
if ($Eval) {
    $swEval = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Step "合并后评测（base 0.5B vs sft 0.5B）"
    $TestPath = Join-Path $ProjectRoot "data/llamafactory/shop_param_test.json"
    Push-Location $ProjectRoot
    try {
    & $VenvPy apps/shop-agent/scripts/eval_sft_before_after.py `
        --base ./models/Qwen2.5-0.5B-Instruct `
        --sft  ./models/Qwen2.5-0.5B-Instruct-sft `
        --data $TestPath --device cuda --batch-size $BatchSize
        if ($LASTEXITCODE -ne 0) {
            Write-Err "评测失败（退出码 $LASTEXITCODE）。"
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
    Write-Ok "评测完成，结果见 eval_sft_before_after.json"
    $Timings["eval"] = $swEval.Elapsed
}

$swTotal.Stop()
Write-Host "`n[DONE] 0.5B 流程结束（dataset=$DatasetName, config=$Config）。临时训练副本: outputs/_train_merge_05b_$Stamp.yaml" -ForegroundColor Green

# ── 计时汇总 ──────────────────────────────────────────────────────
Write-Host "`n[TIMING] 各阶段耗时:" -ForegroundColor Cyan
$phaseLabels = @{
    "augment"  = "误差增强"
    "register" = "数据集注册/准备"
    "train"    = "训练 LoRA"
    "merge"    = "合并导出"
    "eval"     = "评测"
}
$phaseOrder = @("augment", "register", "train", "merge", "eval")
$sumStaged = [TimeSpan]::Zero
foreach ($k in $phaseOrder) {
    if ($Timings.ContainsKey($k)) {
        $ts = $Timings[$k]
        $sumStaged = $sumStaged.Add($ts)
        Write-Host ("  {0,-16} {1}" -f $phaseLabels[$k], (Format-Span $ts)) -ForegroundColor Green
    }
}
if ($sumStaged.Ticks -gt 0) {
    Write-Host ("  {0,-16} {1}" -f "分阶段合计", (Format-Span $sumStaged)) -ForegroundColor Yellow
}
Write-Host ("  {0,-16} {1}" -f "总墙钟", (Format-Span $swTotal.Elapsed)) -ForegroundColor Yellow
