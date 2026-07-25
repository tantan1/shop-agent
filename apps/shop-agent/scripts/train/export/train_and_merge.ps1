# 微调 + 自动合并脚本：训练 Qwen2.5-1.5B 参数抽取 LoRA，完成后自动 merge 到
# ./models/Qwen2.5-1.5B-Instruct-sft（eval 脚本 --sft 指向它）。
#
# 环境：venv_cuda (CUDA torch 2.6.0+cu124)
# 说明：
#  - 全程用 "python -m llamafactory.cli" 绕过损坏的 pip.exe 启动器。
#  - 顺序：可选「误差驱动增强」→ 训练 → 导出合并。
#  - 训练失败则不会执行合并（避免用半截产物污染 sft 目录）。
#  - 选用增强数据集时，会自动确保 dataset_info.json 注册 shop_param_v1_aug，
#    并把训练配置里的 dataset 改成对应名字（不改动你提交的 yaml，只改临时副本）。
#  - 合并目标固定为 ./models/Qwen2.5-1.5B-Instruct-sft（见 export_qwen_param.yaml）。
#
# 用法：
#   标准训练并合并（用 shop_param_v1）：
#     .\scripts\train_and_merge.ps1
#
#   先按上轮评测结果补数据，再训练并合并：
#     .\scripts\train_and_merge.ps1 -Augment
#
#   指定增强参数 / 评测结果文件：
#     .\scripts\train_and_merge.ps1 -Augment -EvalJson eval_sft_before_after.json `
#         -AugMultiplier 3 -AugMaxPerIntent 120
#
#   仅合并（不训练，adapter 已存在时）：
#     .\scripts\train_and_merge.ps1 -SkipTrain
#
#   从已有 checkpoint 续训（默认是覆盖重训，换数据时必须覆盖）：
#     .\scripts\train_and_merge.ps1 -Resume
#
#   训练但不合并（只想产 adapter）：
#     .\scripts\train_and_merge.ps1 -NoMerge
#
#   合并后自动跑一遍评测验证：
#     .\scripts\train_and_merge.ps1 -Augment -Eval

param(
    [string]$Config = "train_qwen_param.yaml",            # 训练配置（相对项目根）
    [string]$ExportConfig = "export_qwen_param.yaml",     # 导出/合并配置
    [switch]$Augment,                                     # 训练前先跑误差驱动增强
    [string]$Dataset = "",                                # 显式指定数据集名；默认按 -Augment 推断
    [string]$EvalJson = "eval_sft_before_after.json",     # 增强依赖的评测结果
    [int]$AugMultiplier = 3,
    [int]$AugMinPerIntent = 30,
    [int]$AugMaxPerIntent = 120,
    [switch]$SkipTrain,                                   # 跳过训练，只合并（已有 adapter）
    [switch]$NoMerge,                                     # 只训练，不合并
    [switch]$Resume,                                      # 从已有 checkpoint 续训（默认：覆盖重训）
    [switch]$Eval,                                         # 合并后自动评测对比 base vs sft
    [int]$BatchSize = 8                                     # 评测批量大小（透传给 eval 脚本）
)

$ErrorActionPreference = "Stop"

# ── 计时：记录各阶段与总墙钟 ──────────────────────────────────────────
$Timings = @{}
$swTotal = [System.Diagnostics.Stopwatch]::StartNew()
function Format-Span($ts) {
    $h = [int]$ts.TotalHours
    $m = $ts.Minutes
    $s = $ts.Seconds
    $d1 = [int]($ts.Milliseconds / 100)   # 十分之一秒
    if ($h -gt 0) { return ("{0}h {1}m {2}s" -f $h, $m, $s) }
    if ($m -gt 0) { return ("{0}m {1}.{2:D1}s" -f $m, $s, $d1) }
    return ("{0}.{1:D1}s" -f $s, $d1)
}

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPy  = Join-Path $ProjectRoot "venv_cuda\Scripts\python.exe"
$Cfg     = Join-Path $ProjectRoot $Config
$ExpCfg  = Join-Path $ProjectRoot $ExportConfig
$LLFData = Join-Path $ProjectRoot "data\llamafactory"          # dataset_info.json 源
$ProjData = Join-Path $ProjectRoot "data"                      # 训练时拷到的目标（CWD/data）
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

# 1) 决定使用的数据集名（用于训练配置里的 dataset 字段）
$DatasetName = $Dataset
if (-not $DatasetName) {
    $DatasetName = if ($Augment) { "shop_param_v1_aug" } else { "shop_param_v1" }
}
Write-Host "[INFO] 使用数据集: $DatasetName"

# ── 可选：误差驱动增强 ────────────────────────────────────────────────
if ($Augment) {
    $swAug = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Step "误差驱动增强（读 $EvalJson，补数据 -> $DatasetName）"
    $EvalPath = Join-Path $ProjectRoot $EvalJson
    if (-not (Test-Path $EvalPath)) {
        Write-Err "未找到评测结果 $EvalPath，无法增强。请先跑： scripts/eval_sft_before_after.py"
        exit 1
    }
    & $VenvPy scripts/augment_by_errors.py `
        --eval $EvalJson `
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

# 2) 确保 dataset_info.json 注册了要用到的数据集（不改动其他注册）
$swReg = [System.Diagnostics.Stopwatch]::StartNew()
Write-Step "校验数据集注册（data/llamafactory/dataset_info.json）"
$diPath = Join-Path $LLFData "dataset_info.json"
$dataInfo = Get-Content $diPath -Raw -Encoding UTF8 | ConvertFrom-Json -Depth 5
$needRegister = @()
if ($DatasetName -eq "shop_param_v1_aug" -and -not $dataInfo.PSObject.Properties["shop_param_v1_aug"]) {
    $needRegister += "shop_param_v1_aug"
}
if ($needRegister.Count -gt 0) {
    # 注意：$regJson 必须是「内层属性字典」（file_name/formatting/tags），
    # 不能外层再包一层 shop_param_v1_aug，否则写进 json 会嵌套导致 LLaMA-Factory KeyError。
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
    # 合并进现有 json 文件
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
$RunCfg = Join-Path $ProjectRoot "outputs/_train_merge_$Stamp.yaml"
Copy-Item $Cfg $RunCfg -Force
(Get-Content $RunCfg) -replace '^dataset:\s*[A-Za-z0-9_]+.*$', "dataset: $DatasetName" |
    Set-Content $RunCfg

# 读出训练输出目录（用于清理旧 checkpoint / 校验）
$OutputDir = (& $VenvPy -c "import yaml; print(yaml.safe_load(open(r'$RunCfg',encoding='utf-8'))['output_dir'])").Trim()
$OutputDirAbs = Join-Path $ProjectRoot ($OutputDir -replace '^\./','')

if (-not $Resume) {
    # 默认覆盖重训：注入 overwrite_output_dir 并清掉旧 checkpoint，
    # 否则 LLaMA-Factory 会自动从旧 checkpoint 续训，导致「换了数据却没真正重训」。
    Add-Content $RunCfg "`noverwrite_output_dir: true"
    if (Test-Path $OutputDirAbs) {
        Write-Warn "覆盖重训：清理旧输出目录 $OutputDirAbs（含旧 checkpoint）"
        Remove-Item $OutputDirAbs -Recurse -Force
    }
    Write-Ok "训练副本: $RunCfg （overwrite_output_dir=true，从头训练）"
} else {
    Write-Ok "训练副本: $RunCfg （Resume 模式，若有 checkpoint 将续训）"
}

# 4) 训练（复用 train_qwen_param.ps1 的框架调用方式，CWD=项目根）
if (-not $SkipTrain) {
    $swTrain = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Step "训练 LoRA（配置: $Config，数据集: $DatasetName）"
    Push-Location $ProjectRoot
    try {
        # 先确保数据集被拷到 CWD/data（与 train 脚本一致）
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
    Write-Ok "训练完成，adapter -> ./outputs/qwen15b-param-lora"
    $Timings["train"] = $swTrain.Elapsed
} else {
    Write-Warn "跳过训练（SkipTrain），直接合并已有 adapter。"
}

# 5) 合并（导出）到 ./models/Qwen2.5-1.5B-Instruct-sft
if (-not $NoMerge) {
    $swMerge = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Step "合并 LoRA -> ./models/Qwen2.5-1.5B-Instruct-sft"
    if (-not (Test-Path $ExpCfg)) {
        Write-Err "导出配置不存在: $ExpCfg"
        exit 1
    }
    # 校验 adapter 路径与训练输出目录一致
    $adapter = (& $VenvPy -c "import yaml; print(yaml.safe_load(open(r'$ExpCfg',encoding='utf-8'))['adapter_name_or_path'])").Trim()
    if (-not (Test-Path $adapter)) {
        Write-Err "未找到 adapter 目录: $adapter 。请先跑训练（或去掉 -SkipTrain）。"
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
    Write-Ok "合并完成 -> $sftDir （已可作为 --sft 直接评测）"
    $Timings["merge"] = $swMerge.Elapsed
} else {
    Write-Warn "跳过合并（NoMerge），仅产出 adapter。"
}

# 6) 可选：合并后自动评测
if ($Eval) {
    $swEval = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Step "合并后评测（base vs 新 sft）"
    $TestPath = Join-Path $ProjectRoot "data/llamafactory/shop_param_test.json"
    Push-Location $ProjectRoot
    try {
        & $VenvPy scripts/eval_sft_before_after.py `
            --base ./models/Qwen2.5-1.5B-Instruct `
            --sft  ./models/Qwen2.5-1.5B-Instruct-sft `
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
Write-Host "`n[DONE] 流程结束（dataset=$DatasetName, config=$Config）。临时训练副本: outputs/_train_merge_$Stamp.yaml" -ForegroundColor Green

# ── 计时汇总 ──────────────────────────────────────────────────────────
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
