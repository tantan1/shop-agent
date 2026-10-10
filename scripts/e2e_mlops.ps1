$k = "ak_bigdata_internal_2024"
$h = @{ "X-API-Key" = $k; "Content-Type" = "application/json" }
$base = "http://localhost:8000/agent/api/v1/mlops"

# 0) GPU 预检（非致命）
Write-Host "=== GPU precheck ==="
try { docker exec mlops-trainer python -c "import torch; print('trainer cuda_available=', torch.cuda.is_available())" 2>$null } catch {}

# 1) 创建 -> 认领 -> 标注正确 -> 确认训练（smoke，真实 GPU 训练）
$r = Invoke-RestMethod -Uri "$base/tasks" -Method POST -Headers $h -ContentType "application/json" -Body '{"sample_id":"e2e-gpu-001","metric_snapshot":{"tool":"query-order","correct":true},"reason":"real gpu training e2e"}'
$tid = $r.data.id
Write-Host "CREATE -> id=$tid status=$($r.data.status)"
if (-not $tid) { Write-Error "CREATE failed: empty task id. raw=$(ConvertTo-Json $r -Depth 3)"; exit 1 }
$r2 = Invoke-RestMethod -Uri "$base/tasks/$tid/claim" -Method POST -Headers $h -ContentType "application/json" -Body '{}'
Write-Host "CLAIM -> $($r2.data.status)"
$r3 = Invoke-RestMethod -Uri "$base/tasks/$tid/label" -Method POST -Headers $h -ContentType "application/json" -Body '{"label":"correct","annotation":"tool selection correct"}'
Write-Host "LABEL(correct) -> $($r3.data.status)"
$r4 = Invoke-RestMethod -Uri "$base/tasks/$tid/confirm-training" -Method POST -Headers $h -ContentType "application/json" -Body '{"smoke":true,"dataset_ref":"data/llamafactory/shop_unified_v1.json"}'
Write-Host "CONFIRM_TRAINING -> $($r4.data.status)"

# 2) 轮询训练
$dl = (Get-Date).AddMinutes(30)
do {
    Start-Sleep -Seconds 10
    $st = (Invoke-RestMethod -Uri "$base/tasks/$tid" -Method GET -Headers $h).data.status
    Write-Host "training status=$st"
} while ($st -notin @("evaluated","training_failed") -and (Get-Date) -lt $dl)

if ($st -eq "training_failed") { Write-Error "TRAINING FAILED"; exit 1 }
Write-Host ">>> TRAINING OK -> $st"

# 3) 评测
$r5 = Invoke-RestMethod -Uri "$base/tasks/$tid/eval" -Method POST -Headers $h -ContentType "application/json" -Body '{"thresholds":{},"max_samples":32}'
Write-Host "EVAL -> $($r5.data.status)"
$dl = (Get-Date).AddMinutes(30)
do {
    Start-Sleep -Seconds 10
    $t = Invoke-RestMethod -Uri "$base/tasks/$tid" -Method GET -Headers $h
    $st = $t.data.status
    Write-Host "eval status=$st eval_pass=$($t.data.eval_pass)"
} while ($null -eq $t.data.eval_result -and (Get-Date) -lt $dl)

if ($null -eq $t.data.eval_result) { Write-Error "EVAL did not produce result"; exit 1 }
Write-Host ">>> EVAL done: $(ConvertTo-Json $t.data.eval_result -Compress)"

# 4) 发布（仅当 eval_pass；smoke 训练质量通常不达阈值，流水线到达 evaluated 即视为跑通）
if ($t.data.eval_pass -eq $true) {
    $r6 = Invoke-RestMethod -Uri "$base/tasks/$tid/publish" -Method POST -Headers $h -ContentType "application/json" -Body '{}'
    Write-Host "PUBLISH -> $($r6.data.status)"
} else {
    Write-Host "SKIP PUBLISH: eval_pass=$($t.data.eval_pass)（smoke 训练质量未达阈值，属预期；流水线已跑通 training->evaluated）"
}
Start-Sleep -Seconds 3
$final = Invoke-RestMethod -Uri "$base/tasks/$tid" -Method GET -Headers $h
Write-Host "FINAL -> $(ConvertTo-Json $final.data -Compress)"
# 端到端判定：真实训练 + 真实评测已完整跑通（到达 evaluated 且有 eval_result）即视为 PASS
$ok = ($final.data.status -in @("evaluated","published")) -and ($null -ne $final.data.eval_result)
if (-not $ok) { Write-Error "E2E 未跑通（status=$($final.data.status) eval_result=$($null -ne $final.data.eval_result))"; exit 1 }
Write-Host "=== E2E REAL TRAINING+Eval PASS (status=$($final.data.status), eval_pass=$($final.data.eval_pass)) ==="
$tid | Out-File -Encoding ascii e2e_taskid.txt
