$env:PYTHONIOENCODING = 'utf-8'
$hdr = @{"Authorization" = "Bearer local-fixed-key"}
$cid = "conv_redis_hist_001"

Write-Output "=== Round 1: 我的订单号111 (establish history) ==="
$b1 = @{message = "我的订单号111"; conversation_id = $cid; domain = "ecommerce"; stream = $false } | ConvertTo-Json -Compress
$r1 = Invoke-WebRequest -Uri "http://localhost/agent/api/v1/chatagent/agent/chat" -Method POST -ContentType "application/json" -Headers $hdr -Body $b1 -TimeoutSec 30
$resp1 = $r1.Content | ConvertFrom-Json
Write-Output ("R1 reply: " + $resp1.data.message)
($resp1.data.steps | Where-Object { $_.step_name -eq '参数抽取' } | ForEach-Object { Write-Output ("R1 参数抽取: " + ($_.output_data | ConvertTo-Json -Compress)) })

Write-Output "=== Round 2: 这个订单有物流信息吗 (should reuse 111 from history) ==="
$b2 = @{message = "这个订单有物流信息吗"; conversation_id = $cid; domain = "ecommerce"; stream = $false } | ConvertTo-Json -Compress
$r2 = Invoke-WebRequest -Uri "http://localhost/agent/api/v1/chatagent/agent/chat" -Method POST -ContentType "application/json" -Headers $hdr -Body $b2 -TimeoutSec 30
$resp2 = $r2.Content | ConvertFrom-Json
Write-Output ("R2 reply: " + $resp2.data.message)
$resp2.data.steps | ForEach-Object { Write-Output ("  step: {0} -> {1}" -f $_.step_name, ($_.output_data | ConvertTo-Json -Compress)) }

# 判定
$hit111 = $resp2.data.message -match "111"
$hit1111 = $resp2.data.message -match "1111"
if ($hit1111) { Write-Output "RESULT: FAIL (hallucinated 1111)" }
elseif ($hit111) { Write-Output "RESULT: PASS (reused 111 from history)" }
else { Write-Output "RESULT: UNKNOWN (no 111 in reply, check steps)" }
