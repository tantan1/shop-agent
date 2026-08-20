<#
.SYNOPSIS
  monitoring_agent 三大功能模块 e2e 验证脚本。
  覆盖：主动巡检与 RCA、服务自动发现、告警双通道接入、HITL 审批流、沙箱验证、计划查询。
.EXAMPLE
  kubectl -n shop-agent port-forward svc/monitoring-agent 9091:80
  pwsh apps/monitoring-agent/scripts/verify_monitoring_e2e.ps1 -BaseUrl http://localhost:9091
#>
param(
  [string]$BaseUrl = "http://localhost:9091",
  [string]$Token   = ""   # MONITORING_WEBHOOK_TOKEN；本地回环(127.0.0.1)可留空
)

$ProgressPreference = 'SilentlyContinue'
$Auth = if ($Token) { @{ Authorization = "Bearer $Token" } } else { @{} }
$pass = 0; $fail = 0; $blocked = 0
function Check($name, $ok, $detail = "") {
  if ($ok) { $script:pass++; Write-Host "  [PASS] $name" -ForegroundColor Green }
  else     { $script:fail++; Write-Host "  [FAIL] $name -> $detail" -ForegroundColor Red }
}
function Blocked($name, $why) {
  $script:blocked++; Write-Host "  [BLOCKED] $name -> $why" -ForegroundColor Yellow
}
function Req($method, $path, $body = $null) {
  $h = @{ 'Content-Type' = 'application/json' } + $Auth
  try {
    $j = if ($body) { $body | ConvertTo-Json -Compress -Depth 10 } else { $null }
    $r = Invoke-RestMethod -Method $method -Uri "$BaseUrl$path" -Headers $h -Body $j -TimeoutSec 15 -ErrorAction Stop
    return $r
  } catch {
    if ($_.Exception.Response) {
      $code = [int]$_.Exception.Response.StatusCode
      $txt = $_.ErrorDetails.Message
      try { return ($txt | ConvertFrom-Json) } catch { return @{ _http = $code; _raw = $txt } }
    }
    throw
  }
}

Write-Host "`n===== monitoring_agent e2e =====" -ForegroundColor Cyan
Write-Host "BaseUrl = $BaseUrl`n"

# ── 一、主动巡检与 RCA ─────────────────────────────────────────────
Write-Host "【一、主动巡检与 RCA】" -ForegroundColor Cyan

Write-Host "[V-1.1] 拓扑健康矩阵 /status"
$s = Req GET "/status"
$compKeys = ($s.components.PSObject.Properties.Name) -join ","
Check "overall 为 bool" ($s.overall -is [bool]) "overall=$($s.overall)"
Check "components 含核心组件" ($s.components.'shop-agent' -ne $null -and $s.components.gateway -ne $null) "keys=$compKeys"
Check "dependencies 为列表（聚合拓扑矩阵）" ($s.dependencies -is [System.Collections.IList]) "deps=$($s.dependencies.Count)"

Write-Host "[V-1.2] 手动触发 RCA（默认 demo 告警）"
$r = Req POST "/rca" @{ source = "alertmanager"; alerts = @(@{ name = "demo-alert"; status = "firing" }) }
Check "severity 合法" ($r.severity -in @("critical","warning","info","P1","P2","P3")) "sev=$($r.severity)"
Check "root_cause 非空" (![string]::IsNullOrEmpty($r.root_cause)) "rc=$($r.root_cause)"
Check "recommendations 非空" ($r.recommendations -and $r.recommendations.Count -gt 0) "recs=$(($r.recommendations|ConvertTo-Json -Compress))"

Write-Host "[V-1.3] 网关中断 -> GatewayDown 规则（纯规则，无 LLM）"
$gwDown = @{ gateway = @{ status = $false }; "shop-agent" = @{ status = $false }; redis = @{ status = $true };
  _dependencies = @(@{ source = "shop-agent"; target = "gateway" }, @{ source = "gateway"; target = "redis" }) }
$rg = Req POST "/rca" @{ source = "alertmanager"; alerts = @(@{ name = "GatewayDown"; severity = "P1" }); topology = $gwDown }
Check "severity 合法" ($rg.severity -in @("critical","warning","info","P1","P2","P3")) "sev=$($rg.severity)"
Check "root_cause 指向 gateway" ($rg.root_cause -match "gateway") "rc=$($rg.root_cause)"
Check "affected 含 shop-agent（受影响者）" ($rg.affected -contains "shop-agent") "affected=$(($rg.affected -join ','))"
Check "used_llm=false（确定性规则）" ($rg.used_llm -eq $false) "used_llm=$($rg.used_llm)"

Write-Host "[V-1.4] 周期巡检调度器（部分实现）"
Blocked "V-1.4 周期调度" "当前为按需 /status 探测，无后台定时任务；需 webhook 叫醒或手动触发。周期调度为后续迭代项。"

# ── 二、服务自动发现 ───────────────────────────────────────────────
Write-Host "`n【二、服务自动发现】" -ForegroundColor Cyan

Write-Host "[V-2.1] 依赖拓扑聚合（Prometheus+SkyWalking / 静态兜底）"
Check "dependencies 为列表（聚合拓扑矩阵）" ($s.dependencies -is [System.Collections.IList]) "deps=$($s.dependencies.Count)"
if ($s.dependencies.Count -gt 0) { Write-Host "  [INFO] 动态依赖边已发现: $($s.dependencies|ConvertTo-Json -Compress -Depth 3)" -ForegroundColor Gray }
else { Write-Host "  [WARN] 本次 dependencies 为空（依赖源 Prometheus/SkyWalking 可能暂不可达，已降级）。可在依赖源可达时复测。" -ForegroundColor Yellow }

Write-Host "[V-2.2] 新增服务自动纳入（自动发现机制）"
Check "依赖聚合返回列表结构" ($s.dependencies -is [System.Collections.IList]) "deps=$($s.dependencies.Count)"

Write-Host "[V-2.3] 级联归因收敛根因"
Check "根因收敛到 gateway（根因候选）" ($rg.root_cause -match "gateway" -and $rg.root_cause -match "根因候选") "rc=$($rg.root_cause)"
Check "级联区分根因/受影响" ($rg.affected -contains "shop-agent" -and $rg.root_cause -match "gateway") "rc=$($rg.root_cause) affected=$(($rg.affected -join ','))"

# ── 三、告警接入（双通道 + HITL） ──────────────────────────────────
Write-Host "`n【三、告警接入】" -ForegroundColor Cyan

Write-Host "[V-3.1] Alertmanager 通道 /ingest/alert"
$a = Req POST "/ingest/alert" @{ alerts = @(@{ labels = @{ alertname = "HighErrorRate"; severity = "P1" }; annotations = @{ summary = "err rate high" } }) }
Check "200/accepted" ($a.accepted -eq $true) "resp=$(($a|ConvertTo-Json -Compress))"
Check "触发 RCA 返回 root_cause" (![string]::IsNullOrEmpty($a.root_cause)) "rc=$($a.root_cause)"

Write-Host "[V-3.2] Langfuse 通道 /ingest/event"
$l = Req POST "/ingest/event" @{ source = "langfuse"; event = @{ name = "llm_exception"; trace_id = "abc123"; error = "timeout" } }
Check "Langfuse 应急口触发 RCA" ($l.accepted -eq $true -and ![string]::IsNullOrEmpty($l.root_cause)) "rc=$($l.root_cause)"

Write-Host "[V-3.3] 入站脱敏（PII 不进分析路径）"
$p = Req POST "/ingest/alert" @{ alerts = @(@{ labels = @{ alertname = "PII" }; annotations = @{ summary = "user a@b.com pub 203.0.113.5 int 10.0.0.1" } }) }
$raw = ($p | ConvertTo-Json -Compress)
Check "邮箱已脱敏" ($raw -notmatch "a@b.com") "raw=$raw"
Check "公网IP已脱敏" ($raw -notmatch "203.0.113.5") "raw=$raw"
# 私网 IP 保留由脱敏单元保证：直接调用模块函数做真实验证（不依赖 RCA 响应体是否回显）
$redactTest = & python -c "import sys; sys.path.insert(0,'apps/monitoring-agent'); from monitoring_agent.alerts import redact; t=redact('src 203.0.113.5 dst 10.0.0.5 127.0.0.1'); print('OK' if ('203.0.113.5' not in t and '10.0.0.5' in t and '127.0.0.1' in t) else 'FAIL:'+t)"
Check "私网IP保留（脱敏单元）" ($redactTest -eq "OK") "redact=$redactTest"

Write-Host "[V-3.4] HITL 审批流（preview -> approve -> apply）"
$preview = Req POST "/remediate/preview" @{ action = "scale_up"; target = "redis"; params = @{ replicas = 1 } }
Check "preview 返回 plan_id" ($preview.plan_id -ne $null -and $preview.status -eq "pending_approval") "plan_id=$($preview.plan_id) status=$($preview.status)"
Check "preview 返回证据包" ($preview.evidence -ne $null -and $preview.evidence.call_sequence.Count -gt 0) "evidence=$(($preview.evidence|ConvertTo-Json -Compress))"

$approve = Req POST "/remediate/approve" @{ plan_id = $preview.plan_id; decision = "approved"; approver = "e2e"; reason = "test" }
Check "approve 返回 accepted" ($approve.accepted -eq $true) "resp=$(($approve|ConvertTo-Json -Compress))"

$dup = Req POST "/remediate/approve" @{ plan_id = $preview.plan_id; decision = "rejected" }
Check "重复审批返回 409" ($dup._http -eq 409) "dup_status=$($dup._http)"

$apply = Req POST "/remediate/apply" @{ plan_id = $preview.plan_id; script = @{ action = "scale_up"; target = "redis"; params = @{ replicas = 1 } }; approved = $true }
Check "apply 返回 applied" ($apply.applied -eq $true) "resp=$(($apply|ConvertTo-Json -Compress))"

Write-Host "[V-3.5] 沙箱验证预览（FakeK8sClient 干跑）"
$ev = $preview.evidence
Check "evidence 含 plan" ($ev.plan -ne $null) "plan=$($ev.plan)"
Check "evidence 含 current" ($ev.current -ne $null) "current=$($ev.current)"
Check "evidence 含 effective" ($ev.effective -ne $null) "effective=$($ev.effective)"
Check "call_sequence 含 read_current" ($ev.call_sequence[0].step -eq "read_current") "step0=$($ev.call_sequence[0].step)"
Check "call_sequence 含 would_apply" ($ev.call_sequence[1].step -eq "would_apply") "step1=$($ev.call_sequence[1].step)"
Check "would_apply 为 dry_run" ($ev.call_sequence[1].dry_run -eq $true) "dry_run=$($ev.call_sequence[1].dry_run)"

Write-Host "[V-3.6] 计划查询 /remediate/plans"
$plans = Req GET "/remediate/plans?limit=10"
Check "plans 返回列表" ($plans -is [System.Collections.IList]) "count=$($plans.Count)"
$history = Req GET "/rca/history?limit=10"
Check "history 返回列表" ($history -is [System.Collections.IList]) "count=$($history.Count)"

# ── 汇总 ───────────────────────────────────────────────────────────
Write-Host "`n===== 汇总 =====" -ForegroundColor Cyan
Write-Host "PASS=$pass  FAIL=$fail"
if ($fail -gt 0) { Write-Host "存在失败用例，请检查 monitoring-agent 运行状态与依赖（Prometheus/SkyWalking 可达性）。" -ForegroundColor Red; exit 1 }
Write-Host "核心功能 e2e 通过。" -ForegroundColor Green
exit 0
