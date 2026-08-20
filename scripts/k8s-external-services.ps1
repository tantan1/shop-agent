#!/usr/bin/env pwsh
Write-Host "`n=== K8s External Services ===" -ForegroundColor Cyan
Write-Host ("=" * 80)

$nodeIP = kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' 2>$null
if (-not $nodeIP) { $nodeIP = "localhost" }

Write-Host "Node Internal IP: $nodeIP`n"

$svcs = kubectl get svc --all-namespaces -o json 2>$null | ConvertFrom-Json

$found = $false
foreach ($svc in $svcs.items) {
    $ns = $svc.metadata.namespace
    $name = $svc.metadata.name
    $type = $svc.spec.type
    $ports = $svc.spec.ports

    if ($type -eq "NodePort") {
        foreach ($p in $ports) {
            $np = $p.nodePort
            $port = $p.port
            $proto = $p.protocol
            Write-Host ("{0}/{1}" -f $ns, $name) -ForegroundColor Yellow -NoNewline
            Write-Host ("  NodePort  {0}:{1}  ({2})" -f $nodeIP, $np, $proto)
            $found = $true
        }
    }
    elseif ($type -eq "LoadBalancer") {
        $ings = @()
        if ($svc.status.loadBalancer.ingress) {
            $ings = $svc.status.loadBalancer.ingress
        }
        foreach ($ing in $ings) {
            $extIP = if ($ing.ip) { $ing.ip } elseif ($ing.hostname) { $ing.hostname } else { "?" }
            foreach ($p in $ports) {
                Write-Host ("{0}/{1}" -f $ns, $name) -ForegroundColor Yellow -NoNewline
                Write-Host ("  LoadBalancer  {0}:{1}  ({2})" -f $extIP, $p.port, $p.protocol)
                $found = $true
            }
        }
        if ($ings.Count -eq 0) {
            foreach ($p in $ports) {
                Write-Host ("{0}/{1}" -f $ns, $name) -ForegroundColor Yellow -NoNewline
                Write-Host ("  LoadBalancer  <pending>  {0}:{1}" -f $p.port, $p.protocol)
                $found = $true
            }
        }
    }
    elseif ($svc.spec.externalIPs) {
        foreach ($extIP in $svc.spec.externalIPs) {
            foreach ($p in $ports) {
                Write-Host ("{0}/{1}" -f $ns, $name) -ForegroundColor Yellow -NoNewline
                Write-Host ("  ExternalIP  {0}:{1}  ({2})" -f $extIP, $p.port, $p.protocol)
                $found = $true
            }
        }
    }
}

if (-not $found) {
    Write-Host "No externally accessible services found" -ForegroundColor Red
}

Write-Host ""
