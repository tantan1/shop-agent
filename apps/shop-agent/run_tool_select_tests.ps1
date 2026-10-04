# Run all tool-selection funnel / early-stop unit tests in one shot.
#
# Background: test_tool_select_stages.py's import chain triggers
#   src/modules/monitoring/langfuse_callback.py:17 load_dotenv()
# load_dotenv() walks up from CWD for the first .env. If a parent dir has a
# non-UTF-8 .env, it raises UnicodeDecodeError and breaks test collection.
# Fix: drop a valid .env (from .env.example) in CWD to shadow the bad file.
#
# Coverage (directly tied to 3-layer early-stop):
#   - tests/core/test_tool_select_pipeline.py : early-stop (P1/P2/P3 convergence + confident), scope passing, fallback, error/timeout
#   - tests/core/test_tool_select_stages.py   : 4-stage integration + emit_final_scope_as_plan (convergence early-stop isolated)
#
# Usage (from apps/shop-agent):
#   powershell -ExecutionPolicy Bypass -File run_tool_select_tests.ps1

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

# 1) Ensure a valid .env exists to shadow any non-UTF-8 parent .env
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "[setup] created .env from .env.example to shadow bad parent .env" -ForegroundColor Yellow
    } else {
        Write-Warning ".env.example missing; langfuse collection may still fail if a bad parent .env exists"
    }
}

# 2) Pick python (prefer venv)
$py = "python"
if (Test-Path "venv/Scripts/python.exe") { $py = "venv/Scripts/python.exe" }

# 3) Run the related tests
$tests = @(
    "tests/core/test_tool_select_pipeline.py"
    "tests/core/test_tool_select_stages.py"
    "tests/test_funnel_metrics_api.py"
)

Write-Host "=== Running tool-select funnel / early-stop tests ===" -ForegroundColor Cyan
& $py -m pytest $tests -v
exit $LASTEXITCODE
