$ErrorActionPreference = 'Continue'
$log = "outputs/watch_eval_holdout_M8.log"
function Ts { return (Get-Date -Format "yyyy-MM-dd HH:mm:ss") }
Add-Content -Encoding utf8 $log "[$(Ts)] holdout M=8 eval start (no retrain; unseen queries)"

# —— base @ M=8 on holdout ——
Add-Content -Encoding utf8 $log "[$(Ts)] >>> eval base @ M=8 (holdout)"
& .\venv_cuda\Scripts\python.exe scripts/eval_tool_select_sft.py --model ./models/Qwen2.5-1.5B-Instruct --data ./data/lscale_M_holdout.json --M 8 --distract-seed 20260728 --runs 3 --out outputs/eval_M8_holdout_base.json 2>&1 | Out-File -Encoding utf8 -Append $log

# —— SFT-M8 @ M=8 on holdout ——
Add-Content -Encoding utf8 $log "[$(Ts)] >>> eval SFT-M8 @ M=8 (holdout)"
& .\venv_cuda\Scripts\python.exe scripts/eval_tool_select_sft.py --model ./models/Qwen2.5-1.5B-Instruct-tool-select-M8 --data ./data/lscale_M_holdout.json --M 8 --distract-seed 20260728 --runs 3 --out outputs/eval_M8_holdout_sft.json 2>&1 | Out-File -Encoding utf8 -Append $log

Add-Content -Encoding utf8 $log "[$(Ts)] holdout eval DONE. results: outputs/eval_M8_holdout_base.json / outputs/eval_M8_holdout_sft.json"
Write-Host "[holdout] M=8 holdout eval 完成，详见 $log"
