import urllib.request
import json

# 检查 error/timeout outcome
url = 'http://localhost:19090/api/v1/query?query=shop_agent_tool_select_stage_total{outcome=~"error|timeout"}'
with urllib.request.urlopen(url) as resp:
    data = json.loads(resp.read().decode())
    print("=== error/timeout outcomes ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))

# 检查所有 outcome
url2 = 'http://localhost:19090/api/v1/query?query=shop_agent_tool_select_stage_total'
with urllib.request.urlopen(url2) as resp:
    data2 = json.loads(resp.read().decode())
    print("\n=== all outcomes ===")
    for r in data2.get('data', {}).get('result', []):
        print(f"  stage={r['metric']['stage']}, outcome={r['metric']['outcome']}, value={r['value'][1]}")