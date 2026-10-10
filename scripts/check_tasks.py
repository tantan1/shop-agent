import json
with open('benchmark/runtime_eval/tasks.json', encoding='utf-8') as f:
    tasks = json.load(f)
for t in tasks:
    if t['task'] >= 't08':
        print(f"{t['task']}: {t['query'][:100]}")
