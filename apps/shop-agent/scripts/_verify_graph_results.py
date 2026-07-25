"""Verify graph-specific test case answers in evaluation results."""
import json

with open("results_ecommerce_with_graph.json", "r", encoding="utf-8") as f:
    data = json.load(f)

graph_queries = ["IPHONE_15兼容配件", "苹果品牌热销", "AIRPODS_PRO2替代品", "MAGSAFE_CHARGER同品类"]

for item in data:
    q = item.get("question", "")
    for gq in graph_queries:
        if gq[:8] in q:
            print(f"{'='*60}")
            print(f"Q: {item['question'][:80]}")
            print(f"A: {item.get('answer', 'N/A')[:400]}")
            print(f"Graph used: {item.get('graph_used', 'N/A')}")
            print(f"Documents found: {item.get('documents_found', 'N/A')}")
            print()
            break
