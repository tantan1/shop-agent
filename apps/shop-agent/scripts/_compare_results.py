"""对比无图/有图评估结果"""
import json

with open('results_ecommerce_with_graph.json', 'r', encoding='utf-8') as f:
    wg = json.load(f)

graph_qs = [
    'IPHONE_15有哪些兼容配件？',
    'AIRPODS_PRO2的替代品有哪些推荐？',
    '苹果品牌有什么热销商品？',
    'MAGSAFE_CHARGER同品类还有什么热销配件？',
]

print("=" * 70)
print("  图增强测试 — 有图(Graph ON) 结果")
print("=" * 70)

for r in wg.get('full_results', []):
    q = r.get('question', '')
    if q in graph_qs:
        ans = r.get('answer', '')
        ctxs = r.get('contexts', []) or []
        print(f"\nQ: {q}")
        print(f"A ({len(ans)}字): {ans[:400]}")
        print(f"Contexts: {len(ctxs)} docs")
        if ctxs:
            for i, c in enumerate(ctxs[:2]):
                print(f"  [{i}] {c[:120]}...")
        steps = r.get('steps', []) or []
        for s in steps:
            sn = s.get('step_name', '?')
            od = str(s.get('output_data', ''))[:200]
            has_graph = 'graph_context' in od or 'product_relations' in od
            print(f"  [{sn}] graph_hit={has_graph} | {od[:150]}")
        if r.get('error'):
            print(f"  ERROR: {r['error'][:100]}")

# Overall RAGAS comparison
print("\n" + "=" * 70)
print("  全量 RAGAS 指标 (with graph, 8 cases)")
print("=" * 70)
for k, v in wg.get('ragas_metrics', {}).items():
    print(f"  {k}: {v:.4f}")
print("\n  本地指标:")
lm = wg.get('local_metrics', {})
for k, v in lm.items():
    if not k.startswith('avg_step'):
        print(f"  {k}: {v}")
