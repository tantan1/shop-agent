"""Compare graph vs no-graph results for the 4 graph-specific test cases."""
import json

def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

with_graph = load('results_ecommerce_with_graph.json')
no_graph = load('results_ecommerce_no_graph.json')

graph_queries = [
    'IPHONE_15有哪些兼容配件？',
    '苹果品牌有什么热销商品？',
    'AIRPODS_PRO2的替代品有哪些推荐？',
    'MAGSAFE_CHARGER同品类还有什么热销配件？',
]

print("=" * 80)
print("图增强消融对比 - 4 个图特化测试用例")
print("=" * 80)

for gq in graph_queries:
    wg = next((r for r in with_graph['full_results'] if r['question'] == gq), None)
    ng = next((r for r in no_graph['full_results'] if r['question'] == gq), None)
    
    print(f"\n{'─' * 80}")
    print(f"Q: {gq}")
    print(f"{'─' * 80}")
    
    wg_answer = wg['answer'][:200].replace('\n', ' ') if wg else 'N/A'
    ng_answer = ng['answer'][:200].replace('\n', ' ') if ng else 'N/A'
    
    print(f"  有图: {wg_answer}...")
    print(f"  无图: {ng_answer}...")
    print(f"  有图 duration: {wg.get('duration_ms', '?')}ms")
    print(f"  无图 duration: {ng.get('duration_ms', '?')}ms")

# Overall metrics comparison
print(f"\n{'=' * 80}")
print("RAGAS 指标对比")
print(f"{'=' * 80}")

wg_metrics = with_graph.get('ragas_metrics', {})
ng_metrics = no_graph.get('ragas_metrics', {})

for m in ['faithfulness', 'answer_relevancy', 'context_recall']:
    wg_v = wg_metrics.get(m, 0)
    ng_v = ng_metrics.get(m, 0)
    delta = wg_v - ng_v
    sign = '+' if delta > 0 else ''
    print(f"  {m:25s}: 有图={wg_v:.4f}  无图={ng_v:.4f}  ({sign}{delta:.4f})")

# Local metrics
print(f"\n本地启发式指标对比:")
wg_local = with_graph.get('local_metrics', {})
ng_local = no_graph.get('local_metrics', {})
for m in ['avg_answer_relevancy', 'avg_faithfulness', 'avg_answer_completeness', 'avg_latency_ms']:
    wg_v = wg_local.get(m, 0)
    ng_v = ng_local.get(m, 0)
    delta = wg_v - ng_v
    sign = '+' if delta > 0 else ''
    if 'latency' in m:
        print(f"  {m:30s}: 有图={wg_v:.0f}ms  无图={ng_v:.0f}ms  ({sign}{delta:.0f}ms)")
    else:
        print(f"  {m:30s}: 有图={wg_v:.4f}  无图={ng_v:.4f}  ({sign}{delta:.4f})")
