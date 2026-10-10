import os, glob, json

src = 'e:/workspace/shop-agent/monitoring/grafana/dashboards'
dst = 'e:/workspace/shop-agent/monitoring/grafana/provisioning/dashboards'
os.makedirs(dst, exist_ok=True)

# 1) validate the new perf dashboard
perf = os.path.join(dst, 'shop-agent-perf.json')
with open(perf, encoding='utf-8') as f:
    json.load(f)
print('perf JSON OK')

# 2) copy the other existing dashboards into the provisioning path,
#    fixing the stale "${DS_PROMETHEUS}" datasource placeholder -> "prometheus"
for f in glob.glob(os.path.join(src, '*.json')):
    name = os.path.basename(f)
    if name == 'shop-agent-overview.json':
        # legacy overview uses stale metric names; keep it out to avoid empty panels
        continue
    c = open(f, encoding='utf-8').read().replace('${DS_PROMETHEUS}', 'prometheus')
    # also drop any __inputs/__requires that reference the placeholder
    out = os.path.join(dst, name)
    open(out, 'w', encoding='utf-8').write(c)
    try:
        json.load(open(out, encoding='utf-8'))
        print('copied + valid:', name)
    except Exception as e:
        print('copied but INVALID:', name, e)

print('listing:', sorted(os.listdir(dst)))
