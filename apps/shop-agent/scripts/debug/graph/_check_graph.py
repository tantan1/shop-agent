"""检查 NebulaGraph shop_graph 空间中的数据"""
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

config = Config()
config.max_connection_pool_size = 1
config.timeout = 5000
pool = ConnectionPool()
ok = pool.init([('127.0.0.1', 9669)], config)
print('connected:', ok)

session = pool.get_session('root', 'nebula')
session.execute('USE shop_graph')

def safe_str(val):
    """安全提取 NebulaGraph Value 字符串"""
    try:
        if hasattr(val, 'get_sVal'):
            return val.get_sVal().decode('utf-8')
    except Exception:
        pass
    try:
        return str(val)
    except Exception:
        return '?'

# Check target products
print('\n--- 检查目标产品 ---')
target_pids = ['IPHONE_15', 'AIRPODS_PRO2', 'MAGSAFE_CHARGER', 'GALAXY_S24_ULTRA', 'MATE_60_PRO']
for pid in target_pids:
    res = session.execute(f'FETCH PROP ON Product "{pid}" YIELD vertex AS v')
    if res.is_succeeded() and res.row_size() > 0:
        print(f'  {pid}: EXISTS (rows={res.row_size()})')
    else:
        print(f'  {pid}: NOT FOUND')

# Count
res = session.execute('MATCH (v:Product) RETURN count(v) AS c')
if res.is_succeeded():
    for r in res.rows():
        print(f'\nTotal products: {r.values[0].get_iVal()}')

# Test a graph query
print('\n--- 测试 IPHONE_15 同品牌查询 ---')
res = session.execute('''
    GO FROM "IPHONE_15" OVER BELONGS_TO
    YIELD $$.Brand.name AS brand
    | GO FROM $-.brand OVER BELONGS_TO REVERSELY
      WHERE $$.Product.id != "IPHONE_15"
      YIELD $$.Product.name AS name, $$.Product.id AS id
    | LIMIT 5
''')
if res.is_succeeded():
    print(f'  rows={res.row_size()}')
    for r in res.rows():
        name = safe_str(r.values[0])
        pid = safe_str(r.values[1])
        print(f'    {pid}: {name}')
else:
    print(f'  query failed: {res.error_msg()}')

# Test COMPATIBLE_WITH
print('\n--- 测试 IPHONE_15 兼容配件 ---')
res = session.execute('''
    GO FROM "IPHONE_15" OVER COMPATIBLE_WITH
    YIELD $$.Product.name AS name, $$.Product.id AS id
    | LIMIT 5
''')
if res.is_succeeded():
    print(f'  rows={res.row_size()}')
    for r in res.rows():
        name = safe_str(r.values[0])
        pid = safe_str(r.values[1])
        print(f'    {pid}: {name}')

session.release()
pool.close()
print('\nDone')
