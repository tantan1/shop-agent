"""调试 NebulaGraph PIPE 查询"""
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

config = Config()
config.max_connection_pool_size = 1
config.timeout = 5000
pool = ConnectionPool()
pool.init([('127.0.0.1', 9669)], config)

def run(nql, label):
    session = pool.get_session('root', 'nebula')
    session.execute('USE shop_graph')
    print(f'\n--- {label} ---')
    print(f'  nGQL: {nql[:120]}')
    res = session.execute(nql)
    ok = res.is_succeeded()
    rows = res.row_size()
    print(f'  result: ok={ok} rows={rows} err={res.error_msg()}')
    if ok and rows > 0:
        cols = res.keys()
        for j, r in enumerate(res.rows()):
            vals = []
            for i, v in enumerate(r.values):
                try: vals.append(v.get_sVal().decode())
                except: vals.append(str(v))
            print(f'  [{j}] {dict(zip(cols, vals))}')
    session.release()

# Test 1: Basic first hop
run('GO FROM "IPHONE_15" OVER BELONGS_TO YIELD $$.Brand.name AS brand', 'hop1: Brand.name')

# Test 2: First hop with id($$) 
run('GO FROM "IPHONE_15" OVER BELONGS_TO YIELD id($$) AS brand_vid', 'hop1: brand VID')

# Test 3: Full pipe (original NQL_SAME_BRAND)
run('''
    GO FROM "IPHONE_15" OVER BELONGS_TO
    YIELD $$.Brand.name AS brand
    | GO FROM $-.brand OVER BELONGS_TO REVERSELY
      WHERE $$.Product.id != "IPHONE_15"
      YIELD $$.Product.name AS name, $$.Product.id AS id
    | LIMIT 5
''', 'full pipe: same brand')

# Test 4: Pipe with id($$) as brand
run('''
    GO FROM "IPHONE_15" OVER BELONGS_TO
    YIELD id($$) AS brand
    | GO FROM $-.brand OVER BELONGS_TO REVERSELY
      WHERE $$.Product.id != "IPHONE_15"
      YIELD $$.Product.name AS name, $$.Product.id AS id
    | LIMIT 5
''', 'pipe: id($$) as brand')

# Test 5: Without pipe
run('''
    GO FROM "苹果" OVER BELONGS_TO REVERSELY
    WHERE $$.Product.id != "IPHONE_15"
    YIELD $$.Product.name AS name, $$.Product.id AS id
    | LIMIT 5
''', 'direct: from 苹果 reversed')

pool.close()
