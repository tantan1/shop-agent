"""调试品牌查询"""
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
    print(f'  nGQL: {nql[:150]}')
    try:
        res = session.execute(nql)
        print(f'  ok={res.is_succeeded()} rows={res.row_size()} err={res.error_msg()}')
        if res.is_succeeded() and res.row_size() > 0:
            cols = res.keys()
            for r in res.rows():
                vals = []
                for v in r.values:
                    try: vals.append(v.get_sVal().decode())
                    except: vals.append(str(v))
                print(f'  {dict(zip(cols, vals))}')
    except Exception as e:
        print(f'  ERROR: {e}')
    session.release()

# Test 1: Simple GO FROM brand reversed
run('GO FROM "苹果" OVER BELONGS_TO REVERSELY YIELD $$.Product.name AS name, $$.Product.id AS id LIMIT 5', 'basic reversed')

# Test 2: With ORDER BY
run('GO FROM "苹果" OVER BELONGS_TO REVERSELY YIELD $$.Product.name AS name, $$.Product.id AS id, $$.Product.sales AS sales | ORDER BY $-.sales DESC | LIMIT 5', 'ordered reversed')

# Test 3: Without pipe
run('GO FROM "苹果" OVER BELONGS_TO REVERSELY YIELD $$.Product.name AS name, $$.Product.id AS id, $$.Product.sales AS sales ORDER BY $$.Product.sales DESC LIMIT 5', 'direct order')

pool.close()
