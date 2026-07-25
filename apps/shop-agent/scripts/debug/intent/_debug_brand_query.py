"""Debug brand query for 苹果品牌热销"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()

from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

config = Config()
config.max_connection_pool_size = 2
pool = ConnectionPool()
pool.init([('127.0.0.1', 9669)], config)

s = pool.get_session('root', 'nebula')
s.execute('USE shop_graph')

# Test 1: Direct REVERSELY brand query
nql = 'GO FROM "苹果" OVER BELONGS_TO REVERSELY YIELD $$.Product.name AS name, $$.Product.id AS id, $$.Product.sales AS sales | ORDER BY $-.sales DESC | LIMIT 5'
print("Query:", nql)
r = s.execute(nql)
print("succeeded:", r.is_succeeded())
print("error:", r.error_msg() if hasattr(r, 'error_msg') else 'N/A')
if r.is_succeeded():
    print(f"rows: {r.row_size()}")
    for row in r.rows():
        print(f"  name={row.values[0]}, id={row.values[1]}, sales={row.values[2]}")

# Test 2: simpler version without ORDER BY
nql2 = 'GO FROM "苹果" OVER BELONGS_TO REVERSELY YIELD $$.Product.name AS name, $$.Product.id AS id | LIMIT 5'
print("\nQuery 2:", nql2)
r2 = s.execute(nql2)
print("succeeded:", r2.is_succeeded())
if r2.is_succeeded():
    print(f"rows: {r2.row_size()}")
    for row in r2.rows():
        print(f"  name={row.values[0]}, id={row.values[1]}")

# Test 3: check if "苹果" vertex exists
nql3 = 'FETCH PROP ON Brand "苹果" YIELD properties(vertex)'
print("\nQuery 3:", nql3)
r3 = s.execute(nql3)
print("succeeded:", r3.is_succeeded())
if r3.is_succeeded():
    print(f"rows: {r3.row_size()}")
    for row in r3.rows():
        print(row.values)

# Test 4: check BELONGS_TO edges 
nql4 = 'GO FROM "苹果" OVER BELONGS_TO YIELD edge AS e'
print("\nQuery 4 (forward BELONGS_TO):", nql4)
r4 = s.execute(nql4)
print("succeeded:", r4.is_succeeded())
print(f"rows: {r4.row_size()}")

# Test 5: REVERSELY without pipe ORDER BY
nql5 = 'GO FROM "苹果" OVER BELONGS_TO REVERSELY YIELD $$.Product.name AS name | LIMIT 3'
print("\nQuery 5:", nql5)
r5 = s.execute(nql5)
print("succeeded:", r5.is_succeeded())
if r5.is_succeeded():
    print(f"rows: {r5.row_size()}")
    for row in r5.rows():
        print(f"  {row.values[0]}")

s.release()
pool.close()
