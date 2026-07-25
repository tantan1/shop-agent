"""Debug SAME_CATEGORY_HOT pipe query"""
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

vid = "MAGSAFE_CHARGER"

# Test 1: Full pipe as-is
nql1 = f'''GO FROM "{vid}" OVER IN_CATEGORY
YIELD id($$) AS cat
| GO FROM $-.cat OVER IN_CATEGORY REVERSELY
  YIELD $$.Product.name AS name, $$.Product.id AS id,
        $$.Product.sales AS sales
| YIELD name, id WHERE $$.Product.id != "{vid}"
| ORDER BY sales DESC | LIMIT 5'''
print(f"Test 1 (full pipe): {nql1[:120]}...")
r1 = s.execute(nql1)
print(f"  succeeded: {r1.is_succeeded()}, rows: {r1.row_size()}")
if not r1.is_succeeded():
    print(f"  error: {r1.error_msg()}")

# Test 2: Simplified pipe (no ORDER BY)
nql2 = f'''GO FROM "{vid}" OVER IN_CATEGORY
YIELD id($$) AS cat
| GO FROM $-.cat OVER IN_CATEGORY REVERSELY
  YIELD $$.Product.name AS name, $$.Product.id AS id
| YIELD name, id WHERE id != "{vid}"'''
print(f"\nTest 2 (simplified): {nql2[:120]}...")
r2 = s.execute(nql2)
print(f"  succeeded: {r2.is_succeeded()}, rows: {r2.row_size()}")
for row in r2.rows():
    print(f"  {row.values[0]}, {row.values[1]}")

# Test 3: Even simpler (no WHERE)
nql3 = f'''GO FROM "{vid}" OVER IN_CATEGORY
YIELD id($$) AS cat
| GO FROM $-.cat OVER IN_CATEGORY REVERSELY
  YIELD $$.Product.name AS name, $$.Product.id AS id'''
print(f"\nTest 3 (no WHERE): {nql3[:120]}...")
r3 = s.execute(nql3)
print(f"  succeeded: {r3.is_succeeded()}, rows: {r3.row_size()}")
for row in r3.rows():
    print(f"  {row.values[0]}, {row.values[1]}")

# Test 4: Test with IPHONE_15
vid2 = "IPHONE_15"
nql4 = f'''GO FROM "{vid2}" OVER IN_CATEGORY
YIELD id($$) AS cat
| GO FROM $-.cat OVER IN_CATEGORY REVERSELY
  YIELD $$.Product.name AS name, $$.Product.id AS id
| YIELD name, id WHERE id != "{vid2}"'''
print(f"\nTest 4 (IPHONE_15): {nql4[:120]}...")
r4 = s.execute(nql4)
print(f"  succeeded: {r4.is_succeeded()}, rows: {r4.row_size()}")
for row in r4.rows():
    print(f"  {row.values[0]}, {row.values[1]}")

s.release()
pool.close()
