"""Debug IN_CATEGORY for MAGSAFE_CHARGER"""
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

# Test 1: IN_CATEGORY from MAGSAFE_CHARGER
nql1 = 'GO FROM "MAGSAFE_CHARGER" OVER IN_CATEGORY YIELD id($$) AS cat'
print(f"Test 1: {nql1}")
r1 = s.execute(nql1)
print(f"  succeeded: {r1.is_succeeded()}, rows: {r1.row_size()}")
for row in r1.rows():
    print(f"  cat={row.values[0]}")

# Test 2: Check if IN_CATEGORY edge exists (no property approach)
nql2 = 'GO FROM "MAGSAFE_CHARGER" OVER IN_CATEGORY YIELD edge AS e'
print(f"\nTest 2: {nql2}")
r2 = s.execute(nql2)
print(f"  succeeded: {r2.is_succeeded()}, rows: {r2.row_size()}")

# Test 3: Check with LOOKUP
nql3 = 'LOOKUP ON Product WHERE Product.id == "MAGSAFE_CHARGER" YIELD properties(vertex)'
print(f"\nTest 3: {nql3}")
r3 = s.execute(nql3)
print(f"  succeeded: {r3.is_succeeded()}, rows: {r3.row_size()}")

# Test 4: Direct edge to Category
nql4 = 'FETCH PROP ON IN_CATEGORY "MAGSAFE_CHARGER"->"配件" YIELD properties(edge)'
print(f"\nTest 4: {nql4}")
r4 = s.execute(nql4)
print(f"  succeeded: {r4.is_succeeded()}, rows: {r4.row_size()}")

# Test 5: Try without pipe
nql5 = 'GO FROM "MAGSAFE_CHARGER" OVER IN_CATEGORY YIELD $$.Category.name AS cat_name'
print(f"\nTest 5: {nql5}")
r5 = s.execute(nql5)
print(f"  succeeded: {r5.is_succeeded()}, rows: {r5.row_size()}")
for row in r5.rows():
    print(f"  cat_name={row.values[0]}")

# Test 6: same for USB_C_CABLE_2M
nql6 = 'GO FROM "USB_C_CABLE_2M" OVER IN_CATEGORY YIELD id($$) AS cat'
print(f"\nTest 6 (USB_C_CABLE_2M): {nql6}")
r6 = s.execute(nql6)
print(f"  succeeded: {r6.is_succeeded()}, rows: {r6.row_size()}")

# Test 7: IN_CATEGORY REVERSELY from 配件
nql7 = 'GO FROM "配件" OVER IN_CATEGORY REVERSELY YIELD $$.Product.name AS name'
print(f"\nTest 7: {nql7}")
r7 = s.execute(nql7)
print(f"  succeeded: {r7.is_succeeded()}, rows: {r7.row_size()}")
for row in r7.rows():
    print(f"  name={row.values[0]}")

s.release()
pool.close()
