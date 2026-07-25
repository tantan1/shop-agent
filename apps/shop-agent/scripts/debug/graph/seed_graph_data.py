"""
================================================================================
NebulaGraph 商品关系知识图谱 — 种子数据脚本
================================================================================

为图增强 RAG 评估插入测试数据，覆盖 graph_service.py 中所有 nGQL 查询模式：

    NQL_SAME_BRAND         Product → BELONGS_TO → Brand → BELONGS_TO (rev) → Product
    NQL_COMPATIBLE         Product → COMPATIBLE_WITH → Product
    NQL_SAME_CATEGORY_HOT  Product → IN_CATEGORY → Category → IN_CATEGORY (rev) → Product
    NQL_SUBSTITUTES        Product → SUBSTITUTE → Product
    NQL_BRAND_HOT_PRODUCTS Brand → BELONGS_TO (rev) → Product
    NQL_PRODUCT_BRAND      Product → BELONGS_TO → Brand
    NQL_BRAND_PARENT       Brand → PARENT_BRAND → Brand

图空间: shop_graph (vid_type=FIXED_STRING(32))

用法:
    # 启动 NebulaGraph 后运行（等待 graphd 就绪）
    python scripts/seed_graph_data.py

    # 如果 NebulaGraph 地址不是 localhost:9669
    python scripts/seed_graph_data.py --addr 192.168.1.100:9669

    # 清除后重新插入
    python scripts/seed_graph_data.py --reset

前置条件:
    docker compose up -d nebula-graphd
    # 等待 graphd healthcheck 通过（约 30 秒）
================================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# 连接配置
# =============================================================================

def _get_config():
    return {
        "addrs": os.getenv("NEBULA_GRAPH_ADDRS", "127.0.0.1:9669"),
        "user": os.getenv("NEBULA_USER", "root"),
        "password": os.getenv("NEBULA_PASSWORD", "nebula"),
        "space": os.getenv("NEBULA_SPACE", "shop_graph"),
    }


# =============================================================================
# 测试数据定义
# =============================================================================

# ── 商品（Product）: VID = 英文 ID，属性 name / id / sales ──
PRODUCTS = [
    # Apple 系
    {"vid": "IPHONE_15",         "name": "iPhone 15",              "id": "IPHONE_15",         "sales": 15000},
    {"vid": "IPHONE_15_PRO",     "name": "iPhone 15 Pro",          "id": "IPHONE_15_PRO",     "sales": 12000},
    {"vid": "IPHONE_15_PRO_MAX", "name": "iPhone 15 Pro Max",      "id": "IPHONE_15_PRO_MAX", "sales": 9000},
    {"vid": "IPHONE_SE_4",       "name": "iPhone SE 4",            "id": "IPHONE_SE_4",       "sales": 5000},
    {"vid": "AIRPODS_PRO2",      "name": "AirPods Pro 2",          "id": "AIRPODS_PRO2",      "sales": 20000},
    {"vid": "AIRPODS_4",         "name": "AirPods 4",              "id": "AIRPODS_4",         "sales": 14000},
    {"vid": "MACBOOK_AIR_M3",    "name": "MacBook Air M3",         "id": "MACBOOK_AIR_M3",    "sales": 7000},
    {"vid": "MACBOOK_PRO_M3",    "name": "MacBook Pro M3",         "id": "MACBOOK_PRO_M3",    "sales": 4500},
    {"vid": "APPLE_WATCH_S9",    "name": "Apple Watch S9",         "id": "APPLE_WATCH_S9",    "sales": 11000},
    {"vid": "APPLE_WATCH_ULTRA", "name": "Apple Watch Ultra 2",    "id": "APPLE_WATCH_ULTRA", "sales": 6000},
    {"vid": "IPAD_AIR_6",        "name": "iPad Air 6",             "id": "IPAD_AIR_6",        "sales": 8000},
    {"vid": "IPAD_PRO_M4",       "name": "iPad Pro M4",            "id": "IPAD_PRO_M4",       "sales": 3500},
    {"vid": "MAGSAFE_CHARGER",   "name": "MagSafe 无线充电器",      "id": "MAGSAFE_CHARGER",   "sales": 25000},
    {"vid": "USB_C_CABLE_2M",    "name": "USB-C 编织数据线 (2m)",   "id": "USB_C_CABLE_2M",    "sales": 35000},
    {"vid": "APPLE_PENCIL_USBC", "name": "Apple Pencil USB-C",     "id": "APPLE_PENCIL_USBC", "sales": 9000},

    # 华为系
    {"vid": "MATE_60_PRO",       "name": "华为 Mate 60 Pro",       "id": "MATE_60_PRO",       "sales": 10000},
    {"vid": "PURA_70_ULTRA",     "name": "华为 Pura 70 Ultra",     "id": "PURA_70_ULTRA",     "sales": 7500},
    {"vid": "FREEBUDS_PRO3",     "name": "华为 FreeBuds Pro 3",    "id": "FREEBUDS_PRO3",     "sales": 12000},
    {"vid": "MATEBOOK_X_PRO",    "name": "华为 MateBook X Pro",    "id": "MATEBOOK_X_PRO",    "sales": 4000},
    {"vid": "WATCH_GT_4",        "name": "华为 Watch GT 4",        "id": "WATCH_GT_4",        "sales": 8500},
    {"vid": "MATE_PAD_PRO",      "name": "华为 MatePad Pro 13.2",  "id": "MATE_PAD_PRO",      "sales": 5500},
    {"vid": "HUAWEI_PENCIL_3",   "name": "华为 M-Pencil 第三代",   "id": "HUAWEI_PENCIL_3",   "sales": 4500},

    # 三星系
    {"vid": "GALAXY_S24_ULTRA",  "name": "Galaxy S24 Ultra",       "id": "GALAXY_S24_ULTRA",  "sales": 9500},
    {"vid": "GALAXY_Z_FLIP6",    "name": "Galaxy Z Flip 6",        "id": "GALAXY_Z_FLIP6",    "sales": 5500},
    {"vid": "GALAXY_BUDS3_PRO",  "name": "Galaxy Buds 3 Pro",      "id": "GALAXY_BUDS3_PRO",  "sales": 10000},
    {"vid": "GALAXY_WATCH6",     "name": "Galaxy Watch 6",         "id": "GALAXY_WATCH6",     "sales": 7000},
    {"vid": "GALAXY_TAB_S9",     "name": "Galaxy Tab S9",          "id": "GALAXY_TAB_S9",     "sales": 3800},

    # 小米系
    {"vid": "XIAOMI_14_ULTRA",   "name": "小米 14 Ultra",          "id": "XIAOMI_14_ULTRA",   "sales": 11000},
    {"vid": "XIAOMI_14",         "name": "小米 14",                "id": "XIAOMI_14",         "sales": 18000},
    {"vid": "REDMI_BUDS_5_PRO",  "name": "Redmi Buds 5 Pro",       "id": "REDMI_BUDS_5_PRO",  "sales": 16000},
    {"vid": "XIAOMI_PAD_6S_PRO", "name": "小米 Pad 6S Pro",        "id": "XIAOMI_PAD_6S_PRO", "sales": 6500},
    {"vid": "XIAOMI_WATCH_S3",   "name": "小米 Watch S3",          "id": "XIAOMI_WATCH_S3",   "sales": 9000},
    {"vid": "XIAOMI_BAND_8_PRO", "name": "小米手环 8 Pro",         "id": "XIAOMI_BAND_8_PRO", "sales": 22000},
]

# ── 品牌（Brand）: VID = 中文品牌名，属性 name ──
BRANDS = [
    {"vid": "苹果",    "name": "Apple"},
    {"vid": "华为",    "name": "华为"},
    {"vid": "三星",    "name": "三星"},
    {"vid": "小米",    "name": "小米"},
    # 集团母公司
    {"vid": "苹果公司", "name": "Apple Inc."},
    {"vid": "三星集团", "name": "三星集团"},
    {"vid": "华为技术", "name": "华为技术有限公司"},
]

# ── 品类（Category）: VID = 中文品类名，属性 name ──
CATEGORIES = [
    {"vid": "手机",   "name": "智能手机"},
    {"vid": "耳机",   "name": "无线耳机"},
    {"vid": "笔记本", "name": "笔记本电脑"},
    {"vid": "手表",   "name": "智能手表"},
    {"vid": "平板",   "name": "平板电脑"},
    {"vid": "配件",   "name": "手机配件"},
    {"vid": "手环",   "name": "智能手环"},
]

# ── 边关系 ──

# Product → Brand (BELONGS_TO)
BELONGS_TO = [
    # Apple
    ("IPHONE_15", "苹果"), ("IPHONE_15_PRO", "苹果"), ("IPHONE_15_PRO_MAX", "苹果"),
    ("IPHONE_SE_4", "苹果"), ("AIRPODS_PRO2", "苹果"), ("AIRPODS_4", "苹果"),
    ("MACBOOK_AIR_M3", "苹果"), ("MACBOOK_PRO_M3", "苹果"),
    ("APPLE_WATCH_S9", "苹果"), ("APPLE_WATCH_ULTRA", "苹果"),
    ("IPAD_AIR_6", "苹果"), ("IPAD_PRO_M4", "苹果"),
    ("MAGSAFE_CHARGER", "苹果"), ("USB_C_CABLE_2M", "苹果"), ("APPLE_PENCIL_USBC", "苹果"),
    # 华为
    ("MATE_60_PRO", "华为"), ("PURA_70_ULTRA", "华为"), ("FREEBUDS_PRO3", "华为"),
    ("MATEBOOK_X_PRO", "华为"), ("WATCH_GT_4", "华为"), ("MATE_PAD_PRO", "华为"),
    ("HUAWEI_PENCIL_3", "华为"),
    # 三星
    ("GALAXY_S24_ULTRA", "三星"), ("GALAXY_Z_FLIP6", "三星"), ("GALAXY_BUDS3_PRO", "三星"),
    ("GALAXY_WATCH6", "三星"), ("GALAXY_TAB_S9", "三星"),
    # 小米
    ("XIAOMI_14_ULTRA", "小米"), ("XIAOMI_14", "小米"), ("REDMI_BUDS_5_PRO", "小米"),
    ("XIAOMI_PAD_6S_PRO", "小米"), ("XIAOMI_WATCH_S3", "小米"), ("XIAOMI_BAND_8_PRO", "小米"),
]

# Brand → Brand (PARENT_BRAND)
PARENT_BRAND = [
    ("苹果", "苹果公司"),
    ("三星", "三星集团"),
    ("华为", "华为技术"),
]

# Product → Category (IN_CATEGORY)
IN_CATEGORY = [
    # 手机
    ("IPHONE_15", "手机"), ("IPHONE_15_PRO", "手机"), ("IPHONE_15_PRO_MAX", "手机"),
    ("IPHONE_SE_4", "手机"), ("MATE_60_PRO", "手机"), ("PURA_70_ULTRA", "手机"),
    ("GALAXY_S24_ULTRA", "手机"), ("GALAXY_Z_FLIP6", "手机"),
    ("XIAOMI_14_ULTRA", "手机"), ("XIAOMI_14", "手机"),
    # 耳机
    ("AIRPODS_PRO2", "耳机"), ("AIRPODS_4", "耳机"),
    ("FREEBUDS_PRO3", "耳机"), ("GALAXY_BUDS3_PRO", "耳机"), ("REDMI_BUDS_5_PRO", "耳机"),
    # 笔记本
    ("MACBOOK_AIR_M3", "笔记本"), ("MACBOOK_PRO_M3", "笔记本"), ("MATEBOOK_X_PRO", "笔记本"),
    # 手表
    ("APPLE_WATCH_S9", "手表"), ("APPLE_WATCH_ULTRA", "手表"),
    ("WATCH_GT_4", "手表"), ("GALAXY_WATCH6", "手表"), ("XIAOMI_WATCH_S3", "手表"),
    # 平板
    ("IPAD_AIR_6", "平板"), ("IPAD_PRO_M4", "平板"),
    ("MATE_PAD_PRO", "平板"), ("GALAXY_TAB_S9", "平板"), ("XIAOMI_PAD_6S_PRO", "平板"),
    # 配件
    ("MAGSAFE_CHARGER", "配件"), ("USB_C_CABLE_2M", "配件"),
    ("APPLE_PENCIL_USBC", "配件"), ("HUAWEI_PENCIL_3", "配件"),
    # 手环
    ("XIAOMI_BAND_8_PRO", "手环"),
]

# Product → Product (COMPATIBLE_WITH) — 兼容配件
COMPATIBLE_WITH = [
    ("MAGSAFE_CHARGER", "IPHONE_15"), ("MAGSAFE_CHARGER", "IPHONE_15_PRO"),
    ("MAGSAFE_CHARGER", "IPHONE_15_PRO_MAX"), ("MAGSAFE_CHARGER", "IPHONE_SE_4"),
    ("USB_C_CABLE_2M", "IPHONE_15_PRO"), ("USB_C_CABLE_2M", "IPHONE_15_PRO_MAX"),
    ("USB_C_CABLE_2M", "MATE_60_PRO"), ("USB_C_CABLE_2M", "XIAOMI_14_ULTRA"),
    ("USB_C_CABLE_2M", "GALAXY_S24_ULTRA"),
    ("AIRPODS_PRO2", "IPHONE_15"), ("AIRPODS_PRO2", "IPHONE_15_PRO"),
    ("AIRPODS_PRO2", "IPHONE_15_PRO_MAX"),
    ("APPLE_PENCIL_USBC", "IPAD_AIR_6"), ("APPLE_PENCIL_USBC", "IPAD_PRO_M4"),
    ("HUAWEI_PENCIL_3", "MATE_PAD_PRO"),
    ("APPLE_WATCH_S9", "IPHONE_15"), ("APPLE_WATCH_S9", "IPHONE_15_PRO"),
    ("FREEBUDS_PRO3", "MATE_60_PRO"), ("FREEBUDS_PRO3", "PURA_70_ULTRA"),
]

# Product → Product (SUBSTITUTE) — 替代品类商品
SUBSTITUTE = [
    ("GALAXY_S24_ULTRA", "IPHONE_15_PRO_MAX"),  # 三星旗舰 vs 苹果旗舰
    ("XIAOMI_14_ULTRA", "IPHONE_15_PRO"),        # 小米旗舰替代
    ("MATE_60_PRO", "IPHONE_15_PRO"),
    ("GALAXY_BUDS3_PRO", "AIRPODS_PRO2"),        # 三星耳机替代
    ("REDMI_BUDS_5_PRO", "AIRPODS_4"),
    ("MATE_PAD_PRO", "IPAD_PRO_M4"),             # 平板替代
    ("GALAXY_TAB_S9", "IPAD_AIR_6"),
    ("GALAXY_WATCH6", "APPLE_WATCH_S9"),         # 手表替代
    ("XIAOMI_WATCH_S3", "APPLE_WATCH_S9"),
]


# =============================================================================
# 连接 & 执行
# =============================================================================

class NebulaSeeder:
    """NebulaGraph 种子数据写入器"""

    def __init__(self, addr: str, user: str, password: str, space: str):
        self._addr = addr
        self._user = user
        self._password = password
        self._space = space
        self._pool = None

    def _get_pool(self):
        if self._pool is not None:
            return self._pool

        from nebula3.gclient.net import ConnectionPool
        from nebula3.Config import Config

        host, port_str = self._addr.rsplit(":", 1) if ":" in self._addr else (self._addr, "9669")
        port = int(port_str)

        config = Config()
        config.max_connection_pool_size = 2
        config.timeout = 10000  # 建 Schema 需要更长超时

        pool = ConnectionPool()
        ok = pool.init([(host, port)], config)
        if not ok:
            raise RuntimeError(f"无法连接到 NebulaGraph: {self._addr}")
        self._pool = pool
        return pool

    def _execute(self, nql: str, check_result: bool = True) -> bool:
        """执行单条 nGQL，返回是否成功"""
        pool = self._get_pool()
        session = pool.get_session(self._user, self._password)
        try:
            # 每个新 session 必须设置图空间
            use_res = session.execute(f"USE {self._space}")
            if not use_res.is_succeeded():
                if check_result:
                    print(f"  [WARN] USE {self._space} failed: {use_res.error_msg()}")
                return False
            res = session.execute(nql)
            ok = res.is_succeeded()
            if not ok and check_result:
                print(f"  [WARN] {nql[:80]}... -> {res.error_msg()}")
            return ok
        finally:
            session.release()

    def run(self, reset: bool = False):
        """完整播种流程"""
        print(f"连接到 NebulaGraph: {self._addr}")
        pool = self._get_pool()

        # ── 1. 注册 Storage Host ──
        # 注意：从 Docker 网络内，storaged 地址为 nebula-storaged:9779
        # nebula-init 容器已负责注册，此处是二次保障
        print("\n[1/5] 检查 Storage Host...")
        self._execute('ADD HOSTS "nebula-storaged":9779', check_result=False)

        # ── 2. 创建图空间 ──
        print(f"\n[2/5] 创建图空间 {self._space}...")
        if reset:
            self._execute(f"DROP SPACE IF EXISTS {self._space}", check_result=False)
            time.sleep(5)  # 等待 DROP 异步完成

        create_sql = (
            f"CREATE SPACE IF NOT EXISTS {self._space}("
            "partition_num=1, replica_factor=1, vid_type=FIXED_STRING(32)"
            ")"
        )
        self._execute(create_sql)

        # NebulaGraph CREATE SPACE 是异步的，需要轮询等待就绪
        print("  等待图空间就绪...", end=" ", flush=True)
        max_wait = 30
        for i in range(max_wait):
            try:
                s = pool.get_session(self._user, self._password)
                res = s.execute(f"USE {self._space}")
                s.release()
                if res.is_succeeded():
                    print("OK")
                    break
            except Exception:
                pass
            time.sleep(1)
            if i % 5 == 4:
                print(f"({i+1}s)", end=" ", flush=True)
        else:
            print("TIMEOUT")
            raise RuntimeError(f"图空间 {self._space} 在 {max_wait}s 内未就绪")

        # ── 3. 创建标签（Tag） ──
        print("\n[3/5] 创建标签（Tags）...")
        session = pool.get_session(self._user, self._password)
        try:
            res = session.execute(f"USE {self._space}")
            if not res.is_succeeded():
                raise RuntimeError(f"USE {self._space} 失败: {res.error_msg()}")

            tags = [
                "CREATE TAG IF NOT EXISTS Product(name string, id string, sales int64)",
                "CREATE TAG IF NOT EXISTS Brand(name string)",
                "CREATE TAG IF NOT EXISTS Category(name string)",
            ]
            for tag_sql in tags:
                res = session.execute(tag_sql)
                if res.is_succeeded():
                    print(f"  OK: {tag_sql.split('(')[0].strip()}")
                else:
                    print(f"  FAIL: {res.error_msg()}")

            # ── 4. 创建边类型（Edge） ──
            print("\n[4/5] 创建边类型（Edges）...")
            edges = [
                "CREATE EDGE IF NOT EXISTS BELONGS_TO()",
                "CREATE EDGE IF NOT EXISTS COMPATIBLE_WITH()",
                "CREATE EDGE IF NOT EXISTS IN_CATEGORY()",
                "CREATE EDGE IF NOT EXISTS SUBSTITUTE()",
                "CREATE EDGE IF NOT EXISTS PARENT_BRAND()",
            ]
            for edge_sql in edges:
                res = session.execute(edge_sql)
                if res.is_succeeded():
                    print(f"  OK: {edge_sql.split('(')[0].strip()}")
                else:
                    print(f"  FAIL: {res.error_msg()}")

            # ── 5. 插入测试数据 ──
            print("\n[5/5] 插入测试数据...")

            # 插入 Product 顶点
            print(f"  插入 {len(PRODUCTS)} 个商品...")
            for p in PRODUCTS:
                nql = (
                    f'INSERT VERTEX Product(name, id, sales) '
                    f'VALUES "{p["vid"]}":("{p["name"]}", "{p["id"]}", {p["sales"]})'
                )
                self._execute(nql)

            # 插入 Brand 顶点
            print(f"  插入 {len(BRANDS)} 个品牌...")
            for b in BRANDS:
                nql = f'INSERT VERTEX Brand(name) VALUES "{b["vid"]}":("{b["name"]}")'
                self._execute(nql)

            # 插入 Category 顶点
            print(f"  插入 {len(CATEGORIES)} 个品类...")
            for c in CATEGORIES:
                nql = f'INSERT VERTEX Category(name) VALUES "{c["vid"]}":("{c["name"]}")'
                self._execute(nql)

            # 插入 BELONGS_TO 边
            print(f"  插入 {len(BELONGS_TO)} 条品牌归属关系...")
            for src, dst in BELONGS_TO:
                nql = f'INSERT EDGE BELONGS_TO() VALUES "{src}" -> "{dst}":()'
                self._execute(nql)

            # 插入 PARENT_BRAND 边
            print(f"  插入 {len(PARENT_BRAND)} 条集团归属关系...")
            for src, dst in PARENT_BRAND:
                nql = f'INSERT EDGE PARENT_BRAND() VALUES "{src}" -> "{dst}":()'
                self._execute(nql)

            # 插入 IN_CATEGORY 边
            print(f"  插入 {len(IN_CATEGORY)} 条品类归属关系...")
            for src, dst in IN_CATEGORY:
                nql = f'INSERT EDGE IN_CATEGORY() VALUES "{src}" -> "{dst}":()'
                self._execute(nql)

            # 插入 COMPATIBLE_WITH 边
            print(f"  插入 {len(COMPATIBLE_WITH)} 条兼容配件关系...")
            for src, dst in COMPATIBLE_WITH:
                nql = f'INSERT EDGE COMPATIBLE_WITH() VALUES "{src}" -> "{dst}":()'
                self._execute(nql)

            # 插入 SUBSTITUTE 边
            print(f"  插入 {len(SUBSTITUTE)} 条替代商品关系...")
            for src, dst in SUBSTITUTE:
                nql = f'INSERT EDGE SUBSTITUTE() VALUES "{src}" -> "{dst}":()'
                self._execute(nql)

        finally:
            session.release()

        print("\n✅ 种子数据写入完成！")
        print(f"\n验证命令（NebulaGraph Console）：")
        print(f"  USE {self._space};")
        print(f"  SHOW TAGS;")
        print(f"  SHOW EDGES;")
        print(f'  MATCH (v:Product) RETURN v.Product.name, v.Product.sales ORDER BY v.Product.sales DESC LIMIT 10;')

    def close(self):
        if self._pool:
            self._pool.close()
            self._pool = None


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NebulaGraph 种子数据脚本")
    parser.add_argument("--addr", default=None, help="graphd 地址 (默认从 .env NEBULA_GRAPH_ADDRS 读取)")
    parser.add_argument("--reset", action="store_true", help="先删除再重建图空间")
    args = parser.parse_args()

    cfg = _get_config()
    addr = args.addr or cfg["addrs"]

    seeder = NebulaSeeder(
        addr=addr,
        user=cfg["user"],
        password=cfg["password"],
        space=cfg["space"],
    )

    try:
        seeder.run(reset=args.reset)
    except ImportError:
        print("\n❌ 缺少 nebula3-python，请先安装：", file=sys.stderr)
        print("   pip install nebula3-python", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 播种失败: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        seeder.close()


if __name__ == "__main__":
    main()
