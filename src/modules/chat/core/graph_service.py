"""
NebulaGraph 商品关系图查询服务
用于增强 RAG：品牌关系、兼容配件、同品类推荐、替代商品等结构化知识

依赖: nebula3-python (pip install nebula3-python)

架构:
    - 单例模式，全局复用一条连接池
    - 图查询失败静默降级，不阻塞主流程
    - 结果转为自然语言文本，注入 RAG context
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional, Dict, List, Any

from src.shared.logger import APILogger

logger = APILogger("graph_service")


# ── nGQL 查询模板 ──────────────────────────────────────────────────


# 查询商品的同品牌其他商品（二跳: Product→Brand→Product）
# {vid} 通过 .format() 填充为双引号字符串字面量；NebulaGraph 不支持参数化 VID
NQL_SAME_BRAND = '''
    GO FROM "{vid}" OVER BELONGS_TO
    YIELD $$.Brand.name AS brand
    | GO FROM $-.brand OVER BELONGS_TO REVERSELY
      WHERE $$.Product.id != "{vid}"
      YIELD $$.Product.name AS name, $$.Product.id AS id
    | LIMIT 5
'''

# 查询商品的兼容配件（一跳: Product→Product）
NQL_COMPATIBLE = '''
    GO FROM "{vid}" OVER COMPATIBLE_WITH
    YIELD $$.Product.name AS name, $$.Product.id AS id
    | LIMIT 5
'''

# 查询同品类热门商品（二跳: Product→Category←Product，按销量排序）
NQL_SAME_CATEGORY_HOT = '''
    GO FROM "{vid}" OVER IN_CATEGORY
    YIELD $$.Category.name AS cat
    | GO FROM $-.cat OVER IN_CATEGORY REVERSELY
      YIELD $$.Product.name AS name, $$.Product.id AS id,
            $$.Product.sales AS sales
    | YIELD name, id WHERE $$.Product.id != "{vid}"
    | ORDER BY sales DESC | LIMIT 5
'''

# 查询商品的替代品
NQL_SUBSTITUTES = '''
    GO FROM "{vid}" OVER SUBSTITUTE
    YIELD $$.Product.name AS name, $$.Product.id AS id
    | LIMIT 5
'''

# 查询某个品牌下的热销商品
NQL_BRAND_HOT_PRODUCTS = '''
    GO FROM "{brand_name}" OVER BELONGS_TO REVERSELY
    YIELD $$.Product.name AS name, $$.Product.id AS id, $$.Product.sales AS sales
    | ORDER BY $-.sales DESC | LIMIT 5
'''

# 查询商品所属品牌及品牌信息
NQL_PRODUCT_BRAND = '''
    GO FROM "{vid}" OVER BELONGS_TO
    YIELD $$.Brand.name AS name
'''

# 查询品牌的所有上级品牌（集团层级）
NQL_BRAND_PARENT = '''
    GO FROM "{brand_name}" OVER PARENT_BRAND
    YIELD $$.Brand.name AS name
'''


# ── 关系类型中文映射 ────────────────────────────────────────────────

RELATION_LABELS: Dict[str, str] = {
    "same_brand": "同品牌",
    "compatible_accessory": "兼容配件",
    "same_category_hot": "同品类热销",
    "substitute": "替代商品",
    "brand_products": "品牌热销",
    "parent_brand": "所属集团",
}


class NebulaGraphService:
    """NebulaGraph 商品关系图服务（单例）"""

    _instance: Optional["NebulaGraphService"] = None
    _pool = None
    _available: bool = False
    _init_attempted: bool = False

    def __new__(cls) -> "NebulaGraphService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "NebulaGraphService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if self._init_attempted:
            return
        self._init_attempted = True
        self._addrs = os.getenv("NEBULA_GRAPH_ADDRS", "127.0.0.1:9669")
        self._user = os.getenv("NEBULA_USER", "root")
        self._pwd = os.getenv("NEBULA_PASSWORD", "nebula")
        self._space = os.getenv("NEBULA_SPACE", "shop_graph")
        self._timeout = int(os.getenv("NEBULA_TIMEOUT", "3000"))
        self._pool_size = int(os.getenv("NEBULA_POOL_SIZE", "4"))
        self._pool = None
        self._available = False

    # =========================================================================
    # 连接管理
    # =========================================================================

    def _ensure_connected(self) -> bool:
        """懒连接 NebulaGraph（首次调用时初始化连接池）。

        连接失败不阻塞主流程，所有图查询返回空结果自然降级。
        """
        if self._available:
            return True

        # 快速路径：已尝试连接但失败，不重复尝试（避免日志刷屏）
        if hasattr(self, "_connect_failed") and self._connect_failed:
            return False

        try:
            from nebula3.gclient.net import ConnectionPool
            from nebula3.Config import Config

            # 解析地址列表（支持逗号分隔多地址）
            addrs = []
            for addr_str in self._addrs.split(","):
                addr_str = addr_str.strip()
                if ":" in addr_str:
                    host, port = addr_str.rsplit(":", 1)
                    addrs.append((host, int(port)))
                else:
                    addrs.append((addr_str, 9669))

            config = Config()
            config.max_connection_pool_size = self._pool_size
            config.timeout = self._timeout
            config.idle_time = 0  # 永不超时回收

            pool = ConnectionPool()
            ok = pool.init(addrs, config)
            if not ok:
                self._connect_failed = True
                logger.warning("NebulaGraph 连接池初始化失败")
                return False

            # 切换到目标图空间
            session = pool.get_session(self._user, self._pwd)
            try:
                res = session.execute(f"USE {self._space}")
                if not res.is_succeeded():
                    logger.warning(
                        f"NebulaGraph 空间切换失败: {self._space}，"
                        f"请先在 Console 中执行 CREATE SPACE 或检查空间名"
                    )
                    self._connect_failed = True
                    return False
            finally:
                session.release()

            self._patch_session_del()
            self._pool = pool
            self._available = True
            logger.info(f"NebulaGraph 连接成功 ({self._addrs})")
            return True

        except ImportError:
            self._connect_failed = True
            logger.warning("nebula3-python 未安装，图功能不可用。安装: pip install nebula3-python")
            return False
        except Exception as e:
            self._connect_failed = True
            logger.warning(f"NebulaGraph 连接失败: {str(e)[:200]}")
            return False

    # =========================================================================
    # nGQL 执行基方法
    # =========================================================================

    def _patch_session_del(self):
        """Monkey-patch Session.__del__，防止 GC 时 signout 触发 Thrift 断言失败。

        nebula3-python 的 Session.__del__ 在进程退出 / KeyboardInterrupt 时会尝试
        signout，此时底层 TCP 连接可能已断开，触发 TCompactProtocol 状态机断言。
        此方法将 __del__ 包装为安全版本，吞掉所有异常。
        """
        try:
            from nebula3.gclient.net.Session import BaseSession
            _original_del = BaseSession.__del__

            def _safe_del(session_self):
                try:
                    _original_del(session_self)
                except Exception:
                    pass  # 进程退出时 TCP 连接已断开，signout 必然失败

            BaseSession.__del__ = _safe_del
        except Exception:
            pass  # 非关键路径，patch 失败不影响功能

    def _execute_nql(self, nql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """同步执行 nGQL 查询，返回 dict 列表。

        Args:
            nql: nGQL 查询模板，使用 {key} 作为占位符（双引号字符串 VID）
            params: 参数字典，传入 None 或空 dict 表示无需填充

        注意：NebulaGraph 不支持参数化 VID（GO FROM $vid），因此用 .format()
              拼接字面量。params 中的值已在上游通过 regex 校验，无注入风险。
        """
        if not self._ensure_connected():
            return []

        session = None
        try:
            session = self._pool.get_session(self._user, self._pwd)
            # 每个新 session 必须设置图空间，否则查询在无空间上下文执行
            use_res = session.execute(f"USE {self._space}")
            if not use_res.is_succeeded():
                logger.warning(f"USE {self._space} 失败")
                return []
            if params:
                nql = nql.format(**params)
            # 拼接后的完整 nGQL（压缩空白，方便 Console 粘贴调试）
            logger.debug(f"执行 nGQL: {' '.join(nql.split())}")
            res = session.execute(nql)

            if not res.is_succeeded():
                logger.debug(f"nGQL 查询无结果: {nql[:80]}...")
                return []

            rows = []
            col_names = res.keys()  # list[str]，与 row.values 按位置对应
            for row in res.rows():
                row_dict = {}
                for i, col_name in enumerate(col_names):
                    val = row.values[i]  # row.values 是 list，非 dict
                    row_dict[col_name] = (
                        val.get_sVal().decode("utf-8") if hasattr(val, "get_sVal")
                        else val.get_iVal() if hasattr(val, "get_iVal")
                        else val.get_dVal() if hasattr(val, "get_dVal")
                        else str(val)
                    )
                rows.append(row_dict)
            return rows

        except Exception as e:
            error_msg = str(e)
            # 传输层损坏（'str' has no 'write' 等）→ 标记重连
            if "write" in error_msg and "object has no attribute" in error_msg:
                self._available = False
                logger.warning(f"nGQL 传输层异常，已标记重连: {error_msg[:120]}")
            else:
                logger.warning(f"nGQL 查询异常: {error_msg[:120]}")
            return []
        finally:
            if session:
                session.release()

    # =========================================================================
    # 商品关系查询
    # =========================================================================

    async def query_product_relations(
        self,
        product_id: str,
    ) -> Dict[str, Any]:
        """查询某个商品的所有关系网络（同品牌、配件、同品类热销、替代品）。

        注意：串行执行 4 条 nGQL，避免多线程并发导致 nebula3 Thrift 协议
        状态机冲突（assert self.state == CLEAR）。4 条本地图查询在毫秒级完成。
        """
        if not self._ensure_connected():
            return {"relations": []}

        loop = asyncio.get_event_loop()
        params = {"vid": product_id}

        # 串行执行，每条查询通过 run_in_executor 避免阻塞事件循环
        same_brand_rows = await loop.run_in_executor(
            None, self._execute_nql, NQL_SAME_BRAND, params
        )
        compatible_rows = await loop.run_in_executor(
            None, self._execute_nql, NQL_COMPATIBLE, params
        )
        same_category_rows = await loop.run_in_executor(
            None, self._execute_nql, NQL_SAME_CATEGORY_HOT, params
        )
        substitutes_rows = await loop.run_in_executor(
            None, self._execute_nql, NQL_SUBSTITUTES, params
        )

        results = [same_brand_rows, compatible_rows, same_category_rows, substitutes_rows]

        relations: List[Dict[str, str]] = []

        # 同品牌
        if isinstance(results[0], list):
            for row in results[0]:
                relations.append({
                    "type": "same_brand",
                    "product": row.get("name", ""),
                    "id": row.get("id", ""),
                })

        # 兼容配件
        if isinstance(results[1], list):
            for row in results[1]:
                relations.append({
                    "type": "compatible_accessory",
                    "product": row.get("name", ""),
                    "id": row.get("id", ""),
                })

        # 同品类热销
        if isinstance(results[2], list):
            for row in results[2]:
                relations.append({
                    "type": "same_category_hot",
                    "product": row.get("name", ""),
                    "id": row.get("id", ""),
                })

        # 替代品
        if isinstance(results[3], list):
            for row in results[3]:
                relations.append({
                    "type": "substitute",
                    "product": row.get("name", ""),
                    "id": row.get("id", ""),
                })

        return {"relations": relations}

    # =========================================================================
    # 品牌信息查询
    # =========================================================================

    async def query_brand_products(self, brand_name: str) -> List[Dict[str, str]]:
        """查询某品牌下的热销商品"""
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(
            None,
            self._execute_nql,
            NQL_BRAND_HOT_PRODUCTS,
            {"brand_name": brand_name},
        )
        return [
            {"type": "brand_products", "product": r.get("name", ""), "id": r.get("id", "")}
            for r in rows
        ]

    # =========================================================================
    # 文本构建（注入 RAG context）
    # =========================================================================

    def build_graph_context(self, relations: List[Dict[str, str]]) -> str:
        """将图查询结果序列化为可注入 LLM prompt 的自然语言文本。

        返回空字符串表示无图数据（prompt 中 graph_context 占位符保底为 ""）。
        """
        if not relations:
            return ""

        lines = ["【商品关系网络 — 来自知识图谱】"]

        # 按类型分组
        grouped: Dict[str, List[str]] = {}
        for r in relations:
            rtype = r.get("type", "unknown")
            label = RELATION_LABELS.get(rtype, rtype)
            if label not in grouped:
                grouped[label] = []
            vid = r.get("id", "")
            name = r.get("product", "")
            if vid:
                grouped[label].append(f"{name} (ID: {vid})" if name else f"ID: {vid}")
            elif name:
                grouped[label].append(name)

        for label, items in grouped.items():
            lines.append(f"- 【{label}】{', '.join(items)}")

        return "\n".join(lines)

    # =========================================================================
    # 高层入口：给 executor 用的统一接口
    # =========================================================================

    async def query_and_build_context(self, question: str) -> str:
        """根据用户问题智能查询图数据库，返回可直接注入的文本。

        当前策略：从问题中识别商品名/品牌名，执行对应的图查询。
        如果问题不含可识别的实体，返回空字符串。

        此方法是 executor step3 的唯一入口，上层无需关心内部 nGQL 细节。
        """
        if not self._ensure_connected():
            return ""

        relations: List[Dict[str, str]] = []

        # 1. 尝试提取商品 ID（字母数字下划线组合，3-30 字符）
        #    使用 re.ASCII 确保 \b 只识别英文单词边界，中文不会被误判为 \w
        import re
        product_ids = re.findall(r'\b([A-Za-z0-9_]{3,30})\b', question, re.ASCII)

        for pid in product_ids:
            # 排除常见非商品词
            if pid.lower() in ("http", "https", "www", "com", "cn", "html"):
                continue
            try:
                result = await self.query_product_relations(pid)
                relations.extend(result.get("relations", []))
            except Exception as e:
                logger.debug(f"商品关系查询失败 (pid={pid}): {str(e)[:80]}")

        # 2. 尝试提取品牌名（中文 2-6 字）
        brand_pattern = r'(?:品牌|牌子)[:：]?\s*([\u4e00-\u9fff]{2,6})'
        brand_match = re.search(brand_pattern, question)
        if brand_match:
            try:
                brand_products = await self.query_brand_products(brand_match.group(1))
                relations.extend(brand_products)
            except Exception as e:
                logger.debug(f"品牌查询失败: {str(e)[:80]}")

        return self.build_graph_context(relations)

    # =========================================================================
    # 资源释放
    # =========================================================================

    def close(self):
        """释放连接池引用，回收单例（应用关闭时调用）。

        注意：不调用 pool.close()，因为 nebula3-python 的 Session.__del__ 会在
        GC 时尝试 signout，若连接已关闭会导致 Thrift 协议断言失败（已知 bug）。
        连接资源由操作系统在进程退出时回收。
        """
        if self._pool:
            self._pool = None
        self._available = False
        self._connect_failed = False
        NebulaGraphService._instance = None
        logger.info("NebulaGraph 连接已释放")
