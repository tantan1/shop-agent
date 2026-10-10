"""配置:从环境变量 / .env 读取,带合理默认。"""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


# Langfuse(只读)
LANGFUSE_PUBLIC_KEY = _env("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = _env("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = _env("LANGFUSE_HOST", "https://cloud.langfuse.com")

# 本地 SQLite 真值库(§2.3)
SQLITE_DB = _env("ANNOTATOR_SQLITE_DB", os.path.join(BASE_DIR, "annotations_idx.db"))

# 复检规模化(§5.3)
AUTO_PASS_MARGIN = float(_env("AUTO_PASS_MARGIN", "0.15"))
AUTO_SPOTCHECK_RATE = float(_env("AUTO_SPOTCHECK_RATE", "0.08"))
SPOTCHECK_PRECISION_FLOOR = float(_env("SPOTCHECK_PRECISION_FLOOR", "0.95"))

# 预标注(§2.4,可选)
PRELABEL_URL = _env("SHOP_AGENT_PRELABEL_URL")

# 工具清单(§2.1 tools()):从 shop-agent 的 /agent/tools 拉取名称+说明,供标注台展示候选与描述
TOOLS_URL = _env("SHOP_AGENT_TOOLS_URL")

# 服务
HOST = _env("HOST", "0.0.0.0")
PORT = int(_env("PORT", "8137"))
PAGE_SIZE = int(_env("PAGE_SIZE", "20"))

# Langfuse trace 名称(与 capture_tool_select 的 session_name 对齐)
TRACE_NAME = "tool_select_review"
