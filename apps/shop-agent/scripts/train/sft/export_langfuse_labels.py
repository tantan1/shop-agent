"""一键触发：从 Langfuse 拉取已标正确样本 → 导出 → 校验 → 回流 SFT。

设计 3：取代原 `GET /mlops/tasks/export`。数据源从 PostgreSQL mlops_review_tasks 换成 Langfuse trace/score。

用法：
    python scripts/train/sft/export_langfuse_labels.py
"""
import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from src.modules.monitoring.langfuse_mlops import export_and_train


def main():
    res = asyncio.run(export_and_train())
    print(res)


if __name__ == "__main__":
    main()
