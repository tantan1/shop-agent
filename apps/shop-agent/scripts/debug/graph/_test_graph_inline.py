"""直接测试 graph_service.query_and_build_context"""
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv()

async def main():
    from src.modules.chat.core.graph_service import NebulaGraphService
    
    gs = NebulaGraphService.get_instance()
    
    questions = [
        "IPHONE_15有哪些兼容配件？",
        "苹果品牌有什么热销商品？",
        "AIRPODS_PRO2的替代品有哪些推荐？",
        "MAGSAFE_CHARGER同品类还有什么热销配件？",
    ]
    
    for q in questions:
        ctx = await gs.query_and_build_context(q)
        print(f"\nQ: {q}")
        if ctx:
            print(f"Graph context ({len(ctx)} chars):")
            print(ctx[:500])
        else:
            print("Graph context: (empty)")
    
    gs.close()

if __name__ == "__main__":
    asyncio.run(main())
