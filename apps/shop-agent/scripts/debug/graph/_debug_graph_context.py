"""Debug: check actual graph_context text for failing queries."""
import asyncio
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
load_dotenv()
os.environ["NEBULA_GRAPH_ENABLED"] = "true"

from src.modules.chat.core.graph_service import NebulaGraphService

async def main():
    graph = NebulaGraphService.get_instance()
    
    for q in ["IPHONE_15有哪些兼容配件？", "MAGSAFE_CHARGER同品类还有什么热销配件？"]:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        context = await graph.query_and_build_context(q)
        print(f"Graph context ({len(context)} chars):")
        print(context if context else "(EMPTY)")
        print()

if __name__ == "__main__":
    asyncio.run(main())
