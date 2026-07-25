#!/usr/bin/env python3
"""测试聊天服务功能"""

import os
import sys
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from dotenv import load_dotenv

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

async def test_chat_agent():
    """测试聊天代理服务"""
    from src.modules.chat.services import ChatAgentService
    from src.modules.chat.schemas import ChatQueryRequest, InsertDocumentRequest
    
    print("=== 测试ChatAgentService ===")
    
    # 创建一个模拟的数据库会话
    class MockAsyncSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
    
    db = MockAsyncSession()
    service = ChatAgentService(db)
    
    try:
        # 测试初始化
        await service._initialize()
        print("✓ 服务初始化成功")
        
        # 测试文档插入
        insert_request = InsertDocumentRequest(
            document="这是一个测试文档，用于验证 ChatAgentService 的文档插入功能。",
            metadata={"source": "test"}
        )
        insert_result = await service.insert_documents(insert_request)
        print(f"✓ 文档插入成功: {insert_result}")
        
        # 测试聊天查询
        chat_request = ChatQueryRequest(
            message="你好，这是一个测试问题"
        )
        chat_result = await service.chat(chat_request)
        print(f"✓ 聊天查询成功: {chat_result.message[:100]}...")
        
        print("\n=== 测试完成 ===")
        print("✅ 所有功能测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        # 检查具体错误
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 检查必要的依赖
    try:
        import langchain_openai
        import pymilvus
        import dashscope
        print("[OK] 所有必要依赖已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        exit(1)
    
    # 检查API密钥
    if not os.getenv("CHAT_TONGYI_API_KEY"):
        print("⚠️  注意: CHAT_TONGYI_API_KEY 未设置，某些功能可能无法正常工作")
        print("设置方法: export CHAT_TONGYI_API_KEY=your_api_key")
    
    # 运行测试
    asyncio.run(test_chat_agent())