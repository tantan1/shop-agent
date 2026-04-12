from fastapi import APIRouter, Depends
from src.modules.auth.dependencies import verify_api_key
from src.shared.responses import success_response

router = APIRouter(prefix="/auth", tags=["认证管理"])


@router.post("/test-auth", response_model=dict, summary="测试认证接口")
async def test_authentication(
    _: None = Depends(verify_api_key)
):
    """
    测试认证功能的接口

    需要在Authorization头中提供有效的Bearer token
    格式：Authorization: Bearer ak_bigdata_internal_2024
    """
    return success_response(
        data={
            "message": "认证成功！",
            "test_result": "API密钥验证通过",
            "api_key_info": "使用固定的内部API密钥"
        },
        message="认证测试成功"
    )