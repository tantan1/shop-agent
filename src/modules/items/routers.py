from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.shared.database import get_db
from src.modules.auth.dependencies import verify_api_key
from src.modules.items.services import EnterpriseService
from src.modules.items.schemas import EnterpriseQueryRequest
from src.shared.responses import success_response

router = APIRouter(prefix="/reports", tags=["数据报表"])


async def get_enterprise_service(db: AsyncSession = Depends(get_db)) -> EnterpriseService:
    """获取企业服务依赖"""
    return EnterpriseService(db)


@router.post("/enterprise-query", response_model=dict, summary="企业信息查询")
async def query_enterprise_info(
    request: EnterpriseQueryRequest,
    _: None = Depends(verify_api_key),
    enterprise_service: EnterpriseService = Depends(get_enterprise_service)
):
    """
    企业信息查询接口

    **请求参数：**
    - **enterprise_name**: 企业名称（可选，支持模糊查询）
    - **credit_code**: 统一社会信用代码（可选，精确匹配）
    - **query_fields**: 查询字段列表，可选值：
      - `basic_info`: 基本信息（法人、注册资本、成立日期、经营范围）
      - `business_status`: 经营状况（经营状态、年营业额、员工数量）
      - `risk_info`: 风险信息（风险等级、诉讼数量、行政处罚数量）
    - **region**: 地区筛选（可选，支持模糊查询）
    - **industry**: 行业筛选（可选，支持模糊查询）

    **注意事项：**
    - 企业名称和统一社会信用代码至少提供一个
    - 需要在Authorization头中提供有效的Bearer token
    - 格式：Authorization: Bearer ak_bigdata_internal_2024

    **示例请求：**
    ```json
    {
        "enterprise_name": "阿里巴巴",
        "query_fields": ["basic_info", "business_status", "risk_info"],
        "region": "浙江省",
        "industry": "互联网"
    }
    ```
    """
    # 查询企业信息
    enterprises = await enterprise_service.query_enterprise_info(request)

    return success_response(
        data=[enterprise.model_dump() for enterprise in enterprises],
        message=f"查询成功，共找到 {len(enterprises)} 条企业信息"
    )
