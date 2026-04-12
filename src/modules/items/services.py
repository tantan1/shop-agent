from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from src.modules.items.schemas import (
    EnterpriseQueryRequest,
    EnterpriseQueryResponse,
    BasicInfo,
    BusinessStatus,
    RiskInfo
)
from src.shared.exceptions import NotFoundException, ValidationException
from src.shared.logger import APILogger

logger = APILogger("enterprise_service")


class EnterpriseService:
    """企业信息查询服务"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def query_enterprise_info(self, request: EnterpriseQueryRequest) -> List[EnterpriseQueryResponse]:
        """查询企业信息"""
        try:
            responses = []

            return responses

        except (NotFoundException, ValidationException):
            # 重新抛出业务异常
            raise
        except Exception as e:
            logger.log_business_event(
                "企业信息查询",
                success=False,
                error=str(e),
                enterprise_name=request.enterprise_name,
                credit_code=request.credit_code[:8] + "****" if request.credit_code else None
            )
            raise ValidationException("查询企业信息失败", str(e))
