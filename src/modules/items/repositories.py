from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from src.modules.items.models import Enterprise
from src.shared.exceptions import DatabaseException
from src.shared.logger import APILogger

logger = APILogger("enterprise_repository")


class EnterpriseRepository:
    """企业信息数据访问层"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def query_enterprises(
        self,
        enterprise_name: Optional[str] = None,
        credit_code: Optional[str] = None,
        region: Optional[str] = None,
        industry: Optional[str] = None
    ) -> List[Enterprise]:
        """查询企业信息"""
        try:
            # 构建查询条件
            conditions = []

            if enterprise_name:
                conditions.append(Enterprise.enterprise_name.like(f"%{enterprise_name}%"))

            if credit_code:
                conditions.append(Enterprise.credit_code == credit_code)

            if region:
                conditions.append(Enterprise.region.like(f"%{region}%"))

            if industry:
                conditions.append(Enterprise.industry.like(f"%{industry}%"))

            # 构建查询语句
            if conditions:
                stmt = select(Enterprise).where(and_(*conditions))
            else:
                stmt = select(Enterprise)

            # 执行查询
            result = await self.db.execute(stmt)
            enterprises = result.scalars().all()

            logger.log_database_operation(
                operation="select",
                table="enterprises",
                success=True,
                count=len(enterprises),
                enterprise_name=enterprise_name,
                credit_code=credit_code[:8] + "****" if credit_code else None
            )

            return list(enterprises)

        except Exception as e:
            logger.log_database_operation(
                operation="select",
                table="enterprises",
                success=False,
                error=str(e)
            )
            raise DatabaseException("查询企业信息失败", str(e))

    async def get_by_name_or_code(
        self,
        enterprise_name: Optional[str] = None,
        credit_code: Optional[str] = None
    ) -> Optional[Enterprise]:
        """根据企业名称或信用代码获取企业信息"""
        try:
            conditions = []

            if enterprise_name:
                conditions.append(Enterprise.enterprise_name == enterprise_name)

            if credit_code:
                conditions.append(Enterprise.credit_code == credit_code)

            if not conditions:
                return None

            # 使用OR条件查询
            stmt = select(Enterprise).where(or_(*conditions))
            result = await self.db.execute(stmt)
            enterprise = result.scalar_one_or_none()

            logger.log_database_operation(
                operation="select",
                table="enterprises",
                success=True,
                found=enterprise is not None,
                enterprise_name=enterprise_name,
                credit_code=credit_code[:8] + "****" if credit_code else None
            )

            return enterprise

        except Exception as e:
            logger.log_database_operation(
                operation="select",
                table="enterprises",
                success=False,
                error=str(e)
            )
            raise DatabaseException("查询企业信息失败", str(e))