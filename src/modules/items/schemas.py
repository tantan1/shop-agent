from typing import Optional, List
from pydantic import BaseModel, Field


class EnterpriseQueryRequest(BaseModel):
    """企业信息查询请求模型"""
    enterprise_name: Optional[str] = Field(None, description="企业名称", max_length=200)
    credit_code: Optional[str] = Field(None, description="统一社会信用代码", max_length=50)
    query_fields: List[str] = Field(
        default=["basic_info", "business_status", "risk_info"],
        description="查询字段列表，可选值：basic_info, business_status, risk_info"
    )
    region: Optional[str] = Field(None, description="地区筛选", max_length=100)
    industry: Optional[str] = Field(None, description="行业筛选", max_length=100)


class BasicInfo(BaseModel):
    """基本信息"""
    legal_person: Optional[str] = Field(None, description="法定代表人")
    register_capital: Optional[str] = Field(None, description="注册资本")
    establish_date: Optional[str] = Field(None, description="成立日期")
    business_scope: Optional[str] = Field(None, description="经营范围")


class BusinessStatus(BaseModel):
    """经营状况"""
    status: Optional[str] = Field(None, description="经营状态")
    annual_revenue: Optional[str] = Field(None, description="年营业额")
    employee_count: Optional[int] = Field(None, description="员工数量")


class RiskInfo(BaseModel):
    """风险信息"""
    risk_level: Optional[str] = Field(None, description="风险等级")
    lawsuit_count: Optional[int] = Field(None, description="诉讼数量")
    penalty_count: Optional[int] = Field(None, description="行政处罚数量")


class EnterpriseQueryResponse(BaseModel):
    """企业信息查询响应模型"""
    enterprise_name: str = Field(..., description="企业名称")
    credit_code: Optional[str] = Field(None, description="统一社会信用代码")
    basic_info: Optional[BasicInfo] = Field(None, description="基本信息")
    business_status: Optional[BusinessStatus] = Field(None, description="经营状况")
    risk_info: Optional[RiskInfo] = Field(None, description="风险信息")

    class Config:
        from_attributes = True
