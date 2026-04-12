from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from src.shared.database import Base


class Enterprise(Base):
    """企业信息模型"""
    __tablename__ = "enterprises"

    id = Column(Integer, primary_key=True, index=True)
    enterprise_name = Column(String(200), nullable=False, index=True, comment="企业名称")
    credit_code = Column(String(50), unique=True, index=True, nullable=True, comment="统一社会信用代码")
    legal_person = Column(String(100), nullable=True, comment="法定代表人")
    register_capital = Column(String(100), nullable=True, comment="注册资本")
    establish_date = Column(String(20), nullable=True, comment="成立日期")
    business_scope = Column(Text, nullable=True, comment="经营范围")

    # 经营状况
    status = Column(String(20), nullable=True, comment="经营状态")
    annual_revenue = Column(String(50), nullable=True, comment="年营业额")
    employee_count = Column(Integer, nullable=True, comment="员工数量")

    # 风险信息
    risk_level = Column(String(20), nullable=True, comment="风险等级")
    lawsuit_count = Column(Integer, default=0, comment="诉讼数量")
    penalty_count = Column(Integer, default=0, comment="行政处罚数量")

    # 地区和行业
    region = Column(String(100), nullable=True, comment="所在地区")
    industry = Column(String(100), nullable=True, comment="所属行业")

    # 系统字段
    created_at = Column(DateTime, default=datetime.now, nullable=False, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False, comment="更新时间")

    def __repr__(self):
        return f"<Enterprise(id={self.id}, name='{self.enterprise_name}', credit_code='{self.credit_code}')>"