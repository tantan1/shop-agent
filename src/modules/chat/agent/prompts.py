"""
通用Agent提示词模板系统
支持多领域模板配置
"""

from typing import Dict, Callable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# =============================================================================
# 提示词模板管理器
# =============================================================================

class PromptTemplateManager:
    """提示词模板管理器"""
    
    _templates: Dict[str, Dict[str, str]] = {}
    _template_funcs: Dict[str, Dict[str, Callable]] = {}
    
    @classmethod
    def register(cls, domain: str, templates: Dict[str, str]) -> None:
        """注册领域模板"""
        cls._templates[domain] = templates
    
    @classmethod
    def register_funcs(cls, domain: str, funcs: Dict[str, Callable]) -> None:
        """注册领域模板生成函数"""
        cls._template_funcs[domain] = funcs
    
    @classmethod
    def get(cls, domain: str, key: str) -> str:
        """获取指定领域和key的模板"""
        if domain in cls._templates and key in cls._templates[domain]:
            return cls._templates[domain][key]
        # 回退到通用模板
        if "general" in cls._templates and key in cls._templates["general"]:
            return cls._templates["general"][key]
        return ""
    
    @classmethod
    def get_prompt_obj(cls, domain: str, key: str, **kwargs) -> ChatPromptTemplate:
        """获取ChatPromptTemplate对象"""
        template = cls.get(domain, key)
        if not template:
            return None
        return ChatPromptTemplate.from_template(template)


# =============================================================================
# 医疗领域模板
# =============================================================================

MEDICAL_TEMPLATES = {
    # 步骤1：问题改写
    "medical_step1_rewrite": """将用户问题改写成适合向量检索的查询关键词。

规则：提取核心医疗关键词，简化口语为标准术语，保持原意。

示例：
输入："我最近头疼，还有点发烧，是不是感冒了？"
输出：头痛 发热 感冒 症状

直接输出关键词，不要解释。""",

    # 步骤2：安全审查
    "medical_step2_safety": """检查用户问题是否涉及医疗敏感内容。

敏感类别：医疗建议（用药/剂量/治疗）、诊断请求、处方相关、预后判断、急危重症（胸痛/呼吸困难/昏迷）、未成年人敏感问题、隐私信息（他人病历）。

判断标准：
- is_safe=true: 问题安全（挂号咨询、科室介绍、就医流程、健康科普等）
- is_safe=false, risk_level=low: 非紧急但需专业指导（用药注意事项、复查时间等）→ 引导至其他渠道
- is_safe=false, risk_level=medium: 需要就医但非立即（持续腹痛、反复发热等）→ 提示就医或咨询医生
- is_safe=false, risk_level=high: 紧急危险症状（胸痛、呼吸困难、意识改变、大量出血等）→ 提示立即就医或拨打120

示例：
输入：我嗓子疼，想挂耳鼻喉科
输出：{{"is_safe": true, "risk_level": "low", "risk_categories": [], "warning_message": ""}}

输入：我胸痛2天了，怎么回事
输出：{{"is_safe": false, "risk_level": "high", "risk_categories": ["急危重症"], "warning_message": "胸痛可能涉及严重心血管疾病，请立即就医或拨打120"}}

严格按JSON格式输出（不要输出其他任何内容）：
```json
{{"is_safe": true, "risk_level": "low", "risk_categories": [], "warning_message": ""}}
```""",

    # 步骤3：检索查询生成
    "medical_step3_query": """根据用户问题生成2-3个检索查询，从不同角度覆盖，使用医疗规范表达，简洁明了适合向量检索。

示例：
输入：心脏病患者可以做胃镜检查吗？
输出：
心脏病患者 胃镜检查 禁忌
无痛胃镜 心血管疾病 风险
胃镜检查前 心脏病 评估

每行一个查询，不要编号或解释。""",

    # 步骤4：回答生成
    "medical_step4_generate": """你是「医院智能客服助手」，为患者提供准确、安全的就医咨询。

# 核心原则
1. **以 RAG 为准**：<context>内的信息视为官方事实，直接引用；无相关内容明确告知"未查询到"
2. **禁止编造**：不基于通用知识臆测医院政策、科室、费用、医保等规则
3. **安全第一**：不诊断、不开药；急症（胸痛/呼吸困难等）提示立即就医或拨打120

# 禁止事项
- 不引用<context>以外的医院信息
- 不提供具体用药剂量、疗程建议
- 不对症状做诊断判断

# 回答结构
直接回答（引用RAG信息）→ 必要补充 → 引导挂号/就诊（如需）

# 当前时间
{current_time}

# 检索到的知识
<context>
{rag_context}
</context>

# 用户问题
{user_question}

# 安全审查结果
{safety_check_result}

# 重要提醒：
{safety_reminder}""",
}


# =============================================================================
# 电商领域模板
# =============================================================================

ECOMMERCE_TEMPLATES = {
    # 步骤1：需求分析
    "ecommerce_step1_analyze": """分析用户购物需求，提取关键信息。

规则：识别用户想要的产品类型、品牌偏好、价格区间、数量需求等。

示例：
输入："我想买一台玩游戏不卡顿的电脑，预算8000左右"
输出：
产品类型：电脑/笔记本
使用场景：游戏
预算范围：7000-9000
性能需求：高性能、不卡顿

请按上述格式输出。""",

    # 步骤2：合规检查
    "ecommerce_step2_compliance": """检查商品信息是否合规。

判断标准：
- 合规=true: 正常商品信息、咨询问题
- 合规=false, issue=违禁品: 涉及违规商品
- 合规=false, issue=敏感内容: 涉及敏感信息
- 合规=false, issue=广告: 明显的广告推广信息

严格按JSON格式输出：
```json
{{"compliant": true, "issue": ""}}
```""",

    # 步骤3：商品检索
    "ecommerce_step3_query": """根据用户需求生成商品检索关键词。

规则：提取产品品类、品牌、功能特点等关键词。

示例：
输入：想要一款续航久的无线耳机
输出：
无线耳机 续航长
蓝牙耳机 低延迟
降噪耳机 长待机

每行一个关键词，不要编号。""",

    # 步骤4：商品推荐
    "ecommerce_step4_generate": """你是专业电商客服助手，根据商品信息为用户推荐合适的产品。

# 推荐原则
1. **匹配需求**：优先推荐符合用户描述的产品
2. **突出卖点**：清晰说明产品的核心优势
3. **诚实客观**：不夸大宣传，如实告知产品特点

# 商品信息
<product_info>
{product_info}
</product_info>

# 用户需求
{user_question}

# 历史咨询
{chat_history}

请基于以上信息生成推荐回复。""",
}


# =============================================================================
# 客服领域模板
# =============================================================================

CUSTOMER_SERVICE_TEMPLATES = {
    # 步骤1：问题分类
    "service_step1_classify": """将用户问题分类到正确的类别。

分类标准：
- category=咨询: 产品/服务信息查询
- category=投诉: 用户表达不满或抱怨
- category=建议: 用户提出改进建议
- category=技术支持: 需要技术帮助
- category=其他: 不属于以上类别

严格按JSON格式输出：
```json
{{"category": "咨询", "keywords": ["关键词1", "关键词2"]}}
```""",

    # 步骤2：敏感检测
    "service_step2_sensitive": """检测用户问题是否包含敏感内容。

敏感内容：政治敏感、色情低俗、暴力恐怖、虚假信息、侵权内容等。

严格按JSON格式输出：
```json
{{"is_safe": true, "reason": ""}}
```""",

    # 步骤3：知识库检索
    "service_step3_query": """将用户问题转化为知识库检索关键词。

规则：简化口语表述，提取核心问题关键词。

示例：
输入：你们的退货政策是怎么样的
输出：退货政策 流程

每行一个关键词。""",

    # 步骤4：回复生成
    "service_step4_generate": """你是专业的客服代表，礼貌、专业地回复用户咨询。

# 回复原则
1. **礼貌友好**：使用敬语，体现专业素养
2. **清晰准确**：简洁明了地回答问题
3. **主动引导**：必要时引导用户采取下一步行动

# 知识库信息
<knowledge>
{knowledge_base}
</knowledge>

# 用户问题
{user_question}

# 问题分类
{category}

请生成专业、友好的回复。""",
}


# =============================================================================
# 通用领域模板
# =============================================================================

GENERAL_TEMPLATES = {
    # 步骤1：问题理解
    "general_step1_understand": """理解用户问题，提取关键信息。

规则：识别问题类型、核心需求、关键实体。

示例：
输入：今天天气怎么样？
输出：
问题类型：信息查询
核心需求：天气预报
关键实体：今天

简洁输出关键信息。""",

    # 步骤2：内容审查
    "general_step2_review": """检查内容是否合规安全。

判断标准：
- is_safe=true: 正常内容
- is_safe=false, reason=敏感: 包含敏感内容
- is_safe=false, reason=违规: 违反规定

严格按JSON格式输出：
```json
{{"is_safe": true, "reason": ""}}
```""",

    # 步骤3：信息检索
    "general_step3_query": """将问题转化为检索关键词。

每行一个关键词，不要编号。""",

    # 步骤4：回答生成
    "general_step4_generate": """请回答用户问题。

# 背景信息
{context}

# 用户问题
{user_question}

请生成准确、有用的回答。""",
}


# =============================================================================
# 安全回复模板（通用）
# =============================================================================

WARNING_TEMPLATES = {
    "default": """
⚠️ **安全提醒**

您的问题涉及以下内容：{risk_categories}

**建议：**
{warning_message}

如有疑问，请联系客服。
""",
    "medical_emergency": """
🚨 **紧急提醒**

您描述的症状可能涉及严重健康风险！

**建议：**
- 立即就医或拨打急救电话：120
- 前往最近的医院急诊

{additional_info}
""",
    "content_filtered": """
⚠️ **内容提示**

您的问题可能包含不适合讨论的内容。

**建议：**
请调整问题表述或咨询相关专业人士。
""",
}


GUIDANCE_TEMPLATES = {
    "default": """
您好！感谢您的咨询。

关于您的问题「{user_question}」，我无法直接提供具体帮助。

**建议您：**
1. 联系客服获得专业指导
2. 查看相关帮助文档
3. 尝试重新表述您的问题

**温馨提示：**
请描述具体需求，以便我更好地帮助您。
""",
    "no_result": """
您好！感谢您的咨询。

抱歉，暂时没有找到与「{user_question}」相关的直接答案。

**您可以尝试：**
1. 拨打客服热线：400-XXX-XXXX
2. 查看常见问题列表
3. 重新描述您的问题

我们会持续更新知识库，为您提供更好的服务。
""",
}


# =============================================================================
# 初始化模板管理器
# =============================================================================

PromptTemplateManager.register("medical", MEDICAL_TEMPLATES)
PromptTemplateManager.register("ecommerce", ECOMMERCE_TEMPLATES)
PromptTemplateManager.register("customer_service", CUSTOMER_SERVICE_TEMPLATES)
PromptTemplateManager.register("general", GENERAL_TEMPLATES)


# =============================================================================
# 兼容旧代码的导出
# =============================================================================

# 医疗领域提示词（保持向后兼容）
QUESTION_REWRITING_TEMPLATE = MEDICAL_TEMPLATES["medical_step1_rewrite"]
SAFETY_CHECK_TEMPLATE = MEDICAL_TEMPLATES["medical_step2_safety"]
RETRIEVAL_QUERY_TEMPLATE = MEDICAL_TEMPLATES["medical_step3_query"]
SYSTEM_PROMPT_TEMPLATE = MEDICAL_TEMPLATES["medical_step4_generate"]

QUESTION_REWRITING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QUESTION_REWRITING_TEMPLATE),
    ("human", "用户问题：{user_question}"),
])

SAFETY_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SAFETY_CHECK_TEMPLATE),
    ("human", "用户问题：{user_question}"),
])

RETRIEVAL_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RETRIEVAL_QUERY_TEMPLATE),
    ("human", "用户问题：{rewritten_question}"),
])

ANSWER_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_TEMPLATE),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "请基于以上信息生成回答。"),
])

# 安全回复模板
SAFETY_WARNING_TEMPLATE = WARNING_TEMPLATES["default"]
GENERAL_GUIDANCE_TEMPLATE = GUIDANCE_TEMPLATES["default"]

# 质量评估模板
ANSWER_QUALITY_EVALUATION_TEMPLATE = """评估AI助手的回答质量。

用户问题：{user_question}
AI回答：{ai_answer}
参考知识：{context}

评估标准：相关性、准确性、完整性、可操作性。

严格按JSON格式输出：
```json
{{"is_solved": true, "quality_score": 8, "reasons": ["回答相关", "引用准确"], "improvement_suggestion": ""}}
```"""

ANSWER_QUALITY_EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ANSWER_QUALITY_EVALUATION_TEMPLATE),
])
