"""
HNSW 参数消融实验 — efSearch recall-latency 曲线
================================================================================

在 Milvus HNSW 索引上，遍历不同 efSearch 值，测量 recall@K 与延迟的关系。

测试维度：
    - efSearch ∈ [8, 16, 24, 32, 50, 64, 100, 128]
    - recall@5, recall@10
    - p50 延迟, p99 延迟, avg 延迟

Ground Truth 获取方式：
    从 Milvus 拉取全量向量，对每条 query 做暴力余弦相似度计算，
    排序取 top-K 作为"真实答案"。

⚠️ 必须在 Milvus 上运行，不能用 Faiss 替代（Segment 架构 + 混合检索
   的 efSearch 代价曲线与 Faiss 单图搜索不同）。

前置条件：
    1. Milvus 运行中（docker-compose up standalone etcd minio）
    2. 集合中包含已嵌入的文档数据
    3. 项目依赖已安装：pip install pymilvus numpy

用法：
    python scripts/benchmark_hnsw_recall.py                          # 使用现有集合
    python scripts/benchmark_hnsw_recall.py --auto-data              # 自动生成测试数据并插入
    python scripts/benchmark_hnsw_recall.py --collection my_coll     # 指定集合名
    python scripts/benchmark_hnsw_recall.py --output results.json    # 输出 JSON
================================================================================
"""

import os
import sys
import json
import time
import statistics
import argparse
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

# =============================================================================
# 常量
# =============================================================================
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
SEARCH_EF_VALUES = [8, 16, 24, 32, 50, 64, 100, 128]
DEFAULT_TOP_K = 10
SEARCH_REPEATS = 10  # 每个 efSearch 值重复搜索次数（用于稳定性）

MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))
DEFAULT_COLLECTION = os.environ.get("MILVUS_COLLECTION", "chat_embeddings")

# =============================================================================
# 测试查询集合（30 条真实风格 + 20 条边缘 case）
# 与知识库内容匹配：冰箱、净水器、电视、公司制度
# =============================================================================

# --- 30 条"真实"客服对话查询 ---
REAL_QUERIES: List[str] = [
    # 冰箱相关 (8 条)
    "电冰箱不制冷了怎么办",
    "冰箱温度应该设置在多少度最省电",
    "冰箱冷藏室结冰是什么原因",
    "NR-JE52TGA-W 这款冰箱的容量是多少",
    "冰箱噪音很大，嗡嗡响是怎么回事",
    "风冷冰箱和直冷冰箱哪个好",
    "冰箱门封条老化需要更换吗",
    "双开门冰箱的冷冻室温度一般是多少",
    # 净水器相关 (7 条)
    "净水器滤芯多久换一次",
    "RO反渗透净水器出来的水能直接喝吗",
    "净水器安装需要什么条件",
    "净水器出水量变小了怎么解决",
    "超滤和反渗透净水器有什么区别",
    "净水器废水比太高怎么调",
    "净水器过滤后的水烧开还有水垢正常吗",
    # 电视相关 (7 条)
    "电视机屏幕出现横条纹是什么故障",
    "智能电视连不上WiFi怎么解决",
    "电视遥控器失灵怎么配对",
    "4K电视和8K电视差别大吗",
    "电视开机后黑屏但有声音",
    "电视机的最佳观看距离是多少",
    "智能电视安装第三方应用的方法",
    # 公司制度/通用客服 (8 条)
    "公司的年假政策是怎样的",
    "如何申请请假流程",
    "员工加班费怎么计算",
    "公司福利有哪些",
    "试用期多久可以转正",
    "公司有没有弹性工作制",
    "请问办公用品的申领流程",
    "出差报销需要提供哪些材料",
]

# --- 20 条边缘 case（同义词、多义词、模糊查询、跨类等）---
EDGE_QUERIES: List[str] = [
    # 同义词 (4 条)
    "冰箱制冷效果不好",        # vs "冰箱不制冷" → 语义相近但表达不同
    "冰柜食物冻得太硬了",      # "冰柜"vs"冰箱" → 用户可能用近义词
    "显示屏不亮了",            # 可能是电视显示屏
    "净水机出水很慢",          # "净水机"vs"净水器"
    # 多义词 / 指代模糊 (4 条)
    "电源打不开",              # 冰箱的电源？电视的电源？
    "这个怎么清理",            # 不知道在说冰箱还是净水器
    "声音太吵了",              # 冰箱噪音？电视噪音？
    "那个按键没反应",          # 遥控器按键？设备按键？
    # 模糊/口语化查询 (4 条)
    "那个大的白色的",          # 可能指冰箱
    "买了个东西不会用",        # 非常模糊
    "水过滤的那个玩意",        # 口语化指净水器
    "屏幕那块出问题了",        # 口语化指电视
    # 跨类混合 (4 条)
    "冰箱和电视哪个更费电",
    "净水器和冰箱的使用寿命分别是多久",
    "家电保修期一般都是几年",
    "大件电器送货上门吗",
    # 长尾/拼写变体 (4 条)
    "bing xiang wen du tiao jie",     # 拼音输入
    "BINXIANGZENMESHEZHI",            # 全大写拼音
    "电冰箱。不制冷。怎么办？",        # 标点异常
    "制冷 效果 差 冰箱",              # 词序颠倒
]

ALL_QUERIES = REAL_QUERIES + EDGE_QUERIES


# =============================================================================
# 自动生成测试文档（当集合中无数据时备用）
# =============================================================================

AUTO_DOCS: List[str] = [
    # 冰箱
    "NR-JE52TGA-W 是一款520升双开门风冷冰箱，冷藏室330升，冷冻室190升，一级能效，日耗电0.89度",
    "冰箱冷藏室最佳温度为3-5℃，冷冻室为-18℃，夏季建议调至较强档位",
    "冰箱不制冷的常见原因：电源插头松动、温控器设置不当、压缩机故障、制冷剂泄漏",
    "风冷冰箱通过风扇循环冷气，食物不易粘连但保湿性较差；直冷冰箱靠蒸发器制冷，保湿好但会结霜",
    "冰箱冷藏室结冰通常由门封条老化漏气、排水孔堵塞、或频繁开关门导致",
    "冰箱正常噪音在38-42分贝之间，异常嗡嗡声可能是压缩机底座松动或管路共振",
    "冰箱门封条老化可自行购买更换，更换时需用吹风机加热新封条使其软化贴合",
    "双开门大容量冰箱的标准冷冻温度是-18℃到-24℃，速冻模式可降至-32℃",
    "冰箱使用5-8年后能效下降约15%，建议定期清理冷凝器散热片保持换热效率",
    "冰箱运输后需静置2-4小时再通电，让压缩机油回流到缸体",
    "冰箱发热是正常现象，两侧或底部散热器在制冷循环中会散发大量热量",
    "冰箱除霜可用热水盆放入冷藏室加速化冰，切勿用利器刮擦蒸发器管路",
    # 净水器
    "RO反渗透净水器的核心是RO膜，过滤精度0.0001微米，能去除重金属、细菌和病毒",
    "PP棉滤芯建议3-6个月更换，活性炭滤芯6-12个月，RO膜2-3年更换一次",
    "净水器出水量变小通常由滤芯堵塞、进水压力不足、或RO膜结垢引起",
    "超滤净水器过滤精度0.01微米，保留矿物质但无法去水垢；反渗透几乎过滤所有离子，出水为纯水",
    "净水器废水比正常为1:1到1:3，即制一杯纯水产生1-3杯废水，比率与进水水质和水温有关",
    "RO净水器过滤后的水烧开一般不会有水垢，如果出现说明RO膜已破损需要更换",
    "净水器安装需要厨下三通接进水、废水管排入下水道、纯水龙头钻孔安装台面",
    "净水器的TDS值小于50即可直饮，TDS计可自行检测，正常的RO净水器出水TDS在5-30之间",
    "净水器长时间不用（>3天）应冲水3-5分钟后再使用，防止滤芯细菌滋生",
    # 电视
    "电视屏幕出现横条纹通常由排线松动、驱动板故障或液晶面板损坏导致",
    "智能电视连不上WiFi的解决方法：重启路由器、检查DNS设置、更新电视系统固件",
    "蓝牙遥控器失灵时可长按主页键+菜单键5秒重新配对，需保持1米内距离",
    "4K电视分辨率为3840×2160约830万像素，8K为7680×4320约3300万像素，目前4K已够用",
    "电视开机黑屏但有声音可能是背光条烧坏或高压板故障，需专业人员检修",
    "55寸电视最佳观看距离2.5-3.5米，65寸电视3-4米，按屏幕高度的3倍计算",
    "安装第三方应用需打开开发者模式，关闭未知来源限制，用U盘安装APK文件",
    "OLED电视色彩好但价格贵寿命略短，QLED电视亮度高性价比好寿命长",
    "电视经常自动关机可能是电源板电容老化或主板过热保护，可先清理通风口灰尘",
    "HDR10和Dolby Vision是主流HDR格式，需片源和电视同时支持才生效",
    # 公司制度
    "公司年假标准：入职满1年享5天，满5年15天，满10年20天，需提前一周申请",
    "请假流程：在OA系统提交请假申请→直属领导审批→HR确认→考勤自动扣除",
    "工作日加班1.5倍工资，休息日加班2倍，法定节假日3倍，调休需提前申请",
    "公司福利包括五险一金、补充公积金、年终奖金、年度体检、带薪培训等",
    "试用期为3-6个月，表现优秀者可提前转正，转正需提交转正申请并由部门负责人评估",
    "公司核心研发岗位实行弹性工作制，核心工作时间10:00-16:00需在岗",
    "办公用品通过行政系统在线申领，每月15日统一配送至工位",
    "出差报销需提供票据原件：交通票、住宿发票、餐饮小票（单餐不超过80元）",
    "公司内部晋升制度：每年春秋两季人才盘点，各级别晋升通道和要求透明公示",
    "新员工入职培训包括：企业文化、安全合规、岗位技能、导师带教四个模块",
    "公司IT服务台提供：电脑维修、软件安装、VPN申请、打印机墨盒更换",
    "员工可以通过内网学习平台免费学习管理、技术、外语等线上课程",
    "定期团建活动：每季度一次部门团建，每年一次全员出游，人均预算公布",
    "公司停车位供不应求时按入职年限排队分配，新能源车优先，每月停车费200元",
    "加班餐补标准：工作日加班至20:00后补贴30元，至22:00后补贴50元",
    "公司健身房位于B1层，24小时开放，刷卡进入，配淋浴间和储物柜",
    "哺乳期女员工每天享1小时哺乳假，可拆分使用或延迟上下班1小时",
    "员工内推奖励：普通岗位3000元/人，高级岗位8000元/人，入职转正后发放",
]


# ---- 噪声文档模板（用于扩充数据集，使 HNSW 图层数增加） ----

_NOISE_TEMPLATES = [
    # 各类产品型号规格噪声（语义与查询主题相近但不完全匹配）
    "{brand}{model}是一款{cate}，功率{power}W，尺寸{width}×{depth}×{height}mm，净重{weight}kg，颜色可选{color}",
    "{brand}{model}特色功能包括：{feature1}、{feature2}、{feature3}，适合{target_user}使用",
    "关于{cate}的常见问题{faq_idx}：{question} 答：{answer}",
    "{brand}{model}保修政策：整机保修{war1}年，主要部件保修{war2}年，提供7×24小时客服热线{cate_hotline}",
    "{cate}选购建议：预算{budget}元左右推荐{brand}{model}，功能均衡适合家庭使用",
    "{brand}{model}用户评价：{review_star}星好评，{review_cmt}，共{review_num}条评价",
    "{brand}{model}的{cate}核心技术采用{tech1}与{tech2}相结合，能耗等级{e_level}级",
    "当{cate}出现{problem}时，请首先检查{check1}是否正常，然后确认{check2}设置是否正确",
    "欢迎购买{brand}{model}，本产品采用环保材料制成，通过{standard}认证",
    "{cate}市场趋势：{year}年智能{cate}渗透率达到{rate}%，主流品牌包括{brand1}、{brand2}和{brand3}",
    "{brand}{model}对比上一代{cate}的升级点：{upgrade1}、{upgrade2}，同时价格降低{price_drop}%",
    "您可以通过{cate}的{mode}模式设置，实现{custom1}和{custom2}的个性化调整",
    "请定期清洁{cate}的{part}部件，建议每{interval}天进行一次维护保养",
    "{brand}{model}支持{smart1}和{smart2}智能控制功能，通过手机APP即可远程操作",
    "购买{brand}{model}赠送配件：{acc1}、{acc2}、{acc3}，价值约{acc_value}元",
]

_BRANDS = ["海尔", "美的", "格力", "海信", "TCL", "创维", "长虹", "松下", "西门子", "三星", "LG", "博世", "飞利浦", "戴森", "方太"]
_CATES = ["冰箱", "电视", "净水器", "空调", "洗衣机", "微波炉", "烤箱", "洗碗机", "吸尘器", "空气净化器"]
_MODEL_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "X", "Y", "Z"]
_COLORS = ["白色", "黑色", "银色", "金色", "灰色", "深蓝色", "玫瑰金"]
_TARGETS = ["家庭用户", "上班族", "老年人", "新婚夫妇", "小户型住户", "母婴家庭", "学生宿舍"]
_WAR_PERIODS = ["1", "3", "5", "10"]
_BUDGETS = ["1000", "2000", "3000", "5000", "8000", "10000", "15000", "20000"]
_STANDARDS = ["国家3C", "节能环保", "ISO9001", "CE", "RoHS", "FCC", "CCC", "UL"]

_VOCAB_CATE = {
    "cate_hotline": ["400-888-", "400-666-", "400-999-", "9510-", "400-800-"],
    "feature": ["智能控制", "变频节能", "静音运行", "快速制冷", "高温自清洁", "WiFi远程", "语音操控", "紫外线杀菌", "多层过滤", "大容量"],
    "question": [
        ("开机后不工作是怎么回事", "请检查电源插头是否插紧，确认开关是否打开，然后按复位键3秒后重新启动。如果仍不工作，请联系售后服务中心"),
        ("耗电量比说明书标称值高怎么办", "日常耗电受环境温度、开门频率、食物摆放密度影响，建议将温控调至适中档位。如超出标称值30%以上请联系售后检测"),
        ("可以自行安装吗", "建议预约专业人员上门安装，自行安装不当可能影响产品性能和安全，且不在保修范围"),
        ("运行时噪音正常范围是多少", "根据国家标准，正常运行时噪音应低于55分贝，如有异常尖锐或撞击声，请立即断电并联系检修"),
        ("如何判断是否需要更换部件或维修", "当出现运行异常、性能明显下降或显示屏故障代码时，建议先看说明书故障排除章节。若无法解决，拨打官方热线预约检修"),
    ],
    "review_cmt": ["质量非常好，推荐购买", "性价比很高，物流也快", "外观漂亮功能齐全", "安装师傅很专业，使用体验好", "用了几个月了没什么问题"],
    "problem": ["不启动", "异常噪音", "漏水", "显示异常", "性能下降", "遥控失灵", "异味", "漏电"],
    "mode": ["节能", "标准", "强力", "静音", "智能", "睡眠"],
    "part": ["滤网", "内胆", "控制面板", "门封", "冷凝器", "排水管", "风扇"],
    "smart": ["WiFi远程控制", "语音助手", "定时预约", "智能感应", "场景联动"],
    "acc": ["说明书", "遥控器", "电源线", "安装工具包", "清洁刷", "备用滤芯", "挂钩套件"],
    "upgrade": ["能效提升", "噪音降低", "容量增加", "新增智能功能", "设计更薄"],
    "tech": ["变频压缩机", "直流无刷电机", "纳米银离子抗菌", "UV紫外线消杀", "PID精准控温", "真空隔热", "石墨烯导热"],
    "year": ["2022", "2023", "2024", "2025"],
    "e_level": ["一", "二", "三"],
}


def generate_noise_docs(n: int, hard_ratio: float = 0.5, seed: int = 42) -> List[str]:
    """生成 n 条噪声文档

    Args:
        n: 总数量
        hard_ratio: 硬噪声比例（0~1）。硬噪声用不相关领域的文本，确保向量空间分布分散，
                    迫使 HNSW 图更深、efSearch 触发的近似误差更明显。
    """
    import random
    rng = random.Random(seed)
    docs = []

    n_hard = int(n * hard_ratio)
    n_soft = n - n_hard

    # 软噪声：同领域模板（与查询共享词汇，语义相近）
    for i in range(n_soft):
        brand = rng.choice(_BRANDS)
        cate = rng.choice(_CATES)
        model = f"{rng.choice(_MODEL_LETTERS)}{rng.choice(_MODEL_LETTERS)}-{rng.randint(100, 999)}"
        template = rng.choice(_NOISE_TEMPLATES)
        v = _VOCAB_CATE
        doc = template.format(
            brand=brand, model=model, cate=cate,
            power=rng.randint(30, 3000),
            width=rng.randint(200, 1800), depth=rng.randint(200, 900), height=rng.randint(300, 2000),
            weight=round(rng.uniform(1.0, 200.0), 1),
            color=rng.choice(_COLORS),
            feature1=rng.choice(v["feature"]), feature2=rng.choice(v["feature"]), feature3=rng.choice(v["feature"]),
            target_user=rng.choice(_TARGETS),
            faq_idx=rng.randint(1, 99),
            question=(q := rng.choice(v["question"]))[0], answer=q[1],
            war1=rng.choice(_WAR_PERIODS), war2=rng.choice(_WAR_PERIODS),
            budget=rng.choice(_BUDGETS),
            review_star=rng.choice(["4.2", "4.5", "4.7", "4.8", "5.0"]),
            review_cmt=rng.choice(v["review_cmt"]),
            review_num=rng.randint(100, 50000),
            tech1=rng.choice(v["tech"]), tech2=rng.choice(v["tech"]),
            e_level=rng.choice(v["e_level"]),
            problem=rng.choice(v["problem"]),
            check1=rng.choice(["电源", "开关", "连接线", "指示灯", "显示屏"]),
            check2=rng.choice(["设置参数", "温度", "模式", "网络连接", "管路"]),
            standard=rng.choice(_STANDARDS),
            year=rng.choice(v["year"]),
            rate=rng.randint(30, 85),
            brand1=rng.choice(_BRANDS), brand2=rng.choice(_BRANDS), brand3=rng.choice(_BRANDS),
            upgrade1=rng.choice(v["upgrade"]), upgrade2=rng.choice(v["upgrade"]),
            price_drop=rng.randint(5, 25),
            mode=rng.choice(v["mode"]),
            custom1=rng.choice(["温度", "风力", "定时", "亮度", "水量"]),
            custom2=rng.choice(["模式", "湿度", "音量", "色温", "流速"]),
            part=rng.choice(v["part"]),
            interval=rng.choice(["7", "15", "30", "90", "180"]),
            smart1=rng.choice(v["smart"]), smart2=rng.choice(v["smart"]),
            acc1=rng.choice(v["acc"]), acc2=rng.choice(v["acc"]), acc3=rng.choice(v["acc"]),
            acc_value=rng.randint(50, 500),
            cate_hotline=f"{rng.choice(v['cate_hotline'])}{rng.randint(1000, 9999)}",
        )
        docs.append(doc)

    # 硬噪声：完全不相关领域的文本（让向量空间分布更散，增加 HNSW 图深度）
    _HARD_NOISE_POOL = [
        "三角形内角和等于180度，这是欧几里得几何的基本定理之一，广泛应用于建筑设计和工程测量",
        "联合国教科文组织世界遗产名录收录了全球超过1000处文化和自然遗产，中国的长城和故宫位列其中",
        "光合作用是植物利用光能将二氧化碳和水转化为葡萄糖和氧气的过程，叶绿体是光合作用的场所",
        "Python是一门解释型、面向对象的高级编程语言，具有动态语义，广泛用于Web开发和数据科学",
        "地中海饮食以橄榄油、鱼类、蔬菜和全谷物为主，被公认为全球最健康的饮食模式之一",
        "量子计算机利用量子比特的叠加态和纠缠态进行计算，在特定问题上理论速度远超传统计算机",
        "《红楼梦》是中国古典四大名著之一，作者曹雪芹，以贾宝玉和林黛玉的爱情悲剧为主线",
        "区块链技术是一种去中心化的分布式账本技术，具有不可篡改、透明可追溯等特性",
        "帕金森定律指出：工作量会自然地膨胀以填满完成工作所允许的时间",
        "莫扎特是维也纳古典乐派的代表人物，其歌剧《费加罗的婚礼》和《魔笛》享誉世界",
        "春秋战国时期百家争鸣，儒家主张仁政德治，法家主张以法治国，道家主张无为而治",
        "COVID-19疫苗采用mRNA技术，通过递送刺突蛋白的基因信息来激发人体免疫反应",
        "地球的赤道周长约40075公里，自转周期约23小时56分4秒，公转周期约365.25天",
        "日本茶道起源于中国宋代的点茶法，经过千利休发扬光大，形成和敬清寂的核心理念",
        "马丘比丘是印加帝国建于15世纪的古城遗址，位于秘鲁安第斯山脉海拔2430米的山脊上",
        "特斯拉线圈是一种谐振变压器电路，由尼古拉·特斯拉于1891年发明，能产生高频高压电",
        "敦煌莫高窟又称千佛洞，始建于前秦时期，现存洞窟735个，壁画4.5万平方米",
        "微生物的发酵作用用于制作面包、酸奶、啤酒等食品，酵母菌是最常用的发酵微生物",
        "皮格马利翁效应又称罗森塔尔效应，指期望与赞美会产生奇迹般的效果",
        "黑洞是时空曲率大到光都无法逃脱的天体，由大质量恒星引力坍缩形成",
        "汉字的演变经历了甲骨文、金文、篆书、隶书、楷书、行书和草书等阶段",
        "神经网络通过反向传播算法更新权重，使用梯度下降来最小化损失函数",
        "火星是太阳系八大行星之一，因表面富含氧化铁而呈红色，直径约为地球的一半",
        "休克疗法最初由萨克斯提出，是通过一系列激进措施快速实现经济体制转型的方案",
        "蒙娜丽莎是达芬奇的代表作，现藏于法国巴黎卢浮宫，每年吸引超过1000万游客参观",
        "克苏鲁神话由洛夫克拉夫特开创，其核心思想是宇宙中存在人类无法理解的古老存在",
        "人工智能的三要素是数据、算法和算力，深度学习是当前AI发展的核心技术路线",
        "南极洲冰盖含有全球约70%的淡水，如果全部融化将使全球海平面上升约60米",
        "水墨画是中国传统绘画形式，以水和墨为主要材料，讲究'墨分五色'和'留白'的技法",
        "进化论由达尔文提出，自然选择是进化的主要机制，物种通过适者生存逐渐改变",
    ]

    for i in range(n_hard):
        base = rng.choice(_HARD_NOISE_POOL)
        # 给每条硬噪声追加一些随机变化使其不完全重复
        suffix = f" 序号{10000 + rng.randint(1, 90000)} | 来源{rng.choice(['百科', '文献', '论文', '报告'])}{rng.randint(1, 100)}号"
        docs.append(base + suffix)

    # 打乱顺序
    rng.shuffle(docs)
    return docs

# =============================================================================
# 辅助函数
# =============================================================================


def cosine_similarity_batch(query_vec: np.ndarray, all_vecs: np.ndarray) -> np.ndarray:
    """对一条 query_vec 与 all_vecs 矩阵做批量余弦相似度（均已归一化）"""
    # 向量已 L2 归一化时，内积 = 余弦相似度
    return np.dot(all_vecs, query_vec)


def compute_ground_truth(
    query_embedding: List[float],
    all_embeddings: np.ndarray,
    doc_ids: List[int],
    top_k: int = DEFAULT_TOP_K,
) -> List[int]:
    """暴力全库扫描，返回 top_k 文档的 Milvus ID 列表（按相似度降序）"""
    q = np.array(query_embedding, dtype=np.float32)
    scores = cosine_similarity_batch(q, all_embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [doc_ids[i] for i in top_indices]


def recall_at_k(pred_indices: List[int], gt_indices: List[int], k: int) -> float:
    """计算 recall@k"""
    pred_set = set(pred_indices[:k])
    gt_set = set(gt_indices[:k])
    if len(gt_set) == 0:
        return 0.0
    return len(pred_set & gt_set) / len(gt_set)


def get_embedding_service():
    """懒加载 EmbeddingService"""
    from src.modules.chat.core.embedding_service import EmbeddingService
    return EmbeddingService.get_instance()


def embed_queries(queries: List[str]) -> List[List[float]]:
    """对 query 列表做 embedding（同步方式）"""
    svc = get_embedding_service()
    embeddings_obj = svc.get_embeddings()
    print(f"  [embed] 正在嵌入 {len(queries)} 条查询...")
    start = time.time()
    embeddings = embeddings_obj.embed_documents(queries)
    elapsed = time.time() - start
    print(f"  [embed] 完成，耗时 {elapsed:.1f}s ({len(queries)/elapsed:.0f} q/s)")
    return embeddings


def embed_documents(docs: List[str]) -> List[List[float]]:
    """对文档列表做 embedding"""
    svc = get_embedding_service()
    embeddings_obj = svc.get_embeddings()
    print(f"  [embed] 正在嵌入 {len(docs)} 条文档...")
    start = time.time()
    embeddings = embeddings_obj.embed_documents(docs)
    elapsed = time.time() - start
    print(f"  [embed] 完成，耗时 {elapsed:.1f}s")
    return embeddings


def get_all_doc_embeddings(collection: Collection) -> Tuple[np.ndarray, List[int], List[str], int]:
    """从 Milvus 拉取全量文档向量，返回 (向量矩阵, Milvus IDs, 文本, 总数)"""
    print(f"  [milvus] 正在拉取全量文档向量...")
    collection.load()

    # 获取总数
    import time
    collection.flush()
    time.sleep(1)  # 等 flush
    num_entities = collection.num_entities
    print(f"  [milvus] 集合中有 {num_entities} 条文档")

    # 分批拉取
    BATCH = 1000
    all_vectors = []
    all_ids = []
    all_texts = []
    offset = 0

    while offset < num_entities:
        limit = min(BATCH, num_entities - offset)
        results = collection.query(
            expr="id >= 0",
            output_fields=["id", "text", "embedding"],
            limit=limit,
            offset=offset,
        )
        for r in results:
            all_vectors.append(r["embedding"])
            all_ids.append(r["id"])
            all_texts.append(r.get("text", ""))
        offset += limit
        print(f"  [milvus]   已拉取 {offset}/{num_entities}...", end="\r")

    print(f"\n  [milvus] 拉取完成，共 {len(all_vectors)} 条向量")
    return np.array(all_vectors, dtype=np.float32), all_ids, all_texts, num_entities


def search_hnsw_with_ef(
    collection: Collection,
    query_embedding: List[float],
    ef: int,
    top_k: int,
) -> Tuple[List[int], float]:
    """用指定 ef 值执行 HNSW 近似搜索，返回 doc_id 列表和延迟(ms)"""
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": ef},
    }
    start = time.perf_counter()
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["id", "text"],
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    hit_ids = []
    for hits in results:
        for hit in hits:
            hit_ids.append(hit.id)
    return hit_ids, elapsed_ms


@dataclass
class EfResult:
    ef: int
    recall_5: float
    recall_10: float
    latencies_ms: List[float]
    p50_ms: float
    p99_ms: float
    avg_ms: float


# =============================================================================
# 自动建集合 + 插入测试文档
# =============================================================================

def embed_documents_batched(docs: List[str], batch_size: int = 32) -> List[List[float]]:
    """分批嵌入，避免大批次 OOM"""
    svc = get_embedding_service()
    embeddings_obj = svc.get_embeddings()
    all_embeddings = []
    total = len(docs)
    print(f"  [embed] 正在分批嵌入 {total} 条文档 (batch_size={batch_size})...")
    start = time.time()
    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        embs = embeddings_obj.embed_documents(batch)
        all_embeddings.extend(embs)
        print(f"  [embed]   进度: {min(i + batch_size, total)}/{total}", end="\r")
    elapsed = time.time() - start
    print(f"\n  [embed] 完成，耗时 {elapsed:.1f}s ({total / elapsed:.0f} docs/s)")
    return all_embeddings


def setup_test_collection(collection_name: str, noise_count: int = 1000) -> Collection:
    """创建测试用集合，插入种子文档 + 噪声文档"""
    print(f"\n{'='*60}")
    print(f"  自动创建测试集合并插入文档 (种子:{len(AUTO_DOCS)} + 噪声:{noise_count})")
    print(f"{'='*60}")

    from src.modules.chat.config import chat_config
    dim = chat_config.embedding_dimension
    print(f"  embedding 维度: {dim}")

    # 删除已有
    if utility.has_collection(collection_name):
        print(f"  删除已有集合 {collection_name}...")
        utility.drop_collection(collection_name)

    # 创建集合
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, "HNSW benchmark test collection")
    coll = Collection(collection_name, schema)

    # 创建 HNSW 索引
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION},
    }
    coll.create_index("embedding", index_params)
    coll.load()

    # 生成所有文档：种子 + 噪声
    noise_docs = generate_noise_docs(noise_count)
    all_docs = AUTO_DOCS + noise_docs
    print(f"  总文档数: {len(all_docs)}")

    # 分批嵌入
    all_embeddings = embed_documents_batched(all_docs, batch_size=64)

    # 分批插入 Milvus
    MILVUS_INSERT_BATCH = 500
    total_inserted = 0
    print(f"  [milvus] 正在分批插入 {len(all_docs)} 条文档...")
    for i in range(0, len(all_docs), MILVUS_INSERT_BATCH):
        batch_texts = all_docs[i : i + MILVUS_INSERT_BATCH]
        batch_embs = all_embeddings[i : i + MILVUS_INSERT_BATCH]
        data = [{"text": t, "embedding": e} for t, e in zip(batch_texts, batch_embs)]
        coll.insert(data)
        total_inserted += len(batch_texts)
        print(f"  [milvus]   已插入 {total_inserted}/{len(all_docs)}...", end="\r")
    coll.flush()
    print(f"\n  [milvus] 插入完成, 当前集合大小: {coll.num_entities}")
    return coll


# =============================================================================
# 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="HNSW efSearch 消融实验")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Milvus 集合名称")
    parser.add_argument("--auto-data", action="store_true", help="自动生成测试数据并插入集合")
    parser.add_argument("--noise-count", type=int, default=1000, help="噪声文档数量 (默认1000，越大 HNSW 图层越深)")
    parser.add_argument("--output", default=None, help="结果输出 JSON 文件路径")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="评估 top-K (默认10)")
    parser.add_argument("--host", default=MILVUS_HOST, help="Milvus 地址")
    parser.add_argument("--port", type=int, default=MILVUS_PORT, help="Milvus 端口")
    args = parser.parse_args()

    print("=" * 70)
    print("  HNSW efSearch 参数消融实验")
    print("=" * 70)
    print(f"  Milvus: {args.host}:{args.port}")
    print(f"  Collection: {args.collection}")
    print(f"  efSearch 候选值: {SEARCH_EF_VALUES}")
    print(f"  测试 Query 数: {len(ALL_QUERIES)} (真实:{len(REAL_QUERIES)} + 边缘:{len(EDGE_QUERIES)})")
    print(f"  每对 (query, ef) 重复搜索: {SEARCH_REPEATS} 次")
    print()

    # ---------- 1. 连接 Milvus ----------
    print("[1/5] 连接 Milvus...")
    try:
        connections.connect("default", host=args.host, port=args.port)
        print("  [OK] 已连接")
    except Exception as e:
        print(f"  [ERROR] 连接失败: {e}")
        print("  请先启动 Milvus: docker-compose up -d standalone etcd minio")
        sys.exit(1)

    # ---------- 2. 准备集合 ----------
    if args.auto_data or not utility.has_collection(args.collection):
        if args.auto_data:
            print(f"\n[2/5] 自动生成测试数据...")
        else:
            print(f"\n[2/5] 集合 '{args.collection}' 不存在，自动生成测试数据...")
        collection = setup_test_collection(args.collection, noise_count=args.noise_count)
    else:
        print(f"\n[2/5] 加载已有集合 '{args.collection}'...")
        collection = Collection(args.collection)
        collection.load()
        num = collection.num_entities
        print(f"  [OK] 集合已加载, 文档数: {num}")
        if num < 10:
            print(f"  [WARN] 文档数太少 ({num}), 建议使用 --auto-data 生成更多数据")
            print(f"  继续使用现有数据...")

    # ---------- 3. 获取全量向量 并计算 Ground Truth ----------
    print(f"\n[3/5] 拉取全量向量并计算 Ground Truth...")
    all_embeddings, doc_ids, all_texts, num_docs = get_all_doc_embeddings(collection)
    print(f"  向量矩阵 shape: {all_embeddings.shape}")

    # 嵌入 queries
    query_embeddings = embed_queries(ALL_QUERIES)
    if len(query_embeddings[0]) != all_embeddings.shape[1]:
        print(f"  [ERROR] 维度不匹配: query={len(query_embeddings[0])}, docs={all_embeddings.shape[1]}")
        sys.exit(1)

    # 为每条 query 计算 ground truth
    print(f"  正在计算 {len(ALL_QUERIES)} 条 query 的 ground truth...")
    gt_start = time.time()
    ground_truths: Dict[int, List[int]] = {}  # query_idx -> [doc_id, ...]
    for i, q_emb in enumerate(query_embeddings):
        gt = compute_ground_truth(q_emb, all_embeddings, doc_ids, args.top_k)
        ground_truths[i] = gt
    gt_elapsed = time.time() - gt_start
    print(f"  Ground Truth 计算完成, 耗时 {gt_elapsed:.1f}s")

    # ---------- 4. 对每个 efSearch 值执行搜索 ----------
    print(f"\n[4/5] 执行 HNSW 消融实验 ({len(SEARCH_EF_VALUES)} 个 ef 值 × {len(ALL_QUERIES)} query × {SEARCH_REPEATS} 次重复)...")

    # 预热：消除首次查询的冷启动延迟偏差（Milvus segment 加载、CPU cache miss 等）
    print(f"  预热中 (3 次全量搜索)...")
    for _ in range(3):
        search_hnsw_with_ef(collection, query_embeddings[0], SEARCH_EF_VALUES[-1], args.top_k)
    print(f"  预热完成，开始正式测量")

    results: List[EfResult] = []

    for ef in SEARCH_EF_VALUES:
        all_recalls_5 = []
        all_recalls_10 = []
        all_latencies = []

        for q_idx, q_emb in enumerate(query_embeddings):
            gt = ground_truths[q_idx]

            for _ in range(SEARCH_REPEATS):
                hits, lat = search_hnsw_with_ef(collection, q_emb, ef, args.top_k)
                all_latencies.append(lat)
                all_recalls_5.append(recall_at_k(hits, gt, 5))
                all_recalls_10.append(recall_at_k(hits, gt, 10))

        avg_recall_5 = statistics.mean(all_recalls_5)
        avg_recall_10 = statistics.mean(all_recalls_10)
        p50 = statistics.median(all_latencies)
        avg_lat = statistics.mean(all_latencies)

        # p99
        sorted_lats = sorted(all_latencies)
        p99_idx = int(0.99 * len(sorted_lats))
        p99 = sorted_lats[min(p99_idx, len(sorted_lats) - 1)]

        results.append(EfResult(
            ef=ef,
            recall_5=avg_recall_5,
            recall_10=avg_recall_10,
            latencies_ms=all_latencies,
            p50_ms=round(p50, 2),
            p99_ms=round(p99, 2),
            avg_ms=round(avg_lat, 2),
        ))

        print(f"  efSearch={ef:>3} | recall@5={avg_recall_5:.4f} recall@10={avg_recall_10:.4f} | p50={p50:.1f}ms p99={p99:.1f}ms avg={avg_lat:.1f}ms")

    # ---------- 5. 输出结果 ----------
    print(f"\n[5/5] 结果汇总")
    print("=" * 70)

    # 表格输出
    print(f"\n{'efSearch':>9} | {'recall@5':>9} | {'recall@10':>10} | {'p50 (ms)':>9} | {'p99 (ms)':>9} | {'avg (ms)':>9}")
    print("-" * 70)
    for r in results:
        print(f"{r.ef:>9} | {r.recall_5:>9.4f} | {r.recall_10:>10.4f} | {r.p50_ms:>9.1f} | {r.p99_ms:>9.1f} | {r.avg_ms:>9.1f}")

    # 找到拐点（recall 边际收益递减 < 1% 且延迟增幅最小的点）
    print("\n--- 拐点分析 ---")
    best_ef = None
    best_score = 0
    for i, r in enumerate(results):
        recall = r.recall_5
        latency = r.avg_ms
        # 归一化后计算综合得分：recall 越高越好，latency 越低越好
        max_recall = max(x.recall_5 for x in results)
        min_latency = min(x.avg_ms for x in results)
        max_latency = max(x.avg_ms for x in results)
        norm_recall = recall / max_recall if max_recall > 0 else 0
        norm_latency = 1.0 - (latency - min_latency) / (max_latency - min_latency) if max_latency > min_latency else 1.0
        score = 0.7 * norm_recall + 0.3 * norm_latency  # recall 权重 70%
        if score > best_score:
            best_score = score
            best_ef = r.ef

    print(f"  综合最优 efSearch: {best_ef}")
    print(f"  决策依据: recall 权重 70%, latency 权重 30%")

    # 输出 JSON
    output = {
        "config": {
            "milvus_host": args.host,
            "milvus_port": args.port,
            "collection": args.collection,
            "num_docs": num_docs,
            "num_queries": len(ALL_QUERIES),
            "num_real_queries": len(REAL_QUERIES),
            "num_edge_queries": len(EDGE_QUERIES),
            "search_repeats": SEARCH_REPEATS,
            "top_k": args.top_k,
            "hnsw_m": HNSW_M,
            "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
        },
        "results": [
            {
                "efSearch": r.ef,
                "recall_at_5": round(r.recall_5, 6),
                "recall_at_10": round(r.recall_10, 6),
                "p50_ms": r.p50_ms,
                "p99_ms": r.p99_ms,
                "avg_ms": r.avg_ms,
            }
            for r in results
        ],
        "best_ef": best_ef,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")

    print("\n" + "=" * 70)
    print("  实验完成")
    print("=" * 70)

    # 清理连接
    connections.disconnect("default")


if __name__ == "__main__":
    main()
