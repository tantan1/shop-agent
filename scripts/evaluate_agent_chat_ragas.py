"""
================================================================================
 agent_chat (通用Agent对话) RAGAS 评估脚本
================================================================================

评估目标：
    - 评估 src/modules/chat/routers.py 中 `/agent/chat` 端点
    - 该端点是一个 4 步骤 RAG Agent pipeline:
        步骤1: 问题理解/改写 → 生成多条检索查询
        步骤2: 内容审查/安全检查 → 判断风险等级
        步骤3: 知识检索 → Milvus 2.6+ 原生混合检索 (Dense + Sparse BM25)
        步骤4: 答案生成 → 基于检索上下文 + LLM 生成回答

RAGAS 评估指标体系：
    Context Precision    - 检索到的上下文中，相关文档排在前面的程度
    Context Recall       - 相关文档被检索到的比例
    Faithfulness         - 生成的答案是否忠实于检索到的上下文（是否编造）
    Answer Relevancy     - 生成的答案是否与问题相关
    Context Relevancy    - 检索到的上下文是否与问题相关
    Answer Correctness   - 答案相对于标准答案的正确性

依赖安装：
    pip install ragas datasets pandas openai

前置条件：
    1. Milvus 向量数据库运行中，且已插入知识库文档
    2. LLM 服务可访问（火山引擎/通义千问）
    3. 嵌入服务可访问
    4. 环境变量已正确配置

用法：
    # 评估所有领域的所有测试用例
    python scripts/evaluate_agent_chat_ragas.py

    # 仅评估指定领域
    python scripts/evaluate_agent_chat_ragas.py --domain medical

    # 仅运行第 3 个测试用例
    python scripts/evaluate_agent_chat_ragas.py --case-index 3

    # 仅运行问题包含"请假"的用例
    python scripts/evaluate_agent_chat_ragas.py --case-name 请假

    # 组合使用：第 1 个电商领域的用例
    python scripts/evaluate_agent_chat_ragas.py --domain ecommerce --case-index 1

    # 仅生成数据集（不运行评估）
    python scripts/evaluate_agent_chat_ragas.py --dry-run

    # 输出到文件
    python scripts/evaluate_agent_chat_ragas.py --output results.json
================================================================================
"""

import os
import sys
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

# ── Langfuse 追踪（可选，用于将评估指标上传到 Langfuse 平台） ──
try:
    from langfuse import Langfuse, observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGfUSE_AVAILABLE = False
    Langfuse = None  # type: ignore
    def observe(*args, **kwargs):  # noqa: E302
        def decorator(func):
            return func
        return decorator


def _get_current_otel_span():
    """获取当前 OpenTelemetry span（Langfuse @observe 底层 span）。"""
    try:
        from opentelemetry import trace
        return trace.get_current_span()
    except ImportError:
        return None


# get_current_trace_id / get_current_langfuse_span 在 langfuse 4.7.1 中不存在，使用 OTel span 替代
def get_current_trace_id():  # noqa: E302
    span = _get_current_otel_span()
    if span and hasattr(span, 'get_span_context'):
        ctx = span.get_span_context()
        if ctx.is_valid:
            return str(ctx.trace_id)
    return None


def get_current_langfuse_span():  # noqa: E302
    return _get_current_otel_span()


# =============================================================================
# 评估数据集定义
# =============================================================================

@dataclass
class EvalCase:
    """单个评估用例"""
    question: str           # 用户问题
    ground_truth: str       # 标准答案（参考答案）
    domain: str = "general" # 业务领域
    expected_context_keywords: List[str] = field(default_factory=list)  # 期望检索到的关键词


# =============================================================================
# 测试用例集（按领域分组）
# 注意：这些用例需要与你已插入 Milvus 的知识库文档内容相匹配
# 请根据实际知识库内容调整问题和参考答案
# =============================================================================

EVALUATION_DATASET = [
    # ---- 通用领域 (general) ----
    EvalCase(
        question="公司员工如何请假？",
        ground_truth="事假:根据情况而定,一般为1天/月，每月病假不超过1天",
        domain="general",
        expected_context_keywords=["请假", "申请", "审批", "人事"]
    ),
    EvalCase(
        question="公司的请假流程是怎样的？",
        ground_truth="请假流程：1) 员工提前填写请假单；2) 直属上级审批；3) 超过3天需部门负责人加签；4) 报人事部备案存档。",
        domain="general",
        expected_context_keywords=["请假", "流程", "审批", "人事部"]
    ),
    EvalCase(
        question="公司有哪些规章制度需要遵守？",
        ground_truth="公司规章制度包括考勤制度、请假制度、办公纪律、安全管理制度、保密制度等，员工入职时应认真学习并签署确认。",
        domain="general",
        expected_context_keywords=["规章制度", "考勤", "纪律", "安全", "保密"]
    ),
    EvalCase(
        question="员工迟到会有什么处罚？",
        ground_truth="迟到处罚标准：前三次迟到每次处罚10元，超出三次后每次处罚50元。同时早退一次处罚10元，月度累计超5次早退视为旷工半天。",
        domain="general",
        expected_context_keywords=["迟到", "处罚", "10元", "50元"]
    ),

    # ---- 电商领域 (ecommerce) ----
    EvalCase(
        question="电冰箱有哪些功能和特点？",
        ground_truth="该款电冰箱采用风冷无霜技术，具有变频节能、智能控温、大容量存储等特点，支持冷藏冷冻分区调节。",
        domain="ecommerce",
        expected_context_keywords=["电冰箱", "风冷", "变频", "智能控温"]
    ),
    EvalCase(
        question="净水器的滤芯多久需要更换一次？",
        ground_truth="净水器滤芯更换周期：PP棉滤芯3-6个月，活性炭滤芯6-12个月，RO反渗透膜24-36个月，具体根据水质和使用频率而定。",
        domain="ecommerce",
        expected_context_keywords=["净水器", "滤芯", "更换", "RO"]
    ),
    EvalCase(
        question="电视有哪些规格参数？",
        ground_truth="电视机包含屏幕尺寸、分辨率、刷新率、HDMI接口数量、是否支持智能系统等规格参数，不同型号存在差异。",
        domain="ecommerce",
        expected_context_keywords=["电视", "规格", "分辨率", "HDMI"]
    ),
    EvalCase(
        question="产品退货流程是什么？",
        ground_truth="退货流程：1) 在订单页面申请退货；2) 填写退货原因；3) 等待客服审核；4) 审核通过后寄回商品；5) 仓库收货后7个工作日内退款。",
        domain="ecommerce",
        expected_context_keywords=["退货", "退款", "订单", "审核"]
    ),

    # ---- 电商领域-图增强测试用例 (graph ablation) ----
    # 这些用例包含商品ID或品牌名，会触发 NebulaGraph 关系查询
    EvalCase(
        question="IPHONE_15有哪些兼容配件？",
        ground_truth="iPhone 15兼容的配件包括MagSafe无线充电器、USB-C编织数据线、AirPods Pro 2无线耳机、Apple Watch S9等Apple生态配件。",
        domain="ecommerce",
        expected_context_keywords=["MagSafe", "USB-C", "AirPods", "兼容"]
    ),
    EvalCase(
        question="苹果品牌有什么热销商品？",
        ground_truth="苹果品牌热销商品包括iPhone 15系列、AirPods系列、MacBook Air M3、Apple Watch S9、iPad Air 6等产品。",
        domain="ecommerce",
        expected_context_keywords=["iPhone", "AirPods", "MacBook", "Apple Watch", "iPad"]
    ),
    EvalCase(
        question="AIRPODS_PRO2的替代品有哪些推荐？",
        ground_truth="AirPods Pro 2的竞品替代包括华为FreeBuds Pro 3、Galaxy Buds 3 Pro、Redmi Buds 5 Pro等无线降噪耳机。",
        domain="ecommerce",
        expected_context_keywords=["FreeBuds", "Galaxy Buds", "Redmi Buds", "替代"]
    ),
    EvalCase(
        question="MAGSAFE_CHARGER同品类还有什么热销配件？",
        ground_truth="手机配件品类中，MagSafe充电器的同品类热销商品包括USB-C编织数据线、Apple Pencil USB-C、华为M-Pencil等配件。",
        domain="ecommerce",
        expected_context_keywords=["USB-C", "数据线", "Apple Pencil", "M-Pencil", "配件"]
    ),
]


# =============================================================================
# RAGAS 评估器
# =============================================================================

class AgentChatRagasEvaluator:
    """
    对 agent_chat 端点进行 RAGAS 评估
    
    评估流程：
    1. 使用测试用例集调用 ChatAgentService.chat_with_agent()
    2. 从响应中提取 contexts（documents_used）和 answer（message）
    3. 构建 RAGAS 所需的 Sample 数据集
    4. 运行 RAGAS 评估指标
    5. 生成评估报告
    """

    def __init__(self, domain: str = None, llm_model: str = None, skip_reranker: bool = False,
                 single_case_index: int = None, single_case_name: str = None,
                 retrieval_mode: str = "hybrid", enable_graph: bool = False):
        """
        初始化评估器
        
        Args:
            domain: 要评估的领域，None 表示所有领域
            llm_model: RAGAS 用于评判的 LLM 模型，默认使用通义千问
            skip_reranker: 是否跳过 BGE-Reranker 重排序
            single_case_index: 仅运行指定序号的用例 (1-based)，None 表示运行全部
            single_case_name: 仅运行问题包含该关键词的用例，None 表示运行全部
            retrieval_mode: 检索模式 — "hybrid"（Dense+BM25）或 "dense-only"（纯向量检索基线）
            enable_graph: 是否启用 NebulaGraph 图增强查询
        """
        self.domain = domain
        self.llm_model = llm_model
        self.skip_reranker = skip_reranker
        self.single_case_index = single_case_index
        self.single_case_name = single_case_name
        self.retrieval_mode = retrieval_mode
        self.enable_graph = enable_graph
        self.results: List[Dict[str, Any]] = []
        self.service = None
        self._original_rerank_flags: Dict[str, bool] = {}
        self._original_hybrid_search = None
        self._original_graph_env = None

        # ── Langfuse 追踪客户端 ──
        self.langfuse_client = None
        self.langfuse_enabled = False
        if LANGFUSE_AVAILABLE:
            try:
                public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
                secret_key = os.getenv("LANGFUSE_SECRET_KEY")
                if public_key and secret_key:
                    self.langfuse_client = Langfuse()
                    self.langfuse_enabled = True
            except Exception:
                pass

    async def _get_service(self):
        """懒加载 ChatAgentService"""
        if self.service is None:
            from src.modules.chat.services import ChatAgentService
            
            class MockSession:
                async def __aenter__(self): return self
                async def __aexit__(self, *args): pass
            
            self.service = ChatAgentService(MockSession())
            await self.service._initialize()
        return self.service

    def _patch_reranker_config(self, disable: bool):
        """Monkey-patch 所有领域配置的 rerank_enabled 为指定值"""
        from src.modules.chat.config import DOMAIN_CONFIGS
        for domain_name, agent_cfg in DOMAIN_CONFIGS.items():
            self._original_rerank_flags[domain_name] = getattr(agent_cfg, 'rerank_enabled', True)
            agent_cfg.rerank_enabled = not disable  # disable=True → rerank_enabled=False

    def _restore_reranker_config(self):
        """恢复所有领域配置的 rerank_enabled 原始值"""
        from src.modules.chat.config import DOMAIN_CONFIGS
        for domain_name, original_flag in self._original_rerank_flags.items():
            if domain_name in DOMAIN_CONFIGS:
                DOMAIN_CONFIGS[domain_name].rerank_enabled = original_flag
        self._original_rerank_flags.clear()

    def _patch_retrieval_mode(self, mode: str):
        """Monkey-patch: 切换检索模式。
        
        "hybrid" → 默认，不修改（Dense+BM25 RRF 融合）
        "dense-only" → 将 MilvusService 实例的 hybrid_search 重定向到 search_similar（纯向量检索基线）
        
        注意：此方法必须在 _get_service() 之后调用，确保 MilvusService 单例已初始化。
        """
        if mode == "hybrid":
            return  # 默认模式，无需 patch
        if mode == "dense-only":
            from src.modules.chat.core.milvus_service import MilvusService
            svc = MilvusService.get_instance()
            
            # Save original bound method for restoration
            self._original_hybrid_search = svc.hybrid_search
            
            def _dense_only_wrapper(query_embedding, query_text=None,
                                    top_k=5, rrf_k=60, extra_field=""):
                """将 hybrid_search 重定向到纯 Dense 向量检索"""
                result = svc.search_similar(query_embedding, top_k=top_k)
                if not hasattr(_dense_only_wrapper, '_logged'):
                    _dense_only_wrapper._logged = True
                    print(f"  [DEBUG] dense-only wrapper 已生效, search_similar 返回 {len(result)} 条")
                return result
            
            # Replace the instance method directly (not via MethodType, to avoid self binding)
            svc.hybrid_search = _dense_only_wrapper
            print(f"  [PATCH] 检索模式: dense-only (纯向量检索，禁用 BM25)\n")
        else:
            raise ValueError(f"不支持的检索模式: {mode}，可选 'hybrid' | 'dense-only'")

    def _restore_retrieval_mode(self):
        """恢复原始 hybrid_search 方法"""
        if self._original_hybrid_search is not None:
            from src.modules.chat.core.milvus_service import MilvusService
            svc = MilvusService.get_instance()
            svc.hybrid_search = self._original_hybrid_search
            self._original_hybrid_search = None

    def _patch_graph_toggle(self, enable: bool):
        """Monkey-patch: 切换 NebulaGraph 图查询开关。
        
        通过覆盖 NEBULA_GRAPH_ENABLED 环境变量实现。
        executor._query_graph_context 在每次调用时读取该变量。
        """
        self._original_graph_env = os.getenv("NEBULA_GRAPH_ENABLED")
        os.environ["NEBULA_GRAPH_ENABLED"] = "true" if enable else "false"
        if enable:
            print(f"  [PATCH] 图增强: 已启用 NebulaGraph 商品关系查询\n")
        else:
            print(f"  [PATCH] 图增强: 已禁用 NebulaGraph\n")

    def _restore_graph_toggle(self):
        """恢复 NEBULA_GRAPH_ENABLED 环境变量"""
        if self._original_graph_env is not None:
            os.environ["NEBULA_GRAPH_ENABLED"] = self._original_graph_env
        elif "NEBULA_GRAPH_ENABLED" in os.environ:
            del os.environ["NEBULA_GRAPH_ENABLED"]
        self._original_graph_env = None

    def _filter_cases(self) -> List[EvalCase]:
        """筛选要评估的测试用例"""
        cases = EVALUATION_DATASET
        
        # 按领域筛选
        if self.domain:
            cases = [c for c in cases if c.domain == self.domain]
        
        # 按关键词匹配筛选（--case-name）
        if self.single_case_name:
            keyword = self.single_case_name.strip()
            cases = [c for c in cases if keyword in c.question]
            if not cases:
                print(f"[WARN] 未找到包含 '{self.single_case_name}' 的测试用例")
        
        # 按序号筛选（--case-index, 1-based）
        if self.single_case_index is not None:
            idx = self.single_case_index - 1  # 转为 0-based
            if 0 <= idx < len(cases):
                cases = [cases[idx]]
            else:
                print(f"[WARN] 序号 {self.single_case_index} 超出范围 (1-{len(cases)})，将运行全部用例")
        
        return list(cases)

    async def run_single_case(self, case: EvalCase) -> Dict[str, Any]:
        """
        执行单个测试用例，调用 agent_chat 服务
        
        返回格式：
        {
            "question": "...",
            "answer": "...",           # Agent 生成的回答
            "contexts": ["...", ...],  # 检索到的文档
            "ground_truth": "...",     # 参考答案
            "domain": "...",
            "steps": [...],            # Agent 执行步骤详情
            "duration_ms": 1234,
            "safety_passed": true
        }
        """
        from src.modules.chat.schemas import ChatRequest

        service = await self._get_service()
        
        start = time.time()
        
        request = ChatRequest(
            message=case.question,
            domain=case.domain,
            stream=False
        )
        
        try:
            response = await service.chat_with_agent(request)
            duration_ms = int((time.time() - start) * 1000)
            
            return {
                "question": case.question,
                "answer": response.message,
                "contexts": response.documents_used or [],
                "ground_truth": case.ground_truth,
                "domain": case.domain,
                "steps": [s.get("step_name", "") for s in response.steps],
                "step_details": response.steps,
                "duration_ms": duration_ms,
                "safety_passed": response.safety_passed,
                "conversation_id": response.conversation_id,
                "cache_hit": response.cache_hit,
            }
        except Exception as e:
            import traceback
            duration_ms = int((time.time() - start) * 1000)
            return {
                "question": case.question,
                "answer": "",
                "contexts": [],
                "ground_truth": case.ground_truth,
                "domain": case.domain,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "duration_ms": duration_ms,
                "safety_passed": False,
            }

    @observe(name="ragas-evaluation-run")
    async def run_evaluation(self) -> Dict[str, Any]:
        """
        运行完整评估
        
        Returns:
            评估汇总结果字典
        """
        cases = self._filter_cases()
        reranker_status = "已禁用 (跳过 BGE-Reranker)" if self.skip_reranker else "已启用 (默认)"
        retrieval_label = {"hybrid": "混合检索 (Dense+BM25)", "dense-only": "纯向量检索 (Dense-only 基线)"}.get(self.retrieval_mode, self.retrieval_mode)
        graph_label = "已启用 (NebulaGraph 图增强)" if self.enable_graph else "已禁用 (默认)"
        print(f"\n{'='*70}")
        print(f"  RAGAS 评估 - agent_chat (通用Agent对话)")
        print(f"  领域: {self.domain or '全部'}")
        print(f"  测试用例数: {len(cases)}")
        print(f"  检索模式: {retrieval_label}")
        print(f"  Reranker: {reranker_status}")
        print(f"  图增强: {graph_label}")
        print(f"{'='*70}\n")
        
        # ---- Monkey-patch: 图增强开关 ----
        self._patch_graph_toggle(self.enable_graph)

        # ---- Monkey-patch: 如果需要跳过 Reranker，禁用所有领域配置的 rerank_enabled ----
        if self.skip_reranker:
            self._patch_reranker_config(disable=True)
            print("  [PATCH] 已禁用所有领域配置的 BGE-Reranker\n")
        try:
            # ---------- Step 1: 运行所有测试用例 ----------
            print("[Step 1] 执行测试用例，调用 agent_chat 服务...")
            
            # 预热：触发模型加载、Milvus连接等初始化，避免首条数据计时不准
            print("  [warmup] 预加载 Embedding 模型...", end=" ")
            try:
                await self._get_service()
                service = self.service
            except Exception as we:
                print(f"\n  [FATAL] 无法初始化 ChatAgentService: {we}")
                print(f"  [HINT] 请检查 PyTorch 安装: pip install torch --index-url https://download.pytorch.org/whl/cpu")
                raise
            from src.modules.chat.core.embedding_service import EmbeddingService
            emb_svc = EmbeddingService.get_instance()
            _ = await emb_svc.embed_query("预热")
            print("完成")
            
            print("  [warmup] 预加载 BGE-Reranker 模型...", end=" ")
            from src.modules.chat.core.reranker_service import RerankerService
            reranker = RerankerService.get_instance()
            _ = reranker.rerank(query="预热", documents=["预热文档"])
            print("完成\n")

            # ---- Monkey-patch: 检索模式 (hybrid vs dense-only) ----
            # 必须在 MilvusService 单例初始化后执行，确保 patch 到正确的实例
            self._patch_retrieval_mode(self.retrieval_mode)

            # ── Langfuse: 记录本地模型的名称和配置到 Trace metadata ──
            if self.langfuse_enabled:
                try:
                    _sp = get_current_langfuse_span()
                    if _sp is not None and not isinstance(_sp, type(None)):
                        # Gather model info
                        from src.modules.chat.config import chat_config
                        emb_model = chat_config.embedding_model
                        emb_dim = chat_config.embedding_dim
                        reranker_model = getattr(reranker, '_model_name', reranker.DEFAULT_MODEL)
                        agent_model = os.getenv("CHAT_MODEL", "doubao-pro-251215")
                        judge_model = self.llm_model or agent_model

                        try:
                            _sp.set_attribute("agent.llm_model", agent_model)
                            _sp.set_attribute("agent.llm_provider", "volcengine")
                            _sp.set_attribute("ragas.judge_llm_model", judge_model)
                            _sp.set_attribute("ragas.judge_llm_provider", "volcengine" if os.getenv("VOLCENGINE_API_KEY") else "dashscope")
                            _sp.set_attribute("embedding.model", emb_model)
                            _sp.set_attribute("embedding.provider", "local")
                            _sp.set_attribute("embedding.dim", emb_dim)
                            _sp.set_attribute("reranker.model", reranker_model)
                            _sp.set_attribute("reranker.provider", "local")
                        except Exception:
                            pass
                        print(f"  [Langfuse] 已记录模型配置到 Trace: emb={emb_model}, reranker={reranker_model}, agent={agent_model}")
                except Exception as _mgmt_err:
                    pass
            
            raw_results = []
            for i, case in enumerate(cases):
                print(f"  [{i+1}/{len(cases)}] [{case.domain}] {case.question[:60]}...", end=" ")
                result = await self.run_single_case(case)
                
                if "error" in result:
                    print(f"FAIL: {result['error'][:80]}")
                    result["_effectiveness_local"] = "[X 调用失败]"
                else:
                    contexts_count = len(result["contexts"])
                    answer_len = len(result["answer"])
                    # 提取每步耗时
                    step_times = ", ".join(
                        f"{s.get('step_name', '?')}={s.get('duration_ms', 0)}ms"
                        for s in result.get("step_details", [])
                    )
                    print(f"OK | 总{result['duration_ms']}ms | {step_times} | {contexts_count}ctx {answer_len}字")
                    
                    # 本地快速回答有效性判断
                    qa_flag = _judge_answer_locally(result)
                    result["_effectiveness_local"] = qa_flag
                    print(f"       → {qa_flag}")
                
                raw_results.append(result)
            
            self.results = raw_results
            
            # ---------- Step 2: 构建 RAGAS 数据集 ----------
            print(f"\n[Step 2] 构建 RAGAS 评估数据集...")
            ragas_samples = self._build_ragas_dataset(raw_results)
            
            # ---------- Step 3: 运行 RAGAS 指标（仅有效样本） ----------
            ragas_scores = {}
            ragas_per_verdicts = []
            if len(ragas_samples) == 0:
                print("[WARN] 没有有效的评估样本（可能是所有用例都失败了或无检索结果），跳过 RAGAS 指标")
            else:
                print(f"\n[Step 3] 运行 RAGAS 评估指标...")
                ragas_scores, ragas_per_verdicts = await self._run_ragas_metrics(ragas_samples)
            
            # 将 RAGAS 逐样本裁决合并到 raw_results 中，供报告使用
            if ragas_per_verdicts:
                sample_idx = 0
                for r in raw_results:
                    if "error" not in r and r.get("answer") and r.get("contexts"):
                        if sample_idx < len(ragas_per_verdicts):
                            r["_effectiveness_ragas"] = ragas_per_verdicts[sample_idx]
                            sample_idx += 1
            
            # ---------- Step 4: 本地简化评估（始终执行，含失败用例统计） ----------
            print(f"\n[Step 4] 运行本地简化评估...")
            local_scores = self._run_local_metrics(raw_results)
            
            # ---------- Step 5: 汇总 ----------
            summary = self._build_summary(ragas_scores, local_scores)
            
            # ── Langfuse: 上传评估指标 ──
            if self.langfuse_enabled and self.langfuse_client:
                try:
                    trace_id = get_current_trace_id()
                    span = get_current_langfuse_span()
                    
                    # 更新 span metadata（评估概览信息）
                    if span is not None and not isinstance(span, type(None)):
                        try:
                            span.set_attribute("ragas.domain", self.domain or "all")
                            span.set_attribute("ragas.total_cases", len(cases))
                            span.set_attribute("ragas.success_cases", summary.get("success_cases", 0))
                            span.set_attribute("ragas.error_cases", summary.get("error_cases", 0))
                            span.set_attribute("ragas.reranker", "disabled" if self.skip_reranker else "enabled")
                            span.set_attribute("ragas.retrieval_mode", self.retrieval_mode)
                            span.set_attribute("ragas.graph_enabled", self.enable_graph)
                        except Exception:
                            pass
                    
                    # 上传 RAGAS 语义指标（langfuse v4.x: score() → create_score()）
                    for metric_name, score_value in (ragas_scores or {}).items():
                        if isinstance(score_value, (int, float)):
                            self.langfuse_client.create_score(
                                name=f"ragas.{metric_name}",
                                value=float(score_value),
                                trace_id=trace_id,
                                data_type="NUMERIC",
                                comment="RAGAS LLM-based evaluation metric",
                            )
                    
                    # 上传本地启发式指标（关键维度）
                    _KEY_LOCAL_SCORES = [
                        "avg_faithfulness_local",
                        "avg_answer_relevancy_local",
                        "avg_context_precision_local",
                        "avg_answer_completeness",
                        "avg_retrieval_effectiveness",
                        "avg_latency_ms",
                    ]
                    for key in _KEY_LOCAL_SCORES:
                        val = (local_scores or {}).get(key)
                        if isinstance(val, (int, float)):
                            self.langfuse_client.create_score(
                                name=f"local.{key}",
                                value=float(val),
                                trace_id=trace_id,
                                data_type="NUMERIC",
                                comment="Local heuristic evaluation metric",
                            )
                    
                    print(f"\n  [Langfuse] 已上传 {len(ragas_scores or {}) + 6} 个评估指标到 Trace")
                except Exception as e:
                    print(f"\n  [Langfuse] 上传评分异常（非致命）: {e}")
            
            return summary
        finally:
            if self.skip_reranker:
                self._restore_reranker_config()
            self._restore_retrieval_mode()
            self._restore_graph_toggle()

    def _build_ragas_dataset(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """
        构建 RAGAS SingleTurnSample 格式的数据集
        
        RAGAS 需要: question, answer, contexts, ground_truth
        """
        samples = []
        for r in results:
            if "error" in r:
                continue
            if not r.get("answer") or not r.get("contexts"):
                continue
            
            samples.append({
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r["ground_truth"],
            })
        
        print(f"  构建了 {len(samples)} 个评估样本")
        return samples

    async def _run_ragas_metrics(self, samples: List[Dict]) -> tuple:
        """
        运行 RAGAS 官方评估指标
        
        依赖：pip install ragas
        
        Returns:
            (avg_scores: dict, per_sample_verdicts: list)
        """
        scores = {}
        per_sample_verdicts = []
        
        # ---- 修复 ragas 与 langchain-community 0.4.x 的兼容性问题 ----
        # ragas 在 llms/base.py 中硬编码了 from langchain_community.chat_models.vertexai import ChatVertexAI
        # 但 langchain-community 0.4.x 已将该模块移至 langchain-google-vertexai 独立包
        # 这里做 monkey-patch，在 ragas 导入前将 ChatVertexAI 注入到旧路径
        import importlib
        try:
            from langchain_google_vertexai import ChatVertexAI as _RealChatVertexAI
            _ = importlib.import_module('langchain_community.chat_models')  # 确保父包已加载
            import sys
            class _FakeVertexAI:
                ChatVertexAI = _RealChatVertexAI
                VertexAI = None  # ragas 同时导入 llms.VertexAI，但该模块缺失
            sys.modules['langchain_community.chat_models.vertexai'] = _FakeVertexAI
            print("  [PATCH] 已将 langchain_google_vertexai.ChatVertexAI 桥接到 langchain_community.chat_models.vertexai")
        except Exception as _patch_err:
            print(f"  [WARN] monkey-patch 失败: {_patch_err}")
        
        # ---- 兼容 ragas 多版本 API ----
        Faithfulness = AnswerRelevancy = None
        ContextPrecision = ContextRecall = ContextRelevancy = None
        AnswerCorrectness = None
        llm_factory = None
        
        try:
            # ragas 0.4.x 推荐路径: from ragas.metrics.collections import ...
            from ragas.metrics.collections import (
                Faithfulness, AnswerRelevancy,
                ContextPrecision, ContextRecall,
                AnswerCorrectness,
            )
            # ContextRelevancy 在 0.4.3 中重命名为 ContextRelevance
            try:
                from ragas.metrics.collections import ContextRelevancy
            except ImportError:
                from ragas.metrics.collections import ContextRelevance as ContextRelevancy
        except ImportError as e2:
            # 回退: ragas 0.2.x/0.3.x: from ragas.metrics import ...
            try:
                from ragas.metrics import (
                    Faithfulness, AnswerRelevancy,
                    ContextPrecision, ContextRecall,
                    ContextRelevancy, AnswerCorrectness,
                )
            except ImportError:
                # ragas 0.1.x: 函数式 API
                try:
                    import ragas.metrics as rm
                    Faithfulness = rm.Faithfulness
                    AnswerRelevancy = rm.AnswerRelevancy
                    ContextPrecision = rm.ContextPrecision
                    ContextRecall = rm.ContextRecall
                    ContextRelevancy = getattr(rm, 'ContextRelevancy', getattr(rm, 'ContextRelevance', None))
                    AnswerCorrectness = rm.AnswerCorrectness
                except Exception:
                    print(f"  [WARN] 无法导入 ragas metrics (0.1.x/0.2.x/0.3.x/0.4.x 均失败): {e2}")
        
        try:
            from ragas.llms import llm_factory
        except ImportError as e3:
            try:
                from ragas.llms.llm import llm_factory
            except ImportError:
                print(f"  [WARN] 无法导入 ragas llm_factory: {e3}")
        
        # 检查关键组件是否都就绪（ragas 0.4.x 用 llm_factory 替代 LangchainLLMWrapper）
        if not all([Faithfulness, llm_factory]):
            missing = []
            if not Faithfulness: missing.append("Faithfulness")
            if not llm_factory: missing.append("llm_factory")
            print(f"  [WARN] ragas 关键组件缺失: {', '.join(missing)}")
            print(f"  [WARN] 请确认 ragas 版本，或运行: pip install -U ragas")
            print(f"  [WARN] 将使用本地启发式评估替代")
            return scores, per_sample_verdicts
        
        try:
            # ---- 确定 LLM 与 API 端点 ----
            judge_model = (
                self.llm_model or
                os.getenv("RAGAS_LLM_MODEL")
            )
            api_key = (
                os.getenv("TONGYI_API_KEY")
            )
            
            # 确定 API base_url
            if api_key:
                base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
                provider_label = "DashScope OpenAI兼容"
            else:
                api_key = os.getenv("VOLCENGINE_API_KEY", "")
                base_url = os.getenv("VOLCENGINE_LLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
                judge_model = os.getenv("CHAT_MODEL", "doubao-pro-251215")
                provider_label = "火山引擎"
            
            # ---- 创建 AsyncOpenAI 客户端（ragas 0.4.x ascore 需要异步客户端） ----
            # ★ Langfuse 追踪：优先使用 langfuse.openai.AsyncOpenAI 自动 trace
            if LANGFUSE_AVAILABLE and self.langfuse_enabled:
                from langfuse.openai import AsyncOpenAI
                _lf_instrumented = True
            else:
                from openai import AsyncOpenAI
                _lf_instrumented = False

            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            # ★ No-Think: AsyncOpenAI 构造器不支持 extra_body，通过 monkey-patch 注入
            _original_create = client.chat.completions.create

            async def _no_think_create(**kwargs):
                if "extra_body" not in kwargs:
                    kwargs["extra_body"] = {"enable_thinking": False}
                return await _original_create(**kwargs)

            client.chat.completions.create = _no_think_create
            
            if _lf_instrumented:
                print(f"  [Langfuse] RAGAS LLM 调用将自动 trace 到当前 Trace")
            
            # ---- 评判 LLM（ragas 0.4.x 现代 API） ----
            # ★ max_tokens=8096
            evaluator_llm = llm_factory(judge_model, client=client, max_tokens=8096)
            print(f"  [RAGAS] 评判模型: {judge_model} ({provider_label})")
            # ── Token 消耗估算 ──
            _n_samples = len(samples)
            _metrics_call_estimates = {
                "Faithfulness": 2, "AnswerRelevancy": 1, "ContextRecall": 1,
            }
            _total_calls = sum(_metrics_call_estimates.values()) * _n_samples
            _calls_per_sample = sum(_metrics_call_estimates.values())
            _est_prompt = _total_calls * 1800  # 截断后每调用 ~1800 prompt tokens
            _est_completion = _total_calls * 250
            _est_cost_plus = (_est_prompt * 2.0 + _est_completion * 8.0) / 1e6
            _est_cost_turbo = (_est_prompt * 0.3 + _est_completion * 0.6) / 1e6
            print(f"  [RAGAS] 预计 LLM 调用: ~{_total_calls} 次 ({_n_samples}样本×{_calls_per_sample}次/样本)")
            print(f"  [RAGAS] 预估 token: ~{_est_prompt//1000}K prompt + ~{_est_completion//1000}K completion")
            print(f"  [RAGAS] 预估费用: 当前模型 ¥{_est_cost_plus:.4f} | qwen-turbo ¥{_est_cost_turbo:.4f}")
            
            # ---- 嵌入模型（AnswerRelevancy / AnswerCorrectness 需要） ----
            embedding_model = os.getenv("RAGAS_EMBEDDING_MODEL") or "text-embedding-v3"
            # 如果是 HuggingFace 模型名（含 /），走本地加载，不走远程 API
            if "/" in embedding_model:
                from ragas.embeddings import HuggingFaceEmbeddings as RagasHfEmbeddings
                ragas_embeddings = RagasHfEmbeddings(
                    model=embedding_model,
                    use_api=False,            # 本地 sentence-transformers
                    normalize_embeddings=True,
                )
                print(f"  [RAGAS] 嵌入模型: {embedding_model} (本地 HuggingFace)")
            else:
                from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
                ragas_embeddings = RagasOpenAIEmbeddings(client=client, model=embedding_model)
                print(f"  [RAGAS] 嵌入模型: {embedding_model} ({provider_label})")
            
        except Exception as e:
            print(f"  [WARN] 无法初始化评判 LLM: {e}")
            print(f"  [WARN] 将使用本地启发式评估替代")
            return {}, per_sample_verdicts

        try:
            # ── 3 个核心 RAGAS 指标 ──
            metrics = [
                Faithfulness(llm=evaluator_llm),
                AnswerRelevancy(llm=evaluator_llm, embeddings=ragas_embeddings, strictness=1),
                ContextRecall(llm=evaluator_llm),
            ]
            print(f"  [RAGAS] 指标: Faithfulness + AnswerRelevancy + ContextRecall")
            
            # ragas 0.4.x 中各指标 ascore 所需 kwargs 不同
            # {metric_class_base_name: lambda sd: kwargs_dict}
            #
            # ★ 截断策略优化 token 消耗：
            #   - 回答截断: Faithfulness/AnswerCorrectness 需逐句分析，长回答(>800字)
            #     的 NLI JSON 输出可达数千 token。截断到 800 字既保留足够信息量，
            #     又避免 max_tokens 耗尽。
            #   - 上下文截断: ContextPrecision 逐条调用 LLM（5 条=5 次），每条上下文
            #     截断到 300 字可大幅减少 prompt 体积（每条省 ~200 token × 5 = 1000 token/样本）。
            _RESPONSE_TRUNCATE = 800
            _CTX_TRUNCATE = 300
            
            def _truncate_response(sd: dict) -> str:
                ans = sd.get("answer") or ""
                if len(ans) > _RESPONSE_TRUNCATE:
                    return ans[:_RESPONSE_TRUNCATE] + "…(已截断)"
                return ans
            
            def _truncate_contexts(sd: dict) -> list:
                ctxs = sd.get("contexts") or []
                return [
                    (c[:_CTX_TRUNCATE] + "…") if len(c) > _CTX_TRUNCATE else c
                    for c in ctxs
                ]
            
            _ASCORE_BUILDERS = {
                "Faithfulness":       lambda sd: dict(user_input=sd["question"], response=_truncate_response(sd), retrieved_contexts=_truncate_contexts(sd)),
                "AnswerRelevancy":    lambda sd: dict(user_input=sd["question"], response=_truncate_response(sd)),
                "ContextPrecision":   lambda sd: dict(user_input=sd["question"], reference=sd["ground_truth"], retrieved_contexts=_truncate_contexts(sd)),
                "ContextRecall":      lambda sd: dict(user_input=sd["question"], retrieved_contexts=_truncate_contexts(sd), reference=sd["ground_truth"]),
                "ContextRelevancy":   lambda sd: dict(user_input=sd["question"], retrieved_contexts=_truncate_contexts(sd)),
                "ContextRelevance":   lambda sd: dict(user_input=sd["question"], retrieved_contexts=_truncate_contexts(sd)),
                "AnswerCorrectness":  lambda sd: dict(user_input=sd["question"], response=_truncate_response(sd), reference=sd["ground_truth"]),
            }
            
            # 每样本各指标并行评估（6 个 ascore → 1 次网络往返）
            # ★ 使用 contextvars 确保 OpenTelemetry trace context 传播到 asyncio.gather 子任务
            #    否则 langfuse.openai.AsyncOpenAI 拿不到当前 trace span，token 消耗不可见
            import contextvars
            
            _metric_idx_map = {m.__class__.__name__: (j+1, len(metrics)) for j, m in enumerate(metrics)}
            
            async def _score_one(sd: dict, metric):
                metric_cls_name = metric.__class__.__name__
                builder = _ASCORE_BUILDERS.get(metric_cls_name)
                if builder is None:
                    return metric, None, f"{metric_cls_name}: SKIP"
                kwargs = builder(sd)
                _m_idx, _m_total = _metric_idx_map[metric_cls_name]
                _q_short = sd["question"][:40]
                print(f"      [{_m_idx}/{_m_total}] {metric.name} 开始... (\"{_q_short}...\")")
                import time; _t0 = time.time()
                try:
                    result = await metric.ascore(**kwargs)
                    _elapsed = time.time() - _t0
                    print(f"      [{_m_idx}/{_m_total}] {metric.name} -> {result.value:.4f} (耗时 {_elapsed:.1f}s)")
                    return metric, result.value, None
                except Exception as e:
                    _elapsed = time.time() - _t0
                    print(f"      [{_m_idx}/{_m_total}] {metric.name} -> ERROR (耗时 {_elapsed:.1f}s): {str(e)[:200]}")
                    raise

            n_metrics = len(metrics)
            metric_names = [m.name for m in metrics]
            print(f"  [RAGAS] 每样本评估 {n_metrics} 个指标: {', '.join(metric_names)}")
            
            for i, sample_data in enumerate(samples):
                print(f"  [{i+1}/{len(samples)}] 评估: \"{sample_data['question'][:60]}...\"")
                print(f"      指标并行执行中...")
                
                # 捕获当前 context（含 OTel span），传入 asyncio.gather 子任务
                ctx = contextvars.copy_context()
                tasks = [ctx.run(_score_one, sample_data, m) for m in metrics]
                results_list = await asyncio.gather(*tasks, return_exceptions=True)
                
                print(f"      -- 样本 [{i+1}/{len(samples)}] 所有指标完成 --")
                
                sample_metrics = {}
                for item in results_list:
                    if isinstance(item, BaseException):
                        import traceback as _tb
                        print(f"    task: ERROR - {str(item)[:300]}")
                        _tb.print_exception(type(item), item, item.__traceback__)
                        continue
                    metric, score_value, err = item
                    metric_name = metric.name
                    if err:
                        print(f"    {metric_name}: {err}")
                        continue
                    scores[metric_name] = scores.get(metric_name, []) + [score_value]
                    sample_metrics[metric_name] = score_value
                
                # 按固定顺序打印
                for m in metrics:
                    v = sample_metrics.get(m.name)
                    if v is not None:
                        print(f"    {m.name}: {v:.4f}")
                
                # ---- Faithfulness 调试：打印答案和上下文，帮助诊断 0% 原因 ----
                faithfulness_score = sample_metrics.get("faithfulness", None)
                if faithfulness_score is not None and faithfulness_score == 0.0:
                    print(f"    [DEBUG faithfulness=0] 问题: {sample_data['question'][:80]}")
                    print(f"    [DEBUG faithfulness=0] 答案(前200字): {sample_data['answer'][:200]}")
                    print(f"    [DEBUG faithfulness=0] 答案总长: {len(sample_data['answer'])} 字")
                    print(f"    [DEBUG faithfulness=0] 上下文数: {len(sample_data['contexts'])}")
                    for ci, ctx in enumerate(sample_data['contexts'][:3]):
                        print(f"    [DEBUG faithfulness=0]   上下文[{ci}](前100字): {ctx[:100]}")
                # ------------------------------------------------------------
                
                # 单条样本回答有效性判断（基于 RAGAS 语义指标）
                verdict = _judge_ragas_effectiveness(sample_metrics)
                per_sample_verdicts.append(verdict)
                print(f"    >>> {verdict}")
            
            # 计算平均分
            avg_scores = {
                name: sum(vals) / len(vals) if vals else 0.0
                for name, vals in scores.items()
            }
            scores = avg_scores
            
        except ImportError:
            print("  [WARN] ragas 未安装或版本不兼容，请运行: pip install -U ragas")
            print("  [WARN] 将使用本地启发式评估替代")
        except Exception as e:
            import traceback as _tb
            print(f"  [WARN] RAGAS 评估异常: {e}")
            print(f"  [WARN] 详细信息: {_tb.format_exc()[-400:]}")
            print(f"  [WARN] 将使用本地启发式评估替代")
        
        return scores, per_sample_verdicts

    def _run_local_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """
        本地启发式评估（不依赖外部 RAGAS / LLM）

        评估维度：
        - context_precision_local: 上下文关键词命中率
        - answer_relevancy_local: 答案与问题的关键词重叠率
        - faithfulness_local: 答案是否基于上下文（简单关键词匹配法）
        - answer_completeness: 答案是否覆盖了参考答案的关键信息
        - retrieval_effectiveness: 检索是否有返回结果
        - safety_check_rate: 安全审查是否正常执行
        - avg_latency_ms: 平均响应时间
        """
        import re
        from collections import Counter

        metrics = {
            "total_cases": len(results),
            "success_cases": 0,
            "error_cases": 0,
            "cache_hits": 0,
            "context_precision_local": [],
            "answer_relevancy_local": [],
            "faithfulness_local": [],
            "answer_completeness": [],
            "retrieval_effectiveness": [],
            "latency_ms": [],
            "safety_passed": 0,
            "steps_completed": [],
            # 每步耗时（按 step_name 分组）
            "step_durations": {},  # {step_name: [duration_ms, ...]}
        }

        def tokenize(text: str) -> set:
            """中文分词（优先使用 jieba，回退到字符 n-gram）"""
            try:
                import jieba
                # jieba 精确模式分词，过滤标点和空白
                words = jieba.lcut(text)
                # 过滤单字（无意义）、空白、标点
                stop_chars = set('，。！？、；：""''【】（）\s\n\r\t')
                tokens = set(w for w in words if len(w) >= 2 and w not in stop_chars and not w.isspace())
                if tokens:
                    return tokens
            except ImportError:
                pass
            
            # 回退：字符级 n-gram
            text = re.sub(r'[，。！？、；：""''【】（）\s\n\r]', ' ', text)
            tokens = set()
            for chunk in text.split():
                for n in range(1, min(4, len(chunk) + 1)):
                    for i in range(len(chunk) - n + 1):
                        tokens.add(chunk[i:i+n])
            return tokens if tokens else set(text)

        for r in results:
            if "error" in r:
                metrics["error_cases"] += 1
                continue

            metrics["success_cases"] += 1
            
            if r.get("cache_hit"):
                metrics["cache_hits"] += 1

            question = r["question"]
            answer = r["answer"]
            contexts = r["contexts"] or []
            ground_truth = r["ground_truth"]
            duration = r.get("duration_ms", 0)
            steps = r.get("steps", [])

            # 响应时间
            metrics["latency_ms"].append(duration)

            # 步骤完成情况
            metrics["steps_completed"].append(len(steps))

            # 每步耗时收集
            for s in r.get("step_details", []):
                step_name = s.get("step_name", "unknown")
                dur = s.get("duration_ms", 0)
                if step_name not in metrics["step_durations"]:
                    metrics["step_durations"][step_name] = []
                metrics["step_durations"][step_name].append(dur)

            # 安全审查
            if r.get("safety_passed"):
                metrics["safety_passed"] += 1

            # ---- Context Precision (本地) ----
            # 判断 context 是否与问题相关
            if contexts:
                question_tokens = tokenize(question)
                context_scores = []
                for ctx in contexts:
                    ctx_lower = ctx.lower() if ctx else ""
                    # 简单评判：context 中的 q_token 覆盖度
                    ctx_tokens = tokenize(ctx_lower)
                    overlap = len(question_tokens & ctx_tokens) / max(len(question_tokens), 1)
                    context_scores.append(overlap)
                
                # 平均上下文相关性
                avg_context_score = sum(context_scores) / len(context_scores) if context_scores else 0
                metrics["context_precision_local"].append(avg_context_score)
                
                # 检索有效性：有检索结果
                metrics["retrieval_effectiveness"].append(1.0)
            else:
                metrics["context_precision_local"].append(0.0)
                metrics["retrieval_effectiveness"].append(0.0)

            # ---- Answer Relevancy (本地) ----
            q_tokens = tokenize(question)
            a_tokens = tokenize(answer) if answer else set()
            overlapped = len(q_tokens & a_tokens)
            relevancy = overlapped / max(len(q_tokens), 1)
            metrics["answer_relevancy_local"].append(min(relevancy, 1.0))

            # ---- Faithfulness (本地) ----
            # 判断答案中的关键声明是否能在上下文中找到
            if contexts and answer:
                a_tokens_set = tokenize(answer)
                all_ctx_text = " ".join(contexts)
                ctx_tokens_set = tokenize(all_ctx_text)
                if a_tokens_set:
                    faith = len(a_tokens_set & ctx_tokens_set) / len(a_tokens_set)
                    metrics["faithfulness_local"].append(min(faith, 1.0))
                else:
                    metrics["faithfulness_local"].append(0.0)
            else:
                metrics["faithfulness_local"].append(0.0)

            # ---- Answer Completeness / Correctness (本地) ----
            gt_tokens = tokenize(ground_truth)
            a_tokens_full = tokenize(answer) if answer else set()
            completeness = len(gt_tokens & a_tokens_full) / max(len(gt_tokens), 1)
            metrics["answer_completeness"].append(min(completeness, 1.0))

        # 计算均值（空列表 → 0.0，避免下游 N/A）
        summary = {}
        for key in ["context_precision_local", "answer_relevancy_local",
                     "faithfulness_local", "answer_completeness",
                     "retrieval_effectiveness", "latency_ms", "steps_completed"]:
            vals = metrics[key]
            summary[f"avg_{key}"] = round(sum(vals) / len(vals), 4) if vals else 0.0
            summary[f"min_{key}"] = round(min(vals), 4) if vals else 0.0
            summary[f"max_{key}"] = round(max(vals), 4) if vals else 0.0

        summary["total_cases"] = metrics["total_cases"]
        summary["success_cases"] = metrics["success_cases"]
        summary["error_cases"] = metrics["error_cases"]
        summary["cache_hits"] = metrics["cache_hits"]
        summary["safety_passed_rate"] = round(
            metrics["safety_passed"] / max(metrics["success_cases"], 1), 4
        )

        # 每步平均耗时
        summary["avg_step_durations"] = {}
        for step_name, durs in metrics["step_durations"].items():
            valid_durs = [d for d in durs if d is not None]
            summary["avg_step_durations"][step_name] = {
                "avg_ms": round(sum(valid_durs) / len(valid_durs), 1) if valid_durs else 0.0,
                "min_ms": round(min(valid_durs), 1) if valid_durs else 0.0,
                "max_ms": round(max(valid_durs), 1) if valid_durs else 0.0,
            }

        return summary

    def _build_summary(self, ragas_scores=None, local_scores=None) -> Dict[str, Any]:
        """构建最终评估汇总"""
        local_scores = local_scores or {}
        
        summary = {
            "evaluation_target": "agent_chat (POST /api/chatagent/agent/chat)",
            "description": "通用Agent多步骤对话 RAGAS 评估",
            "pipeline": [
                "步骤1: 问题改写 (step1_understand)",
                "步骤2: 安全审查 (step2_review)",
                "步骤3: 知识检索 (step3_retrieve, Milvus Hybrid Search)",
                "步骤4: 答案生成 (step4_generate)",
            ],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "domain_filter": self.domain or "all",
            "total_cases": local_scores.get("total_cases", 0),
            "success_cases": local_scores.get("success_cases", 0),
            "error_cases": local_scores.get("error_cases", 0),

            # RAGAS 官方指标（如果可用）
            "ragas_metrics": ragas_scores or {},

            # 本地启发式指标
            "local_metrics": {
                "avg_context_precision": local_scores.get("avg_context_precision_local", "N/A"),
                "avg_answer_relevancy": local_scores.get("avg_answer_relevancy_local", "N/A"),
                "avg_faithfulness": local_scores.get("avg_faithfulness_local", "N/A"),
                "avg_answer_completeness": local_scores.get("avg_answer_completeness", "N/A"),
                "avg_retrieval_effectiveness": local_scores.get("avg_retrieval_effectiveness", "N/A"),
                "avg_latency_ms": local_scores.get("avg_latency_ms", "N/A"),
                "avg_steps_completed": local_scores.get("avg_steps_completed", "N/A"),
                "safety_passed_rate": local_scores.get("safety_passed_rate", "N/A"),
                "cache_hit_count": local_scores.get("cache_hits", 0),
                "avg_step_durations": local_scores.get("avg_step_durations", {}),
            },

            # 详细结果（前3条）
            "sample_results": [
                {
                    "question": r["question"],
                    "answer_preview": (r.get("answer", "") or "")[:200],
                    "contexts_count": len(r.get("contexts", []) or []),
                    "domain": r.get("domain", ""),
                    "duration_ms": r.get("duration_ms", 0),
                    "safety_passed": r.get("safety_passed", False),
                    "effectiveness_local": r.get("_effectiveness_local", ""),
                    "effectiveness_ragas": r.get("_effectiveness_ragas", ""),
                    "error": r.get("error", ""),
                    "step_times": ", ".join(
                        f"{s.get('step_name', '?')}={s.get('duration_ms', 0)}ms"
                        for s in (r.get("step_details", []) or [])
                    ),
                }
                for r in (self.results or [])[:3]
            ],
        }

        return summary


# =============================================================================
# 回答有效性判断辅助函数
# =============================================================================

def _judge_answer_locally(result: Dict) -> str:
    """本地启发式回答有效性判断（Step 1 阶段，无需 RAGAS）"""
    answer = result.get("answer", "") or ""
    contexts = result.get("contexts", []) or []
    
    if not answer:
        return "[- 无效] 无回答内容"
    if len(answer) < 20:
        return f"[- 无效] 答案过短({len(answer)}字)，可能生成失败"
    if not contexts:
        return "[! 无检索] 无上下文，答案可能来自模型通用知识"
    if any(w in answer[:80] for w in ["抱歉", "无法", "不能", "暂无相关信息"]):
        return "[! 拒绝型] 模型未给出实质回答"
    
    # 基础有效：有上下文且有足够长度的答案
    c_count = len(contexts)
    a_len = len(answer)
    if a_len > 200:
        return f"[+ 初步有效] {c_count}条上下文, {a_len}字回答 — 等待RAGAS语义评估"
    else:
        return f"[! 偏短] {c_count}条上下文, 仅{a_len}字 — 可能不够详尽"


def _normalize_ragas_keys(scores: Dict[str, float]) -> Dict[str, float]:
    """统一 RAGAS 指标 key 名到小写（兼容 metric.name 可能返回大小写不同的情况）"""
    if not scores:
        return {}
    # 已知的 key 映射（RAGAS metric.name 可能返回 "faithfulness" 或 "Faithfulness"）
    return {
        k.lower().replace(" ", "_").replace("-", "_"): v
        for k, v in scores.items()
    }


def _judge_ragas_effectiveness(sample_scores: Dict[str, float]) -> str:
    """基于 RAGAS 语义指标判断单条回答是否有效"""
    if not sample_scores:
        return "[? 未知] 无 RAGAS 评分"
    
    scores = _normalize_ragas_keys(sample_scores)
    faithfulness = scores.get("faithfulness", 0)
    answer_relevancy = scores.get("answer_relevancy", 0)
    context_precision = scores.get("context_precision", 0)
    answer_correctness = scores.get("answer_correctness", 0)
    
    # 综合评判
    if faithfulness >= 0.7 and answer_relevancy >= 0.6:
        extra = ""
        if answer_correctness >= 0.6:
            extra = f", 正确性{answer_correctness:.0%}"
        return f"[+ 有效] 忠实度{faithfulness:.0%}, 相关性{answer_relevancy:.0%}{extra}"
    
    if faithfulness >= 0.5 and answer_relevancy >= 0.4:
        parts = []
        if faithfulness < 0.7:
            parts.append(f"忠实度偏低({faithfulness:.0%})")
        if answer_relevancy < 0.6:
            parts.append(f"相关性偏弱({answer_relevancy:.0%})")
        return f"[! 基本有效] {'; '.join(parts)}"
    
    if faithfulness < 0.5 and answer_relevancy >= 0.5:
        return f"[! 可能有幻觉] 忠实度仅{faithfulness:.0%}, 相关内容未被良好利用"
    
    if faithfulness >= 0.5 and answer_relevancy < 0.4:
        return f"[! 偏题] 相关性仅{answer_relevancy:.0%}, 回答可能未聚焦问题"
    
    # 两者都低
    if faithfulness < 0.3:
        return f"[- 无效] 严重幻觉(忠实度{faithfulness:.0%}), 回答不可信"
    return f"[- 无效] 查询-答案关联弱(忠实度{faithfulness:.0%}, 相关性{answer_relevancy:.0%})"


def _print_effectiveness_summary(ragas: Dict[str, float]):
    """在报告中打印回答有效性总览"""
    scores = _normalize_ragas_keys(ragas)
    
    # 诊断：打印实际可用的指标 key，方便排查
    if not scores:
        print(f"    (RAGAS 指标不可用，无法判断)")
        if ragas:
            print(f"    [提示] RAGAS 返回的原始 key: {list(ragas.keys())}")
        return
    
    faithfulness = scores.get("faithfulness", 0)
    answer_relevancy = scores.get("answer_relevancy", 0)
    answer_correctness = scores.get("answer_correctness", 0)
    context_precision = scores.get("context_precision", 0)
    
    if faithfulness == 0 and answer_relevancy == 0:
        print(f"    (RAGAS 指标不可用，无法判断)")
        print(f"    [提示] 可用的 RAGAS keys: {list(scores.keys())}")
        return
    
    # 判断来源
    if faithfulness >= 0.7:
        f_status = "优"
    elif faithfulness >= 0.5:
        f_status = "中"
    else:
        f_status = "差"
    
    if answer_relevancy >= 0.6:
        r_status = "优"
    elif answer_relevancy >= 0.4:
        r_status = "中"
    else:
        r_status = "差"
    
    print(f"    忠实度 (faithfulness):      {faithfulness:.2%}  [{f_status}] — 答案是否基于上下文")
    print(f"    答案相关性 (relevancy):     {answer_relevancy:.2%}  [{r_status}] — 答案是否切题")
    print(f"    上下文精确度 (precision):   {context_precision:.2%}  — 检索文档中有用比例")
    
    # 综合结论
    overall = faithfulness * 0.4 + answer_relevancy * 0.35 + answer_correctness * 0.25
    if overall >= 0.65:
        print(f"    >>> 综合结论: ✅ 回答整体有效 (综合分 {overall:.2%})")
    elif overall >= 0.45:
        print(f"    >>> 综合结论: ⚠️ 回答基本可用，部分维度需优化 (综合分 {overall:.2%})")
    else:
        print(f"    >>> 综合结论: ❌ 回答效果不理想，建议排查 (综合分 {overall:.2%})")


def _print_local_effectiveness_summary(lm: Dict):
    """RAGAS 不可用时，基于本地启发式指标输出回答有效性判断"""
    avg_faith = lm.get("avg_faithfulness", 0) or 0
    avg_relevancy = lm.get("avg_answer_relevancy", 0) or 0
    avg_completeness = lm.get("avg_answer_completeness", 0) or 0
    
    if avg_faith == 0 and avg_relevancy == 0:
        print(f"    (本地指标也不可用，无法判断)")
        return
    
    # 本地指标阈值（比 RAGAS 宽松，因为关键词匹配不如语义匹配精准）
    if avg_faith > 0.4:
        f_status = "优"
    elif avg_faith > 0.15:
        f_status = "中"
    else:
        f_status = "差"
    
    if avg_relevancy > 0.3:
        r_status = "优"
    elif avg_relevancy > 0.1:
        r_status = "中"
    else:
        r_status = "差"
    
    print(f"    忠实度(本地):       {avg_faith:.2%}  [{f_status}] — 答案关键词在上下文中的覆盖度")
    print(f"    答案相关性(本地):   {avg_relevancy:.2%}  [{r_status}] — 答案与问题关键词重叠率")
    print(f"    答案完整度(本地):   {avg_completeness:.2%}  — 答案覆盖参考答案关键信息比例")
    
    overall = avg_faith * 0.4 + avg_relevancy * 0.35 + avg_completeness * 0.25
    if overall >= 0.35:
        print(f"    >>> 综合结论: ✅ 回答初步有效 (本地综合分 {overall:.2%})")
        print(f"    >>> 注意: 本地判断基于关键词匹配，建议安装 ragas 进行语义级评估")
    elif overall >= 0.15:
        print(f"    >>> 综合结论: ⚠️ 回答基本可用，部分维度需优化 (本地综合分 {overall:.2%})")
    else:
        print(f"    >>> 综合结论: ❌ 回答效果不理想，建议排查 (本地综合分 {overall:.2%})")


# =============================================================================
# 报告生成
# =============================================================================

def print_evaluation_report(summary: Dict[str, Any]):
    """打印美化的评估报告"""
    print(f"\n{'='*70}")
    print(f"  [RAGAS] 评估报告 - agent_chat")
    print(f"{'='*70}")
    
    # 基本信息
    print(f"\n[基本信息]")
    print(f"  评估目标: {summary['evaluation_target']}")
    print(f"  评估时间: {summary['timestamp']}")
    print(f"  领域筛选: {summary['domain_filter']}")
    
    # 执行统计
    lm = summary.get("local_metrics", {})
    print(f"\n[执行统计]")
    print(f"  总用例数: {summary['total_cases']}")
    print(f"  成功: {summary['success_cases']} | 失败: {summary['error_cases']}")
    print(f"  缓存命中: {summary.get('local_metrics', {}).get('cache_hit_count', 0)}")
    print(f"  安全审查通过率: {lm.get('safety_passed_rate', 'N/A')}")
    
    # 核心指标
    print(f"\n[核心评估指标]")
    print(f"  {'指标':<30} {'得分':>10} {'说明'}")
    print(f"  {'-'*60}")
    
    # RAGAS 指标（如果可用）
    ragas = summary.get("ragas_metrics", {})
    if ragas:
        print(f"\n  [RAGAS 官方指标（LLM评判）]")
        metric_labels = [
            ("faithfulness", "忠实度", "答案是否忠实于上下文"),
            ("answer_relevancy", "答案相关性", "答案与问题的相关度"),
            ("context_precision", "上下文精确度", "检索结果排序精确度"),
            ("context_recall", "上下文召回率", "相关文档召回率"),
            ("context_relevancy", "上下文相关性", "上下文与问题的相关度"),
            ("answer_correctness", "答案正确性", "答案与参考答案的一致度"),
        ]
        for key, label, desc in metric_labels:
            # 兼容 RAGAS key 名大小写差异（不能用 or 短路，0.0 是合法值）
            score = ragas.get(key)
            if score is None:
                score = ragas.get(key.capitalize())
            if score is None:
                score = ragas.get(key.title())
            # 额外兼容 "context_relevance" → "context_relevancy" 的拼写差异
            if score is None and key == "context_relevancy":
                score = ragas.get("context_relevance")
            if score is not None:
                print(f"  {label:<28} {score:>10.4f}  {desc}")
            else:
                print(f"  {label:<28} {'N/A':>10}  {desc}")
    
    # 回答有效性判断 —— 始终打印（有 RAGAS 用 RAGAS，否则用本地指标兜底）
    print(f"\n  [回答有效性判断]")
    if ragas:
        _print_effectiveness_summary(ragas)
    else:
        # RAGAS 不可用时，基于本地启发式指标给出判断
        _print_local_effectiveness_summary(lm)

    # 本地指标
    print(f"\n  [本地启发式指标]")
    local_metric_labels = [
        ("avg_context_precision", "上下文精确度(本地)", "检索上下文与问题的关键词重叠率"),
        ("avg_answer_relevancy", "答案相关性(本地)", "答案与问题的关键词重叠率"),
        ("avg_faithfulness", "忠实度(本地)", "答案内容在上下文中的存在比例"),
        ("avg_answer_completeness", "答案完整度(本地)", "答案覆盖参考答案关键信息的比例"),
        ("avg_retrieval_effectiveness", "检索有效性", "是否有检索结果返回"),
        ("avg_latency_ms", "平均响应时间(ms)", "从请求到响应的平均耗时"),
        ("avg_steps_completed", "平均步骤数", "Agent pipeline 完成步骤数"),
    ]
    for key, label, desc in local_metric_labels:
        score = lm.get(key)
        if score is not None:
            if isinstance(score, float):
                print(f"  {label:<28} {score:>10.4f}  {desc}")
            else:
                print(f"  {label:<28} {score:>10}  {desc}")
        else:
            print(f"  {label:<28} {'N/A':>10}  {desc}")

    # 每步耗时明细
    print(f"\n  [步骤耗时明细]")
    step_durations = lm.get("avg_step_durations", {})
    if step_durations:
        print(f"  {'步骤':<28} {'平均':>7} {'最小':>7} {'最大':>7}")
        print(f"  {'-'*52}")
        for step_name, dur_info in sorted(step_durations.items()):
            step_short = step_name[:26]
            print(f"  {step_short:<28} {dur_info['avg_ms']:>6.0f}ms {dur_info['min_ms']:>6.0f}ms {dur_info['max_ms']:>6.0f}ms")
    else:
        print(f"  (无步骤耗时数据)")

    # 解读
    print(f"\n[评估解读]")
    _print_interpretation(lm, ragas)

    # 示例结果
    print(f"\n[示例评估结果 (前 3 条)]")
    for i, sample in enumerate(summary.get("sample_results", [])):
        print(f"\n  [{i+1}] Q: {sample['question']}")
        print(f"       A: {sample['answer_preview']}...")
        print(f"       Contexts: {sample['contexts_count']} 条")
        print(f"       Duration: {sample['duration_ms']}ms")
        if sample.get("step_times"):
            print(f"       Steps: {sample['step_times']}")
        if sample.get("effectiveness_local"):
            print(f"       本地判断: {sample['effectiveness_local']}")
        if sample.get("effectiveness_ragas"):
            print(f"       RAGAS判断: {sample['effectiveness_ragas']}")
        if sample.get("error"):
            print(f"       Error: {sample['error'][:100]}")

    print(f"\n{'='*70}")


def _print_interpretation(lm: Dict, ragas: Dict):
    """打印指标解读"""
    # Context Precision
    cp = lm.get("avg_context_precision", 0)
    if isinstance(cp, (int, float)):
        if cp > 0.5:
            print(f"   [PASS] 检索质量良好 - 上下文与问题的平均关键词重叠率 {cp:.2%}")
        elif cp > 0.2:
            print(f"   [WARN] 检索质量一般 - 上下文与问题的平均关键词重叠率仅 {cp:.2%}")
        else:
            print(f"   [FAIL] 检索质量较差 - 上下文与问题的平均关键词重叠率仅 {cp:.2%}")

    # Faithfulness
    f = lm.get("avg_faithfulness", 0)
    if isinstance(f, (int, float)):
        if f > 0.4:
            print(f"   [PASS] 答案忠实度良好 - 生成内容与上下文的平均重叠率 {f:.2%}")
        elif f > 0.15:
            print(f"   [WARN] 答案忠实度一般 - 部分内容可能来自模型通用知识而非检索上下文")
        else:
            print(f"   [FAIL] 答案忠实度较低 - 生成内容可能与检索上下文关联较弱")

    # Retrieval effectiveness
    re = lm.get("avg_retrieval_effectiveness", 0)
    if isinstance(re, (int, float)):
        if re >= 0.9:
            print(f"   [PASS] 检索系统运作正常 - 几乎所有请求都有检索结果")
        elif re > 0.5:
            print(f"   [WARN] 部分请求无检索结果 - 请检查知识库覆盖范围")
        else:
            print(f"   [FAIL] 大量请求无检索结果 - 知识库可能为空或索引有问题")

    # Latency
    lat = lm.get("avg_latency_ms", 0)
    if isinstance(lat, (int, float)):
        if lat < 3000:
            print(f"   [PASS] 响应速度良好 - 平均延迟 {lat:.0f}ms")
        elif lat < 8000:
            print(f"   [WARN] 响应速度一般 - 平均延迟 {lat:.0f}ms，建议优化")
        else:
            print(f"   [FAIL] 响应速度较慢 - 平均延迟 {lat:.0f}ms，需要排查性能瓶颈")


# =============================================================================
# Reranker 消融实验：两轮对比（轻量直接路径）
# =============================================================================

@observe(name="reranker-ablation-experiment")
async def _run_reranker_ablation_direct(args) -> dict:
    """
    轻量 Reranker 消融实验：直接使用 Embedding → Milvus → Reranker → LLM 路径，
    绕过完整的 Agent 管线，快速对比有/无 Reranker 的检索质量差异。
    """
    from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker
    from src.modules.chat.core.embedding_service import LocalEmbeddings
    from src.modules.chat.core.reranker_service import RerankerService
    from openai import AsyncOpenAI

    mode_a_label = "[+] 有 Reranker (默认)"
    mode_b_label = "[-] 无 Reranker (跳过 BGE-Reranker)"

    print(f"\n{'='*70}")
    print(f"  Reranker 消融实验 — 两轮对比 (直接路径)")
    print(f"  测试用例数: {len(EVALUATION_DATASET)}")
    print(f"{'='*70}\n")

    # ---- 初始化基础组件 ----
    print("  初始化基础组件...")
    emb = LocalEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    reranker = RerankerService.get_instance()

    # Milvus 连接
    connections.connect("default", host="localhost", port=19530)
    collection = Collection("chat_embeddings")
    collection.load()
    dim = emb.model.get_embedding_dimension()
    print(f"  Embedding 维度: {dim}, Milvus 文档数: {collection.num_entities}")

    # LLM 客户端 (优先 DashScope，其次 Volcengine)
    tongyi_key = os.getenv("TONGYI_API_KEY", "")
    volc_key = os.getenv("VOLCENGINE_API_KEY", "")
    if tongyi_key:
        api_key = tongyi_key
        llm_model = os.getenv("CHAT_MODEL", "qwen3.6-plus")
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    elif volc_key:
        api_key = volc_key
        # Volcengine Ark 需要 endpoint ID，默认使用 doubao-1.5-pro-32k
        llm_model = os.getenv("CHAT_MODEL", "doubao-1.5-pro-32k-250115")
        base_url = "https://ark.cn-beijing.volces.com/api/v3"
    else:
        raise RuntimeError("请设置 TONGYI_API_KEY 或 VOLCENGINE_API_KEY 环境变量")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    # ★ No-Think: AsyncOpenAI 构造器不支持 extra_body，通过 monkey-patch 注入
    _original_create = client.chat.completions.create

    async def _no_think_create(**kwargs):
        if "extra_body" not in kwargs:
            kwargs["extra_body"] = {"enable_thinking": False}
        return await _original_create(**kwargs)

    client.chat.completions.create = _no_think_create
    print(f"  LLM: {llm_model}, base_url: {base_url}")

    async def _run_one_mode(rerank_enabled: bool) -> tuple:
        """运行一轮评估，返回 (results, summary)"""
        label = mode_a_label if rerank_enabled else mode_b_label
        print(f"\n  {'─'*60}")
        print(f"  运行模式: {label}")
        print(f"  {'─'*60}")

        raw_results = []
        for i, case in enumerate(EVALUATION_DATASET):
            t0 = time.time()
            print(f"    [{i+1}/{len(EVALUATION_DATASET)}] Q: {case.question[:50]}...", end=" ")

            try:
                # Step 1: Embedding
                t1 = time.time()
                query_vec = emb.embed_query(case.question)
                t_emb = (time.time() - t1) * 1000

                # Step 2: Milvus hybrid search
                t2 = time.time()
                docs = _milvus_hybrid_search(collection, query_vec, case.question, top_k=20)
                t_milvus = (time.time() - t2) * 1000

                # Step 3: Reranker (if enabled)
                t3 = time.time()
                doc_texts = [d["content"] for d in docs]
                if rerank_enabled and doc_texts:
                    ranked = reranker.rerank(
                        query=case.question,
                        documents=doc_texts,
                        top_k=5,
                        threshold=0.3
                    )
                    docs_final = [{
                        "content": doc_texts[idx],
                        "score": round(score, 4)
                    } for idx, score, _ in ranked]
                else:
                    docs_final = docs[:5]
                t_rerank = (time.time() - t3) * 1000

                # Step 4: LLM 生成答案
                t4 = time.time()
                answer = await _generate_answer(client, llm_model, case.question, doc_texts)
                t_llm = (time.time() - t4) * 1000

                duration_ms = int((time.time() - t0) * 1000)
                contexts = [d["content"] for d in docs_final]

                print(f"OK | {duration_ms}ms | ctx={len(contexts)} | ans_len={len(answer)}")

                raw_results.append({
                    "question": case.question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": case.ground_truth,
                    "domain": case.domain,
                    "duration_ms": duration_ms,
                    "timing": {"embed_ms": t_emb, "milvus_ms": t_milvus, "rerank_ms": t_rerank, "llm_ms": t_llm},
                    "safety_passed": True,
                    "steps_completed": 4,
                    "cache_hit": False,
                })
            except Exception as e:
                import traceback
                duration_ms = int((time.time() - t0) * 1000)
                print(f"FAIL: {str(e)[:80]}")
                raw_results.append({
                    "question": case.question,
                    "answer": "",
                    "contexts": [],
                    "ground_truth": case.ground_truth,
                    "domain": case.domain,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "duration_ms": duration_ms,
                    "safety_passed": False,
                })

        # 计算本地 + RAGAS 指标
        dummy_evaluator = AgentChatRagasEvaluator()
        dummy_evaluator.results = raw_results

        ragas_samples = dummy_evaluator._build_ragas_dataset(raw_results)
        ragas_scores, ragas_verdicts = {}, []
        if ragas_samples:
            print(f"\n    运行 RAGAS 评估 ({len(ragas_samples)} 个样本)...")
            ragas_scores, ragas_verdicts = await dummy_evaluator._run_ragas_metrics(ragas_samples)

        local_scores = dummy_evaluator._run_local_metrics(raw_results)
        dummy_evaluator.results = []
        # RerankerService.reset_instance()  # 如果需要重新加载模型

        summary = dummy_evaluator._build_summary(ragas_scores, local_scores)
        return raw_results, summary, ragas_scores if isinstance(ragas_scores, dict) else {}

    # ---- 两轮运行 ----
    print(f"\n{'─'*70}")
    print(f"  第 1 轮: {mode_a_label}")
    print(f"{'─'*70}")
    results_a, summary_a, ragas_a = await _run_one_mode(rerank_enabled=True)

    print(f"\n{'─'*70}")
    print(f"  第 2 轮: {mode_b_label}")
    print(f"{'─'*70}")
    results_b, summary_b, ragas_b = await _run_one_mode(rerank_enabled=False)

    # ---- 清理 ----
    connections.disconnect("default")

    # ---- 对比分析 ----
    lat_a = [r.get("duration_ms", 0) for r in results_a if "error" not in r]
    lat_b = [r.get("duration_ms", 0) for r in results_b if "error" not in r]
    rerank_a = [r.get("timing", {}).get("rerank_ms", 0) for r in results_a if "error" not in r]
    ctx_a = [len(r.get("contexts", [])) for r in results_a if "error" not in r]
    ctx_b = [len(r.get("contexts", [])) for r in results_b if "error" not in r]

    def _p50(vals): return float(np.percentile(vals, 50)) if vals else 0
    def _avg(vals): return float(np.mean(vals)) if vals else 0

    print(f"\n{'='*70}")
    print(f"  Reranker 消融实验 — 对比报告")
    print(f"{'='*70}")
    print(f"  {'指标':<32} {'有 Reranker':>14} {'无 Reranker':>14} {'差异':>10}")
    print(f"  {'-'*72}")
    print(f"  {'端到端 p50 延迟':<28} {_p50(lat_a):>12.0f}ms {_p50(lat_b):>12.0f}ms {_p50(lat_a)-_p50(lat_b):>+8.0f}ms")
    print(f"  {'Reranker 耗时 p50':<28} {_p50(rerank_a):>12.0f}ms {'--':>14} {'--':>10}")
    print(f"  {'检索上下文平均数':<28} {_avg(ctx_a):>12.1f} {_avg(ctx_b):>12.1f} {_avg(ctx_a)-_avg(ctx_b):>+8.1f}")
    print(f"  {'─'*72}")

    def _to_scalar(v):
        """安全转为标量：处理 rag 评估失败时遗留的 list 数据"""
        if isinstance(v, list):
            return sum(v) / len(v) if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    _ALL_RAGAS_KEYS = ["faithfulness", "answer_relevancy", "context_precision",
                        "context_recall", "context_relevancy", "answer_correctness"]
    for key in _ALL_RAGAS_KEYS:
        v_a = _to_scalar(ragas_a.get(key, 0) or summary_a.get("ragas_metrics", {}).get(key, 0) or 0)
        v_b = _to_scalar(ragas_b.get(key, 0) or summary_b.get("ragas_metrics", {}).get(key, 0) or 0)
        if v_a or v_b:
            diff = v_a - v_b
            sign = "+" if diff > 0 else ""
            rel = f"({diff/max(abs(v_a),0.001)*100:+.0f}%)" if abs(v_a) > 0.001 else ""
            print(f"  {'RAGAS ' + key:<28} {v_a:>14.4f} {v_b:>14.4f} {sign}{diff:>9.4f} {rel}")

    print(f"\n  [消融结论]")
    reranker_lat = _p50(rerank_a)
    f_a = _to_scalar(ragas_a.get("faithfulness", 0) or 0)
    f_b = _to_scalar(ragas_b.get("faithfulness", 0) or 0)
    print(f"  1. Reranker p50 延迟: {reranker_lat:.0f}ms")
    print(f"  2. Faithfulness: 有={f_a:.4f}, 无={f_b:.4f} (差值={f_a-f_b:+.4f})")
    print(f"  3. 检索上下文: 有={_avg(ctx_a):.1f}篇, 无={_avg(ctx_b):.1f}篇 (过滤{_avg(ctx_b)-_avg(ctx_a):.1f}篇)")

    ablation_summary = {
        "experiment": "reranker_ablation_direct",
        "mode_a": {"label": mode_a_label, "results": results_a, "summary": summary_a},
        "mode_b": {"label": mode_b_label, "results": results_b, "summary": summary_b},
        "comparison": {
            "p50_latency_ms": {"with": _p50(lat_a), "without": _p50(lat_b), "reranker_only": _p50(rerank_a)},
            "avg_contexts": {"with": _avg(ctx_a), "without": _avg(ctx_b)},
            "ragas": {k: {"with": _to_scalar(ragas_a.get(k, 0)), "without": _to_scalar(ragas_b.get(k, 0))}
                       for k in _ALL_RAGAS_KEYS},
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── Langfuse: 上传消融实验对比指标 ──
    _abl_langfuse_enabled = LANGFUSE_AVAILABLE
    if _abl_langfuse_enabled:
        try:
            _pub = os.getenv("LANGFUSE_PUBLIC_KEY")
            _sec = os.getenv("LANGFUSE_SECRET_KEY")
            if not (_pub and _sec):
                _abl_langfuse_enabled = False
        except Exception:
            _abl_langfuse_enabled = False

    if _abl_langfuse_enabled:
        try:
            _abl_client = Langfuse()
            trace_id = get_current_trace_id()

            # RAGAS 指标对比分数
            for key in _ALL_RAGAS_KEYS:
                v_a = _to_scalar(ragas_a.get(key, 0) or summary_a.get("ragas_metrics", {}).get(key, 0) or 0)
                v_b = _to_scalar(ragas_b.get(key, 0) or summary_b.get("ragas_metrics", {}).get(key, 0) or 0)
                if v_a or v_b:
                    _abl_client.score(
                        trace_id=trace_id,
                        name=f"ablation.{key}.with_reranker",
                        value=v_a,
                        comment="RAGAS metric with BGE-Reranker",
                    )
                    _abl_client.score(
                        trace_id=trace_id,
                        name=f"ablation.{key}.without_reranker",
                        value=v_b,
                        comment="RAGAS metric without BGE-Reranker",
                    )

            # 延迟对比
            _abl_client.score(
                trace_id=trace_id,
                name="ablation.p50_latency_ms.with_reranker",
                value=_p50(lat_a),
            )
            _abl_client.score(
                trace_id=trace_id,
                name="ablation.p50_latency_ms.without_reranker",
                value=_p50(lat_b),
            )

            print(f"\n  [Langfuse] 已上传消融实验对比指标")
        except Exception as e:
            print(f"\n  [Langfuse] 消融实验上传评分异常（非致命）: {e}")

    return ablation_summary


def _milvus_hybrid_search(collection, query_vec: list, query_text: str, top_k: int = 20):
    """Milvus 检索：优先混合检索，无 sparse_bm25 字段时回退纯 Dense 检索"""
    from pymilvus import AnnSearchRequest, RRFRanker

    # 检查是否有 sparse_bm25 字段
    schema_fields = {f.name for f in collection.schema.fields}
    has_sparse = "sparse_bm25" in schema_fields

    dense_req = AnnSearchRequest(
        data=[query_vec],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": max(50, top_k * 2)}},
        limit=top_k * 2
    )

    if has_sparse:
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse_bm25",
            param={"metric_type": "BM25"},
            limit=top_k * 2
        )
        reranker = RRFRanker(k=60)
        results = collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=reranker,
            limit=top_k,
            output_fields=["text"]
        )
    else:
        # 纯 Dense 检索
        results = collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": max(50, top_k * 2)}},
            limit=top_k,
            output_fields=["text"]
        )

    docs = []
    for hits in results:
        for hit in hits:
            docs.append({
                "content": hit.entity.get("text", ""),
                "score": round(hit.distance, 4)
            })
    return docs


async def _generate_answer(client, model: str, question: str, contexts: list) -> str:
    """使用 LLM 基于检索上下文生成答案"""
    if not contexts:
        context_str = "(无检索结果，请基于通用知识回答)"
    else:
        context_str = "\n---\n".join(
            f"[{i+1}] {ctx[:500]}" for i, ctx in enumerate(contexts[:8])
        )

    sys_msg = (
        "你是一个电商客服助手。请基于下面提供的「参考资料」回答问题。"
        "如果参考资料充足，优先引用资料中的信息；如果资料不足，基于常识回答并注明。"
        "回答要简洁、准确、有帮助。"
    )
    user_msg = f"【参考资料】\n{context_str}\n\n【用户问题】{question}\n\n请基于以上参考资料回答："

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"[LLM调用失败: {str(e)[:100]}]"


def _run_reranker_ablation(args) -> dict:
    """Reranker 消融实验入口 (异步包装)"""
    return asyncio.run(_run_reranker_ablation_direct(args))


# =============================================================================
# 主入口
# =============================================================================

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RAGAS 评估 agent_chat 端点"
    )
    parser.add_argument(
        "--domain", "-d",
        type=str,
        default=None,
        choices=["medical", "ecommerce", "customer_service", "general"],
        help="只评估指定领域 (默认: 所有领域)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="将评估结果保存到 JSON 文件"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印测试用例，不实际执行评估"
    )
    parser.add_argument(
        "--ragas-only",
        action="store_true",
        help="仅使用 RAGAS 官方指标（需要安装 ragas）"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="RAGAS 评判模型"
    )
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="跳过 BGE-Reranker 重排序，直接使用 Milvus 混合检索结果 (单次评估)"
    )
    parser.add_argument(
        "--compare-reranker",
        action="store_true",
        help="Reranker 消融实验模式：分别跑有/无 Reranker 两轮评估并对比"
    )
    parser.add_argument(
        "--case-index", "-n",
        type=int,
        default=None,
        metavar="N",
        help="仅运行第 N 个测试用例 (1-based, 配合 --domain 或单独使用)"
    )
    parser.add_argument(
        "--case-name", "-q",
        type=str,
        default=None,
        metavar="KEYWORD",
        help="仅运行问题包含 KEYWORD 的测试用例 (模糊匹配)"
    )
    parser.add_argument(
        "--token-report",
        action="store_true",
        help="仅打印 token 消耗预估，不实际执行评估"
    )
    parser.add_argument(
        "--retrieval-mode", "-r",
        type=str,
        default="hybrid",
        choices=["hybrid", "dense-only"],
        help="检索模式: hybrid=混合检索(Dense+BM25, 默认), dense-only=纯向量检索基线"
    )
    parser.add_argument(
        "--enable-graph", "-g",
        action="store_true",
        default=False,
        help="启用 NebulaGraph 图增强查询 (需要 NebulaGraph 服务运行中)"
    )

    args = parser.parse_args()

    # 检查环境变量
    required_envs = ["VOLCENGINE_API_KEY"]
    missing = [e for e in required_envs if not os.getenv(e)]
    if missing:
        print(f"[WARN] 缺少环境变量: {', '.join(missing)}")
        print("   部分功能可能不可用\n")

    if args.dry_run:
        # 仅打印测试用例
        evaluator = AgentChatRagasEvaluator(
            domain=args.domain,
            single_case_index=args.case_index,
            single_case_name=args.case_name,
            retrieval_mode=args.retrieval_mode,
            enable_graph=args.enable_graph,
        )
        cases = evaluator._filter_cases()
        
        print(f"\n{'='*70}")
        print(f"  [DATASET] 评估数据集预览")
        print(f"  领域: {args.domain or '全部'} | 用例数: {len(cases)}")
        print(f"{'='*70}")
        
        for i, case in enumerate(cases):
            print(f"\n  [{i+1}] 领域: {case.domain}")
            print(f"      问题: {case.question}")
            print(f"      参考答案: {case.ground_truth[:100]}...")
            print(f"      期望关键词: {case.expected_context_keywords}")
        return

    if args.token_report:
        # ── Token 消耗预估报告 ──
        evaluator = AgentChatRagasEvaluator(
            domain=args.domain,
            single_case_index=args.case_index,
            single_case_name=args.case_name,
            retrieval_mode=args.retrieval_mode,
            enable_graph=args.enable_graph,
        )
        cases = evaluator._filter_cases()
        n_samples = len(cases)
        ctx_count = 5  # 默认上下文数
        
        print(f"\n{'='*70}")
        print(f"  [TOKEN REPORT] RAGAS 评估 Token 消耗预估")
        print(f"  用例数: {n_samples} | 上下文数/用例: ~{ctx_count}")
        print(f"{'='*70}")
        
        print(f"\n  {'指标':<25} {'LLM调用/样本':>12} {'prompt/次':>10} {'completion/次':>10}")
        print(f"  {'-'*57}")
        
        metrics_info = [
            ("Faithfulness", 2, 1800, 250),
            ("AnswerRelevancy", 1, 1600, 150),
            ("ContextRecall", 1, 2000, 400),
        ]
        
        _total_calls = 0
        _total_prompt = 0
        _total_completion = 0
        for name, calls, prompt, compl in metrics_info:
            _total_calls += calls * n_samples
            _total_prompt += calls * n_samples * prompt
            _total_completion += calls * n_samples * compl
            print(f"  {name:<25} {calls:>12} {prompt:>10} {compl:>10}")
        
        _cost_plus = (_total_prompt * 2.0 + _total_completion * 8.0) / 1e6
        _cost_turbo = (_total_prompt * 0.3 + _total_completion * 0.6) / 1e6
        print(f"\n  总计 (轻量模式):")
        print(f"    LLM 调用: ~{_total_calls} 次")
        print(f"    Prompt tokens: ~{_total_prompt//1000}K")
        print(f"    Completion tokens: ~{_total_completion//1000}K")
        print(f"    费用(qwen3.6-plus): ¥{_cost_plus:.4f}")
        print(f"    费用(qwen-turbo):    ¥{_cost_turbo:.4f} (省 {(1-_cost_turbo/_cost_plus)*100:.0f}%)")
        print(f"\n{'='*70}\n")
        return

    # 运行评估
    if args.compare_reranker:
        # ---- Reranker 消融实验模式：跑两轮 ----
        summary = await _run_reranker_ablation_direct(args)
    else:
        # ---- 单次评估模式 ----
        evaluator = AgentChatRagasEvaluator(
            domain=args.domain, llm_model=args.llm_model,
            skip_reranker=args.skip_reranker,
            single_case_index=args.case_index,
            single_case_name=args.case_name,
            retrieval_mode=args.retrieval_mode,
            enable_graph=args.enable_graph,
        )
        summary = await evaluator.run_evaluation()

    # 输出报告
    if not args.compare_reranker:
        # 消融实验模式在 _run_reranker_ablation 内部已打印对比报告
        print_evaluation_report(summary)

    # 保存
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not args.compare_reranker:
            # 单次评估：添加详细结果
            summary["full_results"] = [
                {
                    "question": r["question"],
                    "answer": r.get("answer", ""),
                    "contexts": r.get("contexts", []),
                    "ground_truth": r.get("ground_truth", ""),
                    "domain": r.get("domain", ""),
                    "duration_ms": r.get("duration_ms", 0),
                    "safety_passed": r.get("safety_passed", False),
                    "steps": r.get("step_details", []),
                    "error": r.get("error", ""),
                }
                for r in (evaluator.results or [])
            ]
        
        def _convert_for_json(obj):
            """递归转换 numpy/datetime 类型为 Python 原生类型，确保 JSON 序列化成功"""
            if isinstance(obj, dict):
                return {k: _convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_convert_for_json(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif hasattr(obj, 'isoformat'):  # datetime/date
                return obj.isoformat()
            return obj
        
        summary_clean = _convert_for_json(summary)
        output_path.write_text(json.dumps(summary_clean, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[SAVED] 详细评估结果已保存到: {output_path}")

    # ── Langfuse: 刷新缓冲区，确保所有追踪数据已发送 ──
    if LANGFUSE_AVAILABLE:
        try:
            _pub = os.getenv("LANGFUSE_PUBLIC_KEY")
            _sec = os.getenv("LANGFUSE_SECRET_KEY")
            if _pub and _sec:
                _flush_client = Langfuse()
                _flush_client.flush()
                print(f"\n[Langfuse] 追踪数据已刷新")
        except Exception as e:
            print(f"\n[Langfuse] flush 异常（非致命）: {e}")


if __name__ == "__main__":
    asyncio.run(main())
