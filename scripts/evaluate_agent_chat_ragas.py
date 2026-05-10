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

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()


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
    # EvalCase(
    #     question="公司的请假流程是怎样的？",
    #     ground_truth="请假流程：1) 员工提前填写请假单；2) 直属上级审批；3) 超过3天需部门负责人加签；4) 报人事部备案存档。",
    #     domain="general",
    #     expected_context_keywords=["请假", "流程", "审批", "人事部"]
    # ),
    # EvalCase(
    #     question="公司有哪些规章制度需要遵守？",
    #     ground_truth="公司规章制度包括考勤制度、请假制度、办公纪律、安全管理制度、保密制度等，员工入职时应认真学习并签署确认。",
    #     domain="general",
    #     expected_context_keywords=["规章制度", "考勤", "纪律", "安全", "保密"]
    # ),
    # EvalCase(
    #     question="员工迟到会有什么处罚？",
    #     ground_truth="员工迟到处罚标准：月度累计迟到3次以内口头警告，3-5次书面警告并扣绩效分，5次以上记过处分。",
    #     domain="general",
    #     expected_context_keywords=["迟到", "处罚", "警告", "绩效"]
    # ),

    # ---- 电商领域 (ecommerce) ----
    # EvalCase(
    #     question="电冰箱有哪些功能和特点？",
    #     ground_truth="该款电冰箱采用风冷无霜技术，具有变频节能、智能控温、大容量存储等特点，支持冷藏冷冻分区调节。",
    #     domain="ecommerce",
    #     expected_context_keywords=["电冰箱", "风冷", "变频", "智能控温"]
    # ),
    # EvalCase(
    #     question="净水器的滤芯多久需要更换一次？",
    #     ground_truth="净水器滤芯更换周期：PP棉滤芯3-6个月，活性炭滤芯6-12个月，RO反渗透膜24-36个月，具体根据水质和使用频率而定。",
    #     domain="ecommerce",
    #     expected_context_keywords=["净水器", "滤芯", "更换", "RO"]
    # ),
    # EvalCase(
    #     question="电视有哪些规格参数？",
    #     ground_truth="电视机包含屏幕尺寸、分辨率、刷新率、HDMI接口数量、是否支持智能系统等规格参数，不同型号存在差异。",
    #     domain="ecommerce",
    #     expected_context_keywords=["电视", "规格", "分辨率", "HDMI"]
    # ),

    # EvalCase(
    #     question="产品退货流程是什么？",
    #     ground_truth="退货流程：1) 在订单页面申请退货；2) 填写退货原因；3) 等待客服审核；4) 审核通过后寄回商品；5) 仓库收货后7个工作日内退款。",
    #     domain="ecommerce",
    #     expected_context_keywords=["退货", "退款", "订单", "审核"]
    # ),

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

    def __init__(self, domain: str = None, llm_model: str = None):
        """
        初始化评估器
        
        Args:
            domain: 要评估的领域，None 表示所有领域
            llm_model: RAGAS 用于评判的 LLM 模型，默认使用通义千问
        """
        self.domain = domain
        self.llm_model = llm_model
        self.results: List[Dict[str, Any]] = []
        self.service = None

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

    def _filter_cases(self) -> List[EvalCase]:
        """筛选要评估的测试用例"""
        if self.domain:
            return [c for c in EVALUATION_DATASET if c.domain == self.domain]
        return list(EVALUATION_DATASET)

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

    async def run_evaluation(self) -> Dict[str, Any]:
        """
        运行完整评估
        
        Returns:
            评估汇总结果字典
        """
        cases = self._filter_cases()
        print(f"\n{'='*70}")
        print(f"  RAGAS 评估 - agent_chat (通用Agent对话)")
        print(f"  领域: {self.domain or '全部'}")
        print(f"  测试用例数: {len(cases)}")
        print(f"{'='*70}\n")
        
        # ---------- Step 1: 运行所有测试用例 ----------
        print("[Step 1] 执行测试用例，调用 agent_chat 服务...")
        
        # 预热：触发模型加载、Milvus连接等初始化，避免首条数据计时不准
        print("  [warmup] 预加载 Embedding 模型...", end=" ")
        await self._get_service()
        service = self.service
        from src.modules.chat.core.embedding_service import EmbeddingService
        emb_svc = EmbeddingService.get_instance()
        _ = await emb_svc.embed_query("预热")
        print("完成")
        
        print("  [warmup] 预加载 BGE-Reranker 模型...", end=" ")
        from src.modules.chat.core.reranker_service import RerankerService
        reranker = RerankerService.get_instance()
        _ = reranker.rerank(query="预热", documents=["预热文档"])
        print("完成\n")
        
        raw_results = []
        for i, case in enumerate(cases):
            print(f"  [{i+1}/{len(cases)}] [{case.domain}] {case.question[:60]}...", end=" ")
            result = await self.run_single_case(case)
            
            if "error" in result:
                print(f"FAIL: {result['error'][:80]}")
                result["_effectiveness_local"] = "[❌ 调用失败]"
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
            # ragas_per_verdicts 顺序与 ragas_samples 一致；映射回 raw_results 的对应项
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
        
        return summary

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
            # ---- 创建 AsyncOpenAI 客户端（ragas 0.4.x ascore 需要异步客户端） ----
            from openai import AsyncOpenAI

            judge_model = (
                self.llm_model or
                os.getenv("CHAT_MODEL") or
                os.getenv("RAGAS_LLM_MODEL") or
                "qwen-plus"
            )
            api_key = (
                os.getenv("TONGYI_API_KEY") or
                os.getenv("DASHSCOPE_API_KEY") or
                os.getenv("CHAT_TONGYI_API_KEY")
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
            
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            
            # ---- 评判 LLM（ragas 0.4.x 现代 API） ----
            evaluator_llm = llm_factory(judge_model, client=client)
            print(f"  [RAGAS] 评判模型: {judge_model} ({provider_label})")
            
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
            metrics = [
                Faithfulness(llm=evaluator_llm),
                AnswerRelevancy(llm=evaluator_llm, embeddings=ragas_embeddings, strictness=1),   # 1 个反向问题，而非默认 3 个
                ContextPrecision(llm=evaluator_llm),
                ContextRecall(llm=evaluator_llm),
                ContextRelevancy(llm=evaluator_llm),
                AnswerCorrectness(llm=evaluator_llm, weights=[1.0, 0.0]),  # 纯 factuality，跳过 embedding 相似度
            ]
            
            # ragas 0.4.x 中各指标 ascore 所需 kwargs 不同
            # {metric_class_base_name: lambda sd: kwargs_dict}
            _ASCORE_BUILDERS = {
                "Faithfulness":       lambda sd: dict(user_input=sd["question"], response=sd["answer"], retrieved_contexts=sd["contexts"]),
                "AnswerRelevancy":    lambda sd: dict(user_input=sd["question"], response=sd["answer"]),
                "ContextPrecision":   lambda sd: dict(user_input=sd["question"], reference=sd["ground_truth"], retrieved_contexts=sd["contexts"]),
                "ContextRecall":      lambda sd: dict(user_input=sd["question"], retrieved_contexts=sd["contexts"], reference=sd["ground_truth"]),
                "ContextRelevancy":   lambda sd: dict(user_input=sd["question"], retrieved_contexts=sd["contexts"]),
                "ContextRelevance":   lambda sd: dict(user_input=sd["question"], retrieved_contexts=sd["contexts"]),
                "AnswerCorrectness":  lambda sd: dict(user_input=sd["question"], response=sd["answer"], reference=sd["ground_truth"]),
            }
            
            # 每样本各指标并行评估（6 个 ascore → 1 次网络往返）
            async def _score_one(sd: dict, metric):
                metric_cls_name = metric.__class__.__name__
                builder = _ASCORE_BUILDERS.get(metric_cls_name)
                if builder is None:
                    return metric, None, f"{metric_cls_name}: SKIP"
                kwargs = builder(sd)
                result = await metric.ascore(**kwargs)
                return metric, result.value, None

            for i, sample_data in enumerate(samples):
                print(f"  [{i+1}/{len(samples)}] 评估: {sample_data['question'][:50]}...")
                
                tasks = [_score_one(sample_data, m) for m in metrics]
                results_list = await asyncio.gather(*tasks, return_exceptions=True)
                
                sample_metrics = {}
                for item in results_list:
                    if isinstance(item, BaseException):
                        print(f"    task: ERROR - {str(item)[:80]}")
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
            summary["avg_step_durations"][step_name] = {
                "avg_ms": round(sum(durs) / len(durs), 1) if durs else 0.0,
                "min_ms": round(min(durs), 1) if durs else 0.0,
                "max_ms": round(max(durs), 1) if durs else 0.0,
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
        return "[❌ 无效] 无回答内容"
    if len(answer) < 20:
        return f"[❌ 无效] 答案过短({len(answer)}字)，可能生成失败"
    if not contexts:
        return "[⚠️ 无检索] 无上下文，答案可能来自模型通用知识"
    if any(w in answer[:80] for w in ["抱歉", "无法", "不能", "暂无相关信息"]):
        return "[⚠️ 拒绝型] 模型未给出实质回答"
    
    # 基础有效：有上下文且有足够长度的答案
    c_count = len(contexts)
    a_len = len(answer)
    if a_len > 200:
        return f"[✅ 初步有效] {c_count}条上下文, {a_len}字回答 — 等待RAGAS语义评估"
    else:
        return f"[⚠️ 偏短] {c_count}条上下文, 仅{a_len}字 — 可能不够详尽"


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
        return "[❓ 未知] 无 RAGAS 评分"
    
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
        return f"[✅ 有效] 忠实度{faithfulness:.0%}, 相关性{answer_relevancy:.0%}{extra}"
    
    if faithfulness >= 0.5 and answer_relevancy >= 0.4:
        parts = []
        if faithfulness < 0.7:
            parts.append(f"忠实度偏低({faithfulness:.0%})")
        if answer_relevancy < 0.6:
            parts.append(f"相关性偏弱({answer_relevancy:.0%})")
        return f"[⚠️ 基本有效] {'; '.join(parts)}"
    
    if faithfulness < 0.5 and answer_relevancy >= 0.5:
        return f"[⚠️ 可能有幻觉] 忠实度仅{faithfulness:.0%}, 相关内容未被良好利用"
    
    if faithfulness >= 0.5 and answer_relevancy < 0.4:
        return f"[⚠️ 偏题] 相关性仅{answer_relevancy:.0%}, 回答可能未聚焦问题"
    
    # 两者都低
    if faithfulness < 0.3:
        return f"[❌ 无效] 严重幻觉(忠实度{faithfulness:.0%}), 回答不可信"
    return f"[❌ 无效] 查询-答案关联弱(忠实度{faithfulness:.0%}, 相关性{answer_relevancy:.0%})"


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
    avg_faith = lm.get("avg_faithfulness_local", 0) or 0
    avg_relevancy = lm.get("avg_answer_relevancy_local", 0) or 0
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
            # 兼容 RAGAS key 名大小写差异
            score = ragas.get(key) or ragas.get(key.capitalize()) or ragas.get(key.title())
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
        help="RAGAS 评判模型 (默认使用 CHAT_MODEL 环境变量，即 qwen3.6-flash-2026-04-16)"
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
        evaluator = AgentChatRagasEvaluator(domain=args.domain)
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

    # 运行评估
    evaluator = AgentChatRagasEvaluator(domain=args.domain, llm_model=args.llm_model)
    summary = await evaluator.run_evaluation()

    # 输出报告
    print_evaluation_report(summary)

    # 保存
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 添加详细结果
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
        
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[SAVED] 详细评估结果已保存到: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
