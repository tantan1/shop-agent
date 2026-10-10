"""工具选择可配置迭代式 Pipeline 调用器。

见 docs/architecture/tool-select-pipeline-design.md §5。
维护 scope（下一层候选范围）与观测缓冲；统一做早停判断、候选软过滤传递与降级。
各层 scored_candidates 仅用于观测日志，绝不参与跨层融合决策。
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from src.modules.chat.core.tool_select_stage import CandidateScore, StageResult, ToolSelectStage
from src.modules.chat.schemas import PlannedAction, ToolPlan
from src.modules.monitoring.metrics import (
    tool_select_stage_total,
    tool_select_stage_duration_ms,
    tool_select_candidates_in,
    tool_select_candidates_out,
    tool_select_confidence,
    tool_select_exit_total,
    tool_select_final_total,
)

logger = logging.getLogger(__name__)

# 未配置 timeout_ms 时的兜底超时（毫秒），避免单层无界等待。
DEFAULT_STAGE_TIMEOUT_MS = 5000


@dataclass(frozen=True)
class PipelinePolicy:
    """Pipeline 的策略参数集合（阈值 / 降级 / 超时）。

    仅为「参数对象」，用于收缩构造参数个数并满足 ruff `max-args` 约束；
    不读 YAML——外部配置化（Phase 3）由 `ToolSelectConfig` 组装后传入。
    """

    thresholds: Dict[str, float] = field(default_factory=dict)
    fallback: Dict[str, str] = field(default_factory=dict)
    timeouts: Dict[str, int] = field(default_factory=dict)
    # 无层早停时，把末级收窄集组装为 ToolPlan（与旧 _select_tools_for_intent 输出一致）。
    # 默认 False：保持原「need_llm」降级契约（Phase 1 框架测试依赖），
    # Phase 4 接入具体 Stage 时由调用方开启。
    emit_final_scope_as_plan: bool = False
    # 收敛阈值：某层把候选收窄到该数量以内即提前结束（成本漏斗）。
    # 默认 1（仅单工具收敛才停）；调用方可放宽到 2，使 ≤2 工具也提前结束、
    # 避免触发更贵层级（默认 P3 LLM）。与 emit_final_scope_as_plan 的 ≤2 判定一致。
    convergence_threshold: int = 1


class ToolSelectPipeline:
    def __init__(
        self,
        stages: List[ToolSelectStage],
        all_tools: List[str],
        deps: Dict[str, Any],
        policy: Optional[PipelinePolicy] = None,
    ) -> None:
        self.stages = stages
        self._all_tools = list(all_tools)
        self._tool_set = set(self._all_tools)  # 注册表成员校验用（O(1)）
        self._deps = deps
        self._policy = policy or PipelinePolicy()

    async def select(self, query: str) -> ToolPlan:
        scope: List[str] = list(self._all_tools)  # 当前候选 scope，传给各层
        observations: List[CandidateScore] = []  # 各层打分，仅用于末尾观测日志

        # 一次性预计算 query embedding，供 P1 FAISS 与 P2 线性头复用，
        # 避免每层各自打一次 vLLM（原先 P1/P2 各 embed 一次，P2 还走了跨事件循环的
        # 同步 embed，p95 被拖到 1-2s）。vLLM 返回已归一化向量，与训练/推理一致。
        # 计时纳入分层监控（见 _record_embedding_metrics）：否则工具选择真实总耗时
        # （含 embedding 网络往返，可能 5-50ms，远大于 P0/P1/P2 各自 <5ms）会被
        # 「每层耗时」面板系统性低估。
        embeddings_svc = self._deps.get("embeddings")
        query_embedding = None
        if embeddings_svc is not None:
            embed_start = time.monotonic()
            embed_ok = False
            try:
                query_embedding = np.array(
                    await embeddings_svc.embed_query(query), dtype=np.float32
                )
                embed_ok = True
            except Exception as e:
                logger.warning(f"query embedding 预计算失败，由各 stage 自行 embed: {e}")
            self._record_embedding_metrics((time.monotonic() - embed_start) * 1000, embed_ok)

        context = {
            "thresholds": self._policy.thresholds,
            "query_embedding": query_embedding,
            **self._deps,
        }
        last_source: Optional[str] = None
        final_scope: List[str] = list(scope)

        for stage in self.stages:
            result, latency_ms = await self._run_stage(stage, query, scope, context)
            self._record_stage_metrics(stage, result, latency_ms, scope)
            self._log_stage_result(stage, result, latency_ms)

            # 错误处理：按 fallback.on_stage_failure 策略（skip 时本层不计入观测）
            if result.error:
                aborted = self._handle_error(stage, result)
                if aborted is not None:
                    self._record_exit(stage.name, aborted)
                    return self._complete(aborted)
            else:
                observations.extend(result.scored_candidates)
                if result.scored_candidates:
                    scope = [s.tool for s in result.scored_candidates]
                last_source = result.source
                final_scope = list(scope)

            # 收敛早停：本层把候选收窄到收敛阈值以内即提前结束，
            # 避免触发更贵层级（默认 P3 LLM）。即便各层 tool=None，
            # 只要候选集收敛（≤阈值）也能让 P1/P2 成为退出层，
            # 让成本漏斗真正生效（不再 100% 落入 p3_llm）。
            # 首层（P0）豁免：保证 P1 FAISS 始终被观测、漏斗不跳过 P1。
            # 同防御项：收敛后的候选必须全落在注册表内，否则不得触发早停。
            # 收敛早停：本层把候选收窄到收敛阈值以内即提前结束，
            # 避免触发更贵层级（默认 P3 LLM）。首层（P0）豁免，
            # 保证 P1 FAISS 始终被观测、漏斗不跳过 P1。
            # 防御项：候选必须全落在注册表内，否则不得触发早停（越界工具名红线）。
            if (
                not result.error
                and stage is not self.stages[0]
                and 0 < len(scope) <= self._policy.convergence_threshold
                and all(t in self._tool_set for t in scope)
            ):
                plan = self._final_scope_plan(scope, result.source, observations)
                self._record_exit(result.source, plan)
                return self._complete(plan, observations)

            # 早停：本层足够置信 → 直接返回，不触发更贵层级（成本漏斗）
            # 防御项：① 失败层不得参与早停；② 工具名必须落在注册表（all_tools）内，
            # 否则会把幻觉/越界工具名包装成 plan_complete 下发（红队 B2）
            threshold = self._policy.thresholds.get(stage.name, 0.9)
            if (
                result.tool
                and not result.error
                and result.tool in self._tool_set
                and result.confidence >= threshold
            ):
                plan = ToolPlan(
                    actions=[
                        PlannedAction(
                            name=result.tool,
                            source=result.source,
                            confidence=result.confidence,
                        )
                    ],
                    source=result.source,
                    stop_condition="plan_complete",
                )
                self._record_exit(stage.name, plan)
                return self._complete(plan, observations)

        # 无层早停：不自行融合打分，按 fallback 策略降级（观测日志见下）
        logger.info(
            "tool_select_no_early_exit",
            extra={
                "observations": [
                    (s.tool, round(s.score, 3) if s.score is not None else None, s.source)
                    for s in observations
                ]
            },
        )

        # emit_final_scope_as_plan：把末级收窄集组装为 ToolPlan
        # （与旧 ReActAgent._select_tools_for_intent 输出语义一致：返回「收窄后的候选集」）
        if self._policy.emit_final_scope_as_plan and final_scope:
            plan = self._final_scope_plan(final_scope, last_source or "pipeline", observations)
            self._record_exit(last_source or "pipeline", plan)
            return self._complete(plan, observations)

        plan = self._fallback_plan()
        self._record_exit("fallback", plan)
        return self._complete(plan, observations)

    def _final_scope_plan(
        self, final_scope: List[str], source: str, observations: Optional[List[CandidateScore]] = None
    ) -> ToolPlan:
        """无早停时，把末级收窄出的候选集组装成 ToolPlan，并透传各层真实打分（设计 1）。

        真实分数来自 ``observations`` 中 ``score is not None`` 的候选（P1 余弦 / P2 softmax /
        P0 规则强弱分）；选择器层（P3）与未产分透传路径 score=None，不覆盖真实分数。
        无真实分数可透传时 confidence 保持 None（捕获侧按"不确定/低置信"处理）。

        ``stop_condition`` 按收窄集规模判定：<=2 视为可确定性执行（plan_complete），
        P0/P1 单工具与 P2 收敛 ≤2 同理，并延伸到 P3。不复用 ``ToolPlan.is_confident``
        （其枚举仅覆盖 p0/p1/p2），此处用统一规模阈值，行为在 p0/p1/p2 场景下完全一致。
        """
        score_map = {s.tool: s.score for s in (observations or []) if s.score is not None}
        actions = [
            PlannedAction(name=n, source=source, confidence=score_map.get(n))
            for n in final_scope
        ]
        stop_condition = "plan_complete" if 0 < len(actions) <= 2 else "need_llm"
        return ToolPlan(actions=actions, source=source, stop_condition=stop_condition)

    async def _run_stage(
        self,
        stage: ToolSelectStage,
        query: str,
        scope: List[str],
        context: Dict[str, Any],
    ) -> Tuple[StageResult, float]:
        """执行单层并强制超时；返回 (本层结果, 耗时毫秒)。

        超时与任何未捕获异常统一转为 `error`，交由调用方按 on_stage_failure 处置。
        """
        started = time.monotonic()
        timeout = self._policy.timeouts.get(stage.name) or DEFAULT_STAGE_TIMEOUT_MS
        try:
            result = await asyncio.wait_for(
                stage.run(query, scope, context), timeout=timeout / 1000
            )
        except asyncio.TimeoutError:
            result = StageResult(tool=None, source=stage.name, error="stage timeout")
        except Exception as exc:
            # 空消息异常（如 raise RuntimeError()）的 str(exc) 为 ""，属假值，
            # 会让失败层被误判为成功层；回落为异常类名以保证 error 恒为真值。
            result = StageResult(
                tool=None, source=stage.name, error=str(exc) or type(exc).__name__
            )
        return result, (time.monotonic() - started) * 1000

    @staticmethod
    def _log_stage_result(stage: ToolSelectStage, result: StageResult, latency_ms: float) -> None:
        """每层结构化日志（design §7.2）。"""
        logger.info(
            "tool_select_stage_result",
            extra={
                "stage": stage.name,
                "tool": result.tool,
                "confidence": result.confidence,
                "candidates_count": len(result.scored_candidates),
                "latency_ms": round(latency_ms, 3),
                "error": result.error,
            },
        )

    @staticmethod
    def _record_stage_metrics(
        stage: ToolSelectStage, result: StageResult, latency_ms: float, scope_in: List[str]
    ) -> None:
        """每层 Prometheus 指标采集（design §7 监控）。

        - stage_total：按 hit/miss/error/timeout 计数（瓶颈/错误定位）
        - stage_duration_ms：每层耗时（定位最贵层级，通常是 P3 LLM）
        - candidates_in/out：收窄漏斗（看每层把候选从多少筛到多少）
        - confidence：各层置信度（判断早停阈值是否可达）
        """
        name = stage.name
        tool_select_stage_duration_ms.labels(stage=name).observe(latency_ms)
        if result.error:
            outcome = "timeout" if "timeout" in (result.error or "").lower() else "error"
        elif not result.scored_candidates:
            outcome = "miss"
        else:
            outcome = "hit"
        tool_select_stage_total.labels(stage=name, outcome=outcome).inc()
        tool_select_candidates_in.labels(stage=name).observe(len(scope_in))
        tool_select_candidates_out.labels(stage=name).observe(len(result.scored_candidates))
        # 各层置信度分布：早停阈值是否可达、是否过于宽松，全靠这个指标判断。
        # 只记录有真实打分的层（P3 选择器层 score 为 None，不参与）。
        for cand in result.scored_candidates:
            score = getattr(cand, "score", None)
            if score is not None:
                tool_select_confidence.labels(stage=name).observe(float(score))

    @staticmethod
    def _record_embedding_metrics(latency_ms: float, ok: bool) -> None:
        """query embedding 预计算耗时纳入分层监控（见 select() 注释）。

        以独立 stage 标签 ``embedding_prepare`` 与 P0/P1/P2 并列出现在
        「每层执行耗时」面板，使工具选择真实总耗时（含 vLLM 网络往返）可见。
        候选/置信度类指标对预计算无意义，故只记 duration 与 total。
        """
        tool_select_stage_duration_ms.labels(stage="embedding_prepare").observe(latency_ms)
        tool_select_stage_total.labels(
            stage="embedding_prepare", outcome="hit" if ok else "error"
        ).inc()

    def _record_exit(self, exit_stage: str, plan: ToolPlan) -> None:
        """记录终止层级（成本漏斗）与最终结果形态。

        exit_stage 期望落在 P0/P1/P2（便宜层级提前结束）；若是 p3 或 fallback，
        说明成本漏斗未生效——最贵层级（P3 LLM）被 100% 触发。
        """
        tool_select_exit_total.labels(stage=exit_stage, stop_condition=plan.stop_condition).inc()
        # 最终结果形态：候选集是否收敛（1/2/3+）决定下游能否确定性执行。
        n = len(plan.actions or [])
        bucket = "1" if n == 1 else ("2" if n == 2 else "3+")
        tool_select_final_total.labels(
            source=plan.source, stop_condition=plan.stop_condition, candidates=bucket
        ).inc()

    def _handle_error(self, stage: ToolSelectStage, result: StageResult) -> Optional[ToolPlan]:
        """按 on_stage_failure 策略处理失败层；返回非 None 表示立即终止。"""
        policy = self._policy.fallback.get("on_stage_failure", "skip")
        logger.warning(
            "tool_select_stage_error", extra={"stage": stage.name, "error": result.error}
        )
        if policy in ("abort", "hitl"):
            # hitl 在代码层等价于 need_llm：返回空结果交上游按部署策略处置
            return ToolPlan(actions=[], source="fallback", stop_condition="need_llm")
        return None

    @staticmethod
    def _build_ranking(observations: Optional[List["CandidateScore"]]) -> List[Dict[str, Any]]:
        """由各层 scored_candidates 去重聚合出全量候选排序(按真实 score 降序)。

        多阶段可能对同一工具打分,取最高分;score=None(P3 LLM 等不产分路径)不计入排序,
        避免 None 污染 margin。返回 [{tool, confidence, source}],供 MLOps 捕获 top_tools/margin。
        """
        best: Dict[str, Tuple[float, str]] = {}
        for s in observations or []:
            if s.score is None:
                continue
            prev = best.get(s.tool)
            if prev is None or s.score > prev[0]:
                best[s.tool] = (s.score, s.source)
        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
        return [
            {"tool": t, "confidence": round(sc, 4), "source": src}
            for t, (sc, src) in ranked
        ]

    @staticmethod
    def _complete(plan: ToolPlan, observations: Optional[List["CandidateScore"]] = None) -> ToolPlan:
        """Pipeline 收尾日志（design §7.2）。

        若 plan 尚未携带 candidate_ranking,则从全量观测(各层 scored_candidates)聚合,
        透传真实打分供 MLOps 捕获 top_tools / margin(§5.3 自动银标闸门)。
        """
        if not plan.candidate_ranking and observations:
            ranking = ToolSelectPipeline._build_ranking(observations)
            if ranking:
                plan = plan.model_copy(update={"candidate_ranking": ranking})
        logger.info(
            "tool_select_pipeline_complete",
            extra={
                "source": plan.source,
                "tool": plan.actions[0].name if plan.actions else None,
                "confidence": plan.actions[0].confidence if plan.actions else None,
                "stop_condition": plan.stop_condition,
            },
        )
        return plan

    def _fallback_plan(self) -> ToolPlan:
        """无层跨阈值时的降级结果；融合决策交由上游/策略，本方法不计算分数。"""
        if self._policy.fallback.get("on_all_stages_failed") == "default_tool" and self._all_tools:
            return ToolPlan(
                actions=[
                    PlannedAction(name=self._all_tools[0], source="default", confidence=0.0)
                ],
                source="default",
                stop_condition="plan_complete",
            )
        # 默认 need_llm：交上游按部署策略处置（云端 ReAct / HITL / 本地小模型）
        return ToolPlan(actions=[], source="fallback", stop_condition="need_llm")
