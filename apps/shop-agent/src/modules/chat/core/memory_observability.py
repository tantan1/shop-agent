"""
记忆系统可观测性：指标埋点与结构化日志
"""

from src.modules.monitoring.metrics import (
    forgetting_job_counter,
    forgetting_job_duration,
    memory_block_counter,
    memory_context_size,
    memory_recall_counter,
    memory_recall_duration,
)
from src.shared.logger import APILogger

logger = APILogger("memory_observability")


class MemoryObservability:
    """记忆系统观测埋点"""

    @staticmethod
    def record_recall(layer: str, duration: float, count: int = 1) -> None:
        memory_recall_counter.labels(layer=layer).inc(count)
        memory_recall_duration.labels(layer=layer).observe(duration)

    @staticmethod
    def record_context_size(layer: str, chars: int) -> None:
        memory_context_size.labels(layer=layer).observe(chars)

    @staticmethod
    def record_block_created(block_type: str) -> None:
        memory_block_counter.labels(block_type=block_type).inc()

    @staticmethod
    def record_forgetting_job(action: str, duration: float, count: int = 1) -> None:
        forgetting_job_counter.labels(action=action).inc(count)
        forgetting_job_duration.observe(duration)

    @staticmethod
    def log_memory_operation(operation: str, user_id: str, **kwargs) -> None:
        logger.log_business_event(
            f"memory_{operation}",
            user_id=user_id,
            **kwargs,
        )
