"""回归守卫：验证 orchestrator_params 已迁移到分层管道（Phase 1 契约）。

确保后续重构不会把调用方改回旧的「L1 直接定稿」实现，
也不会破坏 run_pipeline / McpSchemaProvider 的接入点。
"""
from src.modules.chat.core.layered_param_extractor import run_pipeline, McpSchemaProvider
from src.modules.chat.agent import orchestrator_params


def test_orchestrator_imports_pipeline_symbols():
    # 调用方必须直接引用分层管道的核心符号
    assert orchestrator_params.__dict__.get("run_pipeline") is run_pipeline
    assert McpSchemaProvider is not None


def test_orchestrator_module_has_fallback_path():
    # 管道异常时有降级分支（向后兼容旧 action-based 抽取），通过源码存在性校验
    import inspect

    src = inspect.getsource(orchestrator_params)
    assert "run_pipeline(" in src
    assert "降级" in src  # 异常降级到旧抽取
