"""K8s 自愈执行器（demo 故障注入/修复）。

仅用于在 shop-agent 命名空间内切换目标 Deployment 的副本数（0 ↔ 1）：
- ``inject_fault``：``scale --replicas=0``，制造真实故障（不自动恢复）；
- ``remediate``：``scale --replicas=1``，恢复服务。

鉴权来源（按优先级）：
1. in-cluster：容器内挂载的 SA token（monitoring-agent-remediate Role），
   仅 ``deployments/scale`` 的 patch 权限，最小攻击面。
2. 本地开发：``KUBECONFIG`` 指向的 kubeconfig（便于 compose/本地联调）。

安全约束（04 §8 / 05 自愈「建议+审批」）：
- 本模块**只暴露 scale 切换**，不提供 delete/exec/任意 patch；
- 目标 deployment 名需命中白名单 ``_ALLOWED_TARGETS``，杜绝任意资源操作；
- 所有调用经 ``/demo/inject-fault``、``/demo/remediate`` 显式鉴权入口，
  不在告警/超时任何自动路径被调用（纯人工确认闭环）。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("monitoring_agent.k8s")

# 允许被自愈操作的 deployment 白名单（namespace 固定为当前 NAMESPACE）。
# 当前仅 redis（选它作故障目标：已有降级逻辑、PVC 持久、恢复干净）。
_ALLOWED_TARGETS = frozenset({"redis"})

NAMESPACE = os.getenv("POD_NAMESPACE", "shop-agent")


class K8sUnavailable(Exception):
    """K8s 客户端初始化或调用失败（in-cluster 无 SA / kubeconfig 缺失等）。"""


def _client():
    """惰性初始化 K8s ApiClient，优先 in-cluster 配置，回退 kubeconfig。"""
    try:
        from kubernetes import client, config  # 可选依赖
    except ImportError as exc:  # pragma: no cover
        raise K8sUnavailable("kubernetes 客户端未安装") from exc

    try:
        config.load_incluster_config()
    except Exception:
        kubeconfig = os.getenv("KUBECONFIG")
        try:
            config.load_kube_config(config_file=kubeconfig)
        except Exception as exc:
            raise K8sUnavailable(
                "K8s 配置不可用（非 in-cluster 且 KUBECONFIG 缺失）"
            ) from exc
    return client.AppsV1Api()


def _check_target(name: str) -> None:
    if name not in _ALLOWED_TARGETS:
        raise ValueError(
            f"目标 {name!r} 不在白名单 {sorted(_ALLOWED_TARGETS)}，拒绝操作"
        )


def set_replicas(name: str, replicas: int) -> dict:
    """把目标 deployment 的副本数调整为 ``replicas``（patch deployments/scale）。

    返回包含目标/命名空间/期望副本数的结果 dict。目标不在白名单则抛 ValueError。
    """
    _check_target(name)
    if replicas < 0:
        raise ValueError("replicas 不能为负")
    api = _client()
    body = {"spec": {"replicas": replicas}}
    try:
        api.patch_namespaced_deployment_scale(
            name=name, namespace=NAMESPACE, body=body
        )
    except Exception as exc:
        raise K8sUnavailable(f"scale {name} -> {replicas} 失败: {exc}") from exc
    logger.info("K8s scale: deployment=%s ns=%s replicas=%d", name, NAMESPACE, replicas)
    return {"name": name, "namespace": NAMESPACE, "replicas": replicas}


def get_replicas(name: str) -> int | None:
    """读取目标 deployment 当前期望副本数（仅 get，用于校验/展示）。"""
    _check_target(name)
    api = _client()
    try:
        scale = api.read_namespaced_deployment_scale(name=name, namespace=NAMESPACE)
        return int(scale.spec.replicas or 0)
    except Exception as exc:
        raise K8sUnavailable(f"读取 {name} 副本数失败: {exc}") from exc


# ── 业务封装：故障注入 / 恢复（语义层）─────────────────────────────────────
def inject_fault(target: str = "redis") -> dict:
    """注入真实故障：目标副本数置 0（不自动恢复，须显式 remediate）。"""
    return set_replicas(target, 0)


def remediate(target: str = "redis") -> dict:
    """恢复服务：目标副本数置 1。

    用 PVC 持久数据重启，无需重建数据；Deployment controller 会调度回原节点。
    """
    return set_replicas(target, 1)
