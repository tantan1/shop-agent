#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""契约生成器（方案① T10）：从 OpenAPI 生成 client / model / mock / 属性测试。

把可推导部分从 AI 手里拿走，确保契约层面的确定性。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def validate_spec(spec_path: Path) -> bool:
    """校验 OpenAPI spec 合法性。非法 → 报错退出非 0。"""
    try:
        spec_text = spec_path.read_text(encoding="utf-8")
        import yaml
        data = yaml.safe_load(spec_text) if spec_path.suffix in (".yaml", ".yml") else json.loads(spec_text)
        assert isinstance(data, dict), "spec 必须是对象"
        assert "openapi" in data or "swagger" in data, "缺少 openapi/swagger 版本"
        return True
    except Exception as exc:
        print(f"[ERROR] spec 校验失败：{exc}", file=sys.stderr)
        return False


def generate_models(spec_path: Path, output_dir: Path) -> Optional[Path]:
    """生成 pydantic v2 models。"""
    try:
        models_path = output_dir / "models.py"
        subprocess.run(
            [
                sys.executable, "-m", "datamodel_code_generator",
                "--input", str(spec_path),
                "--output", str(models_path),
                "--input-file-type", "openapi",
                "--output-model-type", "pydantic_v2.BaseModel",
                "--target-python-version", "3.11",
            ],
            capture_output=True,
            check=False,
        )
        if models_path.exists() and models_path.stat().st_size > 0:
            return models_path
        return None
    except Exception as exc:
        print(f"[WARN] models 生成失败：{exc}", file=sys.stderr)
        return None


def generate_client(spec_path: Path, output_dir: Path) -> Optional[Path]:
    """生成 typed client（含 retry / 超时骨架）。"""
    try:
        client_path = output_dir / "client.py"
        spec_text = spec_path.read_text(encoding="utf-8")
        # 轻量实现：生成基于 httpx 的 client 骨架
        header = '''"""自动生成的 typed client（基于 OpenAPI spec）。"""
from __future__ import annotations

import httpx
from typing import Any, Optional

BASE_URL: str = ""
TIMEOUT: float = 30.0
MAX_RETRIES: int = 3


class Client:
    def __init__(self, base_url: str = BASE_URL, timeout: float = TIMEOUT) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(MAX_RETRIES):
            try:
                return self._client.request(method, path, **kwargs)
            except httpx.TransportError:
                if attempt == MAX_RETRIES - 1:
                    raise
        raise RuntimeError("unreachable")

'''
        client_path.write_text(header, encoding="utf-8")
        return client_path
    except Exception as exc:
        print(f"[WARN] client 生成失败：{exc}", file=sys.stderr)
        return None


def generate_schemathesis_tests(spec_path: Path, output_dir: Path) -> Optional[Path]:
    """生成 schemathesis 属性测试（第 1 层契约镜像测试，免审）。"""
    try:
        test_path = output_dir / "test_contract_schemathesis.py"
        header = '''"""自动生成的 schemathesis 属性测试（第 1 层契约镜像测试，免审）。"""
from __future__ import annotations

import schemathesis
from schemathesis import checks

schema = schemathesis.from_path("__spec__")


@schema.parametrize()
@checks
def test_api_contract(case: schemathesis.Case) -> None:
    case.call()
'''
        content = header.replace("__spec__", str(spec_path))
        _write(test_path, content)
        return test_path
    except Exception as exc:
        print(f"[WARN] schemathesis 测试生成失败：{exc}", file=sys.stderr)
        return None


def generate_mock_server(spec_path: Path, output_dir: Path) -> Optional[Path]:
    """生成 mock server（供 test 阶段依赖）。"""
    try:
        mock_path = output_dir / "mock_server.py"
        content = '''"""自动生成的 mock server（基于 OpenAPI spec）。"""
from __future__ import annotations

from typing import Any

MOCK_RESPONSES: dict[str, Any] = {}


def get_mock_response(path: str, method: str) -> tuple[int, Any]:
    key = f"{method.upper()} {path}"
    if key in MOCK_RESPONSES:
        return MOCK_RESPONSES[key]
    return 404, {"error": "not found"}
'''
        _write(mock_path, content)
        return mock_path
    except Exception as exc:
        print(f"[WARN] mock server 生成失败：{exc}", file=sys.stderr)
        return None


def generate(spec_path: Path, output_dir: Path) -> int:
    """主入口：给定合法 openapi.yaml，生成全部契约产物。"""
    if not spec_path.exists():
        print(f"[ERROR] spec 文件不存在：{spec_path}", file=sys.stderr)
        return 2

    if not validate_spec(spec_path):
        return 2

    models = generate_models(spec_path, output_dir)
    client = generate_client(spec_path, output_dir)
    tests = generate_schemathesis_tests(spec_path, output_dir)
    mock = generate_mock_server(spec_path, output_dir)

    summary = {
        "spec": str(spec_path),
        "output_dir": str(output_dir),
        "models": str(models) if models else None,
        "client": str(client) if client else None,
        "tests": str(tests) if tests else None,
        "mock": str(mock) if mock else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="契约生成器：OpenAPI -> models/client/tests/mock")
    ap.add_argument("--spec", required=True, help="OpenAPI spec 文件路径")
    ap.add_argument("--out-dir", required=True, help="输出目录")
    args = ap.parse_args()
    return generate(Path(args.spec), Path(args.out_dir))


if __name__ == "__main__":
    sys.exit(main())
