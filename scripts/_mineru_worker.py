"""
MinerU Worker - 独立子进程脚本 (magic-pdf v1.3.x)
在隔离的 venv 中运行，不污染主项目依赖

用法（由 convert_document.py 自动调用）:
    venv_mineru/Scripts/python scripts/_mineru_worker.py --input a.pdf --output a.md --device cpu

输出: JSON 格式结果到 stdout
    {"success": true, "word_count": 1234, "page_count": 10}
    {"success": false, "error": "error message"}
"""

import sys
import json
import argparse
import time
import tempfile
import os
from pathlib import Path

# ---- 修复 Windows HuggingFace symlink 权限错误 (WinError 1314) ----
def _copy_instead_of_symlink(src, dst, **kwargs):
    import shutil
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

if hasattr(os, "symlink"):
    os.symlink = _copy_instead_of_symlink
if hasattr(os, "link"):
    os.link = _copy_instead_of_symlink
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _patch_torch_load():
    """
    PyTorch 2.6+ 将 torch.load 的默认 weights_only 改为 True，
    导致 magic_pdf 加载 doclayout_yolo 自定义模型时报错。
    此补丁恢复 weights_only=False 的默认行为。
    """
    try:
        import torch
        _original_load = torch.load

        def _patched_load(f, map_location=None, pickle_module=None, *,
                          weights_only=False, mmap=None, **kwargs):
            return _original_load(
                f,
                map_location=map_location,
                pickle_module=pickle_module,
                weights_only=weights_only,
                mmap=mmap,
                **kwargs,
            )

        torch.load = _patched_load
        torch.serialization.load = _patched_load
        print("[Worker] torch.load 已补丁 (weights_only 默认恢复为 False)", file=sys.stderr)
    except ImportError:
        pass  # Torch 未安装，无需补丁


def main():
    parser = argparse.ArgumentParser(description="MinerU PDF to Markdown worker")
    parser.add_argument("--input", required=True, help="Input PDF path")
    parser.add_argument("--output", required=True, help="Output Markdown path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    try:
        _patch_torch_load()  # 必须在 magic_pdf 导入之前打补丁
        from magic_pdf.tools.common import do_parse
        import fitz  # PyMuPDF
    except ImportError as e:
        print(json.dumps({
            "success": False,
            "error": f"magic_pdf 依赖缺失: {e}。请在 mineru venv 中运行: pip install magic-pdf PyMuPDF opencv-python-headless"
        }))
        sys.exit(0)

    try:
        input_path = args.input
        output_path = args.output

        # 检查设备
        device = args.device
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("[Worker] CUDA 不可用，降级到 CPU", file=sys.stderr)
                    device = "cpu"
            except ImportError:
                print("[Worker] PyTorch 未安装，使用 CPU", file=sys.stderr)
                device = "cpu"

        # 获取 PDF 页数
        page_count = 0
        try:
            doc = fitz.open(input_path)
            page_count = len(doc)
            doc.close()
        except Exception:
            pass

        # 读取 PDF 字节
        with open(input_path, 'rb') as f:
            pdf_bytes = f.read()

        print(f"[Worker] 开始转换: {input_path} (device={device}, pages={page_count})", file=sys.stderr)

        start_time = time.time()

        # magic-pdf 1.3.x: 使用 do_parse 函数
        # do_parse 会在 output_dir 下创建 {pdf_name}/{method}/ 目录结构
        fname_stem = Path(input_path).stem
        with tempfile.TemporaryDirectory() as tmpdir:
            do_parse(
                output_dir=tmpdir,
                pdf_file_name=fname_stem,
                pdf_bytes_or_dataset=pdf_bytes,
                model_list=[],           # 使用内置模型
                parse_method='auto',     # auto: 先尝试 txt 提取，文字不足时自动回退到 OCR
                f_dump_md=True,
                f_dump_middle_json=False,
                f_dump_model_json=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False,
                f_draw_span_bbox=False,
                f_draw_layout_bbox=False,
                f_draw_model_bbox=False,
                f_draw_line_sort_bbox=False,
                f_draw_char_bbox=False,
            )

            # 查找生成的 .md 文件
            # 目录结构: tmpdir/{pdf_name}/{method}/{pdf_name}.md
            md_candidates = list(Path(tmpdir).rglob("*.md"))
            if not md_candidates:
                raise RuntimeError("do_parse 未生成 .md 文件")

            md_generated = md_candidates[0]  # 取第一个匹配的

            # 读取生成的 markdown 内容
            md_content = md_generated.read_text(encoding='utf-8')

            # 写入最终输出路径
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

        elapsed = time.time() - start_time

        print(json.dumps({
            "success": True,
            "word_count": len(md_content),
            "page_count": page_count,
            "processing_time": round(elapsed, 2),
        }))

    except Exception as e:
        import traceback
        print(json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }))
        sys.exit(0)


if __name__ == "__main__":
    main()
