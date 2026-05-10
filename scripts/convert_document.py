"""
文档转换测试脚本
使用 MinerU 和 Docling 将 PDF/DOC 转换为 Markdown 格式

依赖安装:
    pip install mineru==3.1.6  # PDF转Markdown
    pip install docling==2.93.0  # DOC转Markdown

用法:
    # 转换单个PDF
    python scripts/convert_document.py --input ./documents/sample.pdf --output ./output/

    # 转换单个DOC
    python scripts/convert_document.py --input ./documents/sample.docx --output ./output/

    # 批量转换目录
    python scripts/convert_document.py --input ./documents/ --output ./output/ --batch

    # 使用指定工具
    python scripts/convert_document.py --input ./sample.pdf --tool mineru
    python scripts/convert_document.py --input ./sample.docx --tool docling
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

# ---- 修复 Windows HuggingFace symlink 权限错误 (WinError 1314) ----
# Windows 默认不允许普通用户创建符号链接，huggingface_hub 缓存模型时会报错
# 通过 monkey-patch 将 symlink/link 替换为文件复制
_original_symlink = getattr(os, "symlink", None)
_original_link = getattr(os, "link", None)

def _copy_instead_of_link(src, dst, **kwargs):
    """用 shutil.copy2 替代 os.symlink/os.link（绕过 Windows 权限限制）"""
    import shutil
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

if hasattr(os, "symlink"):
    os.symlink = _copy_instead_of_link
if hasattr(os, "link"):
    os.link = _copy_instead_of_link
# 同时也设置环境变量（双保险）
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ConversionResult:
    """转换结果"""
    input_path: str
    output_path: str
    tool: str
    success: bool
    error_message: str = ""
    processing_time: float = 0.0
    page_count: int = 0
    word_count: int = 0
    file_size: int = 0


# ============================================================================
# MinerU PDF 转换器 (v3.1.6)
# ============================================================================

class MinerUConverter:
    """
    MinerU 文档转换器（支持两种模式）:

    模式 1 - 内联模式：magic_pdf 可直接 import（不推荐，会有依赖冲突）
    模式 2 - 子进程模式（推荐）：magic_pdf 运行在独立的 venv_mineru 中

    设置子进程模式:
        python -m venv venv_mineru
        venv_mineru\\Scripts\\pip install magic-pdf PyMuPDF
    """

    MINERU_VENV = "venv_mineru"  # 独立 venv 目录名
    WORKER_SCRIPT = "scripts/_mineru_worker.py"  # 子进程 worker

    def __init__(self, device: str = "cpu"):
        """
        Args:
            device: 推理设备，'cuda' 或 'cpu'
        """
        self.device = device
        self._model = None
        self._initialized = False
        self._use_subprocess = False  # 是否使用外部 venv

    def _initialize(self):
        """延迟初始化模型（优先尝试内联，不行则切换到子进程模式）"""
        if self._initialized:
            return

        # 模式 1: 尝试直接导入 magic_pdf (v1.3.x)
        try:
            from magic_pdf.tools.common import do_parse
            # 验证核心依赖
            import fitz  # PyMuPDF

            if self.device == "cuda":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        print("警告: CUDA 不可用，将使用 CPU")
                        self.device = "cpu"
                except ImportError:
                    print("警告: PyTorch 未安装或不支持 CUDA")
                    self.device = "cpu"

            self._initialized = True
            self._use_subprocess = False
            print(f"MinerU 初始化完成（内联模式），使用设备: {self.device}")

        except ImportError:
            # 模式 2: 尝试使用独立 venv_mineru
            self._try_init_subprocess()

    def _try_init_subprocess(self):
        """尝试启用子进程模式（使用独立 venv）"""
        import subprocess

        project_root = str(Path(__file__).resolve().parent.parent)
        venv_dir = Path(project_root) / self.MINERU_VENV

        python_exe = venv_dir / "Scripts" / "python.exe"
        if not python_exe.exists():
            python_exe = venv_dir / "bin" / "python"  # Linux/Mac fallback
        if not python_exe.exists():
            raise ImportError(
                f"magic_pdf 未安装，且未找到独立 MinerU venv ({venv_dir})\n\n"
                f"推荐做法（零依赖冲突）:\n"
                f"  1. cd {project_root}\n"
                f"  2. python -m venv {self.MINERU_VENV}\n"
                f'  3. {python_exe} -m pip install magic-pdf PyMuPDF\n\n'
                f"或跳过 MinerU 直接使用 Docling:\n"
                f"  python scripts/convert_document.py --input a.pdf --tool docling"
            )

        # 验证 magic_pdf 是否可用 (v1.3.x)
        result = subprocess.run(
            [str(python_exe), "-c", "from magic_pdf.tools.common import do_parse; print('OK')"],
            capture_output=True, text=True, timeout=30,
            cwd=project_root, encoding='utf-8', errors='replace',
        )
        if "OK" not in result.stdout:
            raise ImportError(
                f"venv_mineru 存在但 magic_pdf 未安装。请运行:\n"
                f'  {python_exe} -m pip install magic-pdf PyMuPDF opencv-python-headless'
            )

        self._use_subprocess = True
        self._initialized = True
        print(f"MinerU 初始化完成（子进程模式），venv: {venv_dir}")

    def convert(
        self,
        input_path: str,
        output_path: str,
        extract_images: bool = True,
        extract_tables: bool = True,
    ) -> ConversionResult:
        """
        将 PDF 转换为 Markdown

        Returns:
            ConversionResult: 转换结果
        """
        import time
        start_time = time.time()

        result = ConversionResult(
            input_path=input_path,
            output_path=output_path,
            tool="mineru",
            success=False,
        )

        try:
            self._initialize()

            input_file = Path(input_path)
            if not input_file.exists():
                raise FileNotFoundError(f"文件不存在: {input_path}")

            result.file_size = input_file.stat().st_size

            if self._use_subprocess:
                self._convert_via_subprocess(input_path, output_path, result, start_time)
            else:
                self._convert_inline(input_path, output_path, result, start_time,
                                     extract_images, extract_tables)

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
            print(f"[MinerU] 转换失败: {e}")

        return result

    def _convert_via_subprocess(self, input_path: str, output_path: str,
                                 result: ConversionResult, start_time: float):
        """通过子进程调用独立 venv 中的 MinerU"""
        import subprocess
        import time

        project_root = str(Path(__file__).resolve().parent.parent)
        venv_dir = Path(project_root) / self.MINERU_VENV
        python_exe = str(venv_dir / "Scripts" / "python.exe")
        if not Path(python_exe).exists():
            python_exe = str(venv_dir / "bin" / "python")

        worker = str(Path(project_root) / self.WORKER_SCRIPT)

        cmd = [
            python_exe, worker,
            "--input", str(Path(input_path).resolve()),
            "--output", str(Path(output_path).resolve()),
            "--device", self.device,
        ]

        print(f"[MinerU] 子进程模式: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                               cwd=project_root, encoding='utf-8', errors='replace')

        if proc.returncode != 0:
            raise RuntimeError(f"Worker 异常退出 (code={proc.returncode}): {proc.stderr[:500]}")

        try:
            data = json.loads(proc.stdout.strip().split("\n")[-1])
        except json.JSONDecodeError:
            raise RuntimeError(f"Worker 输出解析失败: {proc.stdout[:500]}")

        if data.get("success"):
            result.success = True
            result.word_count = data.get("word_count", 0)
            result.page_count = data.get("page_count", 0)
            result.processing_time = data.get("processing_time", time.time() - start_time)
            print(f"[MinerU] 转换完成: {output_path}")
            print(f"  - 页数: {result.page_count}")
            print(f"  - 字数: {result.word_count}")
            print(f"  - 耗时: {result.processing_time:.2f}s")
        else:
            raise RuntimeError(data.get("error", "Unknown error"))

    def _convert_inline(self, input_path: str, output_path: str,
                         result: ConversionResult, start_time: float,
                         extract_images: bool, extract_tables: bool):
        """内联模式：直接调用 magic_pdf v1.3.x API"""
        import time
        import tempfile
        from magic_pdf.tools.common import do_parse

        # 获取 PDF 页数
        try:
            import fitz
            doc = fitz.open(input_path)
            result.page_count = len(doc)
            doc.close()
        except ImportError:
            result.page_count = 0

        # 读取 PDF 字节
        with open(input_path, 'rb') as f:
            pdf_bytes = f.read()

        fname_stem = Path(input_path).stem
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[MinerU] 开始转换（内联模式）: {input_path}")

        # 使用临时目录作为 magic-pdf 的输出目录
        with tempfile.TemporaryDirectory() as tmpdir:
            do_parse(
                output_dir=tmpdir,
                pdf_file_name=fname_stem,
                pdf_bytes_or_dataset=pdf_bytes,
                model_list=[],
                parse_method='auto',  # auto: 先尝试 txt 提取，文字不足时自动回退到 OCR
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
            md_candidates = list(Path(tmpdir).rglob("*.md"))
            if not md_candidates:
                raise RuntimeError("do_parse 未生成 .md 文件")

            md_content = md_candidates[0].read_text(encoding='utf-8')

            # 写入最终输出
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

        result.success = True
        result.word_count = len(md_content)
        result.processing_time = time.time() - start_time

        print(f"[MinerU] 转换完成: {output_path}")
        print(f"  - 页数: {result.page_count}")
        print(f"  - 字数: {result.word_count}")
        print(f"  - 耗时: {result.processing_time:.2f}s")



# ============================================================================
# Docling 转换器 (v2.93.0)
# ============================================================================

class DoclingConverter:
    """
    Docling 文档转换器
    用于将 DOC/DOCX/PDF 等格式转换为 Markdown

    Docling 特性:
    - 支持多种文档格式
    - 表格结构保持
    - 图片和公式识别
    - 简洁易用的 API
    """

    def __init__(self):
        self._initialized = False

    def _initialize(self):
        """初始化 Docling"""
        if self._initialized:
            return

        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

            self._initialized = True
            print("Docling 初始化完成")

        except ImportError as e:
            raise ImportError(
                f"Docling 未安装或依赖不完整。请运行: pip install docling==2.93.0\n"
                f"错误详情: {e}"
            )

    def convert(
        self,
        input_path: str,
        output_path: str,
        extract_images: bool = True,
        extract_tables: bool = True,
    ) -> ConversionResult:
        """
        将文档转换为 Markdown

        Args:
            input_path: 输入文件路径 (支持 .docx, .pdf, .doc 等)
            output_path: 输出 Markdown 文件路径
            extract_images: 是否提取图片
            extract_tables: 是否保留表格结构

        Returns:
            ConversionResult: 转换结果
        """
        import time
        start_time = time.time()

        result = ConversionResult(
            input_path=input_path,
            output_path=output_path,
            tool="docling",
            success=False,
        )

        try:
            self._initialize()

            input_file = Path(input_path)
            if not input_file.exists():
                raise FileNotFoundError(f"文件不存在: {input_path}")

            result.file_size = input_file.stat().st_size

            # 创建输出目录
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"[Docling] 开始转换: {input_path}")

            # 执行转换
            md_content = self._do_convert(
                input_path,
                extract_images=extract_images,
                extract_tables=extract_tables
            )

            # 保存结果
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

            result.success = True
            result.word_count = len(md_content)
            result.processing_time = time.time() - start_time
            result.page_count = md_content.count("\n---") + 1

            print(f"[Docling] 转换完成: {output_path}")
            print(f"  - 字数: {result.word_count}")
            print(f"  - 耗时: {result.processing_time:.2f}s")

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
            print(f"[Docling] 转换失败: {e}")

        return result

    def _do_convert(
        self,
        input_path: str,
        extract_images: bool,
        extract_tables: bool
    ) -> str:
        """
        执行 Docling 转换

        Args:
            input_path: 输入文件路径
            extract_images: 是否提取图片
            extract_tables: 是否提取表格

        Returns:
            Markdown 内容
        """
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        # 创建转换器
        converter = DocumentConverter()

        # 执行转换
        result = converter.convert(input_path)

        # 导出为 Markdown
        md_content = result.document.export_to_markdown()

        return md_content


# ============================================================================
# 批量转换器
# ============================================================================

class BatchConverter:
    """批量文档转换器"""

    def __init__(self, tool: str = "auto"):
        """
        初始化批量转换器

        Args:
            tool: 转换工具，'mineru', 'docling', 或 'auto'（自动选择）
        """
        self.tool = tool
        self._mineru: Optional[MinerUConverter] = None
        self._docling: Optional[DoclingConverter] = None

    def _get_converter(self, file_path: str):
        """根据文件类型获取转换器，mineru 不可用时自动降级到 docling"""
        ext = Path(file_path).suffix.lower()

        if self.tool == "mineru" or (self.tool == "auto" and ext == ".pdf"):
            try:
                if self._mineru is None:
                    self._mineru = MinerUConverter()
                return self._mineru
            except Exception as e:
                print(f"[WARN] MinerU 初始化失败 ({e})，降级到 Docling")
                self.tool = "docling"

        if self.tool == "docling" or self.tool == "auto":
            if self._docling is None:
                self._docling = DoclingConverter()
            return self._docling

        raise ValueError(f"不支持的文件类型: {ext}")

    def convert_file(
        self,
        input_path: str,
        output_path: str,
        **kwargs
    ) -> ConversionResult:
        """转换单个文件"""
        converter = self._get_converter(input_path)
        return converter.convert(input_path, output_path, **kwargs)

    def convert_batch(
        self,
        input_dir: str,
        output_dir: str,
        file_patterns: List[str] = None,
        **kwargs
    ) -> List[ConversionResult]:
        """
        批量转换目录中的文件

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            file_patterns: 文件匹配模式列表

        Returns:
            转换结果列表
        """
        if file_patterns is None:
            file_patterns = ["*.pdf", "*.docx", "*.doc"]

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []
        files_to_convert = []

        # 收集要转换的文件
        for pattern in file_patterns:
            files_to_convert.extend(input_path.glob(pattern))

        print(f"\n{'='*60}")
        print(f"批量转换: {len(files_to_convert)} 个文件")
        print(f"{'='*60}\n")

        for i, file_path in enumerate(files_to_convert, 1):
            print(f"[{i}/{len(files_to_convert)}] 处理: {file_path.name}")

            # 构建输出路径
            output_file = output_path / f"{file_path.stem}.md"

            # 执行转换
            result = self.convert_file(str(file_path), str(output_file), **kwargs)
            results.append(result)

            print()

        # 打印汇总
        self._print_summary(results)

        return results

    def _print_summary(self, results: List[ConversionResult]):
        """打印转换汇总"""
        total = len(results)
        success = sum(1 for r in results if r.success)
        failed = total - success
        total_time = sum(r.processing_time for r in results)
        total_size = sum(r.file_size for r in results)

        print(f"\n{'='*60}")
        print("批量转换完成")
        print(f"{'='*60}")
        print(f"总计: {total} 个文件")
        print(f"成功: {success} 个")
        print(f"失败: {failed} 个")
        print(f"总耗时: {total_time:.2f}s")
        print(f"总大小: {total_size / 1024 / 1024:.2f} MB")
        print(f"{'='*60}")

        if failed > 0:
            print("\n失败文件:")
            for r in results:
                if not r.success:
                    print(f"  - {r.input_path}: {r.error_message}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="文档转换工具 (MinerU + Docling)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/convert_document.py --input sample.pdf --output ./output/
  python scripts/convert_document.py --input sample.docx --tool docling
  python scripts/convert_document.py --input ./docs/ --output ./md/ --batch

支持的格式:
  MinerU:  PDF
  Docling: PDF, DOCX, DOC, HTML, 图片等
        """
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入文件或目录路径"
    )

    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出文件或目录路径"
    )

    parser.add_argument(
        "--tool", "-t",
        choices=["mineru", "docling", "auto"],
        default="auto",
        help="转换工具 (默认: auto 根据文件类型自动选择)"
    )

    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="批量转换模式 (input 必须是目录)"
    )

    parser.add_argument(
        "--no-images",
        action="store_true",
        help="不提取图片"
    )

    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="不保留表格结构"
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="MinerU 使用的设备 (默认: cuda)"
    )

    args = parser.parse_args()

    # 参数处理
    extract_images = not args.no_images
    extract_tables = not args.no_tables

    try:
        if args.batch:
            # 批量转换
            converter = BatchConverter(tool=args.tool)
            converter.convert_batch(
                input_dir=args.input,
                output_dir=args.output,
                extract_images=extract_images,
                extract_tables=extract_tables
            )
        else:
            # 单文件转换
            if Path(args.input).is_dir():
                print("错误: 单文件模式需要指定文件路径，而非目录")
                print("使用 --batch 参数进行批量转换")
                sys.exit(1)

            converter = BatchConverter(tool=args.tool)
            result = converter.convert_file(
                input_path=args.input,
                output_path=args.output,
                extract_images=extract_images,
                extract_tables=extract_tables
            )

            if result.success:
                print(f"\n转换成功!")
                print(f"输出文件: {result.output_path}")
            else:
                print(f"\n转换失败: {result.error_message}")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n用户取消操作")
        sys.exit(130)
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
