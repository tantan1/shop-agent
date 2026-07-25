"""Download only the essential MinerU model files from HuggingFace."""
import os
import shutil
from huggingface_hub import hf_hub_download

LOCAL_DIR = 'e:/workspace/shop-agent/venv_mineru/Lib/site-packages/magic_pdf/resources/models'
REPO = 'opendatalab/PDF-Extract-Kit-1.0'

# (repo_path, local_rel_path) — local_rel_path 可与 repo filename 不同（用于重命名）
# magic-pdf 1.3.x 引用 v3 名称，但仓库中只有 v4/v5，下载时自动重命名
FILES = [
    # ===== Layout 模型 =====
    ('models/Layout/LayoutLMv3/model_final.pth',
     'Layout/LayoutLMv3/model_final.pth'),
    ('models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt',
     'Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt'),

    # ===== OCR 检测模型 (magic-pdf 配置引用 v3，仓库为 v5，下载后重命名) =====
    ('models/OCR/paddleocr_torch/ch_PP-OCRv5_det_infer.pth',
     'OCR/paddleocr_torch/ch_PP-OCRv3_det_infer.pth'),
    ('models/OCR/paddleocr_torch/Multilingual_PP-OCRv3_det_infer.pth',
     'OCR/paddleocr_torch/Multilingual_PP-OCRv3_det_infer.pth'),

    # ===== OCR 识别模型 (中文主要用) =====
    ('models/OCR/paddleocr_torch/ch_PP-OCRv4_rec_server_doc_infer.pth',
     'OCR/paddleocr_torch/ch_PP-OCRv4_rec_server_doc_infer.pth'),

    # ===== 文本方向分类 =====
    ('models/OCR/paddleocr_torch/ch_ptocr_mobile_v2.0_cls_infer.pth',
     'OCR/paddleocr_torch/ch_ptocr_mobile_v2.0_cls_infer.pth'),
]

# en_PP-OCRv3_det_infer.pth 在仓库中不存在，用 Multilingual 版替代（重命名）
POST_RENAME = {
    'OCR/paddleocr_torch/Multilingual_PP-OCRv3_det_infer.pth': [
        'OCR/paddleocr_torch/en_PP-OCRv3_det_infer.pth',
    ],
}

total_size = 0

for src, dst_rel in FILES:
    dst = os.path.join(LOCAL_DIR, dst_rel)
    if os.path.exists(dst):
        size_mb = os.path.getsize(dst) / 1024 / 1024
        print(f'SKIP (exists): {dst_rel} ({size_mb:.1f} MB)')
        total_size += os.path.getsize(dst)
        continue

    print(f'Downloading: {src} ...')
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    path = hf_hub_download(REPO, src, cache_dir=None, local_files_only=False)

    shutil.copy2(path, dst)
    size_mb = os.path.getsize(dst) / 1024 / 1024
    total_size += os.path.getsize(dst)
    print(f'OK: {dst_rel} ({size_mb:.1f} MB)')

# 处理重命名别名（如 en_PP-OCRv3_det_infer.pth 不存在，复制自 Multilingual）
for src_key, aliases in POST_RENAME.items():
    src_path = os.path.join(LOCAL_DIR, src_key)
    if not os.path.exists(src_path):
        print(f'WARN: 源文件不存在，跳过别名复制: {src_key}')
        continue
    for alias in aliases:
        alias_path = os.path.join(LOCAL_DIR, alias)
        if os.path.exists(alias_path):
            continue
        shutil.copy2(src_path, alias_path)
        print(f'COPY (alias): {src_key} -> {alias}')

print(f'\nAll model files ready! (total: {total_size / 1024 / 1024:.1f} MB)')
