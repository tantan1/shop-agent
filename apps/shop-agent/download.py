"""
模型下载工具
用法：
    python download.py bge-m3            # 下载 BGE-M3 embedding 模型
    python download.py qwen3-tokenizer   # 下载 Qwen3 tokenizer（仅词表，~15MB，用于 token 限流）
    python download.py qwen3-1.7b        # 下载 Qwen3-1.7B 完整模型（约 4GB）
"""

import os
import sys
from huggingface_hub import snapshot_download, hf_hub_download

HF_ENDPOINT = "https://hf-mirror.com"
MODELS_DIR = "./models"


def download_bge_m3():
    """下载 BGE-M3 embedding 模型"""
    target_dir = os.path.join(MODELS_DIR, "bge-m3")
    print(f"[bge-m3] 下载到 {target_dir} ...")
    snapshot_download(
        repo_id="BAAI/bge-m3",
        local_dir=target_dir,
        endpoint=HF_ENDPOINT,
        local_dir_use_symlinks=False,
    )
    print("[bge-m3] ✅ 下载完成")


def download_qwen3_tokenizer():
    """下载 Qwen3 系列 tokenizer 文件（仅词表，不下载模型权重）

    Qwen3 全系列（0.6B/1.7B/3.6B/flash/plus）共用同一套 tokenizer。
    使用 Qwen3-0.6B 仓库（最小，无权重文件要过滤）。
    文件清单：
      - tokenizer.json       (~11MB)  BPE 词表（核心文件）
      - tokenizer_config.json(~10KB)  分词器配置
      - vocab.json            (~3MB)  词表
      - merges.txt            (~2MB)  BPE 合并规则
    总计约 16MB，用于 HF tokenizers 库做 Token 消耗预估。
    """
    repo_id = "Qwen/Qwen3-0.6B"
    target_dir = os.path.join(MODELS_DIR, "Qwen3-0.6B")
    os.makedirs(target_dir, exist_ok=True)

    files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]

    for fname in files:
        target_path = os.path.join(target_dir, fname)
        if os.path.exists(target_path):
            print(f"[qwen3-tokenizer] 跳过已存在: {fname}")
            continue
        print(f"[qwen3-tokenizer] 下载 {fname} ...")
        hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            local_dir=target_dir,
            endpoint=HF_ENDPOINT,
            local_dir_use_symlinks=False,
        )

    print(f"[qwen3-tokenizer] ✅ 下载完成 → {target_dir}")
    print(f"[qwen3-tokenizer] 请在 .env 设置: TOKENIZER_PATH={target_dir}/tokenizer.json")


def download_qwen3_1_7b():
    """下载 Qwen3-1.7B 完整模型（约 4GB，含 tokenizer）"""
    repo_id = "Qwen/Qwen3-1.7B"
    target_dir = os.path.join(MODELS_DIR, "Qwen3-1.7B")
    print(f"[qwen3-1.7b] 下载到 {target_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        endpoint=HF_ENDPOINT,
        local_dir_use_symlinks=False,
    )
    print("[qwen3-1.7b] ✅ 下载完成")


COMMANDS = {
    "bge-m3": download_bge_m3,
    "qwen3-tokenizer": download_qwen3_tokenizer,
    "qwen3-1.7b": download_qwen3_1_7b,
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("可用命令:")
        for name in COMMANDS:
            print(f"  python download.py {name}")
        sys.exit(0)

    cmd = sys.argv[1]
    fn = COMMANDS.get(cmd)
    if fn is None:
        print(f"未知命令: {cmd}")
        print(f"可用: {list(COMMANDS.keys())}")
        sys.exit(1)

    fn()