"""
训练 unified 模型：参数抽取 + 工具选择 单模型 SFT（transformers.Trainer + peft QLoRA）。

目标（多任务合并）：
  用一份 Qwen3-1.7B 替代生产两份独立 1.7B（参数抽取 + 工具选择），省约 6.8GB。
  safety 任务已砍（合规主判归云端大模型），本训练不含 safety 数据。

范式对齐（与 train_tool_select_sft.py 一致，呼应微调坑④/⑤）：
  - 4bit QLoRA（nf4, rank16/alpha32, q/k/v/o_proj），lr=2e-4, epochs=3, cosine, bf16。
  - 训练数据用 tokenizer.apply_chat_template 拼接（Qwen3 需 enable_thinking=False，
    与推理端一致，避免模板错位）。
  - assistant 回合才计 loss，prompt 段 -100 mask。
  - 左填充（padding_side=left），避免左右 padding 污染训练。

数据：data/llamafactory/shop_unified_v1.json（2956 条，param 1729 + tool_select 1227）。
  prompt 结构不改（保留原 system），task 仅作元信息——评测端零改动即可直接测。

用法：
  # 冒烟（S 子集 30 条 5 步，验证链路/模板/内存）
  python scripts/train_unified_sft.py --max-samples 30 --smoke

  # 全量训练，写 ./outputs/qwen17b-unified-lora
  python scripts/train_unified_sft.py

  # 训练后把 LoRA merge 回 base（供 eval / Ollama GGUF 转换使用）
  python scripts/train_unified_sft.py --merge \
      --merge-out ./models/Qwen3-1.7B-unified
"""

import sys
import os
import json
import argparse
from typing import Dict, List, Any

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

DEFAULT_DATA = os.path.join(ROOT, "data/llamafactory/shop_unified_v1.json")
DEFAULT_MODEL = os.path.join(ROOT, "models/Qwen3-1.7B")
DEFAULT_OUT = os.path.join(ROOT, "outputs/qwen17b-unified-lora")


# =============================================================================
# 数据：ShareGPT -> 带 loss mask 的训练样本（assistant 回合才训练）
# Qwen3 需 enable_thinking=False：训练/推理模板一致（坑④）
# =============================================================================
def build_examples(path: str, tokenizer, cutoff_len: int, task: str = None,
                   max_samples: int = None) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if task:
        raw = [r for r in raw if r.get("task") == task]
    if max_samples:
        raw = raw[:max_samples]

    examples = []
    skipped = 0
    for rec in raw:
        convs = rec.get("conversations", [])
        if len(convs) < 3:
            skipped += 1
            continue
        full_ids = tokenizer.apply_chat_template(convs, tokenize=True,
                                                 add_generation_prompt=False,
                                                 enable_thinking=False)
        prompt_ids = tokenizer.apply_chat_template(convs[:-1], tokenize=True,
                                                   add_generation_prompt=True,
                                                   enable_thinking=False)
        # transformers 5.x 返回 BatchEncoding（input_ids 为 list/int 序列）
        if hasattr(full_ids, "keys"):
            full_ids = full_ids["input_ids"]
        if hasattr(prompt_ids, "keys"):
            prompt_ids = prompt_ids["input_ids"]
        if hasattr(full_ids, "tolist"):
            full_ids = full_ids.tolist()
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # 右截断（保留末尾 assistant 段）—— M8 长提示超出 cutoff 时
        if len(full_ids) > cutoff_len:
            full_ids = full_ids[-cutoff_len:]
            labels = labels[-cutoff_len:]
        examples.append({"input_ids": full_ids, "labels": labels,
                         "attention_mask": [1] * len(full_ids)})
    print(f"[INFO] 构建样本 {len(examples)} 条（跳过 {skipped} 条格式异常）")
    return examples


def collate_fn(batch, tokenizer):
    """左填充到本批最长；labels 缺失处填 -100，attention_mask 填 0。"""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        L = len(b["input_ids"])
        pad = max_len - L
        input_ids.append([pad_id] * pad + b["input_ids"])
        labels.append([-100] * pad + b["labels"])
        attn.append([0] * pad + [1] * L)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


# =============================================================================
# 模型：4bit QLoRA + LoRA
# =============================================================================
def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 左填充（坑⑤）

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def merge_and_save(base_path: str, adapter_path: str, merge_out: str):
    """把 LoRA 合并回 base，产出独立完整模型（供 eval / Ollama GGUF 转换）。"""
    print(f"[MERGE] base={base_path} adapter={adapter_path} -> {merge_out}")
    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=False)
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(merge_out)
    tok.save_pretrained(merge_out)
    print(f"[MERGE] done -> {merge_out}")


# =============================================================================
# 主流程
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    ap.add_argument("--task", default=None, choices=[None, "param_extract", "tool_select"],
                    help="只训某任务子集（调试/对照用）")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--cutoff-len", type=int, default=2560)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--save-steps", type=int, default=25,
                    help="每 N 步保存一次 checkpoint（默认 25；adapter 仅几 MB，磁盘压力小）")
    ap.add_argument("--resume", action="store_true",
                    help="从 output-dir 下最新 checkpoint 续训（中断后重启用）")
    ap.add_argument("--smoke", action="store_true", help="冒烟：max_steps=5, epochs=1, 频繁存盘")
    ap.add_argument("--merge", action="store_true", help="训练后把 LoRA merge 回 base")
    ap.add_argument("--merge-out", default=os.path.join(ROOT, "models/Qwen3-1.7B-unified"))
    args = ap.parse_args()

    print(f"[INFO] 加载模型 {args.model} ...")
    model, tokenizer = load_model(args.model)

    examples = build_examples(args.data, tokenizer, args.cutoff_len,
                              task=args.task, max_samples=args.max_samples)
    ds = Dataset.from_list(examples)

    over = sum(1 for e in examples if len(e["input_ids"]) >= args.cutoff_len)
    if over:
        print(f"[WARN] {over} 条样本长度达到 cutoff={args.cutoff_len}（可能被右截断丢弃系统提示头部）")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1 if args.smoke else args.epochs,
        max_steps=5 if args.smoke else -1,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=5 if args.smoke else args.save_steps,
        save_total_limit=2,
        report_to="none",
        optim="adamw_torch",
        gradient_checkpointing=True,
    )

    data_collator = lambda b: collate_fn(b, tokenizer)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=data_collator,
    )

    print("[INFO] 开始训练 ...")
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[DONE] LoRA adapter -> {args.output_dir}")

    if args.merge:
        merge_and_save(args.model, args.output_dir, args.merge_out)


if __name__ == "__main__":
    main()
