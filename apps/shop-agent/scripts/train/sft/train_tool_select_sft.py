"""
训练「工具选择」SFT 适配器（QLoRA + transformers.Trainer + peft）。

为什么不用 LLaMA-Factory yaml：本计划「缺失依赖」表明确要求不依赖 llama_factory，
直接用 Trainer + peft 实现，复用参数抽取微调范式（同一套超参与模板纪律）。

范式对齐（呼应微调坑，保证与推理端一致）：
  - 复用 train_qwen_param.yaml 的超参：lora_rank=16, lora_alpha=32,
    lora_target 含 q/k/v/o_proj, lr=2e-4, epochs=3, cosine, bf16, 4bit QLoRA。
  - 训练样本的拼接用 tokenizer.apply_chat_template（qwen 模板），与推理端（eval
    脚本 FC 模式）完全一致的对话模板（坑④ 最致命：模板不一致 → 适配器整体失效）。
  - assistant 回合才计算 loss（prompt 用 -100 mask），只学「选工具」那一段。
  - 左填充（padding_side=left），避免批量推理生成起点被污染、命中率虚低（坑⑤）。

截断注意：M 规模系统提示约 2000+ token，原 param-sft 的 cutoff_len=1024 会切掉
工具列表；本脚本默认 cutoff_len=2560。可用 --only-scale S 跑短提示做快速验证。

用法：
  # 冒烟（S 规模 30 条，5 步，秒级~分钟级，验证配置/数据/模板能否跑通）
  python scripts/train_tool_select_sft.py --only-scale S --max-samples 30 --smoke

  # 全量训练（S+M，写入 ./outputs/qwen15b-tool-select-lora）
  python scripts/train_tool_select_sft.py

  # 训练完把 LoRA merge 回 base，产出独立模型（供 eval --fc 使用）
  python scripts/train_tool_select_sft.py --merge --output-dir outputs/qwen15b-tool-select-lora \
      --merge-out ./models/Qwen2.5-1.5B-Instruct-toolselect
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATA = os.path.join(ROOT, "data/llamafactory/shop_tool_select_v1.json")
DEFAULT_MODEL = os.path.join(ROOT, "models/Qwen2.5-1.5B-Instruct")
DEFAULT_OUT = os.path.join(ROOT, "outputs/qwen15b-tool-select-lora")


# =============================================================================
# 数据：ShareGPT -> 带 loss mask 的训练样本（仅 assistant 段参与训练）
# =============================================================================
def build_examples(path: str, tokenizer, cutoff_len: int, only_scale: str = None,
                   max_samples: int = None) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 按规模过滤（M 提示长，S 可用于快速验证）
    if only_scale:
        raw = [r for r in raw if r.get("scale") == only_scale]
    if max_samples:
        raw = raw[:max_samples]

    examples = []
    skipped = 0
    for rec in raw:
        convs = rec.get("conversations", [])
        if len(convs) < 3:
            skipped += 1
            continue
        # full = system+user+assistant；prompt = system+user（含 generation prompt 标记）
        full_ids = tokenizer.apply_chat_template(convs, tokenize=True,
                                                  add_generation_prompt=False)
        prompt_ids = tokenizer.apply_chat_template(convs[:-1], tokenize=True,
                                                    add_generation_prompt=True)
        if not isinstance(full_ids, list):
            full_ids = full_ids.tolist()
        if not isinstance(prompt_ids, list):
            prompt_ids = prompt_ids.tolist()

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # 右截断（保留末尾 assistant 段，丢弃最早系统提示）—— M 长提示超出 cutoff 时
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
        torch_dtype=torch.bfloat16,
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
    """把 LoRA 合并回 base，产出独立完整模型（供 eval --fc 直接加载）。"""
    print(f"[MERGE] base={base_path} adapter={adapter_path} -> {merge_out}")
    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=False)
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
    ap.add_argument("--only-scale", default=None, help="只训某规模（S/M），M 提示长")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--cutoff-len", type=int, default=2560)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="冒烟：max_steps=5, epochs=1, 频繁存盘")
    ap.add_argument("--merge", action="store_true", help="训练后把 LoRA merge 回 base")
    ap.add_argument("--merge-out", default=os.path.join(ROOT, "models/Qwen2.5-1.5B-Instruct-toolselect"))
    args = ap.parse_args()

    print(f"[INFO] 加载模型 {args.model} ...")
    model, tokenizer = load_model(args.model)

    examples = build_examples(args.data, tokenizer, args.cutoff_len,
                              only_scale=args.only_scale, max_samples=args.max_samples)
    ds = Dataset.from_list(examples)

    # 超出 cutoff 的样本提示（诊断用）
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
        save_steps=5 if args.smoke else 200,
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
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[DONE] LoRA adapter -> {args.output_dir}")

    if args.merge:
        merge_and_save(args.model, args.output_dir, args.merge_out)


if __name__ == "__main__":
    main()
