"""Shared QLoRA training utilities used by train_clean_lora.py and train_extract_lora.py.

GPU-required (bitsandbytes/CUDA) — do not import this on the local M3 Max.
"""
import json
import os
import random

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(config_name: str) -> dict:
    path = os.path.join(BASE_DIR, "configs", config_name)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    full_path = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
    with open(full_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def doc_level_split(records: list[dict], split_cfg: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into train/val/test at the doc_id level (no leakage across splits),
    optionally stratified by a per-doc key (e.g. quality_bin)."""
    rng = random.Random(split_cfg.get("seed", 42))
    stratify_key = split_cfg.get("stratify_key")

    doc_ids = {}
    for rec in records:
        doc_ids.setdefault(rec["doc_id"], rec.get(stratify_key) if stratify_key else "all")

    by_stratum: dict[str, list[str]] = {}
    for doc_id, stratum in doc_ids.items():
        by_stratum.setdefault(stratum, []).append(doc_id)
    for ids in by_stratum.values():
        rng.shuffle(ids)

    val_frac = split_cfg.get("val_fraction", 0.1)
    test_frac = split_cfg.get("test_fraction", 0.1)

    train_ids, val_ids, test_ids = set(), set(), set()
    for ids in by_stratum.values():
        n = len(ids)
        n_val = round(n * val_frac)
        n_test = round(n * test_frac)
        val_ids.update(ids[:n_val])
        test_ids.update(ids[n_val:n_val + n_test])
        train_ids.update(ids[n_val + n_test:])

    train = [r for r in records if r["doc_id"] in train_ids]
    val = [r for r in records if r["doc_id"] in val_ids]
    test = [r for r in records if r["doc_id"] in test_ids]
    return train, val, test


def build_bnb_config(quant_cfg: dict) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )


def build_lora_config(lora_cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=lora_cfg.get("r", 32),
        lora_alpha=lora_cfg.get("alpha", 64),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_base_model_and_tokenizer(base_model: str, quant_cfg: dict):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=build_bnb_config(quant_cfg),
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def to_chat_dataset(examples: list[dict], system_prompt: str, user_fn, assistant_fn) -> Dataset:
    """examples -> HF Dataset of {messages: [...]}, one row per training example."""
    rows = []
    for ex in examples:
        rows.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_fn(ex)},
                {"role": "assistant", "content": assistant_fn(ex)},
            ]
        })
    return Dataset.from_list(rows)


def run_training(config_name: str, train_ds: Dataset, val_ds: Dataset, model, tokenizer,
                  response_template: str = "<|im_start|>assistant\n"):
    cfg = load_config(config_name)
    train_cfg = cfg["train"]

    def formatting_func(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    lora_config = build_lora_config(cfg["lora"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, cfg["adapter_out"] + "-checkpoints"),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        num_train_epochs=train_cfg["num_train_epochs"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg["eval_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        logging_steps=train_cfg["logging_steps"],
        seed=train_cfg["seed"],
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=formatting_func,
        data_collator=collator,
        max_seq_length=cfg["max_seq_length"],
        tokenizer=tokenizer,
    )
    trainer.train()

    adapter_out = os.path.join(BASE_DIR, cfg["adapter_out"])
    trainer.model.save_pretrained(adapter_out)
    tokenizer.save_pretrained(adapter_out)
    print(f"Saved adapter to {adapter_out}")
    return trainer
