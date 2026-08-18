"""LoRA fine-tune the OCR-cleaning adapter on Apple Silicon (MLX) — mirrors train_clean_lora_cuda.py.

Runs locally on the M3 Max, no cloud GPU. Run after distill_labels.py and after
review_sample.py has cleared the hallucination-rate gate.

Usage: python finetune/train_clean_lora_mlx.py
"""
import json
import os

from train_common_mlx import (
    doc_level_split,
    load_config,
    load_jsonl,
    run_training,
    to_messages_records,
)

CONFIG_NAME = "qlora_clean_mlx.yaml"


def main():
    cfg = load_config(CONFIG_NAME)
    records = load_jsonl(cfg["labels_path"])
    print(f"Loaded {len(records)} cleaning label records")

    train, val, test = doc_level_split(records, cfg["split"])
    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "clean_test_split.jsonl")
    with open(test_path, "w", encoding="utf-8") as f:
        for rec in test:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Held-out test split written to {test_path} (used by eval_clean_mlx.py)")

    train_msgs = to_messages_records(
        train, cfg["system_prompt"],
        user_fn=lambda ex: ex["input"],
        assistant_fn=lambda ex: ex["target"],
    )
    val_msgs = to_messages_records(
        val, cfg["system_prompt"],
        user_fn=lambda ex: ex["input"],
        assistant_fn=lambda ex: ex["target"],
    )

    run_training(CONFIG_NAME, train_msgs, val_msgs)


if __name__ == "__main__":
    main()
