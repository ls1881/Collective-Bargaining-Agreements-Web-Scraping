"""LoRA fine-tune the structured-extraction adapter on Apple Silicon (MLX) — mirrors
train_extract_lora_cuda.py.

Runs locally on the M3 Max, no cloud GPU. Run after distill_labels.py and after
review_sample.py has cleared the hallucination-rate gate.

Usage: python finetune/train_extract_lora_mlx.py
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

CONFIG_NAME = "qlora_extract_mlx.yaml"


def load_schema(cfg: dict) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, cfg["schema_path"]), encoding="utf-8") as f:
        return f.read()


def main():
    cfg = load_config(CONFIG_NAME)
    schema_json = load_schema(cfg)
    records = load_jsonl(cfg["labels_path"])
    print(f"Loaded {len(records)} extraction label records")

    train, val, test = doc_level_split(records, cfg["split"])
    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "extract_test_split.jsonl")
    with open(test_path, "w", encoding="utf-8") as f:
        for rec in test:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Held-out test split written to {test_path} (used by eval_extract_mlx.py)")

    def user_fn(ex):
        return f"Schema:\n\n{schema_json}\n\nDocument text:\n\n{ex['input']}"

    def assistant_fn(ex):
        return json.dumps(ex["target"], ensure_ascii=False)

    train_msgs = to_messages_records(train, cfg["system_prompt"], user_fn, assistant_fn)
    val_msgs = to_messages_records(val, cfg["system_prompt"], user_fn, assistant_fn)

    run_training(CONFIG_NAME, train_msgs, val_msgs)


if __name__ == "__main__":
    main()
