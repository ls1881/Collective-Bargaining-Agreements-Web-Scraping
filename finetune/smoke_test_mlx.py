"""Fast sanity check for the MLX (Apple Silicon) training path — run this first, before a
real training run. Exercises the exact same code path as train_clean_lora_mlx.py /
train_extract_lora_mlx.py (model load, LoRA wrap, training loop, early-stopping callback) on a
tiny 4-example slice for a handful of iterations.

Usage: python finetune/smoke_test_mlx.py
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import mlx.optimizers as optim  # noqa: E402
from mlx_lm.tuner.datasets import CacheDataset, ChatDataset  # noqa: E402
from mlx_lm.tuner.trainer import TrainingArgs, train  # noqa: E402
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters  # noqa: E402

from train_common_mlx import (  # noqa: E402
    EarlyStoppingBestCheckpointCallback,
    build_lora_params,
    load_base_model_and_tokenizer,
    load_config,
    load_jsonl,
    to_messages_records,
)


def run_smoke_test(config_name: str, records_path: str, user_fn, assistant_fn):
    print(f"\n=== smoke test: {config_name} ===")
    cfg = load_config(config_name)
    records = load_jsonl(records_path)[:4]
    assert len(records) >= 2, f"need at least 2 records in {records_path} for a smoke test"

    print("Loading base model (first run downloads ~4.3GB from mlx-community, then it's cached)...")
    model, tokenizer = load_base_model_and_tokenizer(cfg["base_model"])
    model.freeze()

    lora_params = build_lora_params(cfg["lora"])
    linear_to_lora_layers(model, cfg["lora"]["num_layers"], lora_params, use_dora=False)
    print_trainable_parameters(model)

    msg_records = to_messages_records(records, cfg["system_prompt"], user_fn, assistant_fn)
    # reuse the same tiny set for train/val — this is an API/config smoke test, not a quality check
    train_set = CacheDataset(ChatDataset(msg_records, tokenizer, mask_prompt=True))
    val_set = CacheDataset(ChatDataset(msg_records, tokenizer, mask_prompt=True))

    smoke_dir = os.path.join(BASE_DIR, "_smoke_test_mlx_output")
    os.makedirs(smoke_dir, exist_ok=True)
    adapter_file = os.path.join(smoke_dir, "adapters.safetensors")
    best_adapter_file = os.path.join(smoke_dir, "best_adapters.safetensors")

    args = TrainingArgs(
        batch_size=1,
        iters=3,
        val_batches=-1,
        steps_per_report=1,
        steps_per_eval=1,
        steps_per_save=1,
        adapter_file=adapter_file,
        max_seq_length=cfg["max_seq_length"],
        grad_checkpoint=True,
        grad_accumulation_steps=1,
    )
    opt = optim.AdamW(learning_rate=2e-4)
    # patience=10 with only 3 iters means this never actually triggers early stopping — the
    # point here is just confirming the callback wiring (best-checkpoint saving) doesn't crash.
    callback = EarlyStoppingBestCheckpointCallback(model, best_adapter_file, patience=10)

    train(
        model=model, optimizer=opt, train_dataset=train_set, val_dataset=val_set,
        args=args, training_callback=callback,
    )

    assert os.path.exists(best_adapter_file), "best checkpoint was never saved — callback didn't fire"
    assert callback.best_val_loss < 50, f"loss suspiciously high ({callback.best_val_loss}) — check data/config"
    print(f"OK — 3 iters completed, best val_loss={callback.best_val_loss:.4f}")

    import shutil
    shutil.rmtree(smoke_dir, ignore_errors=True)


def main():
    run_smoke_test(
        "qlora_clean_mlx.yaml", os.path.join(BASE_DIR, "data", "clean_labels.jsonl"),
        user_fn=lambda ex: ex["input"], assistant_fn=lambda ex: ex["target"],
    )

    from train_extract_lora_mlx import load_schema
    extract_cfg = load_config("qlora_extract_mlx.yaml")
    schema_json = load_schema(extract_cfg)
    run_smoke_test(
        "qlora_extract_mlx.yaml", os.path.join(BASE_DIR, "data", "extract_labels.jsonl"),
        user_fn=lambda ex: f"Schema:\n\n{schema_json}\n\nDocument text:\n\n{ex['input']}",
        assistant_fn=lambda ex: json.dumps(ex["target"], ensure_ascii=False),
    )

    print("\nAll MLX smoke tests passed. Safe to launch the real training runs.")


if __name__ == "__main__":
    main()
