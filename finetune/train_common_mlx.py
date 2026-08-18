"""Shared LoRA training utilities for Apple Silicon (MLX), mirroring train_common_cuda.py.

Runs natively on the M3 Max — no CUDA/bitsandbytes, no cloud GPU needed. Uses mlx-lm's
lower-level Python API directly (not the `mlx_lm.lora` CLI) so we get the same level of
control the CUDA path has: custom early stopping + best-checkpoint tracking, which the CLI
doesn't expose (its `run()` always builds its own reporting callback and ignores one you pass in).
"""
import json
import os
import random

import mlx.core as mx
import mlx.optimizers as optim
import yaml
from mlx.utils import tree_flatten
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import CacheDataset, ChatDataset
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
from mlx_lm.utils import load, save_config

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
    """Identical logic to train_common_cuda.py's doc_level_split — same seed, same behavior,
    so the CUDA and MLX paths train/eval on the exact same split given the same data."""
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

    train_recs = [r for r in records if r["doc_id"] in train_ids]
    val_recs = [r for r in records if r["doc_id"] in val_ids]
    test_recs = [r for r in records if r["doc_id"] in test_ids]
    return train_recs, val_recs, test_recs


def to_messages_records(examples: list[dict], system_prompt: str, user_fn, assistant_fn) -> list[dict]:
    """examples -> list of {"messages": [...]} dicts — mlx-lm's ChatDataset input format
    (equivalent to train_common_cuda.py's to_chat_dataset)."""
    return [
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_fn(ex)},
                {"role": "assistant", "content": assistant_fn(ex)},
            ]
        }
        for ex in examples
    ]


def load_base_model_and_tokenizer(base_model: str):
    model, tokenizer = load(base_model, tokenizer_config={"trust_remote_code": True})
    return model, tokenizer


class EarlyStoppingBestCheckpointCallback(TrainingCallback):
    """Tracks best val loss, saves that checkpoint's weights separately (mlx-lm's built-in
    saving only writes 'latest' + periodic numbered snapshots, no 'best' concept), and raises
    EarlyStop once val loss hasn't improved for `patience` evaluations — the MLX equivalent of
    the CUDA path's load_best_model_at_end=True + EarlyStoppingCallback(patience=10)."""

    class EarlyStop(Exception):
        pass

    def __init__(self, model, best_adapter_path: str, patience: int = 10):
        self.model = model
        self.best_adapter_path = best_adapter_path
        self.patience = patience
        self.best_val_loss = float("inf")
        self.bad_evals = 0

    def on_train_loss_report(self, train_info: dict):
        pass

    def on_val_loss_report(self, val_info: dict):
        val_loss = val_info["val_loss"]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.bad_evals = 0
            weights = dict(tree_flatten(self.model.trainable_parameters()))
            mx.save_safetensors(self.best_adapter_path, weights)
            print(f"  [new best] val_loss={val_loss:.4f} -> saved {self.best_adapter_path}")
        else:
            self.bad_evals += 1
            print(f"  [no improvement] {self.bad_evals}/{self.patience} (best={self.best_val_loss:.4f})")
            if self.bad_evals >= self.patience:
                raise EarlyStoppingBestCheckpointCallback.EarlyStop(
                    f"No val_loss improvement for {self.patience} evals, stopping early."
                )


def build_lora_params(lora_cfg: dict) -> dict:
    return {
        "rank": lora_cfg.get("rank", 16),
        "dropout": lora_cfg.get("dropout", 0.05),
        "scale": lora_cfg.get("scale", 20.0),
    }


def run_training(config_name: str, train_msg_records: list[dict], val_msg_records: list[dict],
                  iters_override: int | None = None):
    """Load the base model, wrap the last N layers in LoRA, train with early stopping, and
    save the best adapter + adapter_config.json to cfg['adapter_out'] — mirrors
    train_common_cuda.py's run_training().

    train_msg_records / val_msg_records: lists of {"messages": [...]} dicts, already built via
    to_messages_records() by the caller (train_clean_lora_mlx.py / train_extract_lora_mlx.py),
    since the user/assistant formatting differs per task exactly like on the CUDA path.
    """
    cfg = load_config(config_name)
    train_cfg = cfg["train"]

    model, tokenizer = load_base_model_and_tokenizer(cfg["base_model"])
    model.freeze()

    lora_params = build_lora_params(cfg["lora"])
    linear_to_lora_layers(model, cfg["lora"]["num_layers"], lora_params, use_dora=False)
    print_trainable_parameters(model)

    train_set = CacheDataset(ChatDataset(train_msg_records, tokenizer, mask_prompt=True))
    val_set = CacheDataset(ChatDataset(val_msg_records, tokenizer, mask_prompt=True))

    adapter_dir = os.path.join(BASE_DIR, cfg["adapter_out"])
    os.makedirs(adapter_dir, exist_ok=True)
    # mlx-lm's train() periodically overwrites "adapters.safetensors" with the LATEST weights
    # (plus numbered snapshots) on its own steps_per_save cadence. Our early-stopping callback
    # needs a SEPARATE file for "best so far" so the two saves (latest vs. best) don't race and
    # clobber each other on the same iteration — we copy best -> adapters.safetensors at the end.
    adapter_file = os.path.join(adapter_dir, "adapters.safetensors")
    best_adapter_file = os.path.join(adapter_dir, "best_adapters.safetensors")

    # iters: mlx-lm counts steps, not epochs. Convert the CUDA config's epoch count to a step
    # budget using this task's actual train-set size, so both backends get comparable exposure
    # to the data. Early stopping will very likely cut this short anyway on a dataset this small.
    steps_per_epoch = max(1, -(-len(train_msg_records) // train_cfg["batch_size"]))
    iters = iters_override or steps_per_epoch * train_cfg["num_epochs"]

    save_config({
        "fine_tune_type": "lora",
        "num_layers": cfg["lora"]["num_layers"],
        "lora_parameters": lora_params,
    }, os.path.join(adapter_dir, "adapter_config.json"))

    training_args = TrainingArgs(
        batch_size=train_cfg["batch_size"],
        iters=iters,
        val_batches=-1,  # use the entire (tiny) val set every time
        steps_per_report=train_cfg["steps_per_report"],
        steps_per_eval=train_cfg["steps_per_eval"],
        steps_per_save=train_cfg["steps_per_eval"],  # snapshot on the same cadence as eval
        adapter_file=adapter_file,
        max_seq_length=cfg["max_seq_length"],
        grad_checkpoint=train_cfg["grad_checkpoint"],
        grad_accumulation_steps=train_cfg["grad_accumulation_steps"],
    )

    opt = optim.AdamW(learning_rate=train_cfg["learning_rate"])

    callback = EarlyStoppingBestCheckpointCallback(
        model, best_adapter_file, patience=train_cfg["early_stopping_patience"],
    )

    print(f"Training for up to {iters} iters ({steps_per_epoch} steps/epoch x "
          f"{train_cfg['num_epochs']} epochs), early stopping patience={train_cfg['early_stopping_patience']}")
    try:
        train(
            model=model,
            optimizer=opt,
            train_dataset=train_set,
            val_dataset=val_set,
            args=training_args,
            training_callback=callback,
        )
    except EarlyStoppingBestCheckpointCallback.EarlyStop as e:
        print(f"Early stopped: {e}")

    # Promote the best-val-loss checkpoint to the standard "adapters.safetensors" name that
    # load_adapters()/eval/infer scripts expect, overwriting the training run's final/latest
    # weights (which may be worse than the best if loss ticked back up before stopping).
    if os.path.exists(best_adapter_file):
        import shutil
        shutil.copyfile(best_adapter_file, adapter_file)
    else:
        print("  [!] no best checkpoint was ever saved (val loss never improved past inf) — "
              "keeping the final/latest weights instead.")

    print(f"Saved best adapter (val_loss={callback.best_val_loss:.4f}) to {adapter_dir}")
    return model, tokenizer, adapter_dir
