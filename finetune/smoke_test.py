"""Fast sanity check for the QLoRA training path — run this FIRST on a fresh GPU instance,
before launching a real training run. It exercises the exact same code path as
train_clean_lora.py / train_extract_lora.py (model load, 4-bit quant, LoRA wrap, SFTTrainer,
DataCollatorForCompletionOnlyLM) on a tiny 4-example slice, for 3 steps. If your environment
has a version/API mismatch (the single biggest risk on a paid GPU clock), this catches it in
under a minute instead of partway through a real run.

GPU-required. Usage: python finetune/smoke_test.py
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from train_common import (  # noqa: E402
    build_lora_config,
    load_base_model_and_tokenizer,
    load_config,
    load_jsonl,
    to_chat_dataset,
)


def run_smoke_test(config_name: str, records_path: str, user_fn, assistant_fn):
    print(f"\n=== smoke test: {config_name} ===")
    cfg = load_config(config_name)
    records = load_jsonl(records_path)[:4]
    assert len(records) >= 2, f"need at least 2 records in {records_path} for a smoke test"

    print("Loading base model in 4-bit (this is the slow part, ~1-3 min)...")
    model, tokenizer = load_base_model_and_tokenizer(cfg["base_model"], cfg["quant"])

    from peft import get_peft_model
    from transformers import TrainingArguments
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

    lora_config = build_lora_config(cfg["lora"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = to_chat_dataset(records, cfg["system_prompt"], user_fn, assistant_fn)

    def formatting_func(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    response_template = "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, "_smoke_test_output"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        max_steps=3,
        bf16=True,
        gradient_checkpointing=True,
        # Must match train_common.run_training's setting — the whole point of this smoke test
        # is to catch config problems (including this exact gotcha) before the real run.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        formatting_func=formatting_func,
        data_collator=collator,
        max_seq_length=cfg["max_seq_length"],
        tokenizer=tokenizer,
    )
    result = trainer.train()

    final_loss = result.training_loss
    assert final_loss == final_loss, "loss is NaN"  # NaN != NaN
    assert final_loss < 50, f"loss suspiciously high ({final_loss}) — check data/config"
    print(f"OK — 3 steps completed, training_loss={final_loss:.4f}")

    import shutil
    shutil.rmtree(args.output_dir, ignore_errors=True)


def main():
    run_smoke_test(
        "qlora_clean.yaml", os.path.join(BASE_DIR, "data", "clean_labels.jsonl"),
        user_fn=lambda ex: ex["input"], assistant_fn=lambda ex: ex["target"],
    )

    from train_extract_lora import load_schema
    extract_cfg = load_config("qlora_extract.yaml")
    schema_json = load_schema(extract_cfg)
    run_smoke_test(
        "qlora_extract.yaml", os.path.join(BASE_DIR, "data", "extract_labels.jsonl"),
        user_fn=lambda ex: f"Schema:\n\n{schema_json}\n\nDocument text:\n\n{ex['input']}",
        assistant_fn=lambda ex: json.dumps(ex["target"], ensure_ascii=False),
    )

    print("\nAll smoke tests passed. Safe to launch the real training runs.")


if __name__ == "__main__":
    main()
