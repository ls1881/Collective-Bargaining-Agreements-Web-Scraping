"""Evaluate the cleaning adapter: CER reduction vs. raw-OCR baseline, [ILLEGIBLE] placement.

Three modes:
  1. --generate --compare (recommended, GPU-required): generates with BOTH the plain base
     model (no adapter — same prompt, zero-shot, via peft's disable_adapter() context so the
     model is only loaded once) AND the fine-tuned adapter, on the same held-out test set, and
     prints both metric tables plus a summary delta — the evidence for "was fine-tuning
     necessary." The existing baseline_cer column only measures raw-input-vs-gold (how much
     editing was needed at all), not what an unfinetuned model already achieves when prompted.
     python finetune/eval_clean_cuda.py --generate --compare

  2. --generate (fine-tuned only, GPU-required):
     python finetune/eval_clean_cuda.py --generate

  3. Evaluate only (local, no GPU): computes metrics from a predictions JSONL you already
     have (e.g. produced on the GPU instance and copied down).
     python finetune/eval_clean_cuda.py --predictions finetune/data/clean_predictions.jsonl
"""
import argparse
import json
import os
from collections import defaultdict

import jiwer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SPLIT_PATH = os.path.join(BASE_DIR, "data", "clean_test_split.jsonl")
DEFAULT_PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "clean_predictions.jsonl")
BASELINE_PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "clean_predictions_baseline.jsonl")

ILLEGIBLE_TOKEN = "[ILLEGIBLE]"


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _generate_for_records(model, tokenizer, test_records, system_prompt):
    import torch
    predictions = []
    for rec in test_records:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rec["input"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            # Measured max target in the labeled dataset is 697 tokens; 1536 gives headroom.
            out = model.generate(**inputs, max_new_tokens=1536, do_sample=False)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predictions.append({**rec, "prediction": text.strip()})
    return predictions


def generate_predictions(cfg_name: str = "qlora_clean.yaml", compare: bool = False):
    """GPU-required: load base model + adapter, generate on the held-out test split.
    If compare=True, also generates with the adapter disabled (peft's disable_adapter()
    context — no second model load needed) and returns (baseline_preds, finetuned_preds)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from train_common_cuda import build_bnb_config, load_config

    cfg = load_config(cfg_name)
    adapter_path = os.path.join(BASE_DIR, cfg["adapter_out"])

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=build_bnb_config(cfg["quant"]),
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    test_records = load_jsonl(TEST_SPLIT_PATH)

    if compare:
        print("Generating with adapter DISABLED (zero-shot base model baseline)...")
        with model.disable_adapter():
            baseline_predictions = _generate_for_records(model, tokenizer, test_records, cfg["system_prompt"])
        with open(BASELINE_PREDICTIONS_PATH, "w", encoding="utf-8") as f:
            for rec in baseline_predictions:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(baseline_predictions)} predictions to {BASELINE_PREDICTIONS_PATH}")

        print("Generating with adapter ENABLED (fine-tuned)...")
        finetuned_predictions = _generate_for_records(model, tokenizer, test_records, cfg["system_prompt"])
        with open(DEFAULT_PREDICTIONS_PATH, "w", encoding="utf-8") as f:
            for rec in finetuned_predictions:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(finetuned_predictions)} predictions to {DEFAULT_PREDICTIONS_PATH}")
        return baseline_predictions, finetuned_predictions

    predictions = _generate_for_records(model, tokenizer, test_records, cfg["system_prompt"])
    with open(DEFAULT_PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(predictions)} predictions to {DEFAULT_PREDICTIONS_PATH}")
    return predictions


def compute_metrics(predictions: list[dict]):
    by_bin = defaultdict(list)
    for rec in predictions:
        by_bin[rec.get("quality_bin", "unknown")].append(rec)

    print(f"{'bin':<10} {'n':>5} {'raw_input_cer':>14} {'model_cer':>12} {'reduction':>11}")
    overall_baseline, overall_model = [], []
    for bin_name, recs in sorted(by_bin.items()):
        baseline_cers = [jiwer.cer(r["target"], r["input"]) for r in recs]
        model_cers = [jiwer.cer(r["target"], r["prediction"]) for r in recs]
        overall_baseline.extend(baseline_cers)
        overall_model.extend(model_cers)
        b, m = sum(baseline_cers) / len(baseline_cers), sum(model_cers) / len(model_cers)
        reduction = (b - m) / b if b > 0 else 0.0
        print(f"{bin_name:<10} {len(recs):>5} {b:>14.4f} {m:>12.4f} {reduction:>10.1%}")

    b, m = sum(overall_baseline) / len(overall_baseline), sum(overall_model) / len(overall_model)
    overall_cer = m
    print(f"{'overall':<10} {len(predictions):>5} {b:>14.4f} {m:>12.4f} {(b - m) / b if b > 0 else 0:>10.1%}")

    # [ILLEGIBLE] placement sanity check on severe-bin docs: does the model mark unrecoverable
    # spans instead of hallucinating text? Flag if the model's rate diverges a lot from gold's.
    severe = by_bin.get("severe", [])
    if severe:
        gold_rate = sum(r["target"].count(ILLEGIBLE_TOKEN) for r in severe) / len(severe)
        model_rate = sum(r["prediction"].count(ILLEGIBLE_TOKEN) for r in severe) / len(severe)
        print(f"\n[severe bin] avg [ILLEGIBLE] markers per doc — gold: {gold_rate:.2f}, model: {model_rate:.2f}")
        if model_rate < gold_rate * 0.5:
            print("  [!] Model uses [ILLEGIBLE] far less than gold on severe-bin docs — "
                  "possible hallucination risk, review manually.")

    return overall_cer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate predictions on GPU before evaluating")
    parser.add_argument("--compare", action="store_true",
                         help="Also generate with the adapter disabled (zero-shot base model) and print "
                              "both tables — the actual evidence for whether fine-tuning helped. Implies --generate.")
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS_PATH, help="Path to existing predictions JSONL")
    args = parser.parse_args()

    if args.compare:
        baseline_predictions, finetuned_predictions = generate_predictions(compare=True)

        print("\n" + "=" * 70)
        print("BASELINE — plain base model, adapter disabled, same prompt (zero-shot)")
        print("=" * 70)
        baseline_cer = compute_metrics(baseline_predictions)

        print("\n" + "=" * 70)
        print("FINE-TUNED — base model + clean-lora adapter")
        print("=" * 70)
        finetuned_cer = compute_metrics(finetuned_predictions)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        delta = (baseline_cer - finetuned_cer) / baseline_cer if baseline_cer > 0 else 0.0
        print(f"Zero-shot base model overall CER: {baseline_cer:.4f}")
        print(f"Fine-tuned adapter overall CER:   {finetuned_cer:.4f}")
        print(f"Fine-tuning reduced CER by:       {delta:.1%}")
        if finetuned_cer >= baseline_cer:
            print("[!] Fine-tuning did NOT improve on the zero-shot base model on this test set — "
                  "worth investigating.")
        return

    if args.generate:
        predictions = generate_predictions()
    else:
        predictions = load_jsonl(args.predictions)

    compute_metrics(predictions)


if __name__ == "__main__":
    main()
