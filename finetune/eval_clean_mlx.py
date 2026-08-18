"""Evaluate the MLX cleaning adapter: CER reduction vs. raw-OCR baseline, [ILLEGIBLE] placement.
Mirrors eval_clean_cuda.py.

Three modes:
  1. --generate --compare (recommended): generates with BOTH the plain base model (no adapter
     — same prompt, zero-shot) AND the fine-tuned adapter, on the same held-out test set, and
     prints both metric tables back to back. This is the actual evidence for "was fine-tuning
     necessary" — the existing baseline_cer column only measures raw-input-vs-gold (how much
     editing was needed at all), not what an unfinetuned model already achieves when prompted.
     python finetune/eval_clean_mlx.py --generate --compare

  2. --generate (fine-tuned only, as before):
     python finetune/eval_clean_mlx.py --generate

  3. Evaluate only, from predictions you already generated:
     python finetune/eval_clean_mlx.py --predictions finetune/data/clean_predictions_mlx.jsonl
"""
import argparse
import json
import os
from collections import defaultdict

import jiwer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SPLIT_PATH = os.path.join(BASE_DIR, "data", "clean_test_split.jsonl")
DEFAULT_PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "clean_predictions_mlx.jsonl")
BASELINE_PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "clean_predictions_mlx_baseline.jsonl")

ILLEGIBLE_TOKEN = "[ILLEGIBLE]"


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_predictions(cfg_name: str = "qlora_clean_mlx.yaml", use_adapter: bool = True,
                          out_path: str = None) -> list[dict]:
    from mlx_lm import generate
    from mlx_lm.utils import load

    from train_common_mlx import load_config

    cfg = load_config(cfg_name)
    out_path = out_path or (DEFAULT_PREDICTIONS_PATH if use_adapter else BASELINE_PREDICTIONS_PATH)

    if use_adapter:
        adapter_path = os.path.join(BASE_DIR, cfg["adapter_out"])
        print(f"Loading base model + adapter from {adapter_path}")
        model, tokenizer = load(cfg["base_model"], adapter_path=adapter_path,
                                 tokenizer_config={"trust_remote_code": True})
    else:
        print(f"Loading PLAIN base model (no adapter) — zero-shot baseline")
        model, tokenizer = load(cfg["base_model"], tokenizer_config={"trust_remote_code": True})

    test_records = load_jsonl(TEST_SPLIT_PATH)
    predictions = []
    for rec in test_records:
        messages = [
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": rec["input"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        # Measured max target in the labeled dataset is 697 tokens; 1536 gives headroom.
        text = generate(model, tokenizer, prompt, max_tokens=1536)
        predictions.append({**rec, "prediction": text.strip()})

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(predictions)} predictions to {out_path}")
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
    parser.add_argument("--generate", action="store_true", help="Generate predictions locally before evaluating")
    parser.add_argument("--compare", action="store_true",
                         help="Also generate with the plain base model (no adapter) and print both "
                              "tables — the actual evidence for whether fine-tuning helped. Implies --generate.")
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS_PATH, help="Path to existing predictions JSONL")
    args = parser.parse_args()

    if args.compare:
        print("\n" + "=" * 70)
        print("BASELINE — plain base model, no adapter, same prompt (zero-shot)")
        print("=" * 70)
        baseline_predictions = generate_predictions(use_adapter=False)
        baseline_cer = compute_metrics(baseline_predictions)

        print("\n" + "=" * 70)
        print("FINE-TUNED — base model + clean-lora-mlx adapter")
        print("=" * 70)
        finetuned_predictions = generate_predictions(use_adapter=True)
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
                  "worth investigating (more data, different hyperparameters, or the base model "
                  "may already be strong enough at this task without fine-tuning).")
        return

    if args.generate:
        predictions = generate_predictions(use_adapter=True)
    else:
        predictions = load_jsonl(args.predictions)

    compute_metrics(predictions)


if __name__ == "__main__":
    main()
