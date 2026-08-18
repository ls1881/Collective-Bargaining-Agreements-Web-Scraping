"""Evaluate the MLX extraction adapter: JSON parse rate, per-field exact-match/F1, and
null-vs-present F1. Mirrors eval_extract_cuda.py.

Three modes:
  1. --generate --compare (recommended): generates with BOTH the plain base model (no adapter
     — same prompt, zero-shot) AND the fine-tuned adapter, on the same held-out test set, and
     prints both metric tables plus a summary delta — the evidence for "was fine-tuning
     necessary" (a capable base model can often already emit plausible-looking JSON zero-shot;
     the real questions are parse reliability and whether it hallucinates non-null values).
     python finetune/eval_extract_mlx.py --generate --compare

  2. --generate (fine-tuned only):
     python finetune/eval_extract_mlx.py --generate

  3. Evaluate only, from predictions you already generated:
     python finetune/eval_extract_mlx.py --predictions finetune/data/extract_predictions_mlx.jsonl
"""
import argparse
import json
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SPLIT_PATH = os.path.join(BASE_DIR, "data", "extract_test_split.jsonl")
DEFAULT_PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "extract_predictions_mlx.jsonl")
BASELINE_PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "extract_predictions_mlx_baseline.jsonl")

SCALAR_FIELDS = ["sector", "weekly_hours"]
NESTED_SCALAR_GROUPS = {
    "probation_period_days": ["operai", "impiegati", "quadri", "dirigenti"],
    "notice_period_days": ["operai", "impiegati", "quadri", "dirigenti"],
    "leave_entitlements": ["annual_leave_days", "sick_leave_terms", "parental_leave_terms"],
}


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_predictions(cfg_name: str = "qlora_extract_mlx.yaml", use_adapter: bool = True,
                          out_path: str = None) -> list[dict]:
    from mlx_lm import generate
    from mlx_lm.utils import load

    from train_common_mlx import load_config
    from train_extract_lora_mlx import load_schema

    cfg = load_config(cfg_name)
    schema_json = load_schema(cfg)
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
        user_content = f"Schema:\n\n{schema_json}\n\nDocument text:\n\n{rec['input']}"
        messages = [
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        # Measured against the labeled dataset: target JSON runs up to 2086 tokens.
        text = generate(model, tokenizer, prompt, max_tokens=3072)
        predictions.append({"doc_id": rec["doc_id"], "target": rec["target"], "prediction_raw": text.strip()})

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(predictions)} predictions to {out_path}")
    return predictions


def parse_prediction(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None


def wage_increases_f1(gold: list, pred: list) -> tuple[float, float, float]:
    def norm(item):
        return (
            item.get("effective_date"), item.get("level_or_category"),
            item.get("amount_eur"), item.get("amount_pct"),
        )
    gold_set, pred_set = {norm(g) for g in (gold or [])}, {norm(p) for p in (pred or [])}
    tp = len(gold_set & pred_set)
    precision = tp / len(pred_set) if pred_set else (1.0 if not gold_set else 0.0)
    recall = tp / len(gold_set) if gold_set else (1.0 if not pred_set else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def scalar_field_stats(field: str, gold, pred, stats: dict):
    gold_present = gold is not None
    pred_present = pred is not None
    if gold_present and pred_present:
        stats[field]["tp"] += 1
        stats[field]["exact_match"] += int(gold == pred)
        stats[field]["exact_total"] += 1
    elif gold_present and not pred_present:
        stats[field]["fn"] += 1
    elif not gold_present and pred_present:
        stats[field]["fp"] += 1
    else:
        stats[field]["tn"] += 1


def compute_metrics(predictions: list[dict]) -> dict:
    n = len(predictions)
    parsed = [{**p, "prediction": parse_prediction(p["prediction_raw"])} for p in predictions]
    parse_failures = sum(1 for p in parsed if p["prediction"] is None)
    parse_failure_rate = parse_failures / n if n else 0.0
    print(f"Parse failure rate: {parse_failures}/{n} ({parse_failure_rate:.1%})\n")

    valid = [p for p in parsed if p["prediction"] is not None]
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "exact_match": 0, "exact_total": 0})

    for rec in valid:
        gold, pred = rec["target"], rec["prediction"]
        for field in SCALAR_FIELDS:
            scalar_field_stats(field, gold.get(field), pred.get(field), stats)
        for group, subfields in NESTED_SCALAR_GROUPS.items():
            gold_group, pred_group = gold.get(group) or {}, pred.get(group) or {}
            for sub in subfields:
                scalar_field_stats(f"{group}.{sub}", gold_group.get(sub), pred_group.get(sub), stats)

    wage_f1s = []
    for rec in valid:
        _, _, f1 = wage_increases_f1(rec["target"].get("wage_increases", []), rec["prediction"].get("wage_increases", []))
        wage_f1s.append(f1)

    print(f"{'field':<36} {'null_f1':>8} {'exact_match':>12} {'n':>5}")
    low_f1_fields = []
    field_f1s = []
    for field, s in sorted(stats.items()):
        if s["tp"] + s["fp"] + s["fn"] == 0:
            # Field was null in gold for every test example AND the model never predicted a
            # non-null value either — zero opportunities to test non-null handling in this
            # sample (not the same as failure; precision/recall are undefined here, not 0).
            # Report N/A and exclude from the average/flagging rather than silently scoring it
            # as if the model got it wrong.
            exact = s["exact_match"] / s["exact_total"] if s["exact_total"] else float("nan")
            print(f"{field:<36} {'N/A':>8} {exact:>12.2f} {s['tn']:>5}  (always null in gold+pred, {s['tn']} tn)")
            continue
        precision = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0
        recall = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0.0
        null_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        field_f1s.append(null_f1)
        exact = s["exact_match"] / s["exact_total"] if s["exact_total"] else float("nan")
        print(f"{field:<36} {null_f1:>8.2f} {exact:>12.2f} {s['tp'] + s['fp'] + s['fn'] + s['tn']:>5}")
        if null_f1 < 0.7:
            low_f1_fields.append(field)

    avg_wage_f1 = sum(wage_f1s) / len(wage_f1s) if wage_f1s else 0.0
    print(f"{'wage_increases (set F1)':<36} {avg_wage_f1:>8.2f} {'-':>12} {len(valid):>5}")
    if avg_wage_f1 < 0.7:
        low_f1_fields.append("wage_increases")

    if low_f1_fields:
        print(f"\n[!] Fields below F1 0.7 threshold — revise schema/prompt or consider dropping: {low_f1_fields}")

    avg_field_f1 = (sum(field_f1s) + avg_wage_f1) / (len(field_f1s) + 1) if field_f1s else avg_wage_f1
    return {"parse_failure_rate": parse_failure_rate, "avg_field_f1": avg_field_f1}


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
        baseline_stats = compute_metrics(baseline_predictions)

        print("\n" + "=" * 70)
        print("FINE-TUNED — base model + extract-lora-mlx adapter")
        print("=" * 70)
        finetuned_predictions = generate_predictions(use_adapter=True)
        finetuned_stats = compute_metrics(finetuned_predictions)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'':<30} {'baseline':>12} {'fine-tuned':>12}")
        print(f"{'JSON parse failure rate':<30} {baseline_stats['parse_failure_rate']:>12.1%} "
              f"{finetuned_stats['parse_failure_rate']:>12.1%}")
        print(f"{'Avg field F1 (incl. wages)':<30} {baseline_stats['avg_field_f1']:>12.2f} "
              f"{finetuned_stats['avg_field_f1']:>12.2f}")
        if (finetuned_stats["avg_field_f1"] <= baseline_stats["avg_field_f1"]
                and finetuned_stats["parse_failure_rate"] >= baseline_stats["parse_failure_rate"]):
            print("[!] Fine-tuning did NOT improve either metric over the zero-shot base model on "
                  "this test set — worth investigating (more data, different hyperparameters, or "
                  "the base model may already be strong enough at this task).")
        return

    if args.generate:
        predictions = generate_predictions(use_adapter=True)
    else:
        predictions = load_jsonl(args.predictions)

    compute_metrics(predictions)


if __name__ == "__main__":
    main()
