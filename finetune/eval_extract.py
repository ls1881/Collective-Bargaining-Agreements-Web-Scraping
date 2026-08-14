"""Evaluate the extraction adapter: JSON parse rate, per-field exact-match/F1,
and null-vs-present F1 (to catch hallucinated non-null values).

Two modes, same pattern as eval_clean.py:
  python finetune/eval_extract.py --generate                                   # GPU-required
  python finetune/eval_extract.py --predictions finetune/data/extract_predictions.jsonl  # local
"""
import argparse
import json
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SPLIT_PATH = os.path.join(BASE_DIR, "data", "extract_test_split.jsonl")
DEFAULT_PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "extract_predictions.jsonl")

SCALAR_FIELDS = ["sector", "weekly_hours"]
NESTED_SCALAR_GROUPS = {
    "probation_period_days": ["operai", "impiegati", "quadri", "dirigenti"],
    "notice_period_days": ["operai", "impiegati", "quadri", "dirigenti"],
    "leave_entitlements": ["annual_leave_days", "sick_leave_terms", "parental_leave_terms"],
}
LIST_FIELDS = ["wage_increases"]


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_predictions(cfg_name: str = "qlora_extract.yaml") -> list[dict]:
    """GPU-required: load base model + adapter, generate on the held-out test split."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from train_common import build_bnb_config, load_config
    from train_extract_lora import load_schema

    cfg = load_config(cfg_name)
    schema_json = load_schema(cfg)
    adapter_path = os.path.join(BASE_DIR, cfg["adapter_out"])

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=build_bnb_config(cfg["quant"]), device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    test_records = load_jsonl(TEST_SPLIT_PATH)
    predictions = []
    for rec in test_records:
        user_content = f"Schema:\n\n{schema_json}\n\nDocument text:\n\n{rec['input']}"
        messages = [
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        predictions.append({"doc_id": rec["doc_id"], "target": rec["target"], "prediction_raw": text})

    with open(DEFAULT_PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(predictions)} predictions to {DEFAULT_PREDICTIONS_PATH}")
    return predictions


def parse_prediction(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # simple repair attempt: extract the outermost {...} span
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


def compute_metrics(predictions: list[dict]):
    n = len(predictions)
    parsed = [{**p, "prediction": parse_prediction(p["prediction_raw"])} for p in predictions]
    parse_failures = sum(1 for p in parsed if p["prediction"] is None)
    print(f"Parse failure rate: {parse_failures}/{n} ({parse_failures / n:.1%})\n")

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

    wage_precisions, wage_recalls, wage_f1s = [], [], []
    for rec in valid:
        p, r, f1 = wage_increases_f1(rec["target"].get("wage_increases", []), rec["prediction"].get("wage_increases", []))
        wage_precisions.append(p)
        wage_recalls.append(r)
        wage_f1s.append(f1)

    print(f"{'field':<36} {'null_f1':>8} {'exact_match':>12} {'n':>5}")
    low_f1_fields = []
    for field, s in sorted(stats.items()):
        precision = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0
        recall = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0.0
        null_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate predictions on GPU before evaluating")
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS_PATH, help="Path to existing predictions JSONL")
    args = parser.parse_args()

    if args.generate:
        predictions = generate_predictions()
    else:
        predictions = load_jsonl(args.predictions)

    compute_metrics(predictions)


if __name__ == "__main__":
    main()
