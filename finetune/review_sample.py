"""Build a human spot-check batch from distilled labels, stratified across quality bins.

Local-only, no GPU/API required. Run after distill_labels.py and before any training —
this is the hard quality gate described in the plan: review finetune/data/review_batch.csv
and fix prompts/regenerate labels if >15-20% show hallucination before training on them.

Usage: python finetune/review_sample.py [--n-clean 40] [--n-extract 40]
"""
import argparse
import csv
import json
import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_LABELS_PATH = os.path.join(DATA_DIR, "clean_labels.jsonl")
EXTRACT_LABELS_PATH = os.path.join(DATA_DIR, "extract_labels.jsonl")
REVIEW_CSV_PATH = os.path.join(DATA_DIR, "review_batch.csv")

RANDOM_SEED = 42
BIN_TARGET_SHARE = {"clean": 0.375, "noisy": 0.375, "severe": 0.25}  # ~15/15/10 of 40


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stratified_pick(records: list[dict], n: int, key: str, rng: random.Random) -> list[dict]:
    by_key = {}
    for rec in records:
        by_key.setdefault(rec.get(key, "unknown"), []).append(rec)
    for bucket in by_key.values():
        rng.shuffle(bucket)

    picked = []
    for bin_name, share in BIN_TARGET_SHARE.items():
        target = round(n * share)
        picked.extend(by_key.get(bin_name, [])[:target])

    if len(picked) < n:
        picked_ids = {id(r) for r in picked}
        leftovers = [r for r in records if id(r) not in picked_ids]
        rng.shuffle(leftovers)
        picked.extend(leftovers[: n - len(picked)])

    return picked[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clean", type=int, default=40)
    parser.add_argument("--n-extract", type=int, default=40)
    args = parser.parse_args()

    rng = random.Random(RANDOM_SEED)

    clean_records = load_jsonl(CLEAN_LABELS_PATH)
    extract_records = load_jsonl(EXTRACT_LABELS_PATH)

    clean_sample = stratified_pick(clean_records, args.n_clean, "quality_bin", rng)
    extract_sample = rng.sample(extract_records, min(args.n_extract, len(extract_records)))

    rows = []
    for rec in clean_sample:
        rows.append({
            "task": "clean",
            "doc_id": rec["doc_id"],
            "quality_bin": rec.get("quality_bin", ""),
            "input": rec["input"],
            "model_output": rec["target"],
            "human_ok": "",
            "human_notes": "",
        })
    for rec in extract_sample:
        rows.append({
            "task": "extract",
            "doc_id": rec["doc_id"],
            "quality_bin": "",
            "input": rec["input"][:2000],  # truncate for spreadsheet readability
            "model_output": json.dumps(rec["target"], ensure_ascii=False),
            "human_ok": "",
            "human_notes": "",
        })

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REVIEW_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "doc_id", "quality_bin", "input", "model_output", "human_ok", "human_notes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(clean_sample)} clean + {len(extract_sample)} extract records to {REVIEW_CSV_PATH}")
    print("Fill in human_ok (y/n) and human_notes for each row. If >15-20% of either task is marked 'n' "
          "(hallucination: invented numbers/names/values not supported by input), revise the prompts in "
          "finetune/prompts/ and regenerate labels before training.")


if __name__ == "__main__":
    main()
