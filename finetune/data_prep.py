"""Build a filtered corpus inventory and a stratified distillation sample.

Local-only, no GPU required. Reads Italy_scraping/Italy_metadata.csv + italy_txts/,
writes finetune/data/full_corpus_index.csv and finetune/data/distill_sample.csv.

Usage: python finetune/data_prep.py
"""
import csv
import json
import os
import random
import re
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
METADATA_CSV = os.path.join(ROOT_DIR, "italy_scraping", "Italy_metadata.csv")
TXTS_DIR = os.path.join(ROOT_DIR, "italy_scraping", "italy_txts")

DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

DISTILL_SAMPLE_SIZE = 750
QUALITY_TARGET_PROPORTIONS = {"clean": 0.50, "noisy": 0.35, "severe": 0.15}
RANDOM_SEED = 42

_GARBAGE_RE = re.compile(r"[^a-zA-ZàèéìòùÀÈÉÌÒÙ0-9\s.,;:()\-'\"%€/]")
_DIGIT_RUN_RE = re.compile(r"\d{5,}")


def ocr_quality(text: str) -> dict:
    char_count = len(text)
    if char_count == 0:
        return {"char_count": 0, "garbage_ratio": 1.0, "digit_run_ratio": 0.0, "bin": "severe"}

    garbage_chars = len(_GARBAGE_RE.findall(text))
    garbage_ratio = garbage_chars / char_count

    digit_run_chars = sum(len(m.group()) for m in _DIGIT_RUN_RE.finditer(text))
    digit_run_ratio = digit_run_chars / char_count

    if char_count < 500 or garbage_ratio > 0.10:
        bin_ = "severe"
    elif garbage_ratio > 0.02:
        bin_ = "noisy"
    else:
        bin_ = "clean"

    return {
        "char_count": char_count,
        "garbage_ratio": round(garbage_ratio, 4),
        "digit_run_ratio": round(digit_run_ratio, 4),
        "bin": bin_,
    }


def load_metadata_rows() -> list[dict]:
    with open(METADATA_CSV, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_inventory() -> tuple[list[dict], dict]:
    rows = load_metadata_rows()
    report = {
        "total_metadata_rows": len(rows),
        "dropped_missing_path": 0,
        "dropped_file_not_found": 0,
        "dropped_zero_byte": 0,
        "kept": 0,
        "quality_bin_counts": defaultdict(int),
    }

    inventory = []
    for row in rows:
        raw_path = row.get("local_txt_path", "")
        if not raw_path or raw_path == "MISSING":
            report["dropped_missing_path"] += 1
            continue

        basename = os.path.basename(raw_path)
        txt_path = os.path.join(TXTS_DIR, basename)

        if not os.path.exists(txt_path):
            report["dropped_file_not_found"] += 1
            continue

        size = os.path.getsize(txt_path)
        if size == 0:
            report["dropped_zero_byte"] += 1
            continue

        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        quality = ocr_quality(text)
        report["quality_bin_counts"][quality["bin"]] += 1
        report["kept"] += 1

        inventory.append({
            "id_accordo": row.get("id accordo", ""),
            "ccnl_code": row.get("CCNL CNEL idfk", ""),
            "titolo": row.get("Titolo", ""),
            "txt_path": txt_path,
            "char_count": quality["char_count"],
            "garbage_ratio": quality["garbage_ratio"],
            "quality_bin": quality["bin"],
        })

    return inventory, report


def stratified_sample(inventory: list[dict], sample_size: int) -> list[dict]:
    rng = random.Random(RANDOM_SEED)
    by_bin = defaultdict(list)
    for rec in inventory:
        by_bin[rec["quality_bin"]].append(rec)
    for bucket in by_bin.values():
        rng.shuffle(bucket)

    sample = []
    for bin_name, proportion in QUALITY_TARGET_PROPORTIONS.items():
        target = round(sample_size * proportion)
        available = by_bin.get(bin_name, [])
        sample.extend(available[:target])

    # Top up from any bin with leftovers if we're short (e.g. a bin was too small).
    if len(sample) < sample_size:
        taken_ids = {r["id_accordo"] for r in sample}
        leftovers = [r for r in inventory if r["id_accordo"] not in taken_ids]
        rng.shuffle(leftovers)
        sample.extend(leftovers[: sample_size - len(sample)])

    return sample[:sample_size]


def write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    inventory, report = build_inventory()
    fieldnames = ["id_accordo", "ccnl_code", "titolo", "txt_path", "char_count", "garbage_ratio", "quality_bin"]

    full_index_path = os.path.join(DATA_DIR, "full_corpus_index.csv")
    write_csv(full_index_path, inventory, fieldnames)

    sample = stratified_sample(inventory, DISTILL_SAMPLE_SIZE)
    sample_path = os.path.join(DATA_DIR, "distill_sample.csv")
    write_csv(sample_path, sample, fieldnames)

    report["quality_bin_counts"] = dict(report["quality_bin_counts"])
    report["distill_sample_size"] = len(sample)
    report["distill_sample_bin_counts"] = dict(
        (b, sum(1 for r in sample if r["quality_bin"] == b)) for b in QUALITY_TARGET_PROPORTIONS
    )

    report_path = os.path.join(LOGS_DIR, "data_prep_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(inventory)} rows to {full_index_path}")
    print(f"Wrote {len(sample)} rows to {sample_path}")


if __name__ == "__main__":
    main()
