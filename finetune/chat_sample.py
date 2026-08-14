"""Build a small, manually-labelable sample for in-chat distillation (no API key / no cost).

Unlike distill_labels.py (which assumes automated API calls and can afford whole-document
scale), this picks a tractable subset for a human (Claude, in-conversation) to label:
  - Cleaning: individual ~2000-char CHUNKS (not whole docs — docs run up to 1.3M chars),
    stratified by chunk-level OCR-noise score so short garbled spans inside otherwise-long
    documents actually get sampled.
  - Extraction: whole documents, but capped to a max character count so a full document
    fits comfortably in one read.

Local-only, no GPU/API. Usage: python finetune/chat_sample.py
"""
import csv
import os
import random
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from chunking import chunk_document  # noqa: E402
from data_prep import ocr_quality  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data")
DISTILL_SAMPLE_CSV = os.path.join(DATA_DIR, "distill_sample.csv")

CHAT_CLEAN_SAMPLE_CSV = os.path.join(DATA_DIR, "chat_clean_sample.csv")
CHAT_EXTRACT_SAMPLE_CSV = os.path.join(DATA_DIR, "chat_extract_sample.csv")

CLEAN_CHUNK_TARGETS = {"severe": 30, "noisy": 50, "clean": 30}  # ~110 chunks total
MAX_CHUNKS_PER_DOC = 4  # avoid one long noisy doc dominating the sample
EXTRACT_MAX_CHARS = 8000
EXTRACT_TARGET = 60
RANDOM_SEED = 42


def load_distill_sample() -> list[dict]:
    with open(DISTILL_SAMPLE_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_clean_chunk_pool(docs: list[dict]) -> list[dict]:
    pool = []
    for row in docs:
        text = open(row["txt_path"], encoding="utf-8", errors="replace").read()
        chunks = chunk_document(row["id_accordo"], text)
        rng = random.Random(hash(row["id_accordo"]) % (2**32))
        rng.shuffle(chunks)
        kept = 0
        for rec in chunks:
            if kept >= MAX_CHUNKS_PER_DOC:
                break
            q = ocr_quality(rec["text"])
            pool.append({
                "doc_id": row["id_accordo"],
                "chunk_id": rec["chunk_id"],
                "char_start": rec["char_start"],
                "text": rec["text"],
                "chunk_quality_bin": q["bin"],
                "doc_quality_bin": row["quality_bin"],
            })
            kept += 1
    return pool


def stratified_chunks(pool: list[dict], targets: dict) -> list[dict]:
    rng = random.Random(RANDOM_SEED)
    by_bin = {}
    for rec in pool:
        by_bin.setdefault(rec["chunk_quality_bin"], []).append(rec)
    for bucket in by_bin.values():
        rng.shuffle(bucket)

    sample = []
    for bin_name, target in targets.items():
        sample.extend(by_bin.get(bin_name, [])[:target])
    return sample


def build_extract_sample(docs: list[dict]) -> list[dict]:
    rng = random.Random(RANDOM_SEED)
    candidates = [r for r in docs if int(r["char_count"]) <= EXTRACT_MAX_CHARS]
    by_bin = {}
    for r in candidates:
        by_bin.setdefault(r["quality_bin"], []).append(r)
    for bucket in by_bin.values():
        rng.shuffle(bucket)

    # proportional split across bins present among candidates
    total_available = len(candidates)
    sample = []
    for bin_name, bucket in by_bin.items():
        share = len(bucket) / total_available if total_available else 0
        target = round(EXTRACT_TARGET * share)
        sample.extend(bucket[:target])
    if len(sample) < EXTRACT_TARGET:
        picked_ids = {r["id_accordo"] for r in sample}
        leftovers = [r for r in candidates if r["id_accordo"] not in picked_ids]
        rng.shuffle(leftovers)
        sample.extend(leftovers[: EXTRACT_TARGET - len(sample)])
    return sample[:EXTRACT_TARGET]


def main():
    docs = load_distill_sample()

    print("Building chunk-level cleaning pool (this reads every sampled doc once)...")
    pool = build_clean_chunk_pool(docs)
    print(f"Chunk pool size: {len(pool)}")
    by_bin_count = {}
    for rec in pool:
        by_bin_count[rec["chunk_quality_bin"]] = by_bin_count.get(rec["chunk_quality_bin"], 0) + 1
    print("Chunk-level quality bin counts in pool:", by_bin_count)

    clean_sample = stratified_chunks(pool, CLEAN_CHUNK_TARGETS)
    with open(CHAT_CLEAN_SAMPLE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "chunk_id", "char_start", "text", "chunk_quality_bin", "doc_quality_bin"])
        writer.writeheader()
        writer.writerows(clean_sample)
    print(f"Wrote {len(clean_sample)} chunks to {CHAT_CLEAN_SAMPLE_CSV}")

    extract_sample = build_extract_sample(docs)
    with open(CHAT_EXTRACT_SAMPLE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id_accordo", "ccnl_code", "titolo", "txt_path", "char_count", "garbage_ratio", "quality_bin"])
        writer.writeheader()
        writer.writerows(extract_sample)
    print(f"Wrote {len(extract_sample)} docs to {CHAT_EXTRACT_SAMPLE_CSV}")


if __name__ == "__main__":
    main()
