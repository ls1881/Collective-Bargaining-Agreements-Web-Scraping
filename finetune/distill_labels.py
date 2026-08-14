"""Generate distillation training labels via a frontier LLM (Claude).

Local-only, no GPU required — makes API calls against finetune/data/distill_sample.csv
(produced by data_prep.py). Requires ANTHROPIC_API_KEY in the environment.

Writes (append/resume-safe):
  finetune/data/clean_labels.jsonl    {doc_id, chunk_id, input, target, quality_bin}
  finetune/data/extract_labels.jsonl  {doc_id, input, target}

Usage:
  python finetune/distill_labels.py                 # run both tasks over the full sample
  python finetune/distill_labels.py --task clean     # cleaning labels only
  python finetune/distill_labels.py --task extract   # extraction labels only (requires clean_labels.jsonl)
  python finetune/distill_labels.py --limit 20        # smoke test on first 20 docs
"""
import argparse
import csv
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from chunking import chunk_document, reassemble  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
SAMPLE_CSV = os.path.join(DATA_DIR, "distill_sample.csv")
SCHEMA_JSON = os.path.join(BASE_DIR, "schema.json")
CLEAN_LABELS_PATH = os.path.join(DATA_DIR, "clean_labels.jsonl")
EXTRACT_LABELS_PATH = os.path.join(DATA_DIR, "extract_labels.jsonl")

MODEL = os.environ.get("DISTILL_MODEL", "claude-sonnet-5")
MAX_WORKERS = int(os.environ.get("DISTILL_CONCURRENCY", "10"))
MAX_TOKENS_CLEAN = 4096
MAX_TOKENS_EXTRACT = 2048

_write_lock = threading.Lock()
_client = anthropic.Anthropic()


def load_prompt(name: str) -> str:
    with open(os.path.join(PROMPTS_DIR, name), encoding="utf-8") as f:
        return f.read()


def load_sample() -> list[dict]:
    with open(SAMPLE_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def already_done_doc_chunk_ids(path: str) -> set[tuple[str, int]]:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done.add((rec["doc_id"], rec.get("chunk_id", 0)))
    return done


def append_jsonl(path: str, record: dict):
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_claude(prompt: str, max_tokens: int) -> str:
    resp = _client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text").strip()


def run_clean_task(sample: list[dict], limit: int | None):
    template = load_prompt("clean_prompt.txt")
    done = already_done_doc_chunk_ids(CLEAN_LABELS_PATH)

    jobs = []
    for row in sample[:limit] if limit else sample:
        text = open(row["txt_path"], encoding="utf-8", errors="replace").read()
        for rec in chunk_document(row["id_accordo"], text):
            if (rec["doc_id"], rec["chunk_id"]) in done:
                continue
            jobs.append((row, rec))

    print(f"[clean] {len(jobs)} chunks to label (skipping {len(done)} already done)")

    def work(job):
        row, rec = job
        prompt = template.format(chunk_text=rec["text"])
        try:
            cleaned = call_claude(prompt, MAX_TOKENS_CLEAN)
        except Exception as e:
            print(f"  [!] clean failed doc={row['id_accordo']} chunk={rec['chunk_id']}: {e}")
            return
        append_jsonl(CLEAN_LABELS_PATH, {
            "doc_id": row["id_accordo"],
            "chunk_id": rec["chunk_id"],
            "input": rec["text"],
            "target": cleaned,
            "quality_bin": row["quality_bin"],
        })

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(work, job) for job in jobs]
        for i, _ in enumerate(as_completed(futures), 1):
            if i % 25 == 0:
                print(f"  ...{i}/{len(jobs)} chunks done")


def reassemble_cleaned_docs() -> dict[str, str]:
    """Group clean_labels.jsonl by doc_id, reassembling cleaned chunks in order."""
    by_doc = {}
    with open(CLEAN_LABELS_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_doc.setdefault(rec["doc_id"], {})[rec["chunk_id"]] = rec["target"]

    return {
        doc_id: reassemble([chunks[cid] for cid in sorted(chunks)])
        for doc_id, chunks in by_doc.items()
    }


def run_extract_task(sample: list[dict], limit: int | None):
    if not os.path.exists(CLEAN_LABELS_PATH):
        print("[extract] clean_labels.jsonl not found — run the clean task first.", file=sys.stderr)
        sys.exit(1)

    template = load_prompt("extract_prompt.txt")
    with open(SCHEMA_JSON, encoding="utf-8") as f:
        schema_json = f.read()

    cleaned_docs = reassemble_cleaned_docs()
    done = {rec["doc_id"] for rec in (json.loads(l) for l in open(EXTRACT_LABELS_PATH, encoding="utf-8"))} \
        if os.path.exists(EXTRACT_LABELS_PATH) else set()

    doc_ids = [row["id_accordo"] for row in (sample[:limit] if limit else sample)]
    jobs = [d for d in doc_ids if d in cleaned_docs and d not in done]
    print(f"[extract] {len(jobs)} docs to label (skipping {len(done)} already done, "
          f"{len(doc_ids) - len(cleaned_docs)} not yet cleaned)")

    def work(doc_id):
        prompt = template.format(schema_json=schema_json, document_text=cleaned_docs[doc_id])
        try:
            raw = call_claude(prompt, MAX_TOKENS_EXTRACT)
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  [!] extract JSON parse failed doc={doc_id}: {e}")
            return
        except Exception as e:
            print(f"  [!] extract failed doc={doc_id}: {e}")
            return
        append_jsonl(EXTRACT_LABELS_PATH, {
            "doc_id": doc_id,
            "input": cleaned_docs[doc_id],
            "target": parsed,
        })

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(work, doc_id) for doc_id in jobs]
        for i, _ in enumerate(as_completed(futures), 1):
            if i % 25 == 0:
                print(f"  ...{i}/{len(jobs)} docs done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["clean", "extract", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N sample docs (smoke test)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    sample = load_sample()

    if args.task in ("clean", "both"):
        run_clean_task(sample, args.limit)
    if args.task in ("extract", "both"):
        run_extract_task(sample, args.limit)


if __name__ == "__main__":
    main()
