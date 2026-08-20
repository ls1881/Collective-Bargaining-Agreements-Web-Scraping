"""Zero-shot Qwen2.5-7B (no adapter) over the Finnish CBA corpus.

An out-of-distribution comparison test (eval_finland_mlx.py) found the Italian-trained
clean-lora-mlx/extract-lora-mlx adapters give mixed-to-negative results on Finnish/Swedish
text: the adapter missed obvious OCR garbage that zero-shot correctly caught, and missed real
wage-increase clauses (dates/percentages) that zero-shot correctly extracted twice; the adapter
only clearly helped on sector-field extraction. Meanwhile zero-shot alone already produced
valid JSON 8/8 times and read real structured data correctly — this isn't a "broken baseline"
situation the way Italian's zero-shot was, so a full Finland-specific retrain wasn't judged
worth the time investment (and lower Finnish-labeling confidence risk) given the much smaller
corpus (499 docs vs. Italy's 6,801). See finetune/README.md for the full writeup.

No chunk-level noisy/severe filtering: tested two heuristics (Italian's garbage-char whitelist,
adapted for Finnish ä/ö/å; a short-token-ratio check) against known-noisy real chunks and
neither reliably separated noise from clean text on this corpus (e.g. a chunk with obvious
inserted-dash OCR noise scored *lower* on short-token-ratio than a clean chunk). Rather than
ship a filter that can't be validated, every chunk gets cleaned unconditionally — kept for
completeness/future use, but not run for this corpus (see next paragraph).

Scope decision: unfiltered cleaning is 42,606 chunks (a many-hours-to-a-day+ job); extraction
is the actually-new capability for Finland (the original scraping pipeline never extracted any
structured fields), and is far cheaper on its own. So this run is `--stage extract` only,
directly against the raw scraped text — Finland already has its own English translation of the
raw text from the original pipeline (`finland_translated_txts/`), so skipping re-cleaning here
loses nothing that wasn't already a gap. `--stage clean` remains available if that's revisited.

Usage:
  python infer_finland_mlx.py --stage clean
  python infer_finland_mlx.py --stage extract
  python infer_finland_mlx.py --stage all
"""
import argparse
import csv
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from chunking import chunk_document, reassemble  # noqa: E402
from infer_batch_mlx import write_text_atomic  # noqa: E402
from train_common_mlx import load_config  # noqa: E402
from train_extract_lora_mlx import load_schema  # noqa: E402

RA_WORK_SCRAPING = ("/Users/lukeschreiber/Downloads/RA Work/"
                     "Collective-Bargaining-Agreements-Web-Scraping/scraping/finland_scraping")
FINLAND_TXTS_DIR = os.path.join(RA_WORK_SCRAPING, "finland_txts")

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CLEANED_TXT_DIR = os.path.join(OUTPUT_DIR, "finland_txts_cleaned")
EXTRACT_PREDICTIONS_JSONL = os.path.join(DATA_DIR, "finland_extract_predictions_raw_mlx.jsonl")
EXTENDED_METADATA_PATH = os.path.join(OUTPUT_DIR, "finland_metadata_extended.csv")

# Reuses the Finland config files for system_prompt/schema/base_model — adapter_out is simply
# never loaded (see load_base_model below), so these still apply zero-shot only.
CLEAN_CFG_NAME = "qlora_clean_mlx_finland.yaml"
EXTRACT_CFG_NAME = "qlora_extract_mlx_finland.yaml"

MAX_EXTRACT_INPUT_CHARS = 8000
CLEAN_BATCH_DOCS = 10   # smaller than Italy's tuned value of 20 since every chunk is now sent
                        # to the model (no filtering), so batches are larger in chunk-count for
                        # the same doc-count; keeps lockstep-decoding tail latency in check.


def load_base_model(cfg: dict):
    from mlx_lm.utils import load
    return load(cfg["base_model"], tokenizer_config={"trust_remote_code": True})


def batched(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def stage_clean():
    from mlx_lm import batch_generate

    os.makedirs(CLEANED_TXT_DIR, exist_ok=True)
    cfg = load_config(CLEAN_CFG_NAME)
    model, tokenizer = load_base_model(cfg)

    files = sorted(f for f in os.listdir(FINLAND_TXTS_DIR) if f.endswith(".txt"))
    print(f"[clean] {len(files)} Finnish docs to scan", flush=True)

    pending_docs = []
    for i, filename in enumerate(files, 1):
        out_path = os.path.join(CLEANED_TXT_DIR, filename)
        if os.path.exists(out_path):
            continue
        text = open(os.path.join(FINLAND_TXTS_DIR, filename), encoding="utf-8", errors="replace").read()
        all_recs = chunk_document(filename, text)
        pending_docs.append((filename, out_path, all_recs))
        if i % 100 == 0:
            print(f"[clean] scanned {i}/{len(files)}", flush=True)

    total_chunks = sum(len(d[2]) for d in pending_docs)
    print(f"[clean] {total_chunks} chunks across {len(pending_docs)} docs to clean "
          f"(no filtering — see module docstring)", flush=True)

    if not pending_docs:
        print("[clean] nothing left to process — done.", flush=True)
        return

    docs_written = 0
    for batch_num, doc_batch in enumerate(batched(pending_docs, CLEAN_BATCH_DOCS), 1):
        prompt_index = []
        prompts = []
        for doc_id, out_path, chunk_recs in doc_batch:
            for rec in chunk_recs:
                messages = [
                    {"role": "system", "content": cfg["system_prompt"]},
                    {"role": "user", "content": rec["text"]},
                ]
                prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True))
                prompt_index.append((doc_id, out_path, rec["chunk_id"]))

        result = batch_generate(model, tokenizer, prompts, max_tokens=1536)

        by_doc = {(doc_id, out_path): {} for doc_id, out_path, _ in doc_batch}
        for (doc_id, out_path, chunk_id), text in zip(prompt_index, result.texts):
            by_doc[(doc_id, out_path)][chunk_id] = text.strip()

        for (doc_id, out_path), chunks in by_doc.items():
            write_text_atomic(out_path, reassemble([chunks[cid] for cid in sorted(chunks)]))
            docs_written += 1

        print(f"[clean] batch {batch_num}: wrote {len(by_doc)} docs ({docs_written}/{len(pending_docs)} total)",
              flush=True)

    print(f"[clean] done — wrote {docs_written} cleaned documents to {CLEANED_TXT_DIR}", flush=True)


def stage_extract():
    from mlx_lm import batch_generate

    cfg = load_config(EXTRACT_CFG_NAME)
    schema_json = load_schema(cfg)
    model, tokenizer = load_base_model(cfg)

    already_done = set()
    if os.path.exists(EXTRACT_PREDICTIONS_JSONL):
        with open(EXTRACT_PREDICTIONS_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    already_done.add(json.loads(line)["doc_id"])

    # Runs directly on the raw scraped text — the "extraction only, skip cleaning" scope
    # decision (see module docstring): Finland already has its own translation of the raw text
    # from the original scraping pipeline, so nothing is lost by not re-cleaning first here.
    files = sorted(f for f in os.listdir(FINLAND_TXTS_DIR) if f.endswith(".txt"))
    pending = []
    for filename in files:
        if filename in already_done:
            continue
        text = open(os.path.join(FINLAND_TXTS_DIR, filename), encoding="utf-8", errors="replace").read()
        truncated_note = None
        if len(text) > MAX_EXTRACT_INPUT_CHARS:
            truncated_note = f"original {len(text)} chars, truncated to {MAX_EXTRACT_INPUT_CHARS}"
            text = text[:MAX_EXTRACT_INPUT_CHARS]
        user_content = f"Schema:\n\n{schema_json}\n\nDocument text:\n\n{text}"
        messages = [
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        pending.append((filename, prompt, truncated_note))

    print(f"[extract] {len(already_done)} docs already done (resumed), {len(pending)} to extract",
          flush=True)
    if not pending:
        print("[extract] nothing left to process.", flush=True)
        return

    done_count = 0
    for batch_num, batch in enumerate(batched(pending, 20), 1):
        prompts = [p[1] for p in batch]
        result = batch_generate(model, tokenizer, prompts, max_tokens=3072)
        with open(EXTRACT_PREDICTIONS_JSONL, "a", encoding="utf-8") as f:
            for (doc_id, _, truncated_note), text in zip(batch, result.texts):
                f.write(json.dumps({"doc_id": doc_id, "raw_output": text.strip(),
                                     "truncated_note": truncated_note}, ensure_ascii=False) + "\n")
        done_count += len(batch)
        print(f"[extract] batch {batch_num}: {done_count}/{len(pending)} done", flush=True)

    print(f"[extract] wrote predictions to {EXTRACT_PREDICTIONS_JSONL}", flush=True)


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


def merge_into_metadata():
    meta_path = os.path.join(RA_WORK_SCRAPING, "finland_metadata.csv")
    with open(meta_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    preds_by_filename = {}
    if os.path.exists(EXTRACT_PREDICTIONS_JSONL):
        with open(EXTRACT_PREDICTIONS_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    preds_by_filename[rec["doc_id"]] = rec

    flat_fields = ["sector", "weekly_hours",
                    "probation_period_days_operai", "probation_period_days_impiegati",
                    "probation_period_days_quadri", "probation_period_days_dirigenti",
                    "notice_period_days_operai", "notice_period_days_impiegati",
                    "notice_period_days_quadri", "notice_period_days_dirigenti",
                    "leave_annual_leave_days", "leave_sick_leave_terms",
                    "leave_parental_leave_terms", "wage_increases_json"]

    parse_failures = 0
    for row in rows:
        matched_filenames = [row.get(f"PDF_{i}", "").strip() for i in range(1, 8)
                              if row.get(f"PDF_{i}", "").strip()]
        pred = None
        for pdf_name in matched_filenames:
            txt_name = os.path.splitext(os.path.basename(pdf_name))[0] + ".txt"
            if txt_name in preds_by_filename:
                pred = parse_prediction(preds_by_filename[txt_name]["raw_output"])
                break
        for field in flat_fields:
            row[field] = ""
        if pred is None:
            parse_failures += 1 if matched_filenames else 0
            continue
        row["sector"] = pred.get("sector") or ""
        row["weekly_hours"] = pred.get("weekly_hours") or ""
        for group, prefix in [("probation_period_days", "probation_period_days"),
                               ("notice_period_days", "notice_period_days")]:
            g = pred.get(group) or {}
            for sub in ["operai", "impiegati", "quadri", "dirigenti"]:
                row[f"{prefix}_{sub}"] = g.get(sub) if g.get(sub) is not None else ""
        leave = pred.get("leave_entitlements") or {}
        row["leave_annual_leave_days"] = leave.get("annual_leave_days") or ""
        row["leave_sick_leave_terms"] = leave.get("sick_leave_terms") or ""
        row["leave_parental_leave_terms"] = leave.get("parental_leave_terms") or ""
        row["wage_increases_json"] = json.dumps(pred.get("wage_increases") or [], ensure_ascii=False)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(EXTENDED_METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        # extrasaction="ignore": some finland_metadata.csv rows have more raw comma-separated
        # fields than the header declares (likely an unescaped comma in a free-text column like
        # Comments), which DictReader stashes under a None key — drop it rather than crash.
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[extract] wrote extended metadata to {EXTENDED_METADATA_PATH} "
          f"({len(rows)} rows, {parse_failures} parse failures)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["clean", "extract", "all"], default="all")
    args = parser.parse_args()

    if args.stage in ("clean", "all"):
        stage_clean()
    if args.stage in ("extract", "all"):
        stage_extract()
        merge_into_metadata()


if __name__ == "__main__":
    main()
