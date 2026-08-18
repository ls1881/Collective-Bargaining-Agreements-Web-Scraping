"""Run the trained MLX adapters over the full corpus: clean -> re-translate -> extract.
Mirrors infer_batch_cuda.py, using mlx_lm.batch_generate() instead of vLLM.

Runs locally on the M3 Max, no cloud GPU. Same cost-reducing design as the CUDA version:
chunk-level filtering for cleaning (only genuinely noisy/severe chunks touch the model),
truncation + flagging for over-length extraction inputs, resumable batched processing.

Usage:
  python finetune/infer_batch_mlx.py --stage clean
  python finetune/infer_batch_mlx.py --stage translate
  python finetune/infer_batch_mlx.py --stage extract
  python finetune/infer_batch_mlx.py --stage all
"""
import argparse
import csv
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from chunking import chunk_document, reassemble  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CLEANED_TXT_DIR = os.path.join(OUTPUT_DIR, "italy_txts_cleaned")
CLEANED_TRANSLATED_DIR = os.path.join(OUTPUT_DIR, "italy_translated_txts_cleaned")
EXTENDED_METADATA_PATH = os.path.join(OUTPUT_DIR, "Italy_metadata_extended.csv")
EXTRACTION_FLAGS_PATH = os.path.join(OUTPUT_DIR, "extraction_flags.csv")

ROOT_DIR = os.path.dirname(BASE_DIR)
ORIGINAL_METADATA_CSV = os.path.join(ROOT_DIR, "italy_scraping", "Italy_metadata.csv")

CLEAN_CFG_NAME = "qlora_clean_mlx.yaml"
EXTRACT_CFG_NAME = "qlora_extract_mlx.yaml"

MAX_EXTRACT_INPUT_CHARS = 8000  # same reasoning as infer_batch_cuda.py — stay in-distribution

CLEAN_BATCH_DOCS = 50    # smaller than the CUDA path's 100 — a single Mac's unified memory is
EXTRACT_BATCH_DOCS = 100  # shared with the whole OS, not a dedicated 24GB GPU pool. Reduce
                          # further (see --batch-docs) if you see memory pressure.
EXTRACT_PREDICTIONS_JSONL = os.path.join(DATA_DIR, "extract_predictions_raw_mlx.jsonl")


def load_corpus_index() -> list[dict]:
    with open(os.path.join(DATA_DIR, "full_corpus_index.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_text_atomic(path: str, text: str):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, path)


def batched(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def load_model_with_adapter(cfg: dict):
    from mlx_lm.utils import load
    adapter_path = os.path.join(BASE_DIR, cfg["adapter_out"])
    model, tokenizer = load(cfg["base_model"], adapter_path=adapter_path,
                             tokenizer_config={"trust_remote_code": True})
    return model, tokenizer


def stage_clean(scope: str = "noisy-severe", batch_docs: int = CLEAN_BATCH_DOCS):
    """Same chunk-level-filtering design as infer_batch_cuda.py's stage_clean — see that
    file's docstring for the full reasoning (most chunks in a doc-level-noisy document are
    themselves chunk-level clean, so filtering at the chunk level avoids wasting compute)."""
    from data_prep import ocr_quality
    from mlx_lm import batch_generate

    from train_common_mlx import load_config

    cfg = load_config(CLEAN_CFG_NAME)
    model, tokenizer = load_model_with_adapter(cfg)

    os.makedirs(CLEANED_TXT_DIR, exist_ok=True)
    corpus = load_corpus_index()
    print(f"[clean] scanning {len(corpus)} corpus rows...")

    import time
    scan_start = time.time()

    pending_docs = []  # (doc_id, out_path, {chunk_id: text}, [chunk_recs needing the model])
    passthrough_doc_count = 0
    passthrough_chunk_count = 0
    for i, row in enumerate(corpus, 1):
        if i % 500 == 0:
            print(f"[clean] scanned {i}/{len(corpus)} rows ({time.time() - scan_start:.0f}s elapsed, "
                  f"{passthrough_doc_count} passthrough, {len(pending_docs)} pending)", flush=True)

        out_path = os.path.join(CLEANED_TXT_DIR, os.path.basename(row["txt_path"]))
        if os.path.exists(out_path):
            continue  # resume-safe

        if scope == "noisy-severe" and row["quality_bin"] not in ("noisy", "severe"):
            text = open(row["txt_path"], encoding="utf-8", errors="replace").read()
            write_text_atomic(out_path, text)
            passthrough_doc_count += 1
            continue

        # Noisy/severe docs can be huge (up to ~1.3M chars) — chunking + per-chunk quality
        # scoring on these is the expensive path, print before starting so a hang is visible
        # and attributable to a specific document rather than looking like silent progress.
        doc_char_count = row.get("char_count", "?")
        print(f"[clean] row {i}/{len(corpus)}: chunking doc {row['id_accordo']} "
              f"({doc_char_count} chars, bin={row['quality_bin']})...", flush=True)
        chunk_start = time.time()

        text = open(row["txt_path"], encoding="utf-8", errors="replace").read()
        all_recs = chunk_document(row["id_accordo"], text)

        base_chunks = {}
        needs_model = []
        for rec in all_recs:
            if scope == "all" or ocr_quality(rec["text"])["bin"] in ("noisy", "severe"):
                needs_model.append(rec)
            else:
                base_chunks[rec["chunk_id"]] = rec["text"]
                passthrough_chunk_count += 1

        print(f"[clean]   doc {row['id_accordo']}: {len(all_recs)} chunks, "
              f"{len(needs_model)} need the model ({time.time() - chunk_start:.1f}s)", flush=True)

        if not needs_model:
            write_text_atomic(out_path, reassemble([base_chunks[cid] for cid in sorted(base_chunks)]))
            continue

        pending_docs.append((row["id_accordo"], out_path, base_chunks, needs_model))

    total_chunks = sum(len(d[3]) for d in pending_docs)
    print(f"[clean] {passthrough_doc_count} docs passed through unchanged (doc-level clean)", flush=True)
    print(f"[clean] {passthrough_chunk_count} additional chunks passed through unchanged "
          f"(chunk-level clean within a doc-level noisy/severe doc)", flush=True)
    print(f"[clean] {total_chunks} chunks across {len(pending_docs)} docs actually need the model, "
          f"in batches of {batch_docs} docs", flush=True)

    if not pending_docs:
        print("[clean] nothing left to process (resume case) — done.", flush=True)
        return

    docs_written = 0
    for batch_num, doc_batch in enumerate(batched(pending_docs, batch_docs), 1):
        prompt_index = []  # (doc_id, out_path, chunk_id)
        prompts = []
        for doc_id, out_path, _base_chunks, chunk_recs in doc_batch:
            for rec in chunk_recs:
                messages = [
                    {"role": "system", "content": cfg["system_prompt"]},
                    {"role": "user", "content": rec["text"]},
                ]
                prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True))
                prompt_index.append((doc_id, out_path, rec["chunk_id"]))

        result = batch_generate(model, tokenizer, prompts, max_tokens=1536)

        by_doc = {(doc_id, out_path): dict(base_chunks) for doc_id, out_path, base_chunks, _ in doc_batch}
        for (doc_id, out_path, chunk_id), text in zip(prompt_index, result.texts):
            by_doc[(doc_id, out_path)][chunk_id] = text.strip()

        for (doc_id, out_path), chunks in by_doc.items():
            text = reassemble([chunks[cid] for cid in sorted(chunks)])
            write_text_atomic(out_path, text)
            docs_written += 1

        print(f"[clean] batch {batch_num}: wrote {len(by_doc)} docs ({docs_written}/{len(pending_docs)} total)", flush=True)

    print(f"[clean] done — wrote {docs_written} cleaned documents to {CLEANED_TXT_DIR}", flush=True)


def stage_translate():
    """Identical to infer_batch_cuda.py's stage_translate — this stage never needs a GPU/MLX
    at all (it's googletrans, CPU + network only), so there's nothing MLX-specific here. Kept
    for interface parity; run either version, they do the same thing."""
    import time

    from googletrans import Translator

    os.makedirs(CLEANED_TRANSLATED_DIR, exist_ok=True)
    translator = Translator()

    files = [f for f in os.listdir(CLEANED_TXT_DIR) if f.endswith(".txt")]
    print(f"[translate] {len(files)} cleaned files to translate")

    for i, filename in enumerate(files, 1):
        out_path = os.path.join(CLEANED_TRANSLATED_DIR, filename)
        if os.path.exists(out_path):
            continue
        text = open(os.path.join(CLEANED_TXT_DIR, filename), encoding="utf-8").read()
        chunks = chunk_document(filename, text)
        translated_parts = []
        for rec in chunks:
            try:
                time.sleep(1)
                res = translator.translate(rec["text"], src="it", dest="en")
                translated_parts.append(res.text)
            except Exception as e:
                print(f"  [!] translate failed {filename} chunk {rec['chunk_id']}: {e}")
                translated_parts.append(rec["text"])
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(translated_parts))
        if i % 100 == 0:
            print(f"  ...{i}/{len(files)} translated")

    print(f"[translate] wrote translations to {CLEANED_TRANSLATED_DIR}")


def stage_extract(batch_docs: int = EXTRACT_BATCH_DOCS):
    from mlx_lm import batch_generate

    from train_common_mlx import load_config
    from train_extract_lora_mlx import load_schema

    cfg = load_config(EXTRACT_CFG_NAME)
    schema_json = load_schema(cfg)
    model, tokenizer = load_model_with_adapter(cfg)

    corpus = load_corpus_index()

    already_done = set()
    if os.path.exists(EXTRACT_PREDICTIONS_JSONL):
        with open(EXTRACT_PREDICTIONS_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    already_done.add(json.loads(line)["doc_id"])

    pending = []  # (doc_id, prompt_token_ids, truncated_flag_or_None)
    for row in corpus:
        doc_id = row["id_accordo"]
        if doc_id in already_done:
            continue
        cleaned_path = os.path.join(CLEANED_TXT_DIR, os.path.basename(row["txt_path"]))
        if not os.path.exists(cleaned_path):
            continue
        text = open(cleaned_path, encoding="utf-8").read()
        truncated_note = None
        if len(text) > MAX_EXTRACT_INPUT_CHARS:
            truncated_note = (f"original {len(text)} chars, truncated to {MAX_EXTRACT_INPUT_CHARS} "
                               f"(out-of-distribution for the trained adapter; treat as partial-document extraction)")
            text = text[:MAX_EXTRACT_INPUT_CHARS]
        user_content = f"Schema:\n\n{schema_json}\n\nDocument text:\n\n{text}"
        messages = [
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        pending.append((doc_id, prompt, truncated_note))

    print(f"[extract] {len(already_done)} docs already done (resumed), {len(pending)} to extract, "
          f"in batches of {batch_docs} docs", flush=True)

    if not pending and not already_done:
        print(f"[extract] nothing to do — no files found in {CLEANED_TXT_DIR}. Run --stage clean first.", flush=True)
        return
    if not pending:
        print("[extract] nothing left to process (resume case).", flush=True)
    else:
        done_count = 0
        for batch_num, batch in enumerate(batched(pending, batch_docs), 1):
            prompts = [p[1] for p in batch]
            result = batch_generate(model, tokenizer, prompts, max_tokens=3072)

            with open(EXTRACT_PREDICTIONS_JSONL, "a", encoding="utf-8") as f:
                for (doc_id, _, truncated_note), text in zip(batch, result.texts):
                    f.write(json.dumps({"doc_id": doc_id, "raw_output": text.strip(), "truncated_note": truncated_note},
                                        ensure_ascii=False) + "\n")

            done_count += len(batch)
            print(f"[extract] batch {batch_num}: {done_count}/{len(pending)} done", flush=True)

    results, flags = {}, []
    with open(EXTRACT_PREDICTIONS_JSONL, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            doc_id, raw = rec["doc_id"], rec["raw_output"]
            if rec.get("truncated_note"):
                flags.append({"doc_id": doc_id, "reason": "input_truncated", "raw_output": rec["truncated_note"]})
            try:
                results[doc_id] = json.loads(raw)
            except json.JSONDecodeError:
                start, end = raw.find("{"), raw.rfind("}")
                parsed = None
                if start != -1 and end != -1:
                    try:
                        parsed = json.loads(raw[start:end + 1])
                    except json.JSONDecodeError:
                        parsed = None
                if parsed is not None:
                    results[doc_id] = parsed
                else:
                    flags.append({"doc_id": doc_id, "reason": "json_parse_failed", "raw_output": raw[:500]})

    merge_into_metadata(results)

    with open(EXTRACTION_FLAGS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "reason", "raw_output"])
        writer.writeheader()
        writer.writerows(flags)

    print(f"[extract] {len(results)} docs extracted, {len(flags)} flagged -> {EXTRACTION_FLAGS_PATH}")


def merge_into_metadata(results: dict):
    with open(ORIGINAL_METADATA_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        base_rows = list(reader)
        base_fields = reader.fieldnames

    extra_fields = [
        "sector", "weekly_hours",
        "probation_period_days_operai", "probation_period_days_impiegati",
        "probation_period_days_quadri", "probation_period_days_dirigenti",
        "notice_period_days_operai", "notice_period_days_impiegati",
        "notice_period_days_quadri", "notice_period_days_dirigenti",
        "leave_annual_leave_days", "leave_sick_leave_terms", "leave_parental_leave_terms",
        "wage_increases_json",
    ]

    for row in base_rows:
        data = results.get(row.get("id accordo", ""))
        if not data:
            for field in extra_fields:
                row[field] = ""
            continue
        row["sector"] = data.get("sector") or ""
        row["weekly_hours"] = data.get("weekly_hours") or ""
        probation = data.get("probation_period_days") or {}
        notice = data.get("notice_period_days") or {}
        leave = data.get("leave_entitlements") or {}
        for cat in ["operai", "impiegati", "quadri", "dirigenti"]:
            row[f"probation_period_days_{cat}"] = probation.get(cat) or ""
            row[f"notice_period_days_{cat}"] = notice.get(cat) or ""
        row["leave_annual_leave_days"] = leave.get("annual_leave_days") or ""
        row["leave_sick_leave_terms"] = leave.get("sick_leave_terms") or ""
        row["leave_parental_leave_terms"] = leave.get("parental_leave_terms") or ""
        row["wage_increases_json"] = json.dumps(data.get("wage_increases") or [], ensure_ascii=False)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(EXTENDED_METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(base_fields) + extra_fields)
        writer.writeheader()
        writer.writerows(base_rows)

    print(f"[extract] wrote extended metadata to {EXTENDED_METADATA_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["clean", "translate", "extract", "all"], default="all",
        help="'all' runs clean+extract only. 'translate' is CPU/network-only (googletrans) "
             "and is never included in 'all' — run it separately.",
    )
    parser.add_argument(
        "--clean-scope", choices=["noisy-severe", "all"], default="noisy-severe",
        help="Same semantics as infer_batch_cuda.py's --clean-scope.",
    )
    parser.add_argument(
        "--batch-docs", type=int, default=None,
        help="Override the default per-batch doc count (50 for clean, 100 for extract). "
             "Reduce this if you see memory pressure — a Mac's unified memory is shared with "
             "the whole OS, not a dedicated GPU VRAM pool.",
    )
    args = parser.parse_args()

    if args.stage in ("clean", "all"):
        stage_clean(scope=args.clean_scope, batch_docs=args.batch_docs or CLEAN_BATCH_DOCS)
    if args.stage == "translate":
        stage_translate()
    if args.stage in ("extract", "all"):
        stage_extract(batch_docs=args.batch_docs or EXTRACT_BATCH_DOCS)


if __name__ == "__main__":
    main()
