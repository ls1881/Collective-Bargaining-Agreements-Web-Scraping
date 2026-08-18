"""Run the trained adapters over the full corpus: clean -> re-translate -> extract.

GPU-required (vLLM). Run on the rented instance after both adapters are trained
and pass their eval gates (§5 of the plan).

Usage:
  python finetune/infer_batch_cuda.py --stage clean
  python finetune/infer_batch_cuda.py --stage translate
  python finetune/infer_batch_cuda.py --stage extract
  python finetune/infer_batch_cuda.py --stage all
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

CLEAN_CFG_NAME = "qlora_clean.yaml"
EXTRACT_CFG_NAME = "qlora_extract.yaml"

# The extract-lora adapter was only ever trained on documents <=8000 chars (chat_sample.py's
# EXTRACT_MAX_CHARS) because that's what could be hand-labeled without an API key. Full-corpus
# documents run up to ~1.3M chars, far beyond the model's trained context and Qwen2.5's usable
# context window at this batch size. Truncate to stay in-distribution and avoid a context-length
# crash mid-batch; flag truncated docs so results can be understood as partial-document extraction.
MAX_EXTRACT_INPUT_CHARS = 8000


CLEAN_BATCH_DOCS = 100  # docs per vLLM generate() call in stage_clean; bounds lost work on interruption
EXTRACT_BATCH_DOCS = 200  # docs per vLLM generate() call in stage_extract
EXTRACT_PREDICTIONS_JSONL = os.path.join(DATA_DIR, "extract_predictions_raw.jsonl")


def load_corpus_index() -> list[dict]:
    with open(os.path.join(DATA_DIR, "full_corpus_index.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_text_atomic(path: str, text: str):
    """Write via a temp file + rename so a killed process can never leave a partially-written
    file that a later resume would mistake for 'already done' (os.path.exists check)."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, path)


def batched(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def build_vllm(base_model: str, adapter_path: str, max_model_len: int):
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    # Loads the base model in bf16 (~14GB weights for 7B), NOT 4-bit — vLLM's bitsandbytes
    # serving path is immature/version-fragile at the pinned vllm==0.5.1, so bf16 is the safer
    # choice here even though training used 4-bit. This needs a real 24GB-class GPU (matches
    # the plan's assumption); a 16GB card will not fit weights + KV cache.
    llm = LLM(model=base_model, dtype="bfloat16", enable_lora=True, max_lora_rank=32,
              max_model_len=max_model_len)
    lora_request = LoRARequest("adapter", 1, adapter_path)
    return llm, lora_request


def stage_clean(scope: str = "noisy-severe"):
    """scope='noisy-severe' (default): only run the model on docs flagged noisy/severe by
    data_prep.py's quality heuristic (~206 docs), and within those docs, only on the
    individual CHUNKS that are themselves chunk-level noisy/severe (matching exactly how
    chat_sample.py sampled the training data — most chunks inside a "noisy" doc are locally
    clean, the doc-level average just crossed the threshold). Chunk-level-clean chunks and
    the other ~6,595 doc-level-clean docs are copied through unchanged, no GPU cost.
    scope='all' runs the model on every doc/chunk regardless (~526K chunks, tens of
    GPU-hours) — only use that if you specifically need the model's output on already-clean
    text too (e.g. to validate it doesn't over-edit)."""
    from data_prep import ocr_quality
    from train_common_cuda import load_config
    from vllm import SamplingParams

    cfg = load_config(CLEAN_CFG_NAME)
    adapter_path = os.path.join(BASE_DIR, cfg["adapter_out"])
    # max_model_len must cover prompt + generation tokens together. Measured: a 2000-char
    # chunk (chunking.py's max chunk size) is ~650 prompt tokens; with a 1536-token generation
    # budget that's ~2200 tokens needed. 3072 leaves real margin.
    llm, lora_request = build_vllm(cfg["base_model"], adapter_path, max_model_len=3072)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    os.makedirs(CLEANED_TXT_DIR, exist_ok=True)
    corpus = load_corpus_index()

    pending_docs = []  # (doc_id, out_path, {chunk_id: text}, [chunk_recs needing the model])
    passthrough_doc_count = 0
    passthrough_chunk_count = 0
    for row in corpus:
        out_path = os.path.join(CLEANED_TXT_DIR, os.path.basename(row["txt_path"]))
        if os.path.exists(out_path):
            continue  # resume-safe

        if scope == "noisy-severe" and row["quality_bin"] not in ("noisy", "severe"):
            # Already clean by heuristic — copy through without spending GPU time on it.
            text = open(row["txt_path"], encoding="utf-8", errors="replace").read()
            write_text_atomic(out_path, text)
            passthrough_doc_count += 1
            continue

        text = open(row["txt_path"], encoding="utf-8", errors="replace").read()
        all_recs = chunk_document(row["id_accordo"], text)

        base_chunks = {}  # chunk_id -> passthrough text (filled in now, overwritten by model output later)
        needs_model = []
        for rec in all_recs:
            if scope == "all" or ocr_quality(rec["text"])["bin"] in ("noisy", "severe"):
                needs_model.append(rec)
            else:
                base_chunks[rec["chunk_id"]] = rec["text"]
                passthrough_chunk_count += 1

        if not needs_model:
            # Every chunk in this doc turned out to be chunk-level clean — write it through now.
            write_text_atomic(out_path, reassemble([base_chunks[cid] for cid in sorted(base_chunks)]))
            continue

        pending_docs.append((row["id_accordo"], out_path, base_chunks, needs_model))

    total_chunks = sum(len(d[3]) for d in pending_docs)
    print(f"[clean] {passthrough_doc_count} docs passed through unchanged (doc-level clean)")
    print(f"[clean] {passthrough_chunk_count} additional chunks passed through unchanged "
          f"(chunk-level clean within a doc-level noisy/severe doc)")
    print(f"[clean] {total_chunks} chunks across {len(pending_docs)} docs actually need the model, "
          f"in batches of {CLEAN_BATCH_DOCS} docs (a crash/interruption loses at most one batch)")

    if not pending_docs:
        print("[clean] nothing left to process (resume case) — done.")
        return

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1536)
    docs_written = 0
    for batch_num, doc_batch in enumerate(batched(pending_docs, CLEAN_BATCH_DOCS), 1):
        prompt_index = []  # (doc_id, out_path, chunk_id)
        prompts = []
        for doc_id, out_path, _base_chunks, chunk_recs in doc_batch:
            for rec in chunk_recs:
                messages = [
                    {"role": "system", "content": cfg["system_prompt"]},
                    {"role": "user", "content": rec["text"]},
                ]
                prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                prompt_index.append((doc_id, out_path, rec["chunk_id"]))

        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

        by_doc = {(doc_id, out_path): dict(base_chunks) for doc_id, out_path, base_chunks, _ in doc_batch}
        for (doc_id, out_path, chunk_id), output in zip(prompt_index, outputs):
            cleaned = output.outputs[0].text.strip()
            by_doc[(doc_id, out_path)][chunk_id] = cleaned

        for (doc_id, out_path), chunks in by_doc.items():
            text = reassemble([chunks[cid] for cid in sorted(chunks)])
            write_text_atomic(out_path, text)
            docs_written += 1

        print(f"[clean] batch {batch_num}: wrote {len(by_doc)} docs ({docs_written}/{len(pending_docs)} total)")

    print(f"[clean] done — wrote {docs_written} cleaned documents to {CLEANED_TXT_DIR}")


def stage_translate():
    """Re-translate cleaned Italian text to English via googletrans (free/unofficial, rate-
    limited, CPU+network only — no GPU involved). DO NOT run this on your paid GPU instance:
    at full-corpus scale this is ~525K chunk-translation calls, each rate-limited to ~1/sec to
    avoid Google blocking the requests, i.e. days of wall-clock time. Run it locally (or on any
    cheap CPU box) against a copy of finetune/output/italy_txts_cleaned/, separately from the
    GPU session — it never needs to touch the GPU. Reuses italy_translator.ipynb's
    googletrans + rate-limiting approach (chunking.py here instead of the notebook's hard cut)."""
    import time

    from googletrans import Translator

    os.makedirs(CLEANED_TRANSLATED_DIR, exist_ok=True)
    translator = Translator()

    files = [f for f in os.listdir(CLEANED_TXT_DIR) if f.endswith(".txt")]
    print(f"[translate] {len(files)} cleaned files to translate — this runs on CPU only, "
          f"do not run on a billed GPU instance")

    for i, filename in enumerate(files, 1):
        out_path = os.path.join(CLEANED_TRANSLATED_DIR, filename)
        if os.path.exists(out_path):
            continue  # resume-safe
        text = open(os.path.join(CLEANED_TXT_DIR, filename), encoding="utf-8").read()
        chunks = chunk_document(filename, text)
        translated_parts = []
        for rec in chunks:
            try:
                time.sleep(1)  # matches italy_translator.ipynb's rate limit; googletrans blocks without it
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


def stage_extract():
    from train_common_cuda import load_config
    from train_extract_lora_cuda import load_schema
    from vllm import SamplingParams

    cfg = load_config(EXTRACT_CFG_NAME)
    schema_json = load_schema(cfg)
    adapter_path = os.path.join(BASE_DIR, cfg["adapter_out"])
    # max_model_len must cover prompt + generation tokens together (vLLM budgets them as one
    # sequence), not just the training-time max_seq_length. Measured: an 8000-char input (our
    # truncation cap) + schema + system prompt is ~5400 prompt tokens; with a 3072-token
    # generation budget that's ~8500 tokens needed. 9216 leaves real margin.
    llm, lora_request = build_vllm(cfg["base_model"], adapter_path, max_model_len=9216)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    corpus = load_corpus_index()

    # Resume-safe: skip docs already recorded in a prior (possibly interrupted) run.
    already_done = set()
    if os.path.exists(EXTRACT_PREDICTIONS_JSONL):
        with open(EXTRACT_PREDICTIONS_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    already_done.add(json.loads(line)["doc_id"])

    pending = []  # (doc_id, prompt, truncated_flag_or_None)
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
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        pending.append((doc_id, prompt, truncated_note))

    print(f"[extract] {len(already_done)} docs already done (resumed), {len(pending)} to extract, "
          f"in batches of {EXTRACT_BATCH_DOCS} docs")

    if not pending and not already_done:
        # Nothing was ever extracted AND nothing is ready to extract — almost certainly means
        # stage_clean() hasn't populated CLEANED_TXT_DIR yet (e.g. `--stage extract` run
        # standalone before `--stage clean`), not an actual resume. EXTRACT_PREDICTIONS_JSONL
        # was never created in this case, so there's nothing to parse below either — bail here
        # instead of crashing on open() for a file that doesn't exist.
        print(f"[extract] nothing to do — no files found in {CLEANED_TXT_DIR}. "
              f"Run --stage clean first.")
        return
    if not pending:
        print("[extract] nothing left to process (resume case).")
    else:
        # Measured against the labeled dataset: target JSON runs up to 2086 tokens (docs with
        # large wage tables). 1024 would truncate those mid-JSON.
        sampling_params = SamplingParams(temperature=0.0, max_tokens=3072)
        done_count = 0
        for batch_num, batch in enumerate(batched(pending, EXTRACT_BATCH_DOCS), 1):
            prompts = [p[1] for p in batch]
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

            with open(EXTRACT_PREDICTIONS_JSONL, "a", encoding="utf-8") as f:
                for (doc_id, _, truncated_note), output in zip(batch, outputs):
                    raw = output.outputs[0].text.strip()
                    f.write(json.dumps({"doc_id": doc_id, "raw_output": raw, "truncated_note": truncated_note},
                                        ensure_ascii=False) + "\n")

            done_count += len(batch)
            print(f"[extract] batch {batch_num}: {done_count}/{len(pending)} done")

    # Parse everything accumulated so far (this run's + any prior resumed run's) into results/flags.
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
        help="'all' runs clean+extract only (the two GPU-bound stages). 'translate' is never "
             "included in 'all' — it's CPU/network-only (googletrans) and must be run "
             "separately, off the billed GPU instance. See stage_translate()'s docstring.",
    )
    parser.add_argument(
        "--clean-scope", choices=["noisy-severe", "all"], default="noisy-severe",
        help="'noisy-severe' (default, recommended): only look inside the ~206 docs flagged "
             "noisy/severe by data_prep.py, and within those, only run the model on the "
             "individual chunks that are themselves chunk-level noisy/severe — most chunks in "
             "a 'noisy' doc turn out to be locally clean once checked individually. Everything "
             "else (the other ~6,595 doc-level-clean docs, plus locally-clean chunks within "
             "noisy docs) is copied through unchanged, no GPU cost. "
             "'all': run the model on every doc/chunk regardless (~526K chunks, tens of "
             "GPU-hours) — expensive, only use this to validate the model doesn't over-edit "
             "already-clean text.",
    )
    args = parser.parse_args()

    if args.stage in ("clean", "all"):
        stage_clean(scope=args.clean_scope)
    if args.stage == "translate":
        stage_translate()
    if args.stage in ("extract", "all"):
        stage_extract()


if __name__ == "__main__":
    main()
