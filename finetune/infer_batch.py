"""Run the trained adapters over the full corpus: clean -> re-translate -> extract.

GPU-required (vLLM). Run on the rented instance after both adapters are trained
and pass their eval gates (§5 of the plan).

Usage:
  python finetune/infer_batch.py --stage clean
  python finetune/infer_batch.py --stage translate
  python finetune/infer_batch.py --stage extract
  python finetune/infer_batch.py --stage all
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


def load_corpus_index() -> list[dict]:
    with open(os.path.join(DATA_DIR, "full_corpus_index.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_vllm(base_model: str, adapter_path: str):
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    llm = LLM(model=base_model, enable_lora=True, max_lora_rank=64)
    lora_request = LoRARequest("adapter", 1, adapter_path)
    return llm, lora_request


def stage_clean():
    from train_common import load_config
    from vllm import SamplingParams

    cfg = load_config(CLEAN_CFG_NAME)
    adapter_path = os.path.join(BASE_DIR, cfg["adapter_out"])
    llm, lora_request = build_vllm(cfg["base_model"], adapter_path)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    os.makedirs(CLEANED_TXT_DIR, exist_ok=True)
    corpus = load_corpus_index()

    all_chunks = []  # (doc_id, chunk_id, out_path, prompt)
    for row in corpus:
        out_path = os.path.join(CLEANED_TXT_DIR, os.path.basename(row["txt_path"]))
        if os.path.exists(out_path):
            continue  # resume-safe
        text = open(row["txt_path"], encoding="utf-8", errors="replace").read()
        for rec in chunk_document(row["id_accordo"], text):
            messages = [
                {"role": "system", "content": cfg["system_prompt"]},
                {"role": "user", "content": rec["text"]},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_chunks.append((row["id_accordo"], rec["chunk_id"], out_path, prompt))

    print(f"[clean] {len(all_chunks)} chunks across {len({c[0] for c in all_chunks})} docs to process")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    prompts = [c[3] for c in all_chunks]
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    by_doc = {}
    for (doc_id, chunk_id, out_path, _), output in zip(all_chunks, outputs):
        cleaned = output.outputs[0].text.strip()
        by_doc.setdefault((doc_id, out_path), {})[chunk_id] = cleaned

    for (doc_id, out_path), chunks in by_doc.items():
        text = reassemble([chunks[cid] for cid in sorted(chunks)])
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

    print(f"[clean] wrote {len(by_doc)} cleaned documents to {CLEANED_TXT_DIR}")


def stage_translate():
    """Re-translate cleaned Italian text to English, reusing italy_translator.ipynb's
    googletrans + chunking approach (chunking.py here instead of the notebook's hard cut)."""
    from googletrans import Translator

    os.makedirs(CLEANED_TRANSLATED_DIR, exist_ok=True)
    translator = Translator()

    files = [f for f in os.listdir(CLEANED_TXT_DIR) if f.endswith(".txt")]
    print(f"[translate] {len(files)} cleaned files to translate")

    for i, filename in enumerate(files, 1):
        out_path = os.path.join(CLEANED_TRANSLATED_DIR, filename)
        if os.path.exists(out_path):
            continue  # resume-safe
        text = open(os.path.join(CLEANED_TXT_DIR, filename), encoding="utf-8").read()
        chunks = chunk_document(filename, text)
        translated_parts = []
        for rec in chunks:
            try:
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
    from train_common import load_config
    from train_extract_lora import load_schema
    from vllm import SamplingParams

    cfg = load_config(EXTRACT_CFG_NAME)
    schema_json = load_schema(cfg)
    adapter_path = os.path.join(BASE_DIR, cfg["adapter_out"])
    llm, lora_request = build_vllm(cfg["base_model"], adapter_path)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    corpus = load_corpus_index()
    doc_ids, prompts, out_paths = [], [], []
    for row in corpus:
        cleaned_path = os.path.join(CLEANED_TXT_DIR, os.path.basename(row["txt_path"]))
        if not os.path.exists(cleaned_path):
            continue
        text = open(cleaned_path, encoding="utf-8").read()
        user_content = f"Schema:\n\n{schema_json}\n\nDocument text:\n\n{text}"
        messages = [
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        doc_ids.append(row["id_accordo"])
        prompts.append(prompt)

    print(f"[extract] {len(prompts)} docs to extract")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    results, flags = {}, []
    for doc_id, output in zip(doc_ids, outputs):
        raw = output.outputs[0].text.strip()
        try:
            parsed = json.loads(raw)
            results[doc_id] = parsed
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            try:
                parsed = json.loads(raw[start:end + 1]) if start != -1 and end != -1 else None
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
    parser.add_argument("--stage", choices=["clean", "translate", "extract", "all"], default="all")
    args = parser.parse_args()

    if args.stage in ("clean", "all"):
        stage_clean()
    if args.stage in ("translate", "all"):
        stage_translate()
    if args.stage in ("extract", "all"):
        stage_extract()


if __name__ == "__main__":
    main()
