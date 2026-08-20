"""Out-of-distribution quality test: does the Italian-trained clean-lora-mlx/extract-lora-mlx
adapters' learned skill transfer to Finnish CBA text, with no retraining?

No gold Finnish reference text exists (that's exactly the hand-labeling effort this test is
meant to avoid committing to blindly), so this can't compute CER/F1 the way eval_clean_mlx.py /
eval_extract_mlx.py do for Italian. Instead: sample real Finnish documents, generate with BOTH
the plain base model (zero-shot) and the adapter, and report what's actually measurable without
gold labels — JSON parse success rate for extraction — plus save full side-by-side text for
manual read-through (the only way to judge cleaning quality here).

Usage: python eval_finland_mlx.py
"""
import json
import os
import random
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from chunking import chunk_document  # noqa: E402
from infer_finland_mlx import FINLAND_TXTS_DIR, RA_WORK_SCRAPING  # noqa: E402
from train_common_mlx import load_config  # noqa: E402
from train_extract_lora_mlx import load_schema  # noqa: E402

OUT_DIR = os.path.join(BASE_DIR, "data")
CLEAN_SAMPLE_OUT = os.path.join(OUT_DIR, "finland_clean_sample_compare.json")
EXTRACT_SAMPLE_OUT = os.path.join(OUT_DIR, "finland_extract_sample_compare.json")

SAMPLE_SIZE = 8
SEED = 42


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


def sample_docs():
    files = sorted(f for f in os.listdir(FINLAND_TXTS_DIR) if f.endswith(".txt"))
    rng = random.Random(SEED)
    return rng.sample(files, min(SAMPLE_SIZE, len(files)))


def run_clean_compare(sample_files):
    from mlx_lm import generate
    from mlx_lm.utils import load

    cfg = load_config("qlora_clean_mlx_finland.yaml")
    adapter_path = os.path.join(BASE_DIR, cfg["adapter_out"])

    # Pick one representative body chunk per doc (skip the first chunk — usually a title page —
    # and last, take a middle one so we're testing on real prose, not front matter).
    chunks_to_test = []
    for filename in sample_files:
        text = open(os.path.join(FINLAND_TXTS_DIR, filename), encoding="utf-8", errors="replace").read()
        recs = chunk_document(filename, text)
        if len(recs) >= 3:
            pick = recs[len(recs) // 2]
        elif recs:
            pick = recs[0]
        else:
            continue
        chunks_to_test.append({"doc_id": filename, "chunk_id": pick["chunk_id"], "input": pick["text"]})

    print(f"Loading PLAIN base model (zero-shot)...")
    model, tokenizer = load(cfg["base_model"], tokenizer_config={"trust_remote_code": True})
    for rec in chunks_to_test:
        messages = [{"role": "system", "content": cfg["system_prompt"]}, {"role": "user", "content": rec["input"]}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        rec["zero_shot_output"] = generate(model, tokenizer, prompt, max_tokens=1536).strip()
        print(f"  zero-shot done: {rec['doc_id']}", flush=True)
    del model

    print(f"Loading base model + clean-lora-mlx adapter (unmodified, Italian-trained)...")
    model, tokenizer = load(cfg["base_model"], adapter_path=adapter_path,
                             tokenizer_config={"trust_remote_code": True})
    for rec in chunks_to_test:
        messages = [{"role": "system", "content": cfg["system_prompt"]}, {"role": "user", "content": rec["input"]}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        rec["adapter_output"] = generate(model, tokenizer, prompt, max_tokens=1536).strip()
        print(f"  adapter done: {rec['doc_id']}", flush=True)

    with open(CLEAN_SAMPLE_OUT, "w", encoding="utf-8") as f:
        json.dump(chunks_to_test, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(chunks_to_test)} cleaning comparison pairs to {CLEAN_SAMPLE_OUT}")


def run_extract_compare(sample_files):
    from mlx_lm import generate
    from mlx_lm.utils import load

    cfg = load_config("qlora_extract_mlx_finland.yaml")
    schema_json = load_schema(cfg)
    adapter_path = os.path.join(BASE_DIR, cfg["adapter_out"])

    docs = []
    for filename in sample_files:
        text = open(os.path.join(FINLAND_TXTS_DIR, filename), encoding="utf-8", errors="replace").read()
        if len(text) > 8000:
            text = text[:8000]
        docs.append({"doc_id": filename, "input": text})

    def run(model, tokenizer, key):
        for rec in docs:
            user_content = f"Schema:\n\n{schema_json}\n\nDocument text:\n\n{rec['input']}"
            messages = [{"role": "system", "content": cfg["system_prompt"]}, {"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            raw = generate(model, tokenizer, prompt, max_tokens=3072).strip()
            rec[key] = raw
            print(f"  {key} done: {rec['doc_id']}", flush=True)

    print("Loading PLAIN base model (zero-shot)...")
    model, tokenizer = load(cfg["base_model"], tokenizer_config={"trust_remote_code": True})
    run(model, tokenizer, "zero_shot_raw")
    del model

    print("Loading base model + extract-lora-mlx adapter (unmodified, Italian-trained)...")
    model, tokenizer = load(cfg["base_model"], adapter_path=adapter_path,
                             tokenizer_config={"trust_remote_code": True})
    run(model, tokenizer, "adapter_raw")

    zero_shot_parsed = sum(1 for d in docs if parse_prediction(d["zero_shot_raw"]) is not None)
    adapter_parsed = sum(1 for d in docs if parse_prediction(d["adapter_raw"]) is not None)

    with open(EXTRACT_SAMPLE_OUT, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}\nSUMMARY (n={len(docs)})\n{'='*60}")
    print(f"Zero-shot JSON parse success: {zero_shot_parsed}/{len(docs)}")
    print(f"Adapter    JSON parse success: {adapter_parsed}/{len(docs)}")
    print(f"Wrote full outputs to {EXTRACT_SAMPLE_OUT} for manual review "
          f"(no gold labels exist, so plausibility/hallucination must be checked by reading)")


def main():
    sample_files = sample_docs()
    print(f"Sampled {len(sample_files)} Finnish docs (seed={SEED}): {sample_files}\n")
    print("--- Cleaning comparison ---")
    run_clean_compare(sample_files)
    print("\n--- Extraction comparison ---")
    run_extract_compare(sample_files)


if __name__ == "__main__":
    main()
