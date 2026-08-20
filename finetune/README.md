# CCNL QLoRA Pipeline

Fine-tuned pipeline for cleaning OCR/translation noise and extracting structured fields from
Italian Collective Bargaining Agreements (CCNLs). Built on top of the scraping pipeline in
[`../italy_scraping`](../italy_scraping) (see [`../README_Italy`](../README_Italy)), which
produces the raw scraped text this pipeline consumes.

## Why this exists

The scraped corpus (~7,900 CCNL records) is bimodal in quality: clean digital-PDF documents
extract fine, but scanned/signature-stamp regions produce OCR garbage that also corrupts
downstream translation. No structured content was extracted from document bodies — the
original metadata only carried administrative fields (dates, titles, signatories), nothing
parsed from the legal text itself.

This pipeline adds two capabilities via QLoRA fine-tuning of **Qwen2.5-7B-Instruct**:

1. **Cleaning** — corrects OCR/noise artifacts in the original Italian text while preserving
   legal terminology, marking truly unrecoverable spans as `[ILLEGIBLE]` rather than inventing
   content.
2. **Extraction** — pulls structured fields (sector, wage increases, weekly hours, probation
   and notice periods, leave entitlements) out of the cleaned text into JSON matching
   [`schema.json`](schema.json).

Training labels were bootstrapped via frontier-LLM distillation on a stratified sample of the
corpus (110 cleaning examples, 60 extraction examples) rather than hand-written rules, since no
ground-truth clean text or structured labels existed beforehand.

## Two training/inference stacks

Cloud GPU access fell through partway into the project, so the pipeline was rebuilt to run
locally on Apple Silicon (M3 Max) as well as on a rented CUDA GPU. Both stacks are maintained
in parallel and produce equivalent adapters:

| | CUDA (`*_cuda.py`) | MLX (`*_mlx.py`) |
|---|---|---|
| Hardware | Rented GPU (24GB+) | Local Apple Silicon, no cloud dependency |
| Stack | `transformers` / `peft` / `bitsandbytes` / `trl` / `vLLM` | `mlx` / `mlx-lm` |
| Base model | `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4 via bitsandbytes) | `mlx-community/Qwen2.5-7B-Instruct-4bit` (pre-quantized) |
| Status | Built and hardened; not run end-to-end (no GPU budget) | **Trained, evaluated, and run over the full corpus** |

Everything described in the rest of this document — training runs, evaluation numbers,
full-corpus results — is from the **MLX stack**, run locally.

## Model architecture

Two separate LoRA adapters off the same base model, rather than one multi-task adapter:

- `adapters/clean-lora-mlx/` — cleaning
- `adapters/extract-lora-mlx/` — extraction

Kept separate because the two tasks have very different output distributions (long free-text
Italian vs. strict short JSON), risk cross-contaminating at this dataset scale if merged, and
hot-swap cheaply at inference since there's no serving-cost benefit to combining them.

LoRA config (both adapters): rank 16, dropout 0.05, last 16 of 28 transformer blocks adapted,
15 epochs with early stopping (patience 10) on validation loss. Full configs in
[`configs/qlora_clean_mlx.yaml`](configs/qlora_clean_mlx.yaml) and
[`configs/qlora_extract_mlx.yaml`](configs/qlora_extract_mlx.yaml).

> **Note on `scale` vs. `alpha`:** MLX's LoRA `scale` parameter is not the same convention as
> PEFT's `alpha/rank` ratio — it directly multiplies the LoRA delta. The first training attempt
> copied the CUDA config's `learning_rate: 2e-4` (tuned for PEFT's convention) and diverged to
> NaN loss by iteration 43. Matching mlx-lm's own tested default (`1e-5`, paired with
> `scale: 20.0`) fixed it; both adapters then converged cleanly.

## Evaluation: was fine-tuning necessary?

Both adapters were evaluated against a **zero-shot baseline** (same base model, no adapter,
same prompt) on an identical held-out test split, to isolate the effect of fine-tuning from the
base model's own capability. Run via:

```
python eval_clean_mlx.py --generate --compare
python eval_extract_mlx.py --generate --compare
```

**Cleaning** (character error rate vs. held-out gold text, `n=10` test documents):

| Quality bin | n | Raw OCR CER | Zero-shot CER | Fine-tuned CER | Reduction |
|---|---|---|---|---|---|
| clean | 3 | 0.021 | 0.015 | 0.010 | 52.8% |
| noisy | 4 | 10.55 | 10.44 | 0.429 | 95.9% |
| severe | 3 | 0.888 | 0.871 | 0.248 | 72.1% |
| **overall** | 10 | 4.49 | 4.44 | 0.249 | **94.4%** |

The overall figure is real but worth reading carefully: it's dominated by the `noisy` bin,
where the zero-shot baseline's CER of 10.44 (>1.0, meaning more edit operations than the
reference has characters) indicates the base model going off the rails on hard inputs rather
than being moderately worse. The `clean`-bin result (52.8% reduction) is the more representative
figure for typical-difficulty documents; the `noisy`/`severe` bins show fine-tuning's biggest
wins are exactly where the base model breaks down.

**Extraction** (`n=6` test documents):

| | Baseline (zero-shot) | Fine-tuned |
|---|---|---|
| JSON parse failure rate | 16.7% | **0%** |
| Avg. field F1 (incl. wage-increase set F1) | 0.57 | **0.75** |

A capable base model can often already emit plausible-looking JSON zero-shot — the real
evidence for fine-tuning here is parse reliability (eliminated entirely) and reduced
hallucination of non-null values on fields that should be null.

## Full-corpus results

The trained adapters were run over the entire scraped corpus via
[`infer_batch_mlx.py`](infer_batch_mlx.py):

```
python infer_batch_mlx.py --stage clean            # chunk-level filtered cleaning
python infer_batch_mlx.py --stage extract           # structured field extraction
python infer_batch_mlx.py --stage translate          # re-translate cleaned text (googletrans)
```

- **Cleaning**: 6,801 corpus rows (1,360 unique source documents — many rows share text across
  duplicate `id_accordo` entries) scanned; only chunks scored `noisy`/`severe` by a cheap
  OCR-quality heuristic were sent to the model (853 of ~2,600 total chunks in the final
  noisy/severe docs), the rest passed through unchanged. Output: `output/italy_txts_cleaned/`
  (1,360 files).
- **Extraction**: run over all 6,801 corpus records against the cleaned text, merged into
  `output/Italy_metadata_extended.csv` (7,505 rows, 19 new columns). Flags logged to
  `output/extraction_flags.csv`:
  - 5,350 `input_truncated` — document exceeded the 8,000-char input cap (some CCNLs run past
    500,000 characters); truncated and honestly flagged as partial-document extraction rather
    than silently dropped. **This is the main known limitation** — long documents only have
    their first ~8,000 characters analyzed, which can miss fields (e.g. wage tables) that
    appear later in the document.
  - 229 `json_parse_failed` — genuine parse failures, ~3.7% of the full corpus (vs. 0% on the
    small held-out eval set — expected drift at 1000x the scale).
- **Translation**: re-translates cleaned Italian text to English via `googletrans` (free/
  unofficial API, rate-limited to 1 request/sec — CPU + network only, no GPU/MLX involved).
  Chunked at 8,000 chars (not the original notebook's 2,000) after empirically finding request
  latency is roughly constant regardless of payload size up to the point the unofficial API
  starts silently failing (~14,200-14,400 chars — it returns the input text unchanged with no
  error, and that ceiling varies by document). A detector catches this signature (output
  byte-identical to input) and automatically retries with the chunk split in half. Completed
  the full corpus — 1,360/1,360 files, 0 unrecovered failures, 3 silent-failures caught and
  resolved by the retry logic. Output: `output/italy_translated_txts_cleaned/`.

`output/` (~424MB total) is **not committed to this repo** — it's fully regenerable from the
adapters via the three commands above, run against a local copy of the scraped corpus. See
[`../README_Italy`](../README_Italy) for where the raw scraped text it depends on lives.

## Repository layout

```
finetune/
├── schema.json                  # extraction target schema (all fields nullable)
├── chunking.py                  # shared ~2000-char chunking (matches italy_translator.ipynb)
├── data_prep.py                 # corpus indexing + OCR-quality binning
├── distill_labels.py            # frontier-LLM label generation
├── review_sample.py             # human spot-check sampling for hallucination gate
├── train_*_lora_{cuda,mlx}.py   # training entry points
├── train_common_{cuda,mlx}.py   # shared training utilities
├── eval_{clean,extract}_{cuda,mlx}.py  # baseline-vs-fine-tuned evaluation
├── infer_batch_{cuda,mlx}.py    # full-corpus clean/extract/translate
├── configs/                     # per-task, per-backend YAML configs
├── data/                        # labels, splits, predictions
├── adapters/                    # trained LoRA adapters (clean-lora-mlx, extract-lora-mlx)
└── output/                      # cleaned corpus, translated corpus, extended metadata, flags
```

## Requirements

```
pip install -r requirements-mlx.txt     # local Apple Silicon training/inference
pip install -r requirements-train-cuda.txt   # cloud GPU training
pip install -r requirements-infer-cuda.txt   # cloud GPU (vLLM) inference
```

`googletrans==3.1.0a0` is required for `--stage translate` only, and is intentionally **not**
in `requirements-mlx.txt` — it downgrades `httpx` in a way that breaks `huggingface_hub` (used
by `mlx-lm` for model loading). Run the translate stage in a separate environment.
