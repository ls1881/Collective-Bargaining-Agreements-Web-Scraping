# Collective Bargaining Agreements — Web Scraping & Structuring

End-to-end pipeline for collecting, cleaning, translating, and structuring national Collective
Bargaining Agreements (CCNLs) from government labor archives, starting with Italy's CNEL
archive. Combines browser-automated scraping/OCR with a fine-tuned LLM pipeline that turns messy
scanned-document text into structured, queryable data.

## What's in here

| Component | Status | What it does |
|---|---|---|
| [`italy_scraping/`](italy_scraping) | Production | Scrapes the CNEL archive, extracts text from 7,500+ PDF/DOC/RTF agreements (OCR fallback for scans), machine-translates Italian → English |
| [`finetune/`](finetune) | Production | QLoRA-fine-tuned Qwen2.5-7B pipeline that cleans OCR noise and extracts structured fields (wages, hours, leave, notice periods) from the scraped text |
| [`spain_scraping/`](spain_scraping) | Early stage | Scraper for the Spanish equivalent archive; not yet run at scale |

## Highlights

- **7,500+ agreements** scraped, OCR'd, and translated from Italy's national CNEL archive.
- Fine-tuned two LoRA adapters (cleaning + structured extraction) on **Qwen2.5-7B-Instruct**,
  evaluated against a zero-shot baseline to isolate the actual contribution of fine-tuning
  (see [`finetune/README.md`](finetune/README.md) for full methodology and numbers):
  - Cleaning: meaningful character-error-rate reduction vs. zero-shot on typical-difficulty
    documents, with much larger gains on the hardest OCR cases where the base model breaks down.
  - Extraction: JSON parse failure rate reduced from 16.7% (zero-shot) to 0%; average field F1
    improved from 0.57 to 0.75.
- Adapters were then run over the **full corpus** to produce a cleaned text corpus, an
  English translation, and an extended metadata CSV with structured fields (sector, wage
  increases, working hours, probation/notice periods, leave entitlements) for every agreement.
- Built and validated **two parallel training/inference stacks** (CUDA for rented GPUs, Apple
  MLX for local Apple Silicon) after cloud GPU access fell through mid-project — the MLX stack
  trained, evaluated, and ran the full pipeline locally with no cloud dependency.

## Repository structure

```
├── italy_scraping/            # Scraper, OCR/text extraction, translation notebooks
│   └── Italy_metadata.csv     # Core administrative database (7,500+ agreement records)
├── spain_scraping/            # Spain archive scraper (early stage)
├── finetune/                  # QLoRA cleaning + structured-extraction pipeline
│   ├── adapters/               # Trained LoRA adapters
│   ├── output/                 # Cleaned corpus, translated corpus, extended metadata CSV
│   └── README.md                # Full pipeline documentation, architecture, eval results
├── README_Italy                # Detailed data-dictionary / field reference for Italy_metadata.csv
└── Italy CBAs Report Structure.pdf
```

## Getting started

Scraping and OCR (`italy_scraping/`) requires Python 3.12, Tesseract OCR on the system path,
and `seleniumbase`/`pdfplumber`/`pytesseract`. The fine-tuning pipeline (`finetune/`) has
separate requirements for its CUDA and MLX backends — see
[`finetune/README.md`](finetune/README.md) for setup and usage of each stage.

## Data source

Source documents are Italian Collective Bargaining Agreements (CCNLs) as filed with CNEL
(Consiglio Nazionale dell'Economia e del Lavoro), Italy's National Council for Economics and
Labour — publicly available government archive data.
