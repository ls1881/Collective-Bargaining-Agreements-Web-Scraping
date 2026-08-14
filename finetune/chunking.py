"""Shared chunking utility for the finetune pipeline.

Ports the noise-stripping + ~2000-char chunking from italy_scraping/italy_translator.ipynb
so distillation labels and full-corpus inference use identical chunk boundaries.
Unlike the original notebook (hard `content[i:i+MAX_CHARS]` cuts), this splits on
paragraph/sentence boundaries near the target size to avoid cutting mid-word/mid-sentence,
which matters for training data quality.
"""
import re

MAX_CHARS = 2000

_DOT_LEADER_RE = re.compile(r"\.{2,}")
_OCR_NOISE_RE = re.compile(r"[o*]{3,}")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def clean_text(text: str) -> str:
    """Strip common OCR artifacts (dot-leader indices, repeated o/* runs)."""
    text = _DOT_LEADER_RE.sub(" ", text)
    text = _OCR_NOISE_RE.sub(" ", text)
    return text


def chunk_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split text into ~max_chars chunks on paragraph/sentence boundaries.

    Falls back to a hard cut only if a single paragraph/sentence exceeds max_chars.
    """
    text = clean_text(text)
    paragraphs = [p for p in _PARA_SPLIT_RE.split(text) if p.strip()]

    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        units = [para] if len(para) <= max_chars else _SENT_SPLIT_RE.split(para)
        for unit in units:
            if len(unit) > max_chars:
                # Single sentence longer than max_chars: hard-cut as last resort.
                flush()
                for i in range(0, len(unit), max_chars):
                    chunks.append(unit[i:i + max_chars].strip())
                continue
            candidate = f"{current}\n\n{unit}" if current else unit
            if len(candidate) > max_chars:
                flush()
                current = unit
            else:
                current = candidate
    flush()
    return chunks


def chunk_document(doc_id: str, text: str, max_chars: int = MAX_CHARS) -> list[dict]:
    """Chunk a document and return records with reassembly metadata."""
    chunks = chunk_text(text, max_chars=max_chars)
    records = []
    pos = 0
    for i, chunk in enumerate(chunks):
        start = text.find(chunk[:50], pos) if chunk else pos
        records.append({
            "doc_id": doc_id,
            "chunk_id": i,
            "text": chunk,
            "char_start": max(start, 0),
        })
        pos = max(start, 0) + len(chunk)
    return records


def reassemble(chunks: list[str]) -> str:
    """Reassemble cleaned chunks (in chunk_id order) back into a full document."""
    return "\n\n".join(chunks)
