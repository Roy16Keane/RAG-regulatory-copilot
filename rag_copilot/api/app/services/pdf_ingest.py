from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF

from api.app.core.config import settings
from api.app.services.chunking import chunk_page_text

def ensure_dirs() -> None:
    settings.RAW_DIR.mkdir(parents=True, exist_ok=True)
    settings.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    settings.DOCS_DIR.mkdir(parents=True, exist_ok=True)

def save_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

def extract_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Returns list of (page_number_1based, text).
    """
    pages: List[Tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text")  # MVP, robust
            pages.append((i + 1, text))
    return pages

def ingest_pdf_bytes(filename: str, pdf_bytes: bytes) -> Dict:
    """
    Phase 1:
    - store raw pdf
    - parse pages
    - chunk
    - write chunks jsonl
    - write doc registry json
    """
    ensure_dirs()

    doc_id = str(uuid.uuid4())
    raw_path = settings.RAW_DIR / f"{doc_id}.pdf"
    save_bytes(raw_path, pdf_bytes)

    pages = extract_pages(raw_path)

    all_chunks = []
    chunk_index = 0

    for page_no, page_text in pages:
        page_chunks = chunk_page_text(
            page_text=page_text,
            page_number=page_no,
            target_chars=settings.TARGET_CHARS,
            overlap_chars=settings.OVERLAP_CHARS,
        )

        for ch in page_chunks:
            chunk_id = f"{doc_id}:{page_no}:{chunk_index}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "filename": filename,
                "page": page_no,
                "chunk_index": chunk_index,
                "text": ch.text,
                "metadata": {
                    "source": "pdf",
                    "page": page_no,
                    "filename": filename,
                }
            })
            chunk_index += 1

    # Persist chunks as JSONL
    chunks_path = settings.CHUNKS_DIR / f"{doc_id}.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for row in all_chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Persist doc registry
    doc_meta = {
        "doc_id": doc_id,
        "filename": filename,
        "raw_path": str(raw_path),
        "chunks_path": str(chunks_path),
        "num_pages": len(pages),
        "num_chunks": len(all_chunks),
    }
    (settings.DOCS_DIR / f"{doc_id}.json").write_text(
        json.dumps(doc_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return doc_meta