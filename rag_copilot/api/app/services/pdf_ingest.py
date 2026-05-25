from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import fitz

from api.app.core.config import settings
from api.app.services.chunking import chunk_page_text
from api.app.services.entity_extraction import extract_entities_llm, merge_entities


def ensure_dirs() -> None:
    settings.RAW_DIR.mkdir(parents=True, exist_ok=True)
    settings.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    settings.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    settings.EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    settings.ENTITIES_DIR.mkdir(parents=True, exist_ok=True)


def save_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def extract_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []

    with fitz.open(pdf_path) as doc:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text")
            pages.append((i + 1, text))

    return pages


def infer_section(page_metadata: Dict, fallback: str | None = None) -> str | None:
    section_titles = page_metadata.get("section_titles") or []

    if section_titles:
        return section_titles[0]

    return fallback


def ingest_pdf_bytes(filename: str, pdf_bytes: bytes) -> Dict:
    """
    Updated flow:
    - store raw PDF
    - parse pages
    - extract document/page entities
    - save entities JSON
    - enrich chunks with page, section, entities
    - write enriched chunks JSONL
    - write doc registry JSON
    """

    ensure_dirs()

    doc_id = str(uuid.uuid4())
    raw_path = settings.RAW_DIR / f"{doc_id}.pdf"
    save_bytes(raw_path, pdf_bytes)

    pages = extract_pages(raw_path)

    full_text = "\n\n".join(
        f"[Page {page_no}]\n{page_text}"
        for page_no, page_text in pages
    )

    document_metadata = extract_entities_llm(
        text=full_text,
        filename=filename,
        page=None,
    )

    page_metadata_items = []

    for page_no, page_text in pages:
        page_metadata = extract_entities_llm(
            text=page_text,
            filename=filename,
            page=page_no,
        )

        page_metadata_items.append({
            "page": page_no,
            **page_metadata,
        })

    page_metadata_lookup = {
        item["page"]: item
        for item in page_metadata_items
    }

    all_entities = merge_entities(page_metadata_items)

    entities_payload = {
        "doc_id": doc_id,
        "filename": filename,
        "document_metadata": document_metadata,
        "page_metadata": page_metadata_items,
        "merged_entities": all_entities,
    }

    entities_path = settings.ENTITIES_DIR / f"{doc_id}.json"
    entities_path.write_text(
        json.dumps(entities_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_chunks = []
    chunk_index = 0
    current_section = None

    for page_no, page_text in pages:
        page_metadata = page_metadata_lookup.get(page_no, {})
        current_section = infer_section(page_metadata, current_section)

        page_chunks = chunk_page_text(
            page_text=page_text,
            page_number=page_no,
            target_chars=settings.TARGET_CHARS,
            overlap_chars=settings.OVERLAP_CHARS,
        )

        for ch in page_chunks:
            chunk_id = f"{doc_id}:{page_no}:{chunk_index}"

            chunk_entities = page_metadata.get("entities", {})

            all_chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "filename": filename,
                "page": page_no,
                "section": current_section,
                "chunk_index": chunk_index,
                "text": ch.text,
                "entities": chunk_entities,
                "confidence_score": None,
                "metadata": {
                    "source": "pdf",
                    "page": page_no,
                    "filename": filename,
                    "section": current_section,
                    "entities": chunk_entities,
                    "document_type": document_metadata.get("document_type"),
                    "document_summary": document_metadata.get("summary"),
                }
            })

            chunk_index += 1

    chunks_path = settings.CHUNKS_DIR / f"{doc_id}.jsonl"

    with chunks_path.open("w", encoding="utf-8") as f:
        for row in all_chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    doc_meta = {
        "doc_id": doc_id,
        "filename": filename,
        "raw_path": str(raw_path),
        "chunks_path": str(chunks_path),
        "entities_path": str(entities_path),
        "num_pages": len(pages),
        "num_chunks": len(all_chunks),
    }

    (settings.DOCS_DIR / f"{doc_id}.json").write_text(
        json.dumps(doc_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return doc_meta