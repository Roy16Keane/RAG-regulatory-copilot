from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List

@dataclass
class Chunk:
    text: str
    page: int

def _normalize_text(t: str) -> str:
    t = t.replace("\x00", " ")
    # collapse whitespace but keep newlines somewhat
    lines = [ln.strip() for ln in t.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()

def chunk_page_text(
    page_text: str,
    page_number: int,
    target_chars: int = 1400,
    overlap_chars: int = 150,
) -> List[Chunk]:
    """
    Simple, safe MVP chunker:
    - Normalize whitespace
    - Split by paragraphs
    - Pack paragraphs until target size
    - Add small overlap by taking tail chars from previous chunk
    """
    text = _normalize_text(page_text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if not cur:
            return
        chunks.append("\n".join(cur).strip())
        cur = []
        cur_len = 0

    for p in paragraphs:
        p_len = len(p) + 1
        if cur_len + p_len <= target_chars:
            cur.append(p)
            cur_len += p_len
        else:
            flush()
            cur.append(p)
            cur_len = p_len

    flush()

    # add overlap (character tail) between chunks
    if overlap_chars > 0 and len(chunks) > 1:
        overlapped = []
        prev_tail = ""
        for i, c in enumerate(chunks):
            if i == 0:
                overlapped.append(c)
            else:
                prev_tail = chunks[i - 1][-overlap_chars:]
                overlapped.append((prev_tail + "\n" + c).strip())
        chunks = overlapped

    return [Chunk(text=c, page=page_number) for c in chunks]