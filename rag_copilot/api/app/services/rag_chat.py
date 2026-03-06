from __future__ import annotations

from typing import Dict, List
from openai import OpenAI

from api.app.core.config import settings
from api.app.services.hybrid_retrieval import hybrid_search
from api.app.services.rag_prompt import SYSTEM_PROMPT, build_user_prompt

client = OpenAI()

import re

def _extract_cited_chunk_ids(answer: str) -> list[str]:
    # captures [chunk_id] patterns, allows multiple like [id1][id2]
    ids = re.findall(r"\[([^\[\]]+)\]", answer or "")
    # keep order, unique
    seen = set()
    out = []
    for x in ids:
        x = x.strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _make_context(chunks: List[Dict], max_chars: int = 12000) -> str:
    """
    Simple context formatter.
    Keeps chunk_id + page so citations are meaningful.
    """
    parts: List[str] = []
    total = 0
    for c in chunks:
        block = f"[chunk_id={c['chunk_id']} page={c.get('page')} file={c.get('filename')}]\n{c.get('text','')}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n---\n".join(parts)

def _citations_from_chunks(chunks: List[Dict], max_snippet_chars: int = 240) -> List[Dict]:
    cites = []
    for c in chunks:
        text = (c.get("text") or "").strip().replace("\n", " ")
        snippet = text[:max_snippet_chars] + ("..." if len(text) > max_snippet_chars else "")
        cites.append({
            "chunk_id": c["chunk_id"],
            "filename": c.get("filename"),
            "page": c.get("page"),
            "chunk_index": c.get("chunk_index"),
            "snippet": snippet,
        })
    return cites

def rag_answer(
    question: str,
    doc_id: str | None = None,
    top_k: int = 8,
    top_k_vec: int = 12,
    top_k_bm25: int = 12,
    alpha: float = 0.65,
) -> Dict:
    retrieved = hybrid_search(
        query=question,
        doc_id=doc_id,
        top_k=top_k,
        top_k_vec=top_k_vec,
        top_k_bm25=top_k_bm25,
        alpha=alpha,
    )

    chunks = retrieved["results"]
    context = _make_context(chunks)

    user_prompt = build_user_prompt(question=question, context=context)

    resp = client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer = resp.choices[0].message.content
    cited_ids = _extract_cited_chunk_ids(answer)
    chunk_map = {c['chunk_id']: c for c in chunks}
    cited_chunks = [chunk_map[cid] for cid in cited_ids if cid in chunk_map]

    return {
        "question": question,
        "doc_id": doc_id,
        "alpha": alpha,
        "retrieval": {
            "top_k": top_k,
            "top_k_vec": top_k_vec,
            "top_k_bm25": top_k_bm25,
        },
        "answer": answer,
        "citations": _citations_from_chunks(cited_chunks if cited_chunks else chunks),
        "chunks_used": cited_ids if cited_ids else [c["chunk_id"] for c in chunks],
    }
