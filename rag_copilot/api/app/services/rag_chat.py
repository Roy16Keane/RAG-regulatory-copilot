from __future__ import annotations

import re
from typing import Dict, List, Any

from openai import OpenAI

from api.app.core.config import settings
from api.app.services.hybrid_retrieval import hybrid_search
from api.app.services.rag_prompt import SYSTEM_PROMPT, build_user_prompt


client = OpenAI(api_key=settings.OPENAI_API_KEY)


def _extract_cited_chunk_ids(answer: str) -> list[str]:
    ids = re.findall(r"\[([^\[\]]+)\]", answer or "")

    seen = set()
    out = []

    for x in ids:
        x = x.strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)

    return out


def _compact_entities(entities: Dict[str, Any], max_items: int = 8) -> Dict[str, List[str]]:
    compacted = {}

    for key, values in (entities or {}).items():
        if isinstance(values, list):
            clean_values = [str(v) for v in values if v]
            compacted[key] = clean_values[:max_items]

    return compacted


def _make_context(chunks: List[Dict], max_chars: int = 12000) -> str:
    parts: List[str] = []
    total = 0

    for c in chunks:
        entities = _compact_entities(c.get("entities", {}))

        block = (
            f"[chunk_id={c['chunk_id']} "
            f"page={c.get('page')} "
            f"section={c.get('section')} "
            f"file={c.get('filename')} "
            f"confidence_score={c.get('hybrid_score', c.get('confidence_score', c.get('score')))}]\n"
            f"entities={entities}\n"
            f"{c.get('text', '')}\n"
        )

        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n---\n".join(parts)


def _citations_from_chunks(chunks: List[Dict], max_snippet_chars: int = 240) -> List[Dict]:
    citations = []

    for c in chunks:
        text = (c.get("text") or "").strip().replace("\n", " ")
        snippet = text[:max_snippet_chars] + ("..." if len(text) > max_snippet_chars else "")

        citations.append({
            "chunk_id": c.get("chunk_id"),
            "filename": c.get("filename"),
            "page": c.get("page"),
            "section": c.get("section"),
            "confidence_score": c.get("hybrid_score", c.get("confidence_score", c.get("score"))),
            "chunk_index": c.get("chunk_index"),
            "snippet": snippet,
            "entities": c.get("entities", {}),
        })

    return citations


def _supporting_chunks(chunks: List[Dict]) -> List[Dict]:
    return [
        {
            "chunk_id": c.get("chunk_id"),
            "doc_id": c.get("doc_id"),
            "filename": c.get("filename"),
            "page": c.get("page"),
            "section": c.get("section"),
            "confidence_score": c.get("hybrid_score", c.get("confidence_score", c.get("score"))),
            "vector_score": c.get("vector_score"),
            "bm25_score": c.get("bm25_score"),
            "entities": c.get("entities", {}),
            "text": c.get("text"),
        }
        for c in chunks
    ]


def _merge_entities_from_chunks(chunks: List[Dict]) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}

    for c in chunks:
        entities = c.get("entities") or {}

        for key, values in entities.items():
            merged.setdefault(key, [])

            if isinstance(values, list):
                merged[key].extend(str(v) for v in values if v)
            elif values:
                merged[key].append(str(values))

    return {
        key: sorted(set(values))
        for key, values in merged.items()
    }


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

    user_prompt = build_user_prompt(
        question=question,
        context=context,
    )

    resp = client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer = resp.choices[0].message.content or ""

    cited_ids = _extract_cited_chunk_ids(answer)
    chunk_map = {c["chunk_id"]: c for c in chunks}

    cited_chunks = [
        chunk_map[cid]
        for cid in cited_ids
        if cid in chunk_map
    ]

    final_chunks = cited_chunks if cited_chunks else chunks

    return {
        "question": question,
        "doc_id": doc_id,
        "alpha": alpha,
        "retrieval": {
            "top_k": top_k,
            "top_k_vec": top_k_vec,
            "top_k_bm25": top_k_bm25,
            "rewritten": retrieved.get("rewritten"),
        },
        "answer": answer,
        "citations": _citations_from_chunks(final_chunks),
        "supporting_chunks": _supporting_chunks(final_chunks),
        "extracted_entities": _merge_entities_from_chunks(final_chunks),
        "chunks_used": [c["chunk_id"] for c in final_chunks],
    }