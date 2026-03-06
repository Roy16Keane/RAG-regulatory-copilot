from __future__ import annotations

import json
from typing import Dict, Generator

from openai import OpenAI
from api.app.core.config import settings
from api.app.services.hybrid_retrieval import hybrid_search
from api.app.services.rag_prompt import SYSTEM_PROMPT, build_user_prompt

client = OpenAI()

def stream_rag_answer(
    question: str,
    doc_id: str | None = None,
    top_k: int = 8,
    top_k_vec: int = 12,
    top_k_bm25: int = 12,
    alpha: float = 0.65,
) -> Generator[str, None, None]:
    retrieved = hybrid_search(
        query=question,
        doc_id=doc_id,
        top_k=top_k,
        top_k_vec=top_k_vec,
        top_k_bm25=top_k_bm25,
        alpha=alpha,
    )

    chunks = retrieved["results"]

    # compact context (reuse the same format as rag_chat)
    parts = []
    total = 0
    max_chars = 12000
    for c in chunks:
        block = f"[chunk_id={c['chunk_id']} page={c.get('page')} file={c.get('filename')}]\n{c.get('text','')}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    context = "\n---\n".join(parts)

    user_prompt = build_user_prompt(question=question, context=context)

    # Send an initial event containing metadata (citations panel can pre-render)
    meta = {
        "question": question,
        "doc_id": doc_id,
        "alpha": alpha,
        "retrieval": {"top_k": top_k, "top_k_vec": top_k_vec, "top_k_bm25": top_k_bm25},
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "filename": c.get("filename"),
                "page": c.get("page"),
                "chunk_index": c.get("chunk_index"),
                "text": c.get("text"),
            }
            for c in chunks
        ],
    }
    yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

    stream = client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        stream=True,
    )

    # Stream token deltas
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
        if delta:
            yield f"event: token\ndata: {json.dumps({'token': delta})}\n\n"

    yield "event: done\ndata: {}\n\n"
