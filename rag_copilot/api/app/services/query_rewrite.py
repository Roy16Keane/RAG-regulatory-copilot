from __future__ import annotations

from openai import OpenAI
from api.app.core.config import settings

client = OpenAI()

REWRITE_SYSTEM = """You rewrite user questions for document retrieval.
Return TWO lines ONLY:
KW: <keyword-optimized query for BM25>
SEM: <semantic query for vector search>
Keep them short and specific. Preserve legal terms, article numbers, named entities.
"""

def rewrite_query(question: str) -> dict:
    resp = client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": question},
        ],
        temperature=0.1,
    )
    text = (resp.choices[0].message.content or "").strip().splitlines()
    kw = question
    sem = question
    for line in text:
        if line.upper().startswith("KW:"):
            kw = line.split(":", 1)[1].strip() or kw
        if line.upper().startswith("SEM:"):
            sem = line.split(":", 1)[1].strip() or sem
    return {"kw": kw, "sem": sem}
