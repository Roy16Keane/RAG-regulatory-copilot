from __future__ import annotations

from typing import List
from openai import OpenAI

from api.app.core.config import settings

client = OpenAI()

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Returns embeddings for a list of texts using OpenAI embeddings.
    Requires env var: OPENAI_API_KEY
    """
    # OpenAI embeddings endpoint supports batching
    client = OpenAI()  # create client when needed
    resp = client.embeddings.create(
        model=settings.OPENAI_EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in resp.data]


