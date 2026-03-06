from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from api.app.core.config import settings
from api.app.services.embeddings import embed_texts

def _client() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_URL)

def ensure_collection(vector_dim: int) -> None:
    qc = _client()
    collections = {c.name for c in qc.get_collections().collections}
    if settings.QDRANT_COLLECTION in collections:
        return

    qc.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=vector_dim,
            distance=Distance.COSINE
        )
    )

def _load_chunks_jsonl(doc_id: str) -> List[Dict]:
    path = settings.CHUNKS_DIR / f"{doc_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def index_doc_to_qdrant(doc_id: str, batch_size: int = 64) -> Dict:
    """
    Loads chunks from data/chunks/<doc_id>.jsonl,
    generates embeddings, and upserts to Qdrant with payload metadata.
    """
    qc = _client()
    chunks = _load_chunks_jsonl(doc_id)

    if not chunks:
        return {"doc_id": doc_id, "indexed": 0}

    # Embed first chunk to learn vector dimension (no hardcoding)
    first_emb = embed_texts([chunks[0]["text"]])[0]
    ensure_collection(vector_dim=len(first_emb))

    indexed = 0

    # Upsert in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        embs = embed_texts(texts)

        points: List[PointStruct] = []
        for c, v in zip(batch, embs):
            # Use a stable integer ID derived from chunk_id (Qdrant point id must be int or uuid)
            # We'll store chunk_id as payload for citations.
            point_id = abs(hash(c["chunk_id"])) % (2**63)

            payload = {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "filename": c.get("filename"),
                "page": c.get("page"),
                "chunk_index": c.get("chunk_index"),
                "text": c["text"],  # store text for easy retrieval/snippets
                **(c.get("metadata") or {})
            }

            points.append(PointStruct(id=point_id, vector=v, payload=payload))

        qc.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
        indexed += len(points)

    return {"doc_id": doc_id, "indexed": indexed, "collection": settings.QDRANT_COLLECTION}


def vector_search(query: str, top_k: int = 8, doc_id: str | None = None) -> List[Dict]:
    qc = _client()
    q_emb = embed_texts([query])[0]

    qfilter = None
    if doc_id:
        qfilter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )

    res = qc.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=q_emb,
        limit=top_k,
        with_payload=True,
        query_filter=qfilter,
    )

    # qdrant-client returns QueryResponse with .points
    results = []
    for pnt in res.points:
        payload = pnt.payload or {}
        results.append({
            "score": float(pnt.score),
            "chunk_id": payload.get("chunk_id"),
            "doc_id": payload.get("doc_id"),
            "filename": payload.get("filename"),
            "page": payload.get("page"),
            "chunk_index": payload.get("chunk_index"),
            "text": payload.get("text"),
        })
    return results