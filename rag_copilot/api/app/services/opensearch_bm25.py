from __future__ import annotations

import json
from typing import Dict, List

from opensearchpy import OpenSearch
from api.app.core.config import settings


def _client() -> OpenSearch:
    return OpenSearch(
        hosts=[settings.OPENSEARCH_URL],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )


def ensure_index() -> None:
    os = _client()

    if os.indices.exists(index=settings.OPENSEARCH_INDEX):
        return

    mapping = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "filename": {"type": "keyword"},
                "page": {"type": "integer"},
                "section": {"type": "text"},
                "chunk_index": {"type": "integer"},
                "text": {"type": "text"},
                "entities": {"type": "object", "enabled": True},
                "metadata": {"type": "object", "enabled": True},
            }
        },
    }

    os.indices.create(index=settings.OPENSEARCH_INDEX, body=mapping)


def index_doc_chunks(doc_id: str, batch_size: int = 200) -> Dict:
    ensure_index()
    os = _client()

    chunks_path = settings.CHUNKS_DIR / f"{doc_id}.jsonl"

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    bulk_lines: List[str] = []
    indexed = 0

    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            _id = row["chunk_id"]

            action = {
                "index": {
                    "_index": settings.OPENSEARCH_INDEX,
                    "_id": _id,
                }
            }

            source = {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "filename": row.get("filename"),
                "page": row.get("page"),
                "section": row.get("section"),
                "entities": row.get("entities", {}),
                "metadata": row.get("metadata", {}),
                "chunk_index": row.get("chunk_index"),
                "text": row["text"],
            }

            bulk_lines.append(json.dumps(action))
            bulk_lines.append(json.dumps(source))

            if len(bulk_lines) >= batch_size * 2:
                resp = os.bulk(body="\n".join(bulk_lines) + "\n")

                if resp.get("errors"):
                    for item in resp.get("items", []):
                        if "index" in item and item["index"].get("error"):
                            raise RuntimeError(item["index"]["error"])

                indexed += len(bulk_lines) // 2
                bulk_lines = []

    if bulk_lines:
        resp = os.bulk(body="\n".join(bulk_lines) + "\n")

        if resp.get("errors"):
            for item in resp.get("items", []):
                if "index" in item and item["index"].get("error"):
                    raise RuntimeError(item["index"]["error"])

        indexed += len(bulk_lines) // 2

    return {
        "doc_id": doc_id,
        "indexed": indexed,
        "index": settings.OPENSEARCH_INDEX,
    }


def bm25_search(query: str, top_k: int = 8, doc_id: str | None = None) -> List[Dict]:
    ensure_index()
    os = _client()

    must = [{"match": {"text": {"query": query}}}]
    filter_ = []

    if doc_id:
        filter_.append({"term": {"doc_id": doc_id}})

    body = {
        "size": top_k,
        "query": {
            "bool": {
                "must": must,
                "filter": filter_,
            }
        },
    }

    resp = os.search(index=settings.OPENSEARCH_INDEX, body=body)
    hits = resp.get("hits", {}).get("hits", [])

    results = []

    for hit in hits:
        source = hit.get("_source", {})

        results.append({
            "score": float(hit.get("_score", 0.0)),
            "confidence_score": float(hit.get("_score", 0.0)),
            "chunk_id": source.get("chunk_id"),
            "doc_id": source.get("doc_id"),
            "filename": source.get("filename"),
            "page": source.get("page"),
            "section": source.get("section"),
            "entities": source.get("entities", {}),
            "metadata": source.get("metadata", {}),
            "chunk_index": source.get("chunk_index"),
            "text": source.get("text"),
        })

    return results