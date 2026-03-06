from __future__ import annotations

from typing import Dict, List, Tuple

from api.app.services.qdrant_store import vector_search
from api.app.services.opensearch_bm25 import bm25_search
from api.app.services.query_rewrite import rewrite_query

def _minmax_norm(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0 for _ in scores]  # all same -> treat as equally relevant
    return [(s - mn) / (mx - mn) for s in scores]

def hybrid_search(
    query: str,
    doc_id: str | None = None,
    top_k: int = 8,
    top_k_vec: int = 12,
    top_k_bm25: int = 12,
    alpha: float = 0.65,
    use_rewrite: bool = True,
) -> Dict:
    """
    Hybrid retrieval:
    - vector_search from Qdrant
    - bm25_search from OpenSearch
    - normalize scores
    - merge by chunk_id
    - compute hybrid score
    """
    rewritten = rewrite_query(query) if use_rewrite else {'kw': query, 'sem': query}
    vec_results = vector_search(query=rewritten['sem'], top_k=top_k_vec, doc_id=doc_id)
    bm25_results = bm25_search(query=rewritten['kw'], top_k=top_k_bm25, doc_id=doc_id)

    vec_scores = [r["score"] for r in vec_results]
    bm25_scores = [r["score"] for r in bm25_results]

    vec_norm = _minmax_norm(vec_scores)
    bm25_norm = _minmax_norm(bm25_scores)

    # build maps by chunk_id
    merged: Dict[str, Dict] = {}

    for r, n in zip(vec_results, vec_norm):
        cid = r["chunk_id"]
        merged[cid] = {
            **r,
            "vector_score": float(r["score"]),
            "bm25_score": 0.0,
            "vector_norm": float(n),
            "bm25_norm": 0.0,
        }

    for r, n in zip(bm25_results, bm25_norm):
        cid = r["chunk_id"]
        if cid not in merged:
            merged[cid] = {
                **r,
                "vector_score": 0.0,
                "bm25_score": float(r["score"]),
                "vector_norm": 0.0,
                "bm25_norm": float(n),
            }
        else:
            merged[cid]["bm25_score"] = float(r["score"])
            merged[cid]["bm25_norm"] = float(n)

    # compute hybrid score
    for cid, item in merged.items():
        item["hybrid_score"] = float(alpha * item["vector_norm"] + (1 - alpha) * item["bm25_norm"])

    # sort
    ranked = sorted(merged.values(), key=lambda x: x["hybrid_score"], reverse=True)

    return {
        "query": query,
        "doc_id": doc_id,
        "alpha": alpha,
        "top_k": top_k,
        "top_k_vec": top_k_vec,
        "top_k_bm25": top_k_bm25,
        "rewritten": rewritten,
        "results": ranked[:top_k],
    }
