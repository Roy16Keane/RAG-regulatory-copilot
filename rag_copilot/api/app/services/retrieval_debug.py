from __future__ import annotations

from typing import Dict

from api.app.services.qdrant_store import vector_search
from api.app.services.opensearch_bm25 import bm25_search
from api.app.services.hybrid_retrieval import hybrid_search
from api.app.services.query_rewrite import rewrite_query

def retrieval_debug(
    query: str,
    doc_id: str | None = None,
    top_k_vec: int = 8,
    top_k_bm25: int = 8,
    top_k_hybrid: int = 8,
    alpha: float = 0.65,
    use_rewrite: bool = True,
) -> Dict:
    rewritten = rewrite_query(query) if use_rewrite else {"kw": query, "sem": query}

    vector_results = vector_search(
        query=rewritten["sem"],
        top_k=top_k_vec,
        doc_id=doc_id,
    )

    bm25_results = bm25_search(
        query=rewritten["kw"],
        top_k=top_k_bm25,
        doc_id=doc_id,
    )

    hybrid_results = hybrid_search(
        query=query,
        doc_id=doc_id,
        top_k=top_k_hybrid,
        top_k_vec=top_k_vec,
        top_k_bm25=top_k_bm25,
        alpha=alpha,
        use_rewrite=use_rewrite,
    )["results"]

    return {
        "query": query,
        "doc_id": doc_id,
        "use_rewrite": use_rewrite,
        "rewritten": rewritten,
        "alpha": alpha,
        "vector_results": vector_results,
        "bm25_results": bm25_results,
        "hybrid_results": hybrid_results,
    }
