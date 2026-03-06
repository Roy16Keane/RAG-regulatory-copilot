from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.app.schemas.ingest import IngestResponse
from api.app.services.pdf_ingest import ingest_pdf_bytes
from api.app.services.qdrant_store import index_doc_to_qdrant, vector_search
from api.app.services.opensearch_bm25 import index_doc_chunks, bm25_search
from api.app.services.hybrid_retrieval import hybrid_search
from api.app.services.rag_chat import rag_answer
from api.app.services.rag_stream import stream_rag_answer

app = FastAPI(title="RAG API", version="0.3.0")

@app.get("/health")
def health():
    return {"status": "ok"}

#  Phase 1: Ingest 
@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    meta = ingest_pdf_bytes(filename=file.filename, pdf_bytes=pdf_bytes)
    return meta

#  Phase 2: Index -> Qdrant 
class IndexQdrantRequest(BaseModel):
    doc_id: str
    batch_size: int = 64

@app.post("/index/qdrant")
def index_qdrant(req: IndexQdrantRequest):
    try:
        return index_doc_to_qdrant(doc_id=req.doc_id, batch_size=req.batch_size)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

# Phase 3: Index -> OpenSearch (BM25) 
class IndexBM25Request(BaseModel):
    doc_id: str
    batch_size: int = 200

@app.post("/index/bm25")
def index_bm25(req: IndexBM25Request):
    try:
        return index_doc_chunks(doc_id=req.doc_id, batch_size=req.batch_size)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#  Retrieval: Vector
class RetrieveVectorRequest(BaseModel):
    query: str
    top_k: int = 8
    doc_id: str | None = None

@app.post("/retrieve/vector")
def retrieve_vector(req: RetrieveVectorRequest):
    try:
        results = vector_search(query=req.query, top_k=req.top_k, doc_id=req.doc_id)
        return {"query": req.query, "top_k": req.top_k, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#  Retrieval: BM25 
class RetrieveBM25Request(BaseModel):
    query: str
    top_k: int = 8
    doc_id: str | None = None

@app.post("/retrieve/bm25")
def retrieve_bm25(req: RetrieveBM25Request):
    try:
        results = bm25_search(query=req.query, top_k=req.top_k, doc_id=req.doc_id)
        return {"query": req.query, "top_k": req.top_k, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Retrieval: Hybrid (Vector + BM25) 
class RetrieveHybridRequest(BaseModel):
    query: str
    top_k: int = 8
    top_k_vec: int = 12
    top_k_bm25: int = 12
    alpha: float = 0.65
    doc_id: str | None = None

@app.post("/retrieve/hybrid")
def retrieve_hybrid(req: RetrieveHybridRequest):
    try:
        return hybrid_search(
            query=req.query,
            doc_id=req.doc_id,
            top_k=req.top_k,
            top_k_vec=req.top_k_vec,
            top_k_bm25=req.top_k_bm25,
            alpha=req.alpha
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Phase 5: RAG Chat (Hybrid -> LLM -> citations) 
class ChatRequest(BaseModel):
    question: str
    doc_id: str | None = None
    top_k: int = 8
    top_k_vec: int = 12
    top_k_bm25: int = 12
    alpha: float = 0.65

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        return rag_answer(
            question=req.question,
            doc_id=req.doc_id,
            top_k=req.top_k,
            top_k_vec=req.top_k_vec,
            top_k_bm25=req.top_k_bm25,
            alpha=req.alpha,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    gen = stream_rag_answer(
        question=req.question,
        doc_id=req.doc_id,
        top_k=req.top_k,
        top_k_vec=req.top_k_vec,
        top_k_bm25=req.top_k_bm25,
        alpha=req.alpha,
    )
    return StreamingResponse(gen, media_type="text/event-stream")
