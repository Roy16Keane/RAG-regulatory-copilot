from pydantic import BaseModel

class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    raw_path: str
    chunks_path: str
    num_pages: int
    num_chunks: int
