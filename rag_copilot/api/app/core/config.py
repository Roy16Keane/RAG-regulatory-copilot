from pathlib import Path
from pydantic import BaseModel
import os

class Settings(BaseModel):

    DATA_DIR: Path = Path("data")
    RAW_DIR: Path = Path("data/raw")
    CHUNKS_DIR: Path = Path("data/chunks")
    DOCS_DIR: Path = Path("data/docs")

    TARGET_CHARS: int = 1400
    OVERLAP_CHARS: int = 150

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")

    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "rag_chunks")

    # OpenSearch
    OPENSEARCH_URL: str = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
    OPENSEARCH_INDEX: str = os.getenv("OPENSEARCH_INDEX", "rag_chunks_bm25")


settings = Settings()
