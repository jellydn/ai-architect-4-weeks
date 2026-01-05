"""FastAPI RAG Server

Main application entry point. Provides HTTP endpoints for ingestion and querying.
"""

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from generation import RAGGenerator
from ingestion import DocumentIngester
from pydantic import BaseModel
from retrieval import RAGRetriever

# Load environment
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Architect - RAG API", description="Week 1: Basic RAG system", version="0.1.0"
)

# Global instances
base_url = os.getenv("OPENAI_BASE_URL")  # For Ollama, LM Studio, etc.

ingester = DocumentIngester(
    chunk_size=int(os.getenv("CHUNK_SIZE", 512)),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50)),
)
retriever = RAGRetriever(
    embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    base_url=base_url,
)
generator = RAGGenerator(
    model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
    temperature=float(os.getenv("LLM_TEMPERATURE", 0.7)),
    base_url=base_url,
)


# Pydantic models for request/response
class IngestRequest(BaseModel):
    file_paths: list[str]


class IngestResponse(BaseModel):
    status: str
    chunks_created: int
    files_processed: int


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "rag-api"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents for RAG.

    Takes file paths, loads, chunks, and indexes them.
    """
    try:
        all_chunks = []

        for filepath in request.file_paths:
            logger.info(f"Ingesting {filepath}")
            chunks = ingester.ingest(filepath)
            all_chunks.extend(chunks)

        # Index for retrieval
        retriever.index(all_chunks)

        return IngestResponse(
            status="success",
            chunks_created=len(all_chunks),
            files_processed=len(request.file_paths),
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system.

    Retrieves relevant documents and generates an answer.
    """
    try:
        if not retriever.documents:
            raise HTTPException(status_code=400, detail="No documents indexed. Call /ingest first.")

        result = generator.rag_answer(request.query, retriever)

        return QueryResponse(
            answer=result["answer"], sources=result["sources"], latency_ms=result["latency_ms"]
        )

    except HTTPException:
        raise  # Let HTTPException pass through unchanged
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"

    uvicorn.run(app, host=host, port=port)
