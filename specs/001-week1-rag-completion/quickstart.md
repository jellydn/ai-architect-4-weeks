# Quickstart: Week 1 RAG System

**Goal**: Get the RAG API running locally in 5 minutes and understand the full flow.

---

## Prerequisites

- Python 3.11+
- OpenAI API key (free tier or paid account)
- `curl` for testing (or Postman/Insomnia for GUI)

---

## 1. Setup (2 minutes)

### Clone and Install

```bash
# Clone repository
git clone https://github.com/jellydn/ai-architect-4-weeks.git
cd ai-architect-4-weeks

# Create and activate virtual environment
uv venv --python 3.11
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
uv pip install -r requirements.txt
```

### Configure API Key

```bash
# Copy example to .env
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...
nano .env  # or use your editor
```

---

## 2. Start the API Server (30 seconds)

```bash
# From repo root
cd week-1
python main.py

# Output:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete
```

The server is now running on `http://localhost:8000`.

### Optional: View API Documentation

Open your browser to:
```
http://localhost:8000/docs
```

This shows the interactive OpenAPI documentation with "Try it out" buttons.

---

## 3. Quick Test (2 minutes)

### 3.1 Health Check

```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "ok",
  "service": "rag-api"
}
```

### 3.2 Ingest Documents

Ingest the sample document:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["data/sample.txt"]
  }'
```

**Response**:
```json
{
  "status": "success",
  "chunks_created": 5,
  "files_processed": 1
}
```

**What happened**:
1. Loaded `data/sample.txt` (raw text)
2. Split it into overlapping chunks (default: 512 chars, 50 char overlap)
3. Generated embeddings for each chunk (text-embedding-3-small via OpenAI)
4. Stored embeddings in memory for retrieval

**Latency**: 2-5 seconds (depends on file size and API latency)

### 3.3 Query the RAG System

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "top_k": 3
  }'
```

**Response** (example):
```json
{
  "answer": "RAG is retrieval-augmented generation, a technique that combines document retrieval with language model generation. It works by first retrieving relevant documents based on a user query, then using those documents as context for the language model to generate an answer. This approach helps reduce hallucinations and provides up-to-date information without fine-tuning.",
  "sources": ["data/sample.txt"],
  "retrieved_chunks": [
    {
      "chunk_id": "data/sample.txt:0",
      "text": "Retrieval-Augmented Generation (RAG) is...",
      "source": "data/sample.txt",
      "similarity_score": 0.87,
      "rank": 1
    }
  ],
  "latency_ms": 1690.5,
  "model": "gpt-3.5-turbo",
  "tokens_used": {
    "prompt_tokens": 245,
    "completion_tokens": 156
  }
}
```

**What happened**:
1. Embedded your query "What is RAG?" (embedding call)
2. Searched for top-3 most similar chunks using cosine similarity
3. Sent retrieved chunks as context to gpt-3.5-turbo
4. LLM generated answer based on context
5. Returned answer + sources + latency breakdown

**Latency**: 2-5 seconds (retrieval: ~500ms, generation: 2-4s)

---

## 4. Test with Your Own Documents

### Add a Custom Document

```bash
# Create a document
cat > data/myfile.txt << 'EOF'
Machine learning is a subset of artificial intelligence...
Deep learning uses neural networks...
EOF

# Ingest it
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["data/myfile.txt"]
  }'

# Query it
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?"
  }'
```

**Note**: Calling `/ingest` multiple times overwrites the index. To append documents, modify `main.py` to accumulate chunks (Week 2 task).

---

## 5. Understand the Architecture

See [architecture.md](../../docs/architecture.md) for a system diagram.

Quick overview:
```
User Query
    ↓
[Embedding] — Query → OpenAI embedding API
    ↓
[Retrieval] — Cosine Similarity Search → Top-3 chunks
    ↓
[Generation] — LLM + Context → gpt-3.5-turbo
    ↓
Answer + Sources + Latency
```

---

## 6. Explore Trade-Offs

See [trade-offs.md](../../docs/trade-offs.md) for design decisions.

Key questions answered:
- Why RAG over fine-tuning?
- Why text-embedding-3-small (not larger)?
- What are failure modes?
- How do we mitigate hallucinations?

---

## 7. Run Tests

```bash
# From repo root
cd week-1
pytest test_rag.py -v

# Run specific test
pytest test_rag.py::TestIngestion::test_ingest_full_pipeline -v

# Run with coverage
pytest test_rag.py --cov=. -v
```

---

## 8. Check Metrics

After running queries, check latency:

```bash
# From a query response, look at latency_ms
# Example:
# "latency_ms": 1690.5
# = retrieval_ms + generation_ms

# For detailed latency breakdown, check server logs:
# INFO: Retrieved top-3 in 0.45s
# INFO: Generated answer in 1.23s
```

---

## 9. Troubleshooting

### OpenAI API Key Error

```
AuthenticationError: Incorrect API key provided
```

**Fix**: Check `.env` file, ensure `OPENAI_API_KEY=sk-...` is correct and uncommented.

### Rate Limit Error

```
RateLimitError: Rate limit exceeded
```

**Fix**: Free tier has 3 requests/minute. Wait or upgrade to paid account.

### File Not Found

```json
{"detail": "File not found: data/nonexistent.txt"}
```

**Fix**: Check file path. Use absolute or relative path from repo root.

### No Documents Indexed

```json
{"detail": "No documents indexed. Call /ingest first."}
```

**Fix**: Call `/ingest` endpoint before `/query`.

---

## 10. Next Steps

### Understand the Code

- **ingestion.py**: DocumentIngester class (load + chunk)
- **retrieval.py**: RAGRetriever class (embed + search)
- **generation.py**: RAGGenerator class (prompt + LLM)
- **main.py**: FastAPI server (endpoints)
- **test_rag.py**: Unit & integration tests

### Week 1 Checkpoint (Friday)

By end of week, you should be able to:
1. ✅ Run `python main.py` and serve API
2. ✅ Ingest documents and verify chunks
3. ✅ Query and get answers with latency
4. ✅ Explain RAG vs fine-tuning trade-offs
5. ✅ Identify 3 failure modes of RAG

### Week 2 (Advanced)

Next week we'll integrate Weaviate vector database for scalability and add reranking for retrieval quality.

---

## API Reference

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Service health check |
| POST | `/ingest` | Load documents and index |
| POST | `/query` | Retrieve and generate answer |
| GET | `/docs` | Interactive API documentation |

### Request/Response Schemas

See [contracts/schemas.json](contracts/schemas.json) for complete request/response definitions.

**IngestRequest**:
```json
{
  "file_paths": ["data/sample.txt"]
}
```

**QueryRequest**:
```json
{
  "query": "Your question here",
  "top_k": 3
}
```

**QueryResponse**:
```json
{
  "answer": "...",
  "sources": ["data/sample.txt"],
  "retrieved_chunks": [...],
  "latency_ms": 1234.5
}
```

---

## Performance Baseline (Week 1)

Measured on sample document (5KB):

| Operation | Latency | Notes |
|-----------|---------|-------|
| Ingest 5KB | 2.3s | Includes embedding API call |
| Query embedding | 0.45s | OpenAI API latency |
| Retrieval (top-3) | 0.05s | Cosine similarity search |
| LLM generation | 1.2s | gpt-3.5-turbo response |
| **Total E2E** | **1.7s** | Query only (after ingest) |

Cost estimate per 1000 queries: ~$0.02 (embedding) + $0.06 (LLM) = $0.08

---

## Support

- **Documentation**: See `docs/` directory
- **Issues**: Report bugs at [GitHub Issues](https://github.com/jellydn/ai-architect-4-weeks/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/jellydn/ai-architect-4-weeks/discussions)

---

**Happy exploring! 🚀**
