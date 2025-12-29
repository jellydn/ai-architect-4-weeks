# AI Architect: 4-Week Sprint - Week 1 Progress

**Goal**: Build a working RAG system + architecture documentation.

## Status

**Branch**: `week-1`  
**Progress**: Day 1 (Setup) - Complete  
**Target**: Friday checkpoint with running API

---

## What's Built

### Project Structure
```
week-1/
├── __init__.py
├── main.py           # FastAPI server
├── ingestion.py      # Document loading & chunking
├── retrieval.py      # Embedding & vector search
├── generation.py     # LLM-based answer generation
└── test_rag.py       # Unit & integration tests

docs/
├── architecture.md   # System design & diagrams
└── trade-offs.md     # Design decisions & rationale

data/
└── sample.txt        # Sample document for testing
```

### Components

1. **DocumentIngester**: Load files, chunk with overlap, produce structured documents
2. **RAGRetriever**: Generate embeddings, store in memory, retrieve by cosine similarity
3. **RAGGenerator**: Prompt templating, LLM calls, answer generation
4. **FastAPI Server**: HTTP endpoints for ingest and query

---

## Setup (Day 1)

### 1. Environment (Modern Python with `uv`)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with Python 3.11+
uv venv --python 3.11
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
uv pip install -r requirements.txt
```

**Why `uv`**: 10-100x faster than pip, built-in Python version management, better lockfile support, modern Python tooling standard.

### 2. Configuration

```bash
# Copy example to .env and fill in your OpenAI API key
cp .env.example .env

# Edit .env with your keys
# OPENAI_API_KEY=sk-...
```

### 3. Type Check & Lint

```bash
# Type check with ty (Astral's Rust-based type checker)
uvx ty check

# Lint with ruff
ruff check week-1/
```

### 4. Run Tests

```bash
# Run unit tests (ingestion, retrieval)
pytest week-1/test_rag.py -v

# Individual test
pytest week-1/test_rag.py::TestIngestion::test_ingest_full_pipeline -v
```

### 5. Test Modules Independently

```bash
# Test ingestion
cd week-1
python ingestion.py

# Test retrieval (requires OpenAI key)
python retrieval.py

# Test generation (requires OpenAI key)
python generation.py
```

---

## API Usage (When Ready)

### Start Server

```bash
cd week-1
python main.py

# Server runs on http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Ingest Documents

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_paths": ["data/sample.txt"]}'

# Response:
# {
#   "status": "success",
#   "chunks_created": 12,
#   "files_processed": 1
# }
```

### Query RAG System

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG and why is it useful?", "top_k": 3}'

# Response:
# {
#   "answer": "RAG is retrieval-augmented generation...",
#   "sources": ["data/sample.txt"],
#   "latency_ms": 2145.3
# }
```

### Health Check

```bash
curl http://localhost:8000/health
```

---

## Documentation

### Architecture
See [docs/architecture.md](docs/architecture.md) for system diagram, component descriptions, and data flow.

### Trade-Offs
See [docs/trade-offs.md](docs/trade-offs.md) for design decisions:
- Why RAG over fine-tuning
- Embedding model choice
- Chunking strategy
- Vector store approach
- Prompt injection mitigation

---

## Metrics (To Be Measured)

- **Ingestion Latency**: Time to load + chunk + embed
- **Retrieval Latency**: Time to embed query + similarity search
- **Generation Latency**: Time for LLM to respond
- **Total E2E Latency**: Full pipeline (ingest → retrieve → generate)
- **Cost per Query**: Token usage × model pricing

---

## Week 1 Deliverables (by Friday)

- [x] Project structure initialized
- [x] Python modules for ingestion, retrieval, generation
- [x] FastAPI endpoints defined
- [x] Tests written (unit + integration)
- [ ] API running and tested
- [ ] Architecture diagram finalized
- [ ] Trade-offs documentation complete
- [ ] Latency metrics measured
- [ ] README with curl examples

---

## Next Steps

**Tuesday (Day 2)**: Ingestion module - test with real documents  
**Wednesday (Day 3)**: Retrieval module - embedding & vector search  
**Thursday (Day 4)**: Integration - full API endpoint testing  
**Friday (Day 5)**: Documentation & checkpoint validation

---

## Resources

**Documentation**:
- [OpenAI Embedding Docs](https://platform.openai.com/docs/guides/embeddings)
- [LangChain Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

**Concepts**:
- [Vector Similarity Search](https://www.pinecone.io/learn/vector-database/)
- [RAG Concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts/)

---

## License

MIT

---

**Week 1 Status**: In Progress 🚀  
**Last Updated**: 2025-12-29
