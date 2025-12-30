# AI Architect: 4-Week Sprint

**Goal**: Build production-ready RAG systems from scratch.

## What is RAG?

**Retrieval-Augmented Generation (RAG)** combines document retrieval with LLM generation. Instead of relying solely on the model's training data, RAG first retrieves relevant documents from your knowledge base, then uses them as context for the LLM to generate grounded answers.

**Why RAG over fine-tuning?**
- **Instant updates**: Swap documents without retraining
- **Transparency**: See which sources informed the answer
- **Cost-effective**: ~$1 per 1000 queries vs $100s for fine-tuning
- **No hallucination**: Answers grounded in actual documents

---

## Quick Start

```bash
# Setup
make setup

# Configure your OpenAI API key
echo "OPENAI_API_KEY=sk-your-key" > .env

# Run tests
make check

# Start server
make server

# Demo (in another terminal)
make demo
```

---

## Project Structure

```
week-1/
├── main.py           # FastAPI server (/health, /ingest, /query)
├── ingestion.py      # Document loading & chunking
├── retrieval.py      # Embedding & vector search
├── generation.py     # LLM answer generation
└── test_rag.py       # Unit & integration tests

docs/
├── architecture.md   # System design & diagrams
└── trade-offs.md     # Design decisions & rationale

data/
└── sample.txt        # Sample RAG document
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ingest` | POST | Ingest documents |
| `/query` | POST | Query the RAG system |

### Examples

```bash
# Health check
curl http://localhost:8000/health

# Ingest documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_paths": ["../data/sample.txt"]}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "top_k": 3}'
```

---

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make setup` | Create venv and install dependencies |
| `make test` | Run all tests |
| `make lint` | Run ruff linter |
| `make typecheck` | Run ty type checker |
| `make check` | Run lint + tests |
| `make server` | Start FastAPI server |
| `make server-dev` | Start with hot reload |
| `make demo` | Ingest sample + query |
| `make help` | Show all commands |

---

## Metrics

| Stage | Latency | Notes |
|-------|---------|-------|
| Ingestion | <100ms/doc | File I/O + chunking |
| Embedding | 50-200ms | OpenAI API (cached) |
| Retrieval | <10ms | In-memory cosine similarity |
| Generation | 500-2000ms | GPT-3.5-turbo |
| **Total** | **600-2500ms** | End-to-end |

**Cost**: ~$1 per 1000 queries (embedding + generation)

---

## Documentation

- **[Architecture](docs/architecture.md)**: System diagram, components, data flow
- **[Trade-offs](docs/trade-offs.md)**: Design decisions and rationale
- **[Local LLM Guide](docs/local-llm.md)**: Run with Ollama, LM Studio, Groq (free)

---

## Week 1 Checklist

- [x] Project structure initialized
- [x] Ingestion module (load, chunk, structure)
- [x] Retrieval module (embed, cache, search)
- [x] Generation module (prompt, LLM, answer)
- [x] FastAPI endpoints (/health, /ingest, /query)
- [x] Unit tests (8 passing)
- [x] Type checking (ty)
- [x] Linting (ruff)
- [x] Architecture documentation
- [x] Trade-offs documentation
- [ ] Live API testing with real documents

---

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.11+ | Modern async, typing |
| Framework | FastAPI | Fast, async, auto-docs |
| Embeddings | text-embedding-3-small | 10x cheaper than large |
| LLM | gpt-3.5-turbo | Fast, cost-effective |
| Vector Store | In-memory (Week 1) | Simple, no setup |
| Type Checker | ty | Fast Rust-based checker |
| Linter | ruff | Fast Rust-based linter |
| Package Manager | uv | 10-100x faster than pip |

---

## Roadmap

| Week | Focus |
|------|-------|
| **1** | RAG Foundation (current) |
| 2 | Weaviate vector store, evaluation |
| 3 | Production hardening, monitoring |
| 4 | Deployment, scaling |

---

## License

MIT
