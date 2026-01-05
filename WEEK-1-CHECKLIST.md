# Week 1: Core LLM & RAG Foundations — Daily Checklist

**Goal**: Think and speak fluently about LLM systems + ship a basic RAG.  
**Deliverables**: Running RAG API + Architecture diagram + Trade-off notes  
**Checkpoint**: You can explain why RAG is chosen, not just how.

---

## Day 1 (Monday): Foundation & Environment Setup

**Theme**: Understand the landscape. Set up to build.

### Learning (90 min)
- [ ] Read: [Attention Is All You Need (Executive Summary)](https://paperswithcode.com/paper/attention-is-all-you-need) — Focus on: Why attention matters, context window, multi-head concept
- [ ] Watch: [LLM Tokenization Explained (5 min)](https://www.youtube.com/results?search_query=llm+tokenization+explained) — BPE, token limits, cost implications
- [ ] Read: [RAG vs Fine-tuning Decision Tree](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/) — Skim to answer: When do we choose RAG?

**Target**: Spend 30 min, then move on. Depth comes from building.

### Setup (90 min)
- [ ] Create GitHub repo: `ai-architect-4weeks` (public, MIT license)
- [ ] Initialize Python project with modern tooling stack:
  ```bash
  # Install uv (Astral's fast package manager)
  curl -LsSf https://astral.sh/uv/install.sh | sh
  
  # Create project with Python 3.11+
  uv venv --python 3.11
  source .venv/bin/activate  # or .venv\Scripts\activate on Windows
  
  # Install dependencies
  uv pip install fastapi uvicorn python-dotenv langchain openai numpy pytest pytest-asyncio pydantic ruff
  
  # (Uses pyproject.toml for reproducible builds)
  ```
- [ ] Create `.env` file with OpenAI API key
- [ ] Verify modern tooling works:
  ```bash
  # Type check with ty (Astral's Rust-based type checker)
  uvx ty check
  
  # Lint code with ruff
  ruff check week-1/
  ```
- [ ] Create project structure:
  ```
  ai-architect-4weeks/
  ├── week-1/
  │   ├── main.py
  │   ├── ingestion.py
  │   ├── retrieval.py
  │   ├── generation.py
  │   └── test_rag.py
  ├── docs/
  │   ├── architecture.md
  │   └── trade-offs.md
  ├── pyproject.toml (optional: modern Python packaging)
  ├── requirements.txt (or let uv manage via pyproject.toml)
  └── README.md
  ```
- [ ] Create `docs/trade-offs.md` file (stub for end of week)

### Hands-On Spike (30 min)
- [ ] Write down: "Why RAG, not fine-tuning, for a Q&A system?" (3-4 sentences)
- [ ] Write down: "What are the failure modes of RAG?" (5 examples)

### Success Criteria
- [ ] Environment runs without errors
- [ ] You can articulate RAG vs fine-tuning trade-offs in plain English
- [ ] Repo is set up and can be shared

**Estimated Time**: 3.5 hours  
**Blocker Risk**: API key setup

---

## Day 2 (Tuesday): Document Ingestion & Chunking

**Theme**: Data in, data structured.

### Learning (60 min)
- [ ] Read: [LangChain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/) — Skim 3–4 loaders (PDF, TXT, URL)
- [ ] Read: [Chunking Strategies](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/) — Fixed size, semantic, overlap concepts
- [ ] Question: "If I chunk on sentence boundaries vs fixed size, what breaks?"

**Target**: 30 min reading + 30 min writing test cases.

### Build (120 min)
- [ ] Implement `ingestion.py`:
  ```python
  class DocumentIngester:
      def load_from_file(self, filepath: str) -> List[str]
      def chunk(self, documents: List[str], chunk_size: int = 512, overlap: int = 50) -> List[str]
      def ingest(self, filepath: str) -> List[dict]  # Returns: [{"id": str, "text": str, "source": str}]
  ```
- [ ] Test with sample document (create `sample.txt` in repo)
- [ ] Measure: Time to ingest 10 documents, size of chunks
- [ ] Log: Document ingestion with latency

### Hands-On Deliverable
- [ ] `ingestion.py` complete and tested
- [ ] Sample test: `pytest test_rag.py::test_ingestion`
- [ ] Measurement: "Time to ingest 1MB of text: X ms"

### Success Criteria
- [ ] `ingest()` function returns structured dicts
- [ ] Chunking is configurable (size + overlap)
- [ ] No errors on sample documents
- [ ] README updated with: "How to run: `python -c 'from ingestion import *'`"

**Estimated Time**: 3 hours  
**Blocker Risk**: Document format parsing (use TXT for now, avoid PDF)

---

## Day 3 (Wednesday): Embeddings & Retrieval

**Theme**: Text → Numbers → Search.

### Learning (60 min)
- [ ] Read: [Embeddings Intro](https://platform.openai.com/docs/guides/embeddings) — Dimensionality, similarity metrics, cost
- [ ] Read: [Vector Similarity Search](https://weaviate.io/blog/vector-similarity-search) — Euclidean vs cosine, why cosine for text
- [ ] Question: "What happens if I embed 10k documents? Cost? Time? Quality?"

### Build (120 min)
- [ ] Implement `retrieval.py`:
  ```python
  class RAGRetriever:
      def __init__(self, embedding_model: str = "text-embedding-3-small")
      def embed(self, texts: List[str]) -> List[List[float]]
      def index(self, documents: List[dict]) -> None
      def retrieve(self, query: str, top_k: int = 3) -> List[dict]  # Returns: [{"text": str, "score": float}]
  ```
- [ ] Use in-memory vector store (e.g., list of [text, embedding, metadata])
- [ ] Test: Embed sample documents, retrieve by query
- [ ] Measure: Embedding latency, retrieval latency for top-3

### Hands-On Deliverable
- [ ] `retrieval.py` complete with cosine similarity search
- [ ] Test: `pytest test_rag.py::test_retrieval`
- [ ] Measurement: "Embedding 1000 docs: X seconds, Cost: $Y, Retrieval latency: Z ms"

### Success Criteria
- [ ] `retrieve(query, top_k=3)` returns sorted docs by similarity
- [ ] Embeddings are cached (don't re-embed on each retrieval)
- [ ] Latency is logged and reported
- [ ] No API errors on valid inputs

**Estimated Time**: 3 hours  
**Blocker Risk**: OpenAI quota/rate limits

---

## Day 4 (Thursday): Prompt Template & Full RAG Pipeline

**Theme**: Putting it together. End-to-end working system.

### Learning (45 min)
- [ ] Read: [LangChain Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/) — Variable substitution, format strings
- [ ] Read: [Generation with Context](https://python.langchain.com/docs/use_cases/question_answering/) — How to avoid hallucination with retrieved context
- [ ] Question: "What makes a good RAG prompt?"

### Build (135 min)
- [ ] Implement `generation.py`:
  ```python
  class RAGGenerator:
      def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.7)
      def generate(self, query: str, context: List[str]) -> str
      def rag_answer(self, query: str, retriever: RAGRetriever) -> dict  # Returns: {"answer": str, "sources": List[str], "latency_ms": float}
  ```
- [ ] Implement `main.py` (FastAPI app):
  ```python
  @app.post("/ingest")
  def ingest_documents(file_paths: List[str]) -> dict
  
  @app.post("/query")
  def query_rag(query: str, top_k: int = 3) -> dict  # Returns: {"answer": str, "sources": List[str]}
  ```
- [ ] Test end-to-end: Ingest → Retrieve → Generate
- [ ] Measure: Total latency (ingestion + retrieval + generation)

### Hands-On Deliverable
- [ ] `generation.py` complete
- [ ] `main.py` (FastAPI) running on `http://localhost:8000`
- [ ] Test full pipeline: `pytest test_rag.py::test_rag_end_to_end`
- [ ] Measurement: "Q&A latency: X ms (retrieval: Y ms, generation: Z ms)"

### Success Criteria
- [ ] POST `/query` returns answer + sources
- [ ] All latencies logged
- [ ] No hardcoded prompts (use template)
- [ ] README has: `curl` example for testing

**Estimated Time**: 3 hours  
**Blocker Risk**: LLM API errors, slow generation

---

## Day 5 (Friday): Architecture, Docs & Checkpoint

**Theme**: Make it understandable. Validate Week 1 goal.

### Build Architecture (60 min)
- [ ] Create `docs/architecture.md` with:
  - System diagram (Mermaid flowchart)
  - Data flow: user query → retrieval → generation → response
  - Component description: Ingester, Retriever, Generator
  - Example latency trace
- [ ] Example Mermaid:
  ```
  graph LR
  A[User Query] --> B[Ingestion]
  B --> C[Chunking]
  C --> D[Embedding]
  D --> E[Vector Store]
  A --> F[Retriever]
  E --> F
  F --> G[LLM]
  G --> H[Answer + Sources]
  ```

### Write Trade-Offs (60 min)
- [ ] Complete `docs/trade-offs.md`:
  - **RAG vs Fine-tuning**: When to choose each (decision tree)
  - **Embedding choice**: Why text-embedding-3-small (cost vs quality)
  - **Chunking strategy**: Why this size + overlap
  - **In-memory store vs DB**: Why temporary now, moving to Weaviate next week
  - **Prompt injection risks**: One risk identified, one mitigation

### Test Checkpoint (30 min)
- [ ] Can you explain RAG in 2 minutes (write it down)?
- [ ] Can you justify your embedding choice over GPT-4 embeddings?
- [ ] Can you identify 3 failure modes of your RAG?
- [ ] Can you explain latency breakdown to a PM?

### Polish (30 min)
- [ ] README.md:
  ```markdown
  # AI Architect Week 1: RAG Foundation
  
  ## What This Does
  Q&A system over documents via RAG (no fine-tuning).
  
  ## Setup
  ```bash
  pip install -r requirements.txt
  python main.py
  ```
  
  ## Trade-Offs
  See docs/trade-offs.md
  
  ## Architecture
  See docs/architecture.md
  
  ## Metrics
  - Ingestion latency: X ms
  - Retrieval latency: Y ms
  - Generation latency: Z ms
  - Total: X+Y+Z ms
  ```
- [ ] Commit all work to `week-1` branch
- [ ] Push to GitHub

### Success Criteria
- [ ] Architecture diagram is clear (someone could rebuild from it)
- [ ] Trade-offs explain the "why" (not just the "how")
- [ ] You pass the checkpoint validation (can speak about RAG fluently)
- [ ] Repo is public and portfolio-ready

**Estimated Time**: 3 hours

---

## Week 1 Checkpoint Validation

**Before moving to Week 2, validate you can:**

1. **Explain RAG**: "Why did you choose RAG over fine-tuning for this task?" (1–2 min answer)
2. **Trade-Off Awareness**: "What are 3 ways your Week 1 RAG could fail?" (Name them: retrieval failure, hallucination, latency)
3. **Architecture Fluency**: Point to diagram and trace a query through the system without hesitation
4. **Measurement Rigor**: "What's your retrieval latency and why is it acceptable?" (Have the number ready)

**If you can't answer all 4, spend extra time on whichever is weakest.**

---

## Time Summary

| Day | Activity | Est. Time | Deliverable |
|-----|----------|-----------|-------------|
| Mon | Learning + setup | 3.5h | Environment ready |
| Tue | Ingestion | 3h | `ingestion.py` + test |
| Wed | Retrieval | 3h | `retrieval.py` + metrics |
| Thu | Generation + API | 3h | `main.py` (FastAPI) + test |
| Fri | Docs + checkpoint | 3h | Architecture + trade-offs + README |
| **Total** | | **15.5h** | **Working RAG API** |

**Pace**: ~3 hours/day, 5 days.  
**Flexibility**: If Day 2 takes 4 hours, compress Day 3 learning. Stay on track for Friday checkpoint.

---

## Success Looks Like

By end of Week 1:
- [ ] GitHub repo has 4 Python modules + FastAPI server
- [ ] Documentation explains architecture + trade-offs
- [ ] You can run: `curl -X POST http://localhost:8000/query -d '{"query": "..."}' -H "Content-Type: application/json"`
- [ ] You understand RAG deeply enough to design it differently next week

**Ready to move to Week 2?** Commit `week-1` branch, then create `week-2` branch.
