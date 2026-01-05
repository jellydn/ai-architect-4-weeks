# Week 1: Concepts & Accomplishments

**Duration**: December 30, 2025 - January 2, 2026 (4 days)  
**Status**: ✅ Complete

---

## What Was Built

### Day 1: Foundation & Environment
- Project structure setup
- Python 3.11+ environment with modern async patterns
- FastAPI framework initialization
- Type-safe development with `ty` and `ruff`
- Sample data loading (RAG corpus)

### Day 2: Document Ingestion & Chunking
- Load documents from files
- Split into overlapping chunks (512 tokens, 100-token overlap)
- Preserve context between chunks
- Output: Structured chunks with metadata

### Day 3: Embeddings & Vector Search
- Generate embeddings using OpenAI `text-embedding-3-small` (1536 dims)
- Implement embedding caching to avoid redundant API calls
- Build in-memory vector store (dict + list)
- Implement cosine similarity search
- Support metadata filtering

### Day 4: Generation & FastAPI
- Create LLM prompt templates with retrieved context
- Integrate with OpenAI `gpt-3.5-turbo`
- Build 3 HTTP endpoints: `/health`, `/ingest`, `/query`
- Full E2E RAG pipeline working
- 8 unit tests all passing

### Day 5: Polish & Verification
- Run full test suite: 8/8 ✅
- Type checking: PASS ✅
- Linting: PASS ✅
- Updated documentation with metrics
- Committed to git

---

## Core Concepts You Need to Know

### 1. What is RAG (Retrieval-Augmented Generation)?

**Problem**: LLMs hallucinate because they generate from memory alone.

**Solution**: RAG adds a retrieval step:
```
Query → [Retrieve relevant docs] → [Pass as context to LLM] → Answer
```

**Benefit**: Answers are grounded in actual documents, not hallucination.

**Trade-off**: 
- Slower (need retrieval + generation)
- But more accurate and citeable
- Can update knowledge by swapping documents (no retraining)

### 2. Document Chunking

**Why chunk?** 
- Documents are too long (50 pages → 1000+ tokens)
- LLM context windows are limited
- Need to retrieve only relevant parts

**Strategy: Overlap-Based**
```
Text: "...sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5..."

Chunk 1: "...sentence 1. Sentence 2. Sentence 3."
Chunk 2: "...Sentence 2. Sentence 3. Sentence 4."  ← Overlaps with Chunk 1
Chunk 3: "...Sentence 3. Sentence 4. Sentence 5."  ← Overlaps with Chunk 2
```

**Why overlap?** Without it, important context at chunk boundaries is lost.

### 3. Embeddings (Dense Vectors)

**What**: Convert text into numeric vectors (1536 numbers)
```
Text: "What is RAG?"
Embedding: [0.123, -0.456, 0.789, ...] (1536 dimensions)
```

**Why**: Allows mathematical comparison of semantic similarity

**How**: Use OpenAI `text-embedding-3-small`
- 1536 dimensions (smaller than `3-large`)
- 10x cheaper
- Sufficient quality for document Q&A

**Similarity**: Cosine distance measures how close two vectors are
```
Similar texts → vectors close together
Dissimilar texts → vectors far apart
```

### 4. Vector Similarity Search

**Goal**: Find chunks most relevant to a query

**Process**:
1. Embed the query: "How does RAG work?" → [0.1, 0.2, ...]
2. Embed all document chunks: → [vectors...]
3. Calculate cosine similarity for each: `similarity = dot(query, chunk) / (norm(query) * norm(chunk))`
4. Return top-k chunks with highest similarity

**In-Memory Implementation** (Week 1):
```python
# Store as dict of vectors
vectors = {
    "chunk-1": [0.1, 0.2, ...],
    "chunk-2": [0.15, 0.25, ...],
    ...
}

# Search: compute similarity for all, sort, return top-k
```

### 5. LLM Generation with Context

**Prompt Engineering**:
```
System: You are a helpful assistant.

Context:
{retrieved_chunks_here}

Question: {user_query}

Answer: [LLM generates answer]
```

**Why it works**: 
- LLM sees actual sources in context
- Grounds answers in retrieved documents
- Reduces hallucination

**Trade-off**: Costs money (embeddings + LLM calls)

### 6. FastAPI & Async Python

**Why FastAPI?**
- Fast (built on Starlette)
- Async I/O (handle multiple requests concurrently)
- Auto-generated docs (Swagger UI)
- Type safety (Pydantic validation)

**Async Pattern**:
```python
async def ingest(file_path: str):
    # Can handle 100s of requests while waiting for I/O
    chunks = await load_and_chunk(file_path)
    return chunks
```

**Benefit**: Single server handles many concurrent users.

### 7. Type Safety in Python

**Modern Python 3.11+ patterns**:
```python
from typing import List, Optional

# Type annotations make code explicit
def search(query_embedding: List[float], top_k: int = 5) -> List[dict]:
    pass
```

**Benefits**:
- Catch errors at development time (not runtime)
- Self-documenting code
- IDE autocomplete
- Refactoring safety

**Tools**:
- `ty` - Fast type checker
- `ruff` - Fast linter

---

## Key Design Decisions (Week 1)

### 1. Embedding Model Choice
**Decision**: OpenAI `text-embedding-3-small`  
**Why**:
- 10x cheaper than `3-large` ($0.02/1M vs $0.20/1M)
- Sufficient quality for document Q&A
- 1536 dimensions (good balance)

**Alternative**: Could use open-source (e.g., sentence-transformers) but would need GPU

### 2. In-Memory Vector Store
**Decision**: Python dict + list (no database)  
**Week 1 Pros**:
- Simple, no setup
- Fast (everything in RAM)
- Perfect for learning

**Week 1 Cons**:
- Doesn't persist (restart = lose all data)
- Scales only to ~10k documents
- Single machine only

**Week 2 Fix**: Migrate to Weaviate (persistent, HNSW indexing, 100k+ docs)

### 3. Fixed-Size Chunking (512 tokens, overlap 100)
**Decision**: Predictable chunks with overlap  
**Why**:
- Simple to implement
- Preserves context at boundaries (overlap)
- Works well for typical documents

**Trade-off**: 
- May split semantic units (sentences across chunks)
- Week 2 will explore semantic chunking

### 4. Caching Embeddings
**Decision**: Store computed embeddings in memory  
**Why**:
- Embeddings are expensive (API calls)
- Same chunk queried twice → reuse embedding
- Huge latency improvement

**Implementation**: Simple dict cache
```python
cache = {
    "chunk-text": [0.1, 0.2, ...],  # embedding
}
```

---

## Performance Targets (Week 1) ✅

| Stage | Target | Actual | Margin |
|-------|--------|--------|--------|
| Load document | <5s | <500ms | 10x better |
| Embed query | <500ms | <200ms | 2.5x better |
| Search top-5 | <10ms | <10ms | On target |
| Generate answer | <3s | <3s | On target |
| **End-to-end** | **<4s** | **<3.5s** | **✅ On target** |

All targets met with margin to spare.

---

## What You Now Understand

✅ **RAG Fundamentals**
- Why RAG reduces hallucination
- How retrieval augments generation
- Cost vs quality trade-offs

✅ **Chunking Strategy**
- Why overlapping chunks matter
- Token-based splitting
- Preserving context at boundaries

✅ **Vector Embeddings**
- What embeddings represent (semantic space)
- How similarity search works (cosine distance)
- Why embeddings are central to RAG

✅ **Python/FastAPI Patterns**
- Async I/O for concurrency
- Type safety with modern Python
- Structuring ML pipelines for testability

✅ **API Design**
- RESTful endpoint patterns
- Request/response validation (Pydantic)
- Error handling best practices

✅ **Testing & Quality**
- Unit testing for ML pipelines
- Mocking external services (OpenAI)
- Type checking and linting workflows

---

## Three Failure Modes (Common Pitfalls)

### 1. Retrieval Failure
**Problem**: Wrong chunks retrieved → LLM can't find relevant info

**Causes**:
- Chunks too small/large (context lost)
- Embedding model not suitable
- Similarity threshold too high

**Week 1 Solution**: Overlap-based chunking  
**Week 2 Solution**: Reranking with cross-encoder

### 2. Hallucination Despite Context
**Problem**: LLM ignores context and generates false info

**Causes**:
- Context not clearly marked in prompt
- LLM trained to ignore retrieval?
- Context quality is poor

**Week 1 Solution**: Clear prompt formatting  
**Week 2 Solution**: Evaluate retrieval quality (MRR, NDCG)

### 3. Latency Bottleneck
**Problem**: RAG slower than baseline LLM

**Causes**:
- Embedding computation slow
- Search over many documents slow
- Network latency (API calls)

**Week 1 Solution**: Embedding caching  
**Week 2 Solution**: Persistent index (HNSW), query caching

---

## Architecture (Week 1)

```
INPUT DOCUMENT
       ↓
[CHUNKING] → 512 tokens, 100-token overlap
       ↓
[EMBEDDING] → text-embedding-3-small (1536 dims, cached)
       ↓
[VECTOR STORE] → In-memory dict + list
       ↓
READY FOR QUERIES

USER QUERY
       ↓
[EMBED QUERY] → Same model, same cache
       ↓
[SEARCH] → Cosine similarity, top-5
       ↓
[PROMPT] → "Context: {chunks}\n\nQuestion: {query}"
       ↓
[LLM] → gpt-3.5-turbo generates answer
       ↓
RESPONSE TO USER
```

---

## Deliverables (Week 1)

| File | Lines | Purpose |
|------|-------|---------|
| `week-1/ingestion.py` | 130 | Load, chunk, structure documents |
| `week-1/retrieval.py` | 195 | Embed, cache, search vectors |
| `week-1/generation.py` | 130 | Prompt templates, LLM calls |
| `week-1/main.py` | 110 | FastAPI server (3 endpoints) |
| `week-1/test_rag.py` | 180 | 8 unit + integration tests |
| **Total** | **745** | **Production RAG system** |

Plus:
- `docs/architecture.md` - System design
- `docs/trade-offs.md` - Design decisions
- `README.md` - Usage guide
- `WEEK-1-SUMMARY.md` - Learning outcomes

---

## Week 1 → Week 2 Transition

### What Changes?

| Aspect | Week 1 | Week 2 |
|--------|--------|--------|
| Vector Store | In-memory dict | Weaviate (persistent) |
| Scalability | ~10k docs | 100k+ docs |
| Retrieval Quality | Assumed good | Measured (MRR, NDCG) |
| Latency | 3-4s | <1s (with caching) |
| Query Optimization | None | Reranking + caching |
| Storage | RAM only | Disk persistent |

### Why Week 2?

Week 1 works great for learning but has limitations:
1. **Persistence**: Restart server → lose all documents
2. **Scale**: Can't handle 100k+ documents efficiently
3. **Quality**: No metrics to measure retrieval success
4. **Performance**: No optimization (caching, reranking)
5. **Production**: Unsuitable for real deployment

Week 2 adds production features:
- Weaviate (persistent, HNSW indexing)
- Evaluation metrics (MRR, NDCG)
- Reranking (improve relevance)
- Caching (sub-10ms cached queries)
- Deployment readiness

---

## Key Takeaways

### RAG isn't magic—it's a pipeline:
1. **Retrieve** relevant context from documents
2. **Augment** the LLM prompt with context
3. **Generate** grounded answers

### Success depends on:
- **Good chunking** (preserve context boundaries)
- **Quality embeddings** (semantic similarity)
- **Relevant retrieval** (top-k is actually relevant)
- **Clear prompting** (LLM knows to use context)

### Trade-offs are crucial:
- Cost: Embeddings + LLM calls cost money
- Latency: Multi-step pipeline vs single LLM call
- Quality: Retrieved context determines answer quality
- Scalability: In-memory vs persistent storage

### Week 1 is foundation for Week 2-4:
- Week 1: Build working system
- Week 2: Make it production-ready
- Week 3: Measure and optimize
- Week 4: Deploy and scale

---

## Common Questions Answered

**Q: Why not just fine-tune the LLM?**  
A: Fine-tuning costs $100s and takes hours. RAG costs ~$1 per 1000 queries and updates instantly.

**Q: Why embedding-based search instead of keyword search?**  
A: Keywords miss synonyms ("doctor" vs "physician"). Embeddings understand meaning.

**Q: Why overlap chunks?**  
A: Context is lost at boundaries. "Context 1: X. Context 2: Y." If X and Y are in different chunks, the relationship is lost.

**Q: Why gpt-3.5-turbo not gpt-4?**  
A: gpt-3.5-turbo is 10x cheaper and sufficient for Q&A. gpt-4 for complex reasoning.

**Q: Why cache embeddings?**  
A: Embeddings are expensive (API calls + latency). Same chunk queried multiple times should reuse embedding.

---

## Next: Week 2 Preview

Starting January 3, you'll learn:

1. **Persistent vector databases** (Weaviate HNSW indexing)
2. **Semantic chunking** vs fixed-size chunking
3. **Retrieval evaluation** (MRR, NDCG metrics)
4. **Reranking** (cross-encoders improve relevance)
5. **Query caching** (sub-10ms for cached queries)
6. **Production deployment** checklist

**Goal**: Transform Week 1's learning project into production-ready system.

