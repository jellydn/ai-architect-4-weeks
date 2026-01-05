# Week 1 Design Trade-Offs

This document captures key architectural decisions, their rationale, and alternatives considered.

## 1. RAG vs Fine-Tuning

**Decision**: Use Retrieval-Augmented Generation (RAG)

| Factor | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Cost** | ~$0.01 per query | $100s-$1000s training |
| **Update speed** | Instant (swap docs) | Hours/days retraining |
| **Transparency** | Can cite sources | Black box |
| **Flexibility** | Works with any docs | Locked to training data |

**Why RAG**: 
- No training infrastructure needed
- Knowledge updates are instant
- Can see which documents informed the answer
- 100x cheaper for our Q&A use case

**When Fine-Tuning is Better**:
- Teaching specific writing style
- Mathematical reasoning
- Code generation with custom patterns

---

## 2. Embedding Model Choice

**Decision**: `text-embedding-3-small` (1536 dimensions)

| Model | Dimensions | Cost/1M tokens | Quality |
|-------|------------|----------------|---------|
| text-embedding-3-small | 1536 | $0.02 | Good |
| text-embedding-3-large | 3072 | $0.13 | Better |
| text-embedding-ada-002 | 1536 | $0.10 | Legacy |

**Why 3-small**:
- 10x cheaper than 3-large
- Sufficient quality for document Q&A
- Faster inference
- Acceptable accuracy trade-off for learning project

**Upgrade Path**: Switch to 3-large if retrieval quality issues emerge.

---

## 3. Chunking Strategy

**Decision**: Fixed-size chunks (512 chars, 50 overlap)

| Strategy | Pros | Cons |
|----------|------|------|
| **Fixed-size** | Simple, deterministic | May split mid-sentence |
| Semantic | Respects boundaries | Complex, slower |
| Recursive | Balanced | More implementation |

**Why Fixed-Size**:
- Predictable chunk count
- Easy to debug
- Fast processing
- Good enough for Week 1 MVP

**Future Improvement** (Week 2+):
- Add sentence-aware chunking
- Consider `langchain.text_splitter.RecursiveCharacterTextSplitter`

---

## 4. Vector Store

**Decision**: In-memory dict/list (Week 1)

| Store | Scalability | Persistence | Complexity |
|-------|-------------|-------------|------------|
| **In-memory** | ~10K docs | None | Zero |
| SQLite + vectors | ~100K docs | File | Low |
| Weaviate | Millions | Full | Medium |
| Pinecone | Billions | Managed | Low (SaaS) |

**Why In-Memory**:
- Zero setup
- Fast iteration during development
- Sufficient for learning with small datasets
- Focus on RAG concepts, not infrastructure

**Week 2 Migration**: Move to Weaviate for persistence and scale.

---

## 5. LLM Choice

**Decision**: `gpt-3.5-turbo`

| Model | Cost/1K tokens | Latency | Quality |
|-------|----------------|---------|---------|
| **gpt-3.5-turbo** | $0.002 | ~500ms | Good |
| gpt-4o | $0.005 | ~800ms | Better |
| gpt-4 | $0.06 | ~2000ms | Best |

**Why 3.5-turbo**:
- 30x cheaper than GPT-4
- 4x faster response
- Sufficient for context-grounded Q&A
- Easy upgrade path if needed

---

## 6. Failure Modes & Mitigation

### Retrieval Failure
**Problem**: Poor chunking loses important context across boundaries.

**Mitigation**:
- Overlap chunks (50 chars)
- Increase top_k for more coverage
- Future: semantic chunking

### Hallucination
**Problem**: LLM may invent details not in context.

**Mitigation**:
- Prompt explicitly says "answer based on context only"
- Return sources for verification
- Future: add confidence scoring

### Latency
**Problem**: Multi-step pipeline (embed → search → generate) adds latency.

**Mitigation**:
- Cache embeddings
- Use fast embedding model
- In-memory search (sub-10ms)
- Future: streaming responses

---

## 7. Cost Analysis

**Estimated cost per 1,000 queries**:

| Component | Calculation | Cost |
|-----------|-------------|------|
| Query embedding | 1K queries × 50 tokens × $0.02/1M | $0.001 |
| Retrieval | In-memory | $0.00 |
| Generation | 1K queries × 500 tokens × $0.002/1K | $1.00 |
| **Total** | | **~$1.00** |

With GPT-4 instead: ~$30/1K queries (30x more)

---

## 8. Future Work (Week 2+)

| Week | Enhancement |
|------|-------------|
| 2 | Weaviate vector store for persistence |
| 2 | Multi-document ingestion |
| 3 | Evaluation framework (retrieval metrics) |
| 3 | Reranking with cross-encoder |
| 4 | Production deployment (Docker, monitoring) |
| 4 | Streaming responses |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Week 1 | RAG over fine-tuning | Cost, flexibility, transparency |
| Week 1 | text-embedding-3-small | 10x cheaper, sufficient quality |
| Week 1 | Fixed-size chunking | Simple MVP, easy to debug |
| Week 1 | In-memory vector store | Zero setup, focus on concepts |
| Week 1 | gpt-3.5-turbo | Cost-effective, fast |
