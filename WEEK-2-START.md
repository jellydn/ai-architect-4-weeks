# Week 2: Production-Grade RAG & Retrieval Optimization

**Status**: Starting January 3, 2026  
**Goal**: In-memory → persistent vector DB, measure retrieval quality, optimize latency

## Timeline

| Day | Focus | Branch |
|-----|-------|--------|
| 1 | Weaviate setup & migration | `002-week2-vector-db` |
| 2 | Chunking strategies & metadata | `002-week2-vector-db` |
| 3 | Reranking & evaluation | `002-week2-vector-db` |
| 4 | Caching & performance | `002-week2-vector-db` |
| 5 | Checkpoint & latency report | `002-week2-vector-db` |

## Key Learning Outcomes

**By end of Week 2, you will understand:**
1. How to migrate from in-memory to persistent vector databases
2. How to evaluate retrieval quality quantitatively (MRR, NDCG)
3. Why reranking improves answers and its latency trade-off
4. How to cache queries and measure cache effectiveness
5. How to profile and optimize ML pipelines end-to-end

## Why This Matters

**Week 1** built a working RAG system. **Week 2** makes it production-ready:
- **Persistence**: Documents survive restarts
- **Scale**: Handle 100k+ documents
- **Quality**: Measure and improve retrieval
- **Speed**: Cache common queries, rerank efficiently
- **Cost**: Optimize embedding and LLM calls

## Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Retrieval MRR | > 0.7 | Reranking improves baseline by 10%+ |
| Reranking latency | < 100ms | Per query |
| Cache hit rate | > 30% | On typical query distribution |
| Cached query latency | < 10ms | Sub-10ms for cached hits |
| E2E latency (w/ cache) | < 1s | 50% improvement from Week 1 |
| Weaviate disk usage | < 500MB | For 1k document test set |

## Getting Started

**Day 1 Quick Start:**
```bash
# Create new branch
git checkout -b 002-week2-vector-db

# Install Weaviate (Docker required)
docker run -d -p 8080:8080 -p 50051:50051 \
  cr.weaviate.io/semitechnologies/weaviate:latest

# Install Python client
pip install weaviate-client

# Start Day 1 implementation
python week-2/vector_db.py --test
```

## Dependencies

- **Docker**: For Weaviate container
- **weaviate-client**: Python client library
- **sentence-transformers**: For reranking (Day 3)
- **numpy, scipy**: For metrics calculation

All additions go in `pyproject.toml` and `requirements-week2.txt`

## Architecture Changes

**Week 1 (In-Memory)**:
```
Document → Chunk → Embed → Dict[vector] → Search
```

**Week 2 (Persistent)**:
```
Document → Chunk → Embed → Weaviate → HNSW Index → Search → Rerank → Answer
```

## Next Steps

1. Start with **Task 3** (Vector DB setup)
2. Run through **Tasks 4-7** in sequence
3. Complete verification on **Task 7**
4. Transition to **Week 3** (evaluation & monitoring)

