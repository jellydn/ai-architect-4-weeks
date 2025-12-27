# Week 2: Production-Grade RAG & Retrieval Optimization — Daily Checklist

**Goal**: Move from demo → production thinking.  
**Deliverables**: RAG v2 with vector DB + reranking + caching + latency report  
**Checkpoint**: You can debug "bad answers" by tracing retrieval, not guessing prompts.

---

## Day 1 (Monday): Vector DB Setup & Migration

**Theme**: In-memory store → production database.

### Learning (60 min)
- [ ] Read: [Weaviate Getting Started](https://weaviate.io/developers/weaviate/quickstart) — Architecture, schema, query API
- [ ] Read: [Vector Database Comparison](https://weaviate.io/blog/weaviate-vs-pinecone-vs-milvus) — Why Weaviate (local dev, reranking built-in)
- [ ] Read: [Metadata Filtering](https://weaviate.io/developers/weaviate/search/filtering) — How to filter by source, date, etc.
- [ ] Question: "If I index 100k docs, what breaks? Memory? Query speed?"

### Setup (90 min)
- [ ] Install Weaviate (local Docker):
  ```bash
  docker run -d -p 8080:8080 -p 50051:50051 \
    -e CLIP_INFERENCE_API="http://host.docker.internal:8000" \
    cr.weaviate.io/semitechnologies/weaviate:latest
  ```
- [ ] Install Python client:
  ```bash
  pip install weaviate-client
  ```
- [ ] Create `week-2/vector_db.py`:
  ```python
  class WeaviateStore:
      def __init__(self, url: str = "http://localhost:8080")
      def create_class(self) -> None  # Define schema
      def index_documents(self, documents: List[dict]) -> None
      def search(self, query_vector: List[float], top_k: int) -> List[dict]
      def delete_all(self) -> None  # For testing
  ```
- [ ] Test connection to Weaviate

### Hands-On (60 min)
- [ ] Migrate `ingestion.py` output to Weaviate indexing
- [ ] Test: Index 100 sample documents
- [ ] Measure: Indexing latency, disk usage

### Success Criteria
- [ ] Weaviate running on localhost:8080
- [ ] Documents indexed with metadata (source, chunk_id, text)
- [ ] No errors on indexing 1k documents
- [ ] Disk usage < 500MB for test set

**Estimated Time**: 3 hours  
**Blocker Risk**: Docker issues, Weaviate connectivity

---

## Day 2 (Tuesday): Chunking Strategies & Metadata Filtering

**Theme**: Smarter retrieval = better answers.

### Learning (60 min)
- [ ] Read: [Semantic Chunking](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/semantic_chunker/) — Sentence vs semantic boundaries
- [ ] Read: [Metadata Filtering in RAG](https://weaviate.io/blog/metadata-filtering-rag) — Pre-filter before vector search
- [ ] Experiment: Compare fixed (512 tokens) vs semantic chunking on same doc

### Build (120 min)
- [ ] Implement `chunking_strategies.py`:
  ```python
  class ChunkingStrategy:
      def chunk(self, text: str) -> List[dict]  # Returns: [{"text": str, "chunk_id": str, "metadata": dict}]
  
  class FixedChunker(ChunkingStrategy):
      def __init__(self, size: int = 512, overlap: int = 50)
  
  class SemanticChunker(ChunkingStrategy):
      def __init__(self, model: str = "text-embedding-3-small")
  ```
- [ ] Implement metadata extraction:
  ```python
  def extract_metadata(text: str, source: str) -> dict:
      # Returns: {"source": str, "section": str, "timestamp": str}
  ```
- [ ] Compare chunking strategies on 3 sample docs:
  - Latency to chunk
  - Number of chunks created
  - Average chunk size
  - Retrieval quality (see Day 3)

### Hands-On Deliverable
- [ ] `chunking_strategies.py` with 2+ strategies
- [ ] Comparison metrics: fixed vs semantic chunking
- [ ] Metadata indexed in Weaviate with every chunk

### Success Criteria
- [ ] Weaviate stores chunks with metadata
- [ ] Metadata filtering works: `where_filter={"source": "doc1.txt"}`
- [ ] Chunking comparison documented in `docs/chunking-analysis.md`
- [ ] No retrieval quality loss vs Week 1

**Estimated Time**: 3 hours  
**Blocker Risk**: Semantic chunking may require running a local embedding model (skip if slow)

---

## Day 3 (Wednesday): Reranking & Retrieval Evaluation

**Theme**: Get the right docs in the right order.

### Learning (60 min)
- [ ] Read: [Cross-Encoder Reranking](https://www.sbert.net/examples/applications/cross-encoders/) — Why reranking improves answers
- [ ] Read: [Retrieval Metrics (MRR, NDCG)](https://towardsdatascience.com/evaluating-search-ranking-a-tutorial-on-mrr-and-ndcg-a4e87f7c3d19) — Measure retrieval quality
- [ ] Question: "Does reranking slow down retrieval? By how much?"

### Build (120 min)
- [ ] Implement `reranking.py`:
  ```python
  class Reranker:
      def rerank(self, query: str, candidates: List[dict], top_k: int = 3) -> List[dict]
      # Option 1: Cross-encoder (SBERT)
      # Option 2: LLM-based reranking (Claude)
  ```
- [ ] Comparison pipeline:
  ```python
  def compare_retrieval(query: str, top_k_before: int = 10) -> dict:
      # BM25 (baseline)
      # Vector search (no rerank)
      # Vector + rerank
      # Return: MRR, NDCG, latency for each
  ```
- [ ] Test on 20 hand-crafted queries (create `test-queries.json`)
- [ ] Measure: Reranking latency, quality improvement (MRR before vs after)

### Hands-On Deliverable
- [ ] `reranking.py` with working reranker
- [ ] Evaluation script: `python evaluate_retrieval.py`
- [ ] Results: `docs/retrieval-metrics.json` showing MRR, NDCG, latency

### Success Criteria
- [ ] Reranking improves MRR by at least 10%
- [ ] Reranking latency < 100ms per query
- [ ] Baseline metrics documented (no rerank)
- [ ] Test queries with human-judged relevance ratings created

**Estimated Time**: 3 hours  
**Blocker Risk**: Slow cross-encoder inference; may need to use LLM-based reranking instead

---

## Day 4 (Thursday): Caching & Performance Tuning

**Theme**: Make it fast. Measure everything.

### Learning (45 min)
- [ ] Read: [Query Caching Strategies](https://redis.io/docs/latest/develop/use/patterns/caching/) — When to cache, what to cache
- [ ] Read: [FastAPI Caching](https://fastapi.tiangolo.com/advanced/response-cache/) — In-process vs Redis caching
- [ ] Question: "What's the cache hit rate for real queries?"

### Build (135 min)
- [ ] Implement `caching.py`:
  ```python
  class QueryCache:
      def __init__(self, ttl_seconds: int = 3600, max_size: int = 10000)
      def get(self, query: str) -> Optional[dict]
      def set(self, query: str, result: dict) -> None
      def stats(self) -> dict  # hit_rate, size, memory_usage
  
  class EmbeddingCache:
      def __init__(self)
      def get(self, text: str) -> Optional[List[float]]
      def set(self, text: str, embedding: List[float]) -> None
  ```
- [ ] Integrate into `main.py`:
  ```python
  @app.post("/query")
  def query_rag(query: str, use_cache: bool = True) -> dict
  ```
- [ ] Add metrics endpoint:
  ```python
  @app.get("/metrics")
  def get_metrics() -> dict  # latency_p50, latency_p99, cache_hit_rate, tokens_used, cost
  ```
- [ ] Performance test: Run 100 queries (50% repeated)

### Hands-On Deliverable
- [ ] `caching.py` with query + embedding cache
- [ ] Metrics endpoint returning latency + cost breakdown
- [ ] Performance report: `docs/performance-report.md`

### Success Criteria
- [ ] Cache hit rate > 30% on repeated queries
- [ ] Cached queries < 50ms latency
- [ ] All latencies logged with buckets (P50, P90, P99)
- [ ] Cost per query calculated and reported

**Estimated Time**: 3 hours  
**Blocker Risk**: High variance in generation latency

---

## Day 5 (Friday): Production Readiness & Documentation

**Theme**: Ship it. Document it. Validate Week 2.

### Build Production Checklist (60 min)
- [ ] Add guardrails to `main.py`:
  ```python
  # Input validation
  @app.post("/query")
  def query_rag(query: str):
      if len(query) > 5000:
          raise HTTPException(400, "Query too long")
      if not query.strip():
          raise HTTPException(400, "Query cannot be empty")
  ```
- [ ] Add error handling + logging:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  
  try:
      result = retriever.retrieve(query)
  except Exception as e:
      logger.error(f"Retrieval failed for query: {query}", exc_info=e)
      raise HTTPException(500, "Retrieval failed")
  ```
- [ ] Add `.env` validation (required fields)
- [ ] Create `docker-compose.yml`:
  ```yaml
  version: '3'
  services:
    weaviate:
      image: cr.weaviate.io/semitechnologies/weaviate:latest
      ports: ["8080:8080", "50051:50051"]
    app:
      build: .
      ports: ["8000:8000"]
      depends_on: [weaviate]
  ```

### Write Design Decisions (60 min)
- [ ] Create `docs/design-decisions.md`:
  - **Vector DB Choice**: Why Weaviate (local, reranking, filtering)
  - **Chunking Strategy**: Why semantic (if chose it), trade-offs vs fixed
  - **Reranking**: When it helps, latency cost
  - **Caching**: Hit rates observed, when to disable
  - **Scaling Concerns**: 100k docs → what breaks? (mention Week 3)

### Architecture v2 Diagram (30 min)
- [ ] Update `docs/architecture.md`:
  - Previous: In-memory vector store
  - Now: Weaviate + reranking + caching + metrics
  - Latency trace with new components
  - Cache hit/miss paths

### Checkpoint Validation (30 min)
- [ ] Can you explain retrieval failure (not generation)? (Example: "Wrong docs retrieved because...")
- [ ] Can you trace a bad query result to the retrieval step?
- [ ] Do you know your MRR and NDCG scores?
- [ ] Can you justify: "We chose Weaviate because..."?
- [ ] Can you show: Latency breakdown (retrieval vs rerank vs generation)?

### Final Deliverables (30 min)
- [ ] README updated with Week 2 changes
- [ ] `docs/design-decisions.md` complete
- [ ] `docker-compose.yml` tested
- [ ] Commit to `week-2` branch

### Success Criteria
- [ ] Weaviate indexed with metadata
- [ ] Reranking improves retrieval quality
- [ ] Caching reduces latency for repeated queries
- [ ] All latencies measured and reported
- [ ] You can debug retrieval problems independently
- [ ] Production checklist passed

**Estimated Time**: 3 hours

---

## Week 2 Checkpoint Validation

**Before moving to Week 3, validate you can:**

1. **Retrieval Debugging**: "This answer is wrong. Is it a retrieval failure or generation failure?" (Explain how you'd trace it)
2. **Design Trade-Offs**: "Why Weaviate? What would break if we used Pinecone instead?"
3. **Metrics Literacy**: State your MRR, NDCG, and cache hit rate from your last test run
4. **Production Thinking**: "What's your P99 latency? Is it acceptable for production?"

**If you can't answer all 4, spend extra time on whichever is weakest.**

---

## Time Summary

| Day | Activity | Est. Time | Deliverable |
|-----|----------|-----------|-------------|
| Mon | Vector DB setup | 3h | Weaviate indexing working |
| Tue | Chunking + metadata | 3h | Chunking comparison |
| Wed | Reranking | 3h | MRR/NDCG metrics |
| Thu | Caching + metrics | 3h | Performance report |
| Fri | Production + docs | 3h | Design decisions + checkpoint |
| **Total** | | **15h** | **Production RAG v2** |

---

## Success Looks Like

By end of Week 2:
- [ ] Weaviate indexing 1k+ documents with metadata
- [ ] Reranking improves MRR by 10%+
- [ ] Cache hit rate > 30% on repeated queries
- [ ] Latency breakdown documented (retrieval, rerank, generation, total)
- [ ] You can explain why this design over alternatives
- [ ] Docker Compose spins up entire stack

**Ready for Week 3?** Commit `week-2`, create `week-3` branch, then focus on evaluation.
