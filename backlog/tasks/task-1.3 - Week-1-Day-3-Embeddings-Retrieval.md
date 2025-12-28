---
id: task-1.3
title: 'Week 1 Day 3: Embeddings & Retrieval'
status: To Do
assignee: []
created_date: '2025-12-28 07:48'
labels:
  - week-1
  - day-3
  - embeddings
  - retrieval
milestone: Week 1
dependencies: []
parent_task_id: task-1
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build embedding and vector similarity search.

**Learning (60 min):**
- OpenAI embeddings API (dimensionality, cost)
- Vector similarity: Euclidean vs cosine distance
- Why cosine similarity for text embeddings

**Build (120 min):**
- Implement RAGRetriever class with:
  - __init__(embedding_model='text-embedding-3-small')
  - embed(texts: List[str]) → List[List[float]]
  - index(documents: List[dict]) → None
  - retrieve(query: str, top_k=3) → List[dict]
- Use in-memory vector store (list of [text, embedding, metadata])
- Test: embed sample documents, retrieve by query
- Measure: embedding latency, retrieval latency

**Output:**
- retrieval.py complete with cosine similarity
- pytest test_rag.py::test_retrieval passes
- Metrics: Embedding 1000 docs latency, cost, retrieval P99
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 RAGRetriever.embed() calls OpenAI API and caches embeddings
- [ ] #2 RAGRetriever.retrieve() returns sorted List[dict] by cosine similarity
- [ ] #3 retrieve(query, top_k=3) returns exactly top_k results
- [ ] #4 Embeddings are cached (no re-embedding on retrieval)
- [ ] #5 Retrieval latency logged and reported
- [ ] #6 pytest test passes for retrieval module
- [ ] #7 Measurement: Embedding latency + cost + retrieval P99 latency documented
<!-- AC:END -->
