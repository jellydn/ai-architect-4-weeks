---
id: task-5
title: 'Week 2 Day 3: Reranking & Retrieval Evaluation'
status: To Do
assignee: []
created_date: '2026-01-03 08:44'
labels: []
milestone: Week 2
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement reranking to improve retrieval quality. Build evaluation metrics (MRR, NDCG) and measure impact of reranking on answer quality.

**Learning Goals**:
- Understand cross-encoder vs dense vector trade-offs
- Learn retrieval evaluation metrics
- Measure quality improvements quantitatively

**Deliverables**:
- week-2/reranking.py with Reranker class
- 20 hand-crafted test queries with human relevance judgments
- Evaluation script: evaluate_retrieval.py
- Metrics report: docs/retrieval-metrics.json
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 week-2/reranking.py implements Reranker class
- [ ] #2 Reranking improves MRR by at least 10%
- [ ] #3 Reranking latency < 100ms per query
- [ ] #4 evaluate_retrieval.py compares 3 approaches: baseline, vector search, vector+rerank
- [ ] #5 docs/retrieval-metrics.json has MRR, NDCG, latency for all approaches
<!-- AC:END -->
