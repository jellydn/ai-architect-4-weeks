---
id: task-3
title: 'Week 2 Day 1: Vector DB Setup & Migration (Weaviate)'
status: To Do
assignee: []
created_date: '2026-01-03 08:43'
labels: []
milestone: Week 2
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move from in-memory vector store to production Weaviate database. Set up local Weaviate instance, create schema, and migrate Week 1 ingestion pipeline to store documents in Weaviate with metadata filtering support.

**Learning Goals**:
- Understand Weaviate architecture and HNSW indexing
- Learn metadata filtering patterns
- Compare in-memory vs persistent vector store trade-offs

**Deliverables**:
- Weaviate running on localhost:8080
- week-2/vector_db.py with WeaviateStore class
- Documents indexed with metadata (source, chunk_id, text)
- Integration tests for indexing and retrieval
- Performance baseline (indexing latency, disk usage)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Weaviate running on localhost:8080 with no errors
- [ ] #2 week-2/vector_db.py implements WeaviateStore class with create_class, index_documents, search methods
- [ ] #3 Successfully index 100 sample documents with metadata
- [ ] #4 No errors on indexing 1k documents
- [ ] #5 Disk usage for test set < 500MB
- [ ] #6 Retrieval performance within 2x of Week 1 in-memory baseline
<!-- AC:END -->
