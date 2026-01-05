---
id: task-4
title: 'Week 2 Day 2: Chunking Strategies & Metadata Filtering'
status: To Do
assignee: []
created_date: '2026-01-03 08:44'
labels: []
milestone: Week 2
dependencies:
  - task-3
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement multiple chunking strategies and metadata extraction. Build semantic chunking on top of Week 1's fixed-size chunking. Evaluate which strategy produces better retrieval results.

**Learning Goals**:
- Compare fixed vs semantic chunking effectiveness
- Understand metadata extraction patterns
- Learn how to filter before vector search

**Deliverables**:
- week-2/chunking_strategies.py with FixedChunker and SemanticChunker
- Metadata extraction function
- Comparison metrics on 3 sample documents
- Analysis document: docs/chunking-analysis.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 week-2/chunking_strategies.py has ChunkingStrategy base class with FixedChunker and SemanticChunker implementations
- [ ] #2 Metadata filtering works in Weaviate: where_filter={"source": "doc.txt"}
- [ ] #3 Chunking comparison metrics documented (latency, chunk count, avg size)
- [ ] #4 No retrieval quality loss vs Week 1
- [ ] #5 docs/chunking-analysis.md with comparison results
<!-- AC:END -->
