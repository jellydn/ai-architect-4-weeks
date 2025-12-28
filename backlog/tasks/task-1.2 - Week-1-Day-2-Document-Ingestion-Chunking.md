---
id: task-1.2
title: 'Week 1 Day 2: Document Ingestion & Chunking'
status: To Do
assignee: []
created_date: '2025-12-28 07:48'
labels:
  - week-1
  - day-2
  - ingestion
milestone: Week 1
dependencies: []
parent_task_id: task-1
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build document loading and chunking pipeline.

**Learning (60 min):**
- LangChain document loaders (PDF, TXT, URL)
- Chunking strategies: fixed-size, semantic, overlap
- Trade-offs: sentence vs fixed-size chunking

**Build (120 min):**
- Implement DocumentIngester class with:
  - load_from_file(filepath) → List[str]
  - chunk(documents, chunk_size=512, overlap=50) → List[str]
  - ingest(filepath) → List[dict]
- Test with sample document
- Measure ingestion latency and chunk sizes

**Output:**
- ingestion.py complete and tested
- pytest test_rag.py::test_ingestion passes
- Measurement: "Time to ingest 1MB of text: X ms"
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 DocumentIngester.load_from_file() loads text and returns List[str]
- [ ] #2 DocumentIngester.chunk() supports configurable size and overlap parameters
- [ ] #3 DocumentIngester.ingest() returns List[dict] with id, text, source fields
- [ ] #4 Sample document (sample.txt) created and tested
- [ ] #5 pytest test passes for ingestion module
- [ ] #6 Ingestion latency measured and logged
- [ ] #7 README updated with: 'How to run: python -c from ingestion import *'
<!-- AC:END -->
