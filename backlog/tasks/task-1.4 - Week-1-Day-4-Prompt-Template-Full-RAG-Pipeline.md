---
id: task-1.4
title: 'Week 1 Day 4: Prompt Template & Full RAG Pipeline'
status: To Do
assignee: []
created_date: '2025-12-28 07:48'
labels:
  - week-1
  - day-4
  - generation
  - api
milestone: Week 1
dependencies: []
parent_task_id: task-1
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement LLM generation and end-to-end FastAPI application.

**Learning (45 min):**
- LangChain prompt templates and variable substitution
- Generation with retrieved context (avoiding hallucination)
- Best practices for RAG prompts

**Build (135 min):**
- Implement RAGGenerator class with:
  - __init__(model='gpt-3.5-turbo', temperature=0.7)
  - generate(query, context) → str
  - rag_answer(query, retriever) → dict with answer, sources, latency_ms
- Implement FastAPI main.py with:
  - POST /ingest (list of file paths)
  - POST /query (query string, top_k parameter)
- Test end-to-end: ingest → retrieve → generate
- Measure total latency breakdown

**Output:**
- generation.py complete
- main.py (FastAPI) running on localhost:8000
- pytest test_rag.py::test_rag_end_to_end passes
- curl example in README
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 RAGGenerator.generate() uses LangChain prompt templates (not hardcoded)
- [ ] #2 RAGGenerator.rag_answer() returns dict with answer, sources, latency_ms
- [ ] #3 FastAPI POST /ingest endpoint accepts List[str] file paths
- [ ] #4 FastAPI POST /query endpoint accepts query string and top_k
- [ ] #5 POST /query returns answer + sources + latency
- [ ] #6 End-to-end test (ingest → retrieve → generate) passes
- [ ] #7 Total latency measured with breakdown: retrieval + generation
- [ ] #8 README includes curl example for testing
<!-- AC:END -->
