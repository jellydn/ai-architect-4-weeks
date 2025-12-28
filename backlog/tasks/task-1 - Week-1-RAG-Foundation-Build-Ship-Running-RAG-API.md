---
id: task-1
title: 'Week 1: RAG Foundation - Build & Ship Running RAG API'
status: To Do
assignee: []
created_date: '2025-12-28 07:48'
labels:
  - rag
  - week-1
  - foundation
  - api
  - core
milestone: Week 1
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Foundation phase of the 4-week AI Architect sprint. Build a working RAG (Retrieval-Augmented Generation) system from scratch.

**Objectives:**
- Understand and explain RAG architecture (why RAG vs fine-tuning)
- Implement document ingestion pipeline (chunking, embedding)
- Build vector database integration (Weaviate)
- Create FastAPI endpoint for RAG queries
- Document system architecture and decisions

**Output Deliverable:**
- Running RAG API (FastAPI on localhost:8000)
- Architecture diagram explaining RAG flow
- Documentation answering: "Why RAG, not fine-tuning?"
- Code organized and tested (pytest)
- Docker setup ready for next phases

**Key Learning Focus:**
- Tokens and context windows
- Embeddings (semantic search)
- Vector databases
- RAG pipeline flow
- Prompt engineering basics

**Time Budget:** 15.5 hours (5 days, 3–4 hours/day)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 FastAPI endpoint accepts POST /query with JSON body and returns structured response
- [ ] #2 Document ingestion pipeline chunks and embeds documents using OpenAI embeddings
- [ ] #3 Weaviate vector DB stores and retrieves documents with reranking
- [ ] #4 System accurately explains why RAG was chosen over fine-tuning (3-sentence writeup)
- [ ] #5 Architecture diagram covers ingestion → embedding → retrieval → generation flow
- [ ] #6 All code tested with pytest (coverage >70%)
- [ ] #7 Docker setup ready (Dockerfile + docker-compose.yml)
- [ ] #8 README includes running instructions and API examples
<!-- AC:END -->
