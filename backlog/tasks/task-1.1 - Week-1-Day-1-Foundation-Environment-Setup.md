---
id: task-1.1
title: 'Week 1 Day 1: Foundation & Environment Setup'
status: To Do
assignee: []
created_date: '2025-12-28 07:48'
labels:
  - week-1
  - day-1
  - setup
  - learning
milestone: Week 1
dependencies: []
parent_task_id: task-1
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Understand LLM landscape and set up Python/FastAPI environment.

**Learning (90 min):**
- Attention & context windows
- Tokenization and cost implications
- RAG vs fine-tuning decision framework

**Setup (90 min):**
- Create GitHub repo (public, MIT license)
- Initialize Python venv with FastAPI, LangChain, OpenAI
- Create project structure with modules (ingestion, retrieval, generation)
- Set up .env with API keys
- Create docs stubs (architecture.md, trade-offs.md)

**Spike (30 min):**
- Write down: "Why RAG, not fine-tuning?"
- Write down: "5 RAG failure modes"

**Success:** Environment ready, can articulate RAG trade-offs
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Python venv created and activated with all dependencies installed
- [ ] #2 Project structure exists: week-1/(main.py, ingestion.py, retrieval.py, generation.py, test_rag.py)
- [ ] #3 docs/(architecture.md, trade-offs.md) created as stubs
- [ ] #4 .env file configured with OpenAI API key
- [ ] #5 GitHub repo public with MIT license
- [ ] #6 README.md exists with basic description
- [ ] #7 Can explain RAG vs fine-tuning in 3-4 sentences
- [ ] #8 Identified 5 failure modes of RAG system
<!-- AC:END -->
