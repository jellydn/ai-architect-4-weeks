---
id: task-1.5
title: 'Week 1 Day 5: Architecture, Docs & Checkpoint'
status: To Do
assignee: []
created_date: '2025-12-28 07:48'
labels:
  - week-1
  - day-5
  - documentation
  - checkpoint
milestone: Week 1
dependencies: []
parent_task_id: task-1
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Document the system, validate understanding, and prepare for Week 2.

**Build Architecture (60 min):**
- Create docs/architecture.md with:
  - System diagram (Mermaid flowchart)
  - Data flow: query → retrieval → generation → response
  - Component descriptions
  - Example latency trace

**Write Trade-Offs (60 min):**
- Complete docs/trade-offs.md with:
  - RAG vs fine-tuning decision tree
  - Embedding choice (text-embedding-3-small vs others)
  - Chunking strategy rationale
  - In-memory store vs production DB (Weaviate next week)
  - Prompt injection risks + mitigation

**Checkpoint Validation (30 min):**
- Explain RAG in 2 minutes
- Justify embedding choice
- Identify 3 failure modes
- Explain latency breakdown to a PM

**Output:**
- docs/architecture.md with diagram
- docs/trade-offs.md complete
- Updated README with metrics
- Commit to week-1 branch, push to GitHub
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 docs/architecture.md includes Mermaid flowchart of entire system
- [ ] #2 Data flow documented: user query → retrieval → generation → response
- [ ] #3 Example latency trace provided (timing for each stage)
- [ ] #4 docs/trade-offs.md explains RAG vs fine-tuning with decision tree
- [ ] #5 docs/trade-offs.md justifies: embedding choice, chunking strategy, in-memory store
- [ ] #6 One prompt injection risk identified + mitigation documented
- [ ] #7 README updated with: Setup, Trade-Offs link, Architecture link, Metrics section
- [ ] #8 Can explain RAG in 2 minutes (coherent, fluent explanation)
- [ ] #9 Can identify 3+ RAG failure modes
- [ ] #10 Repo committed to week-1 branch and pushed to GitHub
<!-- AC:END -->
