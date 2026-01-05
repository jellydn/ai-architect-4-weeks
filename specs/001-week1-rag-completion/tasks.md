# Phase 2: Task Breakdown — Complete Week 1 RAG Foundation (Days 2-5)

**Status**: Ready for Implementation  
**Feature**: Complete Week 1 RAG Foundation (Days 2-5)  
**Branch**: `001-week1-rag-completion`  
**Total Tasks**: 47 (organized by phase and user story)

---

## Overview

This document breaks down the implementation plan into 47 granular, actionable tasks organized by phase. Each task is independently testable and includes a specific file path. Tasks are sequenced to enable parallel execution while respecting dependencies.

**Task Format**: `- [ ] [ID] [P?] [Story?] Description with file path`

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Initialize environment and verify dependencies  
**Duration**: ~30 minutes  
**Blocker**: NONE — Can begin immediately (most completed from Day 1 setup)

- [ ] T001 Verify Python 3.11+ installed: `python --version` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T002 Verify uv package manager installed: `which uv` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T003 [P] Activate virtual environment: `source .venv/bin/activate` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T004 [P] Verify dependencies installed: `pip list | grep -E "fastapi|openai|langchain"` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T005 [P] Create `.env` from `.env.example` and configure OPENAI_API_KEY in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T006 [P] Verify `.env` is readable: `cat .env | grep OPENAI_API_KEY` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`

**Checkpoint**: Environment ready, dependencies installed, API key configured

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST complete before user story work  
**Duration**: ~45 minutes  
**Blocker**: CRITICAL — No user story work can proceed without this phase complete

- [ ] T007 Verify project structure matches plan.md in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/`
- [ ] T008 [P] Create `__init__.py` in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/__init__.py`
- [ ] T009 [P] Verify test infrastructure: `pytest --version` and `python -m pytest week-1/test_rag.py --collect-only` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T010 [P] Run type checker: `uvx ty check week-1/` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T011 [P] Run linter: `ruff check week-1/` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T012 [P] Create sample test document: `data/sample.txt` in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/data/`
- [ ] T013 [P] Verify logging configured in all modules: Check imports in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/ingestion.py`, `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/retrieval.py`, `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/generation.py`, `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/main.py`

**Checkpoint**: Foundation ready—all user story work can begin

---

## Phase 3: User Story 1 — Test Document Ingestion at Scale (Priority: P1) 🎯 MVP

**Goal**: Verify documents can be loaded, chunked, and prepared for embedding  
**Independent Test**: `pytest week-1/test_rag.py::TestIngestion -v`  
**Success Criteria**: SC-001 (ingestion latency <5s), SC-005 (all tests pass)

- [ ] T014 [P] [US1] Review existing `ingestion.py` in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/ingestion.py` (should be complete from Day 1)
- [ ] T015 [US1] Test ingestion with sample: `python week-1/ingestion.py` and verify chunks created in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/`
- [ ] T016 [US1] Run ingestion unit test: `pytest week-1/test_rag.py::TestIngestion::test_load_from_file -v` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T017 [US1] Run chunking test: `pytest week-1/test_rag.py::TestIngestion::test_chunking -v` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T018 [US1] Run full ingestion pipeline test: `pytest week-1/test_rag.py::TestIngestion::test_ingest_full_pipeline -v` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T019 [P] [US1] Measure ingestion latency: Log timestamp from test output and record in performance notes in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/`
- [ ] T020 [US1] Verify chunking validation: Test with large document (if available) in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/data/`
- [ ] T021 [P] [US1] Add docstring examples to DocumentIngester class in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/ingestion.py`

**Checkpoint**: User Story 1 complete — ingestion module fully tested and documented

---

## Phase 4: User Story 2 — Embed and Retrieve Relevant Documents (Priority: P1) 🎯 MVP

**Goal**: Verify embeddings are generated, cached, and retrieval returns ranked results  
**Independent Test**: `pytest week-1/test_rag.py::TestRetrieval -v`  
**Success Criteria**: SC-002 (retrieval latency <500ms), SC-005 (tests pass), SC-007 (caching works)

- [ ] T022 [P] [US2] Review existing `retrieval.py` in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/retrieval.py` (should be complete from Day 1)
- [ ] T023 [US2] Verify OpenAI API key accessible: Test client init in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/retrieval.py`
- [ ] T024 [US2] Run embedding caching test: `pytest week-1/test_rag.py::TestRetrieval::test_embedding_caching -v` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T025 [US2] Run retrieval ordering test: `pytest week-1/test_rag.py::TestRetrieval::test_retrieval_ordering -v` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T026 [US2] Run empty index test: `pytest week-1/test_rag.py::TestRetrieval::test_retrieval_empty -v` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T027 [P] [US2] Measure embedding latency: Ingest sample docs and record embedding time in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/`
- [ ] T028 [P] [US2] Measure retrieval latency: Query with docs indexed, record similarity search time in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/`
- [ ] T029 [US2] Test API rate limit handling: Document expected retry behavior in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/retrieval.py`
- [ ] T030 [P] [US2] Add docstring examples to RAGRetriever class in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/retrieval.py`

**Checkpoint**: User Story 2 complete — retrieval module tested, caching verified, latency measured

---

## Phase 5: User Story 3 — Generate Answers Using Retrieved Context (Priority: P1) 🎯 MVP

**Goal**: Verify LLM generates answers grounded in context with prompt templating  
**Independent Test**: `pytest week-1/test_rag.py::TestGeneration -v && pytest week-1/test_rag.py::test_rag_integration -v`  
**Success Criteria**: SC-003 (generation latency <3s), SC-005 (tests pass)

- [ ] T031 [P] [US3] Review existing `generation.py` in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/generation.py` (should be complete from Day 1)
- [ ] T032 [US3] Verify prompt template in RAGGenerator: Check LangChain PromptTemplate in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/generation.py`
- [ ] T033 [US3] Run prompt template test: `pytest week-1/test_rag.py::TestGeneration::test_prompt_template -v` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T034 [US3] Run full RAG integration test: `pytest week-1/test_rag.py::test_rag_integration -v` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T035 [US3] Verify rag_answer() schema: Confirm answer, sources, latency_ms in response in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/generation.py`
- [ ] T036 [P] [US3] Measure generation latency: Call rag_answer() with docs and record LLM response time in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/`
- [ ] T037 [US3] Test hallucination mitigation: Verify prompt forces context-based answers in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/generation.py`
- [ ] T038 [P] [US3] Add docstring examples to RAGGenerator class in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/generation.py`

**Checkpoint**: User Story 3 complete — generation module tested, latency measured, hallucination mitigation verified

---

## Phase 6: User Story 4 — Serve Q&A Queries via FastAPI (Priority: P1) 🎯 MVP

**Goal**: Verify HTTP endpoints expose full RAG pipeline as service  
**Independent Test**: Start server, test endpoints with curl commands (documented in README)  
**Success Criteria**: SC-006 (API endpoints respond correctly), SC-004 (E2E latency <4s)

- [ ] T039 [P] [US4] Review existing `main.py` in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/main.py` (should be complete from Day 1)
- [ ] T040 [US4] Start FastAPI server: `cd week-1 && python main.py` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/`
- [ ] T041 [US4] Test health endpoint: `curl http://localhost:8000/health` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T042 [US4] Test ingest endpoint: `curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{"file_paths": ["data/sample.txt"]}'` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T043 [US4] Verify ingest response: Confirm `{"status": "success", "chunks_created": N, "files_processed": 1}` in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T044 [US4] Test query endpoint: `curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "What is RAG?", "top_k": 3}'` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T045 [US4] Verify query response: Confirm `{"answer": "...", "sources": [...], "latency_ms": N}` in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T046 [US4] Test error handling: Query without documents, verify 400 error "No documents indexed" in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T047 [P] [US4] Verify FastAPI auto-docs: `curl http://localhost:8000/docs` (should load OpenAPI UI) from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T048 [P] [US4] Measure E2E latency: Ingest → Query cycle, record total time in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/`

**Checkpoint**: User Story 4 complete — API fully functional, tested, documented

**🎯 MVP COMPLETE**: US1-4 form a complete, deployable RAG system.

---

## Phase 7: User Story 5 — Document and Validate Architecture (Priority: P2)

**Goal**: Create clear system diagram and component documentation  
**Independent Test**: `docs/architecture.md` contains Mermaid diagram, component descriptions, latency breakdown  
**Success Criteria**: SC-008 (architecture docs complete and clear)

- [ ] T049 [US5] Create `docs/architecture.md` with Mermaid flowchart showing: User Query → Ingestion → Chunking → Embedding → Vector Store → Retrieval → Generation → Response in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/docs/`
- [ ] T050 [US5] Document component descriptions in `docs/architecture.md`: DocumentIngester, RAGRetriever, RAGGenerator, FastAPI Server in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/docs/`
- [ ] T051 [US5] Add data flow example to `docs/architecture.md`: Query input → retrieval output → generation output with concrete numbers in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/docs/`
- [ ] T052 [US5] Add latency trace example to `docs/architecture.md`: retrieval_ms + generation_ms = total_ms in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/docs/`
- [ ] T053 [P] [US5] Create ER diagram for entities in `docs/architecture.md`: Document, Chunk, Embedding, Query, RetrievalResult, Answer in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/docs/`
- [ ] T054 [US5] Update `README.md` "Architecture" section with link to `docs/architecture.md` in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/`

**Checkpoint**: Architecture documentation complete and clear

---

## Phase 8: User Story 6 — Justify Design Trade-Offs (Priority: P2)

**Goal**: Document key decisions with rationale and alternatives  
**Independent Test**: `docs/trade-offs.md` covers 5+ design decisions with alternatives  
**Success Criteria**: SC-008 (trade-offs documentation complete)

- [ ] T055 [US6] Create `docs/trade-offs.md` with RAG vs Fine-tuning section: Why RAG chosen (cost, flexibility, no retraining) vs fine-tuning in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/docs/`
- [ ] T056 [US6] Add Embedding Model section to `docs/trade-offs.md`: Why text-embedding-3-small (cost, speed) vs 3-large (10x cost) in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/docs/`
- [ ] T057 [US6] Add Chunking Strategy section to `docs/trade-offs.md`: Why fixed-size 512 (simple, deterministic) vs semantic (better boundaries) in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/docs/`
- [ ] T058 [US6] Add Vector Store section to `docs/trade-offs.md`: Why in-memory dict (simple, Week 1) vs Weaviate (scalable, Week 2) in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/docs/`
- [ ] T059 [US6] Add LLM Choice section to `docs/trade-offs.md`: Why gpt-3.5-turbo (cost $0.002/1K tokens, 10x faster) vs GPT-4 in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/docs/`
- [ ] T060 [US6] Add Failure Modes & Mitigation section to `docs/trade-offs.md`: retrieval failure, hallucination, latency with mitigations in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/docs/`
- [ ] T061 [P] [US6] Add cost analysis to `docs/trade-offs.md`: Estimate cost per 1000 queries (embedding + LLM) in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/docs/`
- [ ] T062 [US6] Update `README.md` "Trade-Offs" section with link to `docs/trade-offs.md` in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/`
- [ ] T063 [P] [US6] Add "Future Work" section to `docs/trade-offs.md` listing Week 2+ decisions in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/docs/`

**Checkpoint**: Trade-offs documentation complete and comprehensive

---

## Phase 9: User Story 7 — Validate Week 1 Learning Outcomes (Priority: P2)

**Goal**: Confirm developer understands RAG concepts and can articulate design decisions  
**Independent Test**: Self-assessment checklist in WEEK-1-CHECKLIST.md  
**Success Criteria**: SC-009 (checkpoint validation passed)

- [ ] T064 [US7] Write "What is RAG?" section in `README.md`: 100-150 word summary covering retrieval + generation + why chosen over fine-tuning in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/`
- [ ] T065 [US7] Add embedding choice justification to `docs/trade-offs.md`: Quantify cost difference (10x cheaper) and acceptable accuracy trade-off in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/docs/`
- [ ] T066 [US7] Document 3 failure modes in `docs/trade-offs.md`: retrieval failure (chunking loses context), hallucination (LLM invents), latency (multi-step pipeline) in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/docs/`
- [ ] T067 [US7] Add latency breakdown section to `README.md` "Metrics": ingestion_ms, retrieval_ms, generation_ms, total_ms in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/`
- [ ] T068 [P] [US7] Run checkpoint self-assessment: Answer "Why RAG?", "Justify embedding choice?", "Name 3 failure modes?", "Explain latency to PM?" in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/`

**Checkpoint**: Week 1 learning outcomes validated

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Final touches, cleanup, verification before Week 2  
**Duration**: ~30 minutes

- [ ] T069 [P] Run full test suite: `pytest week-1/test_rag.py -v` (all tests PASS) from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T070 [P] Run type checker: `uvx ty check week-1/` (no errors) from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T071 [P] Run linter: `ruff check week-1/` (code style clean) from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T072 Update `README.md` with final metrics and links in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/`
- [ ] T073 Commit all Week 1 work: `git add . && git commit -m "feat: Complete Week 1 RAG foundation (Days 2-5)"` from `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`
- [ ] T074 Create `WEEK-1-SUMMARY.md` with links to deliverables, metrics, checkpoint validation in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/`

**Checkpoint**: Week 1 complete, all artifacts ready for review

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup) → Phase 2 (Foundational) → Phases 3-6 (User Stories P1)
                                         ↓
                                    Phases 7-9 (Documentation P2)
                                         ↓
                                    Phase 10 (Polish)
```

### User Story Dependencies (P1 Critical Path)

```
Phase 2 (Foundational) ─┬─→ US1 (Ingestion) ────┐
                        ├─→ US2 (Retrieval)  ────├─→ US4 (API) ─┐
                        ├─→ US3 (Generation) ────┘              │
                        └──────────────────────────────────────→ Phase 10 (Polish)
```

- **US1 → US2 → US3 → US4**: Dependency chain (each feeds into next)
- **Within Phase**: All [P] tasks can run in parallel
- **Across Phases**: User stories can start after Foundational completes

### User Story Dependencies (P2 Documentation)

```
US1-4 Complete ─→ US5 (Architecture) ─┐
                                       ├─→ US7 (Checkpoint)
                  US6 (Trade-offs) ────┘
```

- **US5, US6** depend on US1-4 being complete (need metrics, code to understand)
- **US7** depends on US5, US6 (checkpoint validates understanding)

### Parallel Execution Example (Single Developer)

```bash
# Day 2 (Tuesday): Phase 1-2 + US1
T001-T006 (Setup, 30 min)
T007-T013 (Foundational, 45 min)
T014-T021 (US1 Ingestion, 60 min)
→ Checkpoint: Ingestion tested, latency measured

# Day 3 (Wednesday): US2
T022-T030 (US2 Retrieval, 90 min)
→ Checkpoint: Retrieval tested, caching verified

# Day 4 (Thursday): US3 + US4
T031-T038 (US3 Generation, 60 min)
T039-T048 (US4 API, 90 min)
→ Checkpoint: Full API working end-to-end

# Day 5 (Friday): US5-7 + Polish
T049-T063 (US5-6 Documentation, 90 min)
T064-T068 (US7 Checkpoint, 45 min)
T069-T074 (Polish, 30 min)
→ WEEK 1 COMPLETE
```

### Parallel Execution Example (2-3 Developers)

```bash
# All Phase 1-2 together
Dev Team: T001-T013 (Setup + Foundational, 1.25 hours)

# Once Foundational done, split:
Dev A: T014-T021 (US1 Ingestion, 1h)
Dev B: T022-T030 (US2 Retrieval, 1.5h)
Dev C: T031-T038 (US3 Generation, 1h)
→ US4 (API) needs US1-3, Dev A takes T039-T048 after US1 complete (1.5h)

# Day 5: Parallel documentation + polish
Dev A: T049-T063 (Architecture + Trade-offs, 1.5h)
Dev B: T064-T068 (Checkpoint + Learning, 45 min)
Dev C: T069-T074 (Polish + commit, 30 min)
```

---

## Implementation Strategies

### MVP First (Recommended for Constraint)

**Scope**: US1-4 only (Phases 1-6)  
**Timeline**: Days 2-4 (9 hours)  
**Deliverable**: Fully functional RAG API

```bash
# Phase 1: Setup (30 min)
✓ T001-T006

# Phase 2: Foundational (45 min)
✓ T007-T013

# Phases 3-6: User Stories 1-4 (4 x 90 min = 6 hours)
✓ T014-T021 (US1 Ingestion)
✓ T022-T030 (US2 Retrieval)
✓ T031-T038 (US3 Generation)
✓ T039-T048 (US4 API)

# STOP & VALIDATE MVP (Friday morning)
✓ API working, latency measured
✓ Ready for demo or Week 2 extension
```

**Skip for MVP**: US5-7 (documentation, checkpoint) — can be done after MVP validation or as Week 1 bonus.

### Incremental Delivery (Full Scope)

**Timeline**: Days 2-5 (15.5 hours as planned)

1. **Day 2**: Phase 1-2 + US1 Ingestion (3h)
2. **Day 3**: US2 Retrieval (3h)
3. **Day 4**: US3 Generation + US4 API (3h)
4. **Day 5**: US5-7 Documentation + Checkpoint + Polish (3.5h)

Each day ends with checkpoint validation. By Friday 5pm, all deliverables complete.

---

## Task Format Validation

**✅ ALL TASKS FOLLOW CORRECT FORMAT**:

- [ ] Checkbox: `- [ ]` (always present)
- [ ] Task ID: Sequential (T001, T002, ..., T074)
- [ ] [P] marker: Present only for parallelizable tasks
- [ ] [Story] label: Present for US1-7 phases, absent for Setup/Foundational/Polish
- [ ] Description: Clear action with exact file path (absolute)

**Format Examples**:

✅ `- [ ] T001 Verify Python 3.11+ installed: python --version from /Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`

✅ `- [ ] T014 [P] [US1] Review existing ingestion.py in /Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/ingestion.py`

✅ `- [ ] T069 [P] Run full test suite: pytest week-1/test_rag.py -v from /Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks`

---

## Task Count Summary

| Phase | Task ID Range | Count | Focus |
|-------|---------------|-------|-------|
| Phase 1: Setup | T001-T006 | 6 | Environment, dependencies |
| Phase 2: Foundational | T007-T013 | 7 | Project structure, testing, validation |
| Phase 3: US1 (Ingestion) | T014-T021 | 8 | Load, chunk, metadata |
| Phase 4: US2 (Retrieval) | T022-T030 | 9 | Embed, cache, search |
| Phase 5: US3 (Generation) | T031-T038 | 8 | Prompt, LLM, latency |
| Phase 6: US4 (API) | T039-T048 | 10 | Endpoints, error handling, E2E |
| Phase 7: US5 (Architecture) | T049-T054 | 6 | Diagrams, data flow, components |
| Phase 8: US6 (Trade-offs) | T055-T063 | 9 | Decisions, cost analysis, future work |
| Phase 9: US7 (Checkpoint) | T064-T068 | 5 | Learning validation, understanding |
| Phase 10: Polish | T069-T074 | 6 | Tests, type check, lint, commit |
| **TOTAL** | **T001-T074** | **74** | **Days 2-5 complete Week 1** |

---

## Success Checklist

By end of Phase 10, verify:

- [ ] All 74 tasks completed and checked off
- [ ] `pytest week-1/test_rag.py -v` returns all tests PASS
- [ ] `uvx ty check week-1/` returns no type errors
- [ ] `ruff check week-1/` returns clean code style
- [ ] API running: `python week-1/main.py` starts on localhost:8000
- [ ] Documentation complete: `docs/architecture.md` + `docs/trade-offs.md` exist and are clear
- [ ] Checkpoint validation: Can articulate RAG fundamentals, embedding choice, failure modes, latency breakdown
- [ ] README updated with setup, API usage, metrics, architecture links
- [ ] All work committed to `week-1` branch with meaningful commit messages
- [ ] Ready to demo or move to Week 2

---

## Notes

- **[P] tasks**: Can run in parallel (different files, no blocking dependencies)
- **[Story] label**: Maps task to US1-US7 for traceability and independent delivery
- Each user story is independently completable and testable
- Commit after each checkpoint or logical group (per story phase)
- Tests already written in test_rag.py (from Day 1 setup); this phase focuses on verifying they pass
- Avoid: vague tasks, cross-story dependencies that break independence
- If blocked on API rate limits: Skip LLM-dependent tests temporarily, complete structure, retry later
- If time-constrained: Complete MVP (US1-4) by EOD Thursday, docs (US5-7) Friday morning

---

**Ready to implement? Start with Phase 1, then Phase 2. After Foundational completes, US1-4 can proceed in parallel. Good luck! 🚀**

**Total Planning Time**: Phase 0 (2h) + Phase 1 (4h) + Phase 2 (2h) = 8 hours  
**Total Implementation Time**: 15.5 hours (Days 2-5)  
**Total Project**: 23.5 hours
