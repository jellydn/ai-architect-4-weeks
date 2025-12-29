# Tasks: Complete Week 1 RAG Foundation (Days 2-5)

**Input**: Design documents from `/specs/001-week1-rag-completion/`  
**Prerequisites**: plan.md (complete), spec.md (complete), WEEK-1-CHECKLIST.md  
**Branch**: `001-week1-rag-completion` (feature tasks) → merges to `week-1` (implementation)

**Total Tasks**: 47 (Setup: 4, Foundational: 7, US1-4 implementations: 24, US5-7 docs: 8, Polish: 4)

---

## Format: `[ID] [P?] [Story?] Description`

- **[ID]**: Task identifier (T001, T002, etc.) in execution order
- **[P]**: Tasks marked [P] can run in parallel (different files, no dependencies)
- **[Story?]**: Which user story this belongs to (US1-US7), omitted for Setup/Foundational/Polish phases
- **File paths**: Absolute paths for clarity during implementation

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and dependency management  
**Timeline**: ~30 minutes (should already be mostly complete from Day 1)

- [ ] T001 Verify Python 3.11+ and uv installed: `which uv && python --version`
- [ ] T002 Activate venv and verify dependencies: `source .venv/bin/activate && uv pip list | grep -E "fastapi|openai|langchain"`
- [ ] T003 [P] Create `.env` from `.env.example` and configure OPENAI_API_KEY
- [ ] T004 [P] Verify pyproject.toml dependencies match requirements.txt

**Checkpoint**: Environment ready, dependencies installed, .env configured

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before user story implementations  
**Timeline**: ~45 minutes (mostly complete from Day 1, finalizations needed)

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T005 Verify project structure matches plan.md in `/Users/huynhdung/src/tries/2025-12-28-ai-in-4-weeks/week-1/`
- [ ] T006 [P] Create `week-1/__init__.py` (empty or with version marker)
- [ ] T007 [P] Verify test infrastructure: `pytest --version && python -m pytest week-1/test_rag.py --collect-only`
- [ ] T008 [P] Run type checker: `uvx ty check week-1/`
- [ ] T009 [P] Run linter: `ruff check week-1/`
- [ ] T010 Create sample test document: `data/sample.txt` (already exists from Day 1)
- [ ] T011 [P] Verify logging configured in all modules (check imports in ingestion.py, retrieval.py, generation.py, main.py)

**Checkpoint**: Foundation ready - all user story implementations can now begin

---

## Phase 3: User Story 1 - Test Document Ingestion at Scale (Priority: P1) 🎯 MVP

**Goal**: Verify that documents can be loaded, chunked, and prepared for embedding  
**Independent Test**: `pytest week-1/test_rag.py::TestIngestion -v`  
**Success Criteria**: SC-001 (ingestion latency <5s for 1MB), SC-005 (tests pass)

### Implementation for User Story 1

- [ ] T012 [P] [US1] Review existing ingestion.py in `week-1/ingestion.py` (should be complete from Day 1 setup)
- [ ] T013 [US1] Test ingestion with sample document: `python week-1/ingestion.py` and verify output
- [ ] T014 [US1] Run ingestion unit tests: `pytest week-1/test_rag.py::TestIngestion::test_load_from_file -v`
- [ ] T015 [US1] Run chunking tests: `pytest week-1/test_rag.py::TestIngestion::test_chunking -v`
- [ ] T016 [US1] Run full ingestion pipeline test: `pytest week-1/test_rag.py::TestIngestion::test_ingest_full_pipeline -v`
- [ ] T017 [P] [US1] Measure ingestion latency: Log timestamp from test output, record in performance notes
- [ ] T018 [US1] Verify chunking validation: Test with edge case (very large document >10MB if possible)
- [ ] T019 [P] [US1] Add docstring examples to DocumentIngester class in `week-1/ingestion.py`

**Checkpoint**: User Story 1 complete - ingestion module fully tested and documented

---

## Phase 4: User Story 2 - Embed and Retrieve Relevant Documents (Priority: P1) 🎯 MVP

**Goal**: Verify embeddings are generated, cached, and retrieval returns ranked results  
**Independent Test**: `pytest week-1/test_rag.py::TestRetrieval -v`  
**Success Criteria**: SC-002 (retrieval latency <500ms), SC-005 (tests pass), SC-007 (caching works)

### Implementation for User Story 2

- [ ] T020 [P] [US2] Review existing retrieval.py in `week-1/retrieval.py` (should be complete from Day 1 setup)
- [ ] T021 [US2] Ensure OpenAI API key is accessible: Test import and client initialization
- [ ] T022 [US2] Run retrieval caching test: `pytest week-1/test_rag.py::TestRetrieval::test_embedding_caching -v`
- [ ] T023 [US2] Run retrieval ordering test: `pytest week-1/test_rag.py::TestRetrieval::test_retrieval_ordering -v`
- [ ] T024 [US2] Run empty index test: `pytest week-1/test_rag.py::TestRetrieval::test_retrieval_empty -v`
- [ ] T025 [P] [US2] Measure embedding latency: Ingest sample.txt and record time for embedding 10+ chunks
- [ ] T026 [P] [US2] Measure retrieval latency: Query with 3 documents indexed, record query embed + similarity search time
- [ ] T027 [US2] Test API rate limit handling: Add retry logic or document expected behavior in logging
- [ ] T028 [P] [US2] Add docstring examples to RAGRetriever class in `week-1/retrieval.py`

**Checkpoint**: User Story 2 complete - retrieval module fully tested, embeddings caching verified, latency measured

---

## Phase 5: User Story 3 - Generate Answers Using Retrieved Context (Priority: P1) 🎯 MVP

**Goal**: Verify LLM generates answers grounded in context with prompt templating  
**Independent Test**: `pytest week-1/test_rag.py::TestGeneration -v && pytest week-1/test_rag.py::test_rag_integration -v`  
**Success Criteria**: SC-003 (generation latency <3s), SC-005 (tests pass)

### Implementation for User Story 3

- [ ] T029 [P] [US3] Review existing generation.py in `week-1/generation.py` (should be complete from Day 1 setup)
- [ ] T030 [US3] Review prompt template in RAGGenerator class: Verify it includes context and query safely
- [ ] T031 [US3] Run generation prompt template test: `pytest week-1/test_rag.py::TestGeneration::test_prompt_template -v`
- [ ] T032 [US3] Run full RAG integration test: `pytest week-1/test_rag.py::test_rag_integration -v`
- [ ] T033 [US3] Verify rag_answer() returns correct schema: answer (str), sources (list), latency_ms (float)
- [ ] T034 [P] [US3] Measure generation latency: Call rag_answer() with 3 retrieved documents, record LLM response time
- [ ] T035 [US3] Test hallucination mitigation: Verify prompt forces context-based answers (document in generation.py comments)
- [ ] T036 [P] [US3] Add docstring examples to RAGGenerator class in `week-1/generation.py`

**Checkpoint**: User Story 3 complete - generation module fully tested, latency measured, hallucination mitigation verified

---

## Phase 6: User Story 4 - Serve Q&A Queries via FastAPI (Priority: P1) 🎯 MVP

**Goal**: Verify HTTP endpoints expose full RAG pipeline as a service  
**Independent Test**: Start server, test endpoints with curl commands (documented in README)  
**Success Criteria**: SC-006 (API endpoints respond correctly), SC-004 (E2E latency <4s)

### Implementation for User Story 4

- [ ] T037 [P] [US4] Review existing main.py in `week-1/main.py` (should be complete from Day 1 setup)
- [ ] T038 [US4] Start FastAPI server: `cd week-1 && python main.py` (should run on localhost:8000)
- [ ] T039 [US4] Test health endpoint: `curl http://localhost:8000/health`
- [ ] T040 [US4] Test ingest endpoint: `curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{"file_paths": ["data/sample.txt"]}'`
- [ ] T041 [US4] Verify ingest response schema: Should return `{"status": "success", "chunks_created": N, "files_processed": 1}`
- [ ] T042 [US4] Test query endpoint: `curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "What is RAG?", "top_k": 3}'`
- [ ] T043 [US4] Verify query response schema: Should return `{"answer": "...", "sources": [...], "latency_ms": N}`
- [ ] T044 [US4] Test error handling: Query without documents indexed, verify 400 error with message "No documents indexed. Call /ingest first"
- [ ] T045 [P] [US4] Verify FastAPI auto-generated docs: `curl http://localhost:8000/docs` (should load OpenAPI UI)
- [ ] T046 [P] [US4] Measure full E2E latency: Ingest → Query cycle, record total time including ingestion, retrieval, generation

**Checkpoint**: User Story 4 complete - API fully functional, tested, documented

**🎯 MVP COMPLETE**: US1-4 form a complete, deployable RAG system. Stop here if time is constrained.

---

## Phase 7: User Story 5 - Document and Validate Architecture (Priority: P2)

**Goal**: Create clear system diagram and component documentation  
**Independent Test**: `docs/architecture.md` contains Mermaid diagram, component descriptions, latency breakdown  
**Success Criteria**: SC-008 (architecture docs complete and clear)

### Documentation for User Story 5

- [ ] T047 [US5] Create `docs/architecture.md` with:
  - [ ] Mermaid flowchart showing: User Query → Ingestion → Chunking → Embedding → Vector Store → Retrieval → Generation → Response
  - [ ] Component descriptions: DocumentIngester, RAGRetriever, RAGGenerator, FastAPI Server
  - [ ] Data flow example: Query input → retrieval output → generation output
  - [ ] Latency trace example: retrieval_ms + generation_ms = total_ms (with concrete numbers from US2-3)
  - [ ] References to functional requirements (FR-001 through FR-013) in component sections

- [ ] T048 [US5] Add system diagram to `docs/architecture.md` showing all 4 components and endpoints
- [ ] T049 [P] [US5] Create ASCII or Mermaid ER diagram for entities: Document, Chunk, Embedding, Query, RetrievalResult, Answer
- [ ] T050 [US5] Update `README.md` "Architecture" section to link to `docs/architecture.md`

**Checkpoint**: Architecture documentation complete and clear

---

## Phase 8: User Story 6 - Justify Design Trade-Offs (Priority: P2)

**Goal**: Document key decisions with rationale and alternatives  
**Independent Test**: `docs/trade-offs.md` covers 5+ design decisions with alternatives  
**Success Criteria**: SC-008 (trade-offs documentation complete)

### Documentation for User Story 6

- [ ] T051 [US6] Create `docs/trade-offs.md` sections:
  - [ ] **RAG vs Fine-tuning**: Why RAG chosen (cost, flexibility, no retraining) vs fine-tuning (permanent knowledge, specific style)
  - [ ] **Embedding Model**: Why text-embedding-3-small (cost, speed) vs 3-large (cost 10x higher, minimal accuracy gain)
  - [ ] **Chunking Strategy**: Why fixed-size 512 tokens with 50 overlap (simple, deterministic) vs semantic chunking (better boundaries, more compute)
  - [ ] **Vector Store**: Why in-memory dict (simple, Week 1) vs Weaviate (scalable, Week 2 plan)
  - [ ] **LLM Choice**: Why gpt-3.5-turbo (cost $0.002/1K tokens, 10x faster) vs GPT-4 (cost $0.03/1K tokens, overkill for retrieval)
  - [ ] **Prompt Injection Mitigation**: Risk (user query in prompt) + mitigation (context from retrieval, not appended to user input)
  - [ ] **Hallucination Mitigation**: Strategy (force context-based answers in prompt) + limitation (LLM can still hallucinate details)

- [ ] T052 [US6] Add decision tree to `docs/trade-offs.md`: "When to use RAG vs alternatives" (decision matrix)
- [ ] T053 [P] [US6] Add cost analysis to `docs/trade-offs.md`: Estimate cost per 1000 queries (embedding API + LLM API)
- [ ] T054 [US6] Update `README.md` "Trade-Offs" section to link to `docs/trade-offs.md`
- [ ] T055 [P] [US6] Add "Future Work" section to `docs/trade-offs.md` listing Week 2+ decisions (Weaviate, reranking, evaluation)

**Checkpoint**: Trade-offs documentation complete and comprehensive

---

## Phase 9: User Story 7 - Validate Week 1 Learning Outcomes (Priority: P2)

**Goal**: Confirm developer understands RAG concepts and can articulate design decisions  
**Independent Test**: Self-assessment checklist in WEEK-1-CHECKLIST.md  
**Success Criteria**: SC-009 (checkpoint validation passed)

### Checkpoint Validation for User Story 7

- [ ] T056 [US7] **Explain RAG in 2 minutes**: Write 100-150 word summary in README.md under "What is RAG?"
  - [ ] Covers: retrieval + generation architecture
  - [ ] Covers: Why chosen over fine-tuning
  - [ ] Covers: Trade-off (fast, cheap, flexible vs accuracy)

- [ ] T057 [US7] **Justify embedding choice**: Write 2-3 sentences in `docs/trade-offs.md` "Embedding Model" section
  - [ ] Quantify cost difference (3-small is 10x cheaper)
  - [ ] Explain acceptable accuracy trade-off (1-2% loss for retrieval-based task)

- [ ] T058 [US7] **Identify 3 failure modes**: Document in `docs/trade-offs.md` "Failure Modes" section
  - [ ] Retrieval failure: Chunking loses important context
  - [ ] Hallucination: LLM invents details not in context
  - [ ] Latency: Multi-step pipeline (embed query, search, generate) vs single LLM call

- [ ] T059 [US7] **Explain latency breakdown**: Add section to README.md "Metrics"
  - [ ] Ingestion latency: X ms (from SC-001)
  - [ ] Retrieval latency: Y ms (from SC-002)
  - [ ] Generation latency: Z ms (from SC-003)
  - [ ] Total E2E: X+Y+Z ms (from SC-004)

- [ ] T060 [P] [US7] Run checkpoint self-assessment:
  - [ ] Can I explain RAG vs fine-tuning trade-offs in plain English? (write in README)
  - [ ] Can I justify my embedding choice over GPT-4 embeddings? (write in trade-offs.md)
  - [ ] Can I identify 3 failure modes of my RAG? (write in trade-offs.md)
  - [ ] Can I explain latency breakdown to a PM? (write in README Metrics)

**Checkpoint**: Week 1 learning outcomes validated, developer can speak fluently about RAG

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Final touches, cleanup, verification before Week 2  
**Timeline**: ~30 minutes

- [ ] T061 [P] Run full test suite: `pytest week-1/test_rag.py -v` (all tests pass)
- [ ] T062 [P] Run type checker: `uvx ty check week-1/` (no type errors)
- [ ] T063 [P] Run linter: `ruff check week-1/` (code style clean)
- [ ] T064 Update README.md with final metrics and links to architecture/trade-offs docs
- [ ] T065 Commit all Week 1 work to `week-1` branch: `git add . && git commit -m "feat: Complete Week 1 RAG foundation (Days 2-5)"` 
- [ ] T066 Create WEEK-1-SUMMARY.md with:
  - [ ] Links to all deliverables (API, architecture.md, trade-offs.md)
  - [ ] Latency measurements (ingestion, retrieval, generation, E2E)
  - [ ] Cost estimate per 1000 queries
  - [ ] Checkpoint validation checklist (all 4 items checked)
  - [ ] Confidence statement: "I understand RAG system architecture and trade-offs fluently"

**Checkpoint**: Week 1 complete, all artifacts ready for review

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - start immediately (should be quick since Day 1 setup is done)
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - US1, US2, US3, US4 can proceed **in parallel** (different modules: ingestion, retrieval, generation, main)
  - Or sequentially if working alone
- **Documentation (Phase 7-9)**: Depends on US1-4 completion (need measurements, implementation complete)
- **Polish (Phase 10)**: Depends on all prior phases

### User Story Dependencies (P1 - Critical Path)

```
Phase 2 (Foundational) ─┬─→ US1 (Ingestion) ─┐
                        ├─→ US2 (Retrieval)  ├─→ US4 (API) ─┐
                        ├─→ US3 (Generation) ┘              │
                        └──────────────────────────────────→ Phase 10 (Polish)
```

- **US1 → US2 → US3 → US4**: Dependency chain (each feeds into next)
- **Within Phase**: All [P] tasks can run in parallel
- **Across Phases**: User stories can start after Foundational completes

### User Story Dependencies (P2 - Documentation & Learning)

```
US1-4 Complete ─→ US5 (Architecture) ─┐
                                       ├─→ US7 (Checkpoint)
                  US6 (Trade-offs) ────┘
```

- **US5, US6** depend on US1-4 being complete (need metrics, code to understand)
- **US7** depends on US5, US6 (checkpoint validates understanding of architecture + trade-offs)

### Parallel Execution Example (Single Developer)

```bash
# Day 2 (Tuesday): Phase 1-2 + US1
T001-T004 (Setup, 30 min)
T005-T011 (Foundational, 45 min)
T012-T019 (US1 Ingestion, 60 min)
→ Checkpoint: Ingestion tested, latency measured

# Day 3 (Wednesday): US2
T020-T028 (US2 Retrieval, 90 min)
→ Checkpoint: Retrieval tested, caching verified

# Day 4 (Thursday): US3 + US4
T029-T036 (US3 Generation, 60 min)
T037-T046 (US4 API, 90 min)
→ Checkpoint: Full API working end-to-end

# Day 5 (Friday): US5-7 + Polish
T047-T055 (US5-6 Documentation, 90 min)
T056-T060 (US7 Checkpoint, 45 min)
T061-T066 (Polish, 30 min)
→ WEEK 1 COMPLETE
```

### Parallel Execution Example (2-3 Developers)

```bash
# All Phase 1-2 together
Dev Team: T001-T011 (Setup + Foundational, 1 hour)

# Once Foundational done, split:
Dev A: T012-T019 (US1 Ingestion)
Dev B: T020-T028 (US2 Retrieval)
Dev C: T029-T036 (US3 Generation)
→ US4 (API) needs US1-3, Dev A takes T037-T046 after US1 complete

# Day 5: Parallel documentation + polish
Dev A: T047-T055 (Architecture + Trade-offs)
Dev B: T056-T060 (Checkpoint + Learning)
Dev C: T061-T066 (Polish + commit)
```

---

## Implementation Strategy

### MVP First (Recommended for Constraint)

**Scope**: US1-4 only (Phase 1-6)  
**Timeline**: Days 2-4 (9 hours)  
**Deliverable**: Fully functional RAG API

```bash
# Phase 1: Setup (30 min)
✓ Environment ready, dependencies verified

# Phase 2: Foundational (45 min)
✓ Test infrastructure, structure verified

# Phase 3-6: User Stories 1-4 (3 x 90 min = 4.5 hours)
✓ Ingestion → Retrieval → Generation → API
✓ Each story tested independently
✓ Full E2E pipeline working

# STOP & VALIDATE MVP (Friday morning)
✓ API working, latency measured
✓ Ready for demo or Week 2 extension
```

**Skip for MVP**: US5-7 (documentation, checkpoint) - can be done after MVP validation or as Week 1 bonus.

### Incremental Delivery (Full Scope)

**Timeline**: Days 2-5 (15.5 hours as planned)

1. **Day 2**: Phase 1-2 + US1 Ingestion (3h)
2. **Day 3**: US2 Retrieval (3h)
3. **Day 4**: US3 Generation + US4 API (3h)
4. **Day 5**: US5-7 Documentation + Checkpoint + Polish (3.5h)

Each day ends with checkpoint validation. By Friday 5pm, all deliverables complete.

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

## Task Count Summary

| Phase | Count | Focus |
|-------|-------|-------|
| Phase 1: Setup | 4 | Environment, dependencies |
| Phase 2: Foundational | 7 | Project structure, testing, validation |
| Phase 3: US1 (Ingestion) | 8 | Load, chunk, metadata |
| Phase 4: US2 (Retrieval) | 9 | Embed, cache, search |
| Phase 5: US3 (Generation) | 8 | Prompt template, LLM call, latency |
| Phase 6: US4 (API) | 10 | Endpoints, error handling, E2E |
| Phase 7: US5 (Architecture) | 4 | Diagrams, data flow, components |
| Phase 8: US6 (Trade-offs) | 5 | Decisions, cost analysis, future work |
| Phase 9: US7 (Checkpoint) | 5 | Learning validation, understanding |
| Phase 10: Polish | 6 | Tests, type check, lint, commit |
| **TOTAL** | **47** | **Days 2-5 complete Week 1** |

---

## Success Checklist

By end of Phase 10, verify:

- [ ] All 47 tasks completed and checked off
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

**Ready to implement? Start with Phase 1, then Phase 2. After Foundational completes, US1-4 can proceed in parallel. Good luck! 🚀**
