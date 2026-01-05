# Implementation Plan: Complete Week 1 RAG Foundation (Days 2-5)

**Branch**: `001-week1-rag-completion` | **Date**: 2025-12-29 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-week1-rag-completion/spec.md`

## Summary

Complete Week 1 of the 4-week AI Architect sprint by building, testing, and documenting a fully functional Retrieval-Augmented Generation (RAG) system. This feature covers Days 2-5: document ingestion with configurable chunking, vector embeddings and similarity search, LLM-based answer generation with prompt templating, FastAPI HTTP endpoints, and comprehensive architecture/trade-off documentation. By end of Week 1, users will have a working Q&A API, measurable latency baselines, and deep understanding of RAG system design trade-offs.

## Technical Context

**Language/Version**: Python 3.11 (modern async/typing)  
**Primary Dependencies**: FastAPI, uvicorn, LangChain, OpenAI Python client, numpy, pydantic  
**Additional Tooling**: uv (package manager), ruff (linter), ty (type checker via uvx), pytest (testing)  
**Storage**: In-memory vector store (Week 1); escalates to Weaviate Week 2  
**Testing**: pytest with asyncio support; unit + integration tests  
**Target Platform**: Local development + Linux server ready  
**Project Type**: Single Python backend (Web API)  
**Performance Goals**: 
- Ingestion: <5s for 10 documents (1MB total)
- Retrieval: <500ms for embed query + similarity search
- Generation: <3s for LLM response
- Total E2E: <4s per query

**Constraints**: 
- gpt-3.5-turbo context window (4K tokens) sufficient for 3 retrieved documents
- In-memory vector store supports <10k documents
- Local development environment (Python 3.11+)

**Scale/Scope**: 
- 7 user stories, P1 (4 critical path items) + P2 (3 learning/docs items)
- 13 functional requirements, 10 success criteria
- Days 2-5 implementation (15.5 hours total, ~3 hours/day)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle: Architect-First, Not Tutorial-First ✓
- **Requirement**: Every deliverable must explain **why**, not just **how**
- **Status**: PASS
- **Evidence**: Spec includes trade-offs.md + architecture.md as P2 user stories. Success criteria SC-009 requires developer articulate RAG vs fine-tuning decision. Assumptions documented.

### Principle: Running Code + Production Thinking ✓
- **Requirement**: Each week ends with deployable, measurable output. Demo grade rejected.
- **Status**: PASS
- **Evidence**: 7 user stories include full API (FR-008 through FR-010), latency measurement (FR-011), structured responses (FR-012). Success criteria SC-001 through SC-004 mandate specific latency targets.

### Principle: Evaluation Drives Architecture ✓
- **Requirement**: Do not iterate blindly. By Week 3, all output is measured.
- **Status**: PASS (Week 1 baseline)
- **Evidence**: FR-011 requires latency logging for all operations. Success criteria SC-001 through SC-004 define measurable baselines. Week 2+ will compare retrieval quality metrics.

### Principle: Single Flagship Repository ✓
- **Requirement**: All work lives in one repo with clear weekly branches
- **Status**: PASS
- **Evidence**: Feature in branch `001-week1-rag-completion` (feature spec), merges to `week-1` (implementation). Single repository: https://github.com/jellydn/ai-architect-4-weeks

### Principle: Deliverables Are Written, Not Just Code ✓
- **Requirement**: Each week includes architecture diagram + trade-off analysis + decision record + measurement report
- **Status**: PASS
- **Evidence**: User stories US5 (architecture), US6 (trade-offs), US7 (checkpoint validation). Success criteria SC-008 (docs complete), SC-009 (understanding validated).

### Principle: Public-Ready Artifacts ✓
- **Requirement**: By Week 4, code is clean, diagrams clear, writing concise
- **Status**: PASS (Week 1 foundation)
- **Evidence**: Modern tooling (ruff linter, ty type checker, pytest), type hints on all modules. README documents API usage via curl. Mermaid diagram required.

**GATE RESULT**: ✅ **PASS** — All principles satisfied. No violations.

## Project Structure

### Documentation (this feature)

```text
specs/001-week1-rag-completion/
├── spec.md              # Feature specification (complete)
├── plan.md              # This file (implementation plan)
├── checklists/
│   └── requirements.md  # Quality validation checklist (complete)
├── research.md          # Phase 0 output (TBD - research phase)
├── data-model.md        # Phase 1 output (TBD - design phase)
├── quickstart.md        # Phase 1 output (TBD - design phase)
└── contracts/           # Phase 1 output (TBD - API contracts)
```

### Source Code (repository root)

```text
ai-architect-4weeks/
├── week-1/                    # Week 1 RAG implementation
│   ├── __init__.py
│   ├── main.py               # FastAPI app (FR-008, FR-009, FR-010)
│   ├── ingestion.py          # DocumentIngester class (FR-001, FR-002)
│   ├── retrieval.py          # RAGRetriever class (FR-003, FR-004, FR-005)
│   ├── generation.py         # RAGGenerator class (FR-006, FR-007)
│   └── test_rag.py          # Tests: ingestion, retrieval, generation, integration
│
├── docs/                      # Documentation
│   ├── architecture.md       # System diagram, data flow (US5)
│   └── trade-offs.md         # Design decisions & rationale (US6)
│
├── data/                      # Sample documents
│   └── sample.txt            # Test document for Week 1
│
├── pyproject.toml            # Python packaging (dependencies, dev tools)
├── requirements.txt          # Pinned versions (legacy, uv manages via pyproject.toml)
├── .env.example              # Configuration template
├── README.md                 # Setup, API usage, metrics (updated)
└── WEEK-1-CHECKLIST.md       # Daily checklist (updated with modern tooling)
```

**Structure Decision**: Single Python backend (Option 1). Week 1 is a focused API + library exercise. No frontend/mobile needed. As system grows to Week 2+, structure scales horizontally within `week-2/`, `week-3/` folders rather than vertically.

## Complexity Tracking

> No Constitution violations. No complexity justification needed.

All design decisions align with Constitution principles. Technology stack is fixed (Python, FastAPI, LangChain, OpenAI). Scale is bounded (Days 2-5, 15.5 hours, 7 user stories). No deviations required.

---

## Implementation Phases

### Phase 0: Research & Unknowns Resolution

**Status**: TBD (triggered by `/speckit.plan` with `--phase 0`)

**Tasks**:
1. Research LangChain document loaders (best practices for TXT, markdown)
2. Research embedding cost calculation (text-embedding-3-small API pricing)
3. Research cosine similarity implementation (numpy vs scipy performance)
4. Research FastAPI async patterns for I/O-bound operations
5. Research prompt injection mitigation in LangChain templates
6. Research OpenAI API rate limits and retry strategies

**Output**: `research.md` with decisions, rationale, alternatives considered

**Timeline**: ~2-3 hours (learning-by-building approach; fast research, depth from building)

---

### Phase 1: Design & Contracts

**Status**: TBD (triggered by `/speckit.plan` with `--phase 1`)

**Tasks**:

1. **Data Model** (`data-model.md`):
   - Document entities: Chunk, Embedding, Query, RetrievalResult, Answer
   - Validation rules (e.g., chunk size, similarity score bounds)

2. **API Contracts** (`contracts/`):
   - `/health` endpoint (GET)
   - `/ingest` endpoint (POST) with request/response schema
   - `/query` endpoint (POST) with request/response schema
   - OpenAPI schema generation (FastAPI auto-docs)

3. **Quickstart** (`quickstart.md`):
   - How to run FastAPI server locally
   - How to ingest sample documents
   - How to query RAG system
   - Expected latency and cost per operation

4. **Agent Context Update**:
   - Run `.specify/scripts/bash/update-agent-context.sh amp`
   - Update agent context with Week 1 technology decisions

**Output**: data-model.md, contracts/, quickstart.md, updated agent context

**Timeline**: ~2-3 hours

---

### Phase 2: Task Breakdown (triggered by `/speckit.tasks`)

**Status**: TBD (next command after /speckit.plan)

**Output**: `tasks.md` with granular tasks for Days 2-5 implementation

**Structure**:
- Day 2 (Tuesday): Ingestion tasks
- Day 3 (Wednesday): Retrieval tasks
- Day 4 (Thursday): Generation + API tasks
- Day 5 (Friday): Documentation + checkpoint tasks

---

## Success Criteria for Plan Acceptance

- ✅ Technical Context fully specified (no NEEDS CLARIFICATION)
- ✅ Constitution Check passes (all principles satisfied)
- ✅ Project Structure is concrete (no template placeholders)
- ✅ Phases outlined with clear inputs/outputs
- ✅ Timeline realistic (3-4 hours per phase, Days 2-5 implementation)

**Plan Status**: ✅ **READY FOR PHASE 0 RESEARCH**

---

## Next Steps

1. **Phase 0**: Run `/speckit.plan --phase 0` to generate research.md
2. **Phase 1**: Run `/speckit.plan --phase 1` to generate design artifacts
3. **Phase 2**: Run `/speckit.tasks` to break into task-level work items
4. **Implementation**: Begin Day 2 on `week-1` branch using tasks.md as checklist
