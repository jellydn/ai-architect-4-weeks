# Reading Guide: Week 1 Summary

**Start here** if you want to understand what was accomplished and the core concepts.

---

## Quick Overview (5 minutes)

**Start with**: `SUMMARY.txt`
- Day-by-day accomplishments
- 7 core concepts at a glance
- Performance metrics
- Key decisions

---

## Comprehensive Understanding (30 minutes)

**Then read**: `WEEK-1-CONCEPTS.md`
- Deep dive into each concept
- Design decisions explained
- Failure modes and solutions
- Week 1 → 2 transition
- Common questions answered

---

## Implementation Details (15 minutes)

**Code to review**:
1. `week-1/ingestion.py` - How documents are chunked
2. `week-1/retrieval.py` - How embeddings and search work
3. `week-1/generation.py` - How prompts and LLM calls work
4. `week-1/main.py` - How the FastAPI server is structured
5. `week-1/test_rag.py` - How to test ML pipelines

**Architecture**: `docs/architecture.md`
- System design diagrams
- Data flow
- Component interactions

**Trade-offs**: `docs/trade-offs.md`
- Why each decision was made
- Alternatives considered
- Performance trade-offs

---

## Project Status & Next Steps (10 minutes)

**Current state**: `PROJECT-STATUS.md`
- Week 1 complete checklist
- Repository metrics
- Week 2 overview
- Getting started instructions

**Week 2 preparation**: `NEXT-STEPS.md`
- Day 1 quick start
- Prerequisites
- Backlog tasks
- Transition plan

---

## Learning Summary (2 minutes)

**Key takeaway**: `WEEK-1-SUMMARY.md`
- Learning outcomes
- Metrics
- Next steps

---

## Recommended Reading Path

### Path A: Quick Review (7 minutes)
1. This file (you're reading it)
2. SUMMARY.txt (5 min)
3. Jump to NEXT-STEPS.md to start Week 2

### Path B: Learning (45 minutes)
1. SUMMARY.txt (5 min)
2. WEEK-1-CONCEPTS.md (30 min)
3. Review one implementation file (week-1/retrieval.py) (10 min)

### Path C: Deep Dive (2 hours)
1. WEEK-1-CONCEPTS.md (30 min)
2. All 5 implementation files + tests (60 min)
3. docs/architecture.md (15 min)
4. docs/trade-offs.md (15 min)

### Path D: Code-First (1.5 hours)
1. week-1/test_rag.py (10 min) - See what the system does
2. week-1/ingestion.py (10 min) - How documents are processed
3. week-1/retrieval.py (15 min) - The core ML logic
4. week-1/generation.py (10 min) - Prompt engineering
5. week-1/main.py (10 min) - API structure
6. docs/architecture.md (15 min) - Connect the pieces

---

## By Role

### For Learners (Want to understand RAG)
1. WEEK-1-CONCEPTS.md - Core concepts
2. docs/architecture.md - System overview
3. week-1/retrieval.py - Vector search implementation
4. WEEK-1-SUMMARY.md - Key takeaways

### For Engineers (Want to build on this)
1. NEXT-STEPS.md - Week 2 setup
2. week-2/vector_db.py - New code to extend
3. PROJECT-STATUS.md - Context and metrics
4. backlog/tasks/ - What needs doing

### For Managers (Want to understand progress)
1. PROJECT-STATUS.md - Status overview
2. SUMMARY.txt - What was built
3. README.md - Deliverables and usage

### For Researchers (Want metrics and insights)
1. WEEK-1-SUMMARY.md - Metrics
2. docs/trade-offs.md - Design decisions
3. WEEK-1-CONCEPTS.md - Failure modes
4. week-1/test_rag.py - Evaluation approach

---

## Document Map

```
READING-GUIDE.md (you are here)
├── Quick Start
│   ├── SUMMARY.txt                 ← 5-minute overview
│   └── README.md                   ← How to use the system
│
├── Concepts & Theory
│   ├── WEEK-1-CONCEPTS.md          ← 30-minute deep dive
│   └── docs/
│       ├── architecture.md         ← System design
│       └── trade-offs.md           ← Design decisions
│
├── Implementation
│   ├── week-1/                     ← 5 Python files
│   │   ├── ingestion.py            ← Chunking logic
│   │   ├── retrieval.py            ← Vector search
│   │   ├── generation.py           ← LLM prompting
│   │   ├── main.py                 ← FastAPI server
│   │   └── test_rag.py             ← Tests & integration
│   └── week-2/                     ← Next phase starts here
│       └── vector_db.py            ← Weaviate integration
│
├── Planning & Progress
│   ├── PROJECT-STATUS.md           ← Current state
│   ├── WEEK-1-SUMMARY.md           ← Outcomes
│   ├── WEEK-1-CHECKLIST.md         ← What was done
│   ├── WEEK-1-COMPLETION.md        ← Detailed status
│   ├── NEXT-STEPS.md               ← Week 2 start
│   ├── WEEK-2-START.md             ← Week 2 overview
│   └── WEEK-2-CHECKLIST.md         ← Week 2 plan
│
└── Backlog & Tasks
    ├── backlog/                    ← 5 Week 2 tasks
    │   └── tasks/
    │       ├── task-3 ...          ← Vector DB setup
    │       ├── task-4 ...          ← Chunking strategies
    │       ├── task-5 ...          ← Reranking & eval
    │       ├── task-6 ...          ← Caching & perf
    │       └── task-7 ...          ← Checkpoint
    └── AGENTS.md                   ← Development workflow
```

---

## Key Files by Purpose

### To Understand RAG
1. WEEK-1-CONCEPTS.md (concepts 1-5)
2. docs/architecture.md
3. week-1/retrieval.py (the search logic)

### To Run the Code
1. README.md (setup & usage)
2. week-1/main.py (API server)
3. Makefile (run commands)

### To Extend the Code
1. week-2/vector_db.py (new database layer)
2. NEXT-STEPS.md (what to build)
3. backlog/tasks/ (specific tasks)

### To Learn the Design
1. WEEK-1-CONCEPTS.md (all 7 concepts)
2. docs/trade-offs.md (why decisions made)
3. week-1/test_rag.py (how it's tested)

---

## Quick Facts

- **Duration**: 4 days (Dec 30 - Jan 2)
- **Code**: 745 lines across 5 files
- **Tests**: 8 passing (0.67s total)
- **Type coverage**: 100%
- **Documentation**: 7 guides
- **Performance**: All targets met or exceeded
- **Status**: ✅ Production ready for Week 1

---

## Concepts Checklist

After reading, you should understand:

- [ ] What RAG is and when to use it
- [ ] Why overlapping chunks matter
- [ ] How embeddings represent meaning
- [ ] How cosine similarity finds relevant documents
- [ ] How prompt engineering grounds answers
- [ ] Why async/FastAPI matters for scale
- [ ] Why type safety catches bugs early
- [ ] The 3 failure modes and solutions
- [ ] Week 1 limitations and Week 2 solutions
- [ ] Cost vs quality trade-offs

---

## Next: Week 2

Once you've read through, follow this path:

1. **Read**: NEXT-STEPS.md (5 min)
2. **Setup**: Docker + Weaviate (10 min)
3. **Code**: week-2/vector_db.py (already written!)
4. **Test**: python -m week-2.vector_db

Then continue with Days 2-5.

---

**Choose your path above and start reading!**
