# Stacked Pull Requests: 4-Week RAG Learning Path

**Structure**: Each week builds on the previous week's PR, creating a linear dependency chain.

---

## PR Stack Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  Week 4: Production                         │
│          004-week4-production → 003-week3-evals            │
│     (Docker, Kubernetes, Security, Operations)              │
│                      PR #4                                  │
└─────────────────────────────────────────────────────────────┘
                           ↓ (depends on)
┌─────────────────────────────────────────────────────────────┐
│             Week 3: Evaluation & Monitoring                 │
│         003-week3-evals → 002-week2-vector-db              │
│    (Metrics, Testing, Monitoring, Load Testing)             │
│                      PR #3                                  │
└─────────────────────────────────────────────────────────────┘
                           ↓ (depends on)
┌─────────────────────────────────────────────────────────────┐
│         Week 2: Production-Grade RAG                        │
│       002-week2-vector-db → 001-week1-rag-completion       │
│    (Vector DB, Reranking, Caching, Evaluation)              │
│                      PR #2                                  │
└─────────────────────────────────────────────────────────────┘
                           ↓ (depends on)
┌─────────────────────────────────────────────────────────────┐
│          Week 1: RAG Foundation                             │
│       001-week1-rag-completion → main                       │
│     (Core System, Testing, Documentation)                   │
│                      PR #1                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Pull Request Details

### PR #1: Week 1 - RAG Foundation ✅ COMPLETE

**Base**: `main`  
**Head**: `001-week1-rag-completion`  
**URL**: https://github.com/jellydn/ai-architect-4-weeks/pull/1

**Status**: ✅ Complete (Dec 30 - Jan 2, 2026)

**What's Included**:
- Core RAG system (745 lines of code)
- Document ingestion with overlap-based chunking
- Vector embeddings and similarity search
- LLM integration for answer generation
- FastAPI server with 3 working endpoints
- 8/8 unit tests passing
- Type checking and linting: PASS
- Complete documentation

**Key Files**:
- `week-1/ingestion.py` - Document processing
- `week-1/retrieval.py` - Vector search (core ML)
- `week-1/generation.py` - LLM integration
- `week-1/main.py` - FastAPI server
- `week-1/test_rag.py` - Comprehensive tests

**Performance**:
- End-to-end latency: <3.5s ✅
- Ingestion: <500ms/doc ✅
- Retrieval: <200ms ✅
- Test execution: 0.67s ✅

**Next**: Review and merge to main, then proceed to PR #2

---

### PR #2: Week 2 - Production-Grade RAG 🚀 IN PROGRESS

**Base**: `001-week1-rag-completion`  
**Head**: `002-week2-vector-db`  
**URL**: https://github.com/jellydn/ai-architect-4-weeks/pull/2

**Status**: 🚀 Starting (Jan 3, 2026)

**What's Included**:
- Weaviate vector database integration (250 LOC)
- Persistent storage with HNSW indexing
- Fixed and semantic chunking strategies
- Metadata filtering support
- Reranking implementation
- Query caching
- Evaluation metrics (MRR, NDCG)
- Production deployment checklist

**Days**:
- Day 1: Vector DB setup & migration
- Day 2: Chunking strategies & metadata ✅ (chunking_strategies.py added)
- Day 3: Reranking & evaluation
- Day 4: Caching & performance tuning
- Day 5: Checkpoint & latency report

**Success Metrics**:
- Weaviate uptime: 100%
- Index latency: <50ms/doc
- Retrieval MRR: >0.7
- Cache hit rate: >30%
- E2E latency (with cache): <1s

**Dependencies**: PR #1 must be merged first

---

### PR #3: Week 3 - Evaluation & Monitoring 🔍 PLANNED

**Base**: `002-week2-vector-db`  
**Head**: `003-week3-evals`  
**URL**: https://github.com/jellydn/ai-architect-4-weeks/pull/3

**Status**: 🔍 Planned (Jan 8-12, 2026)

**What's Included**:
- Evaluation metrics dashboard
- Monitoring infrastructure
- Quality metric implementation (MRR, NDCG, F1)
- Comparison reports (Week 1 vs Week 2)
- Production monitoring (latency, cost, errors)
- Load testing infrastructure
- Scaling recommendations

**Days**:
- Day 1: Evaluation framework
- Day 2: Quality metrics
- Day 3: Production monitoring
- Day 4: Load testing & scaling
- Day 5: Production checklist & deployment

**Success Metrics**:
- All metrics tracked and visualized
- Comparison report complete
- Production monitoring active
- Load testing results documented
- Deployment-ready system

**Dependencies**: PR #2 must be merged first

---

### PR #4: Week 4 - Production Deployment 🔧 PLANNED

**Base**: `003-week3-evals`  
**Head**: `004-week4-production`  
**URL**: https://github.com/jellydn/ai-architect-4-weeks/pull/4

**Status**: 🔧 Planned (Jan 15-19, 2026)

**What's Included**:
- Docker containerization (multi-stage builds)
- Docker Compose with all dependencies
- Kubernetes manifests (Deployment, Service, ConfigMap)
- Horizontal Pod Autoscaling (HPA)
- Database backup/restore procedures
- API authentication (JWT/OAuth)
- Rate limiting and quotas
- Operations runbook
- Training materials

**Days**:
- Day 1: Docker & containerization
- Day 2: Kubernetes deployment
- Day 3: Database migration & backup
- Day 4: Security & access control
- Day 5: Launch & operations

**Success Metrics**:
- Service deployed on Kubernetes
- Zero-downtime deployments working
- Backup/restore tested
- Security audit passed
- Cost optimized (<$100/month)

**Dependencies**: PR #3 must be merged first

---

## How Stacked PRs Work

**Merged in sequence**:
1. PR #1 (Week 1) merges to `main`
2. PR #2 (Week 2) merges to Week 1 branch (becomes Week 1 + Week 2)
3. PR #3 (Week 3) merges to Week 2 branch (becomes Week 1-3)
4. PR #4 (Week 4) merges to Week 3 branch (becomes Week 1-4)

**Benefits**:
- Each PR can be reviewed independently
- Work progresses in parallel
- Clear dependency chain
- Easy to rebase on main when ready
- Clear separation of concerns by week

**Workflow**:
```bash
# Week 1 PR ready
git checkout 001-week1-rag-completion
# → Review and test
# → Merge to main

# Week 2 PR ready
git checkout 002-week2-vector-db
# → Review and test (based on Week 1)
# → Merge to 001-week1-rag-completion
# → Week 1 branch now includes Week 2
# → Can squash and merge Week 1 to main if needed

# Week 3 PR ready
git checkout 003-week3-evals
# → Review and test (based on Week 2)
# → Merge to 002-week2-vector-db
# → And so on...
```

---

## Status Dashboard

| Week | PR | Status | Base | Days | Commits |
|------|----|---------|----- |------|---------|
| 1 | #1 | ✅ Complete | main | 4 | 27 |
| 2 | #2 | 🚀 Starting | 001 | 5 | +1 (chunking_strategies) |
| 3 | #3 | 🔍 Planned | 002 | 5 | +0 |
| 4 | #4 | 🔧 Planned | 003 | 5 | +0 |

---

## Reviewing PRs

### Week 1 Review Checklist
- [ ] Core RAG system works
- [ ] All 8 tests passing
- [ ] Type checking passing
- [ ] Linting passing
- [ ] Documentation complete
- [ ] Performance targets met
- [ ] Code quality good
- [ ] Ready to merge to main

### Week 2 Review Checklist
- [ ] Weaviate integration working
- [ ] Vector DB setup instructions clear
- [ ] Chunking strategies implemented
- [ ] Metadata filtering works
- [ ] Tests for new components
- [ ] Backward compatible with Week 1
- [ ] Documentation updated
- [ ] Ready to merge to Week 1

### Week 3 Review Checklist
- [ ] Evaluation metrics working
- [ ] Monitoring infrastructure set up
- [ ] Quality metrics calculated
- [ ] Comparison reports generated
- [ ] Load testing completed
- [ ] Deployment checklist done
- [ ] Ready to merge to Week 2

### Week 4 Review Checklist
- [ ] Docker builds successfully
- [ ] Kubernetes manifests valid
- [ ] Security audit passed
- [ ] Backup/restore tested
- [ ] Operations runbook complete
- [ ] Training materials ready
- [ ] Ready to merge to Week 3 and eventually main

---

## Commands Reference

### View PR Details
```bash
# List all PRs
gh pr list

# View PR #1
gh pr view 1

# View PR #2
gh pr view 2

# Get review status
gh pr status
```

### Checkout PR Branch
```bash
# Week 1
git checkout origin/001-week1-rag-completion

# Week 2
git checkout origin/002-week2-vector-db

# Week 3
git checkout origin/003-week3-evals

# Week 4
git checkout origin/004-week4-production
```

### Merge PR
```bash
# Merge Week 1 to main (when ready)
git checkout main
git merge 001-week1-rag-completion

# Merge Week 2 to Week 1 (when ready)
git checkout 001-week1-rag-completion
git merge 002-week2-vector-db

# And so on...
```

---

## Timeline

| Week | Start | End | Status |
|------|-------|-----|--------|
| 1 | Dec 30 | Jan 2 | ✅ Complete |
| 2 | Jan 3 | Jan 7 | 🚀 In Progress |
| 3 | Jan 8 | Jan 12 | 🔍 Planned |
| 4 | Jan 15 | Jan 19 | 🔧 Planned |

---

## Links

- **Repository**: https://github.com/jellydn/ai-architect-4-weeks
- **PR List**: https://github.com/jellydn/ai-architect-4-weeks/pulls
- **Week 1 Files**: `WEEK-1-SUMMARY.md`, `WEEK-1-CONCEPTS.md`
- **Week 2 Files**: `WEEK-2-START.md`, `NEXT-STEPS.md`
- **Week 3 Files**: `WEEK-3-CHECKLIST.md`
- **Week 4 Files**: `WEEK-4-CHECKLIST.md`

---

**Next Step**: Review PR #1 and merge when ready. Then proceed with PR #2.
