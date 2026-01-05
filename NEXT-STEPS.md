# Next Steps: Week 2 Ready to Start

**Status**: Week 1 complete, Week 2 setup done  
**Branch**: `001-week1-rag-completion` (current)  
**Next Branch**: `002-week2-vector-db` (start after Weaviate setup)

---

## What's Ready

✅ **Week 1 Complete**:
- Ingestion, retrieval, generation modules
- FastAPI server with 3 endpoints
- 8/8 tests passing
- Type checking & linting passing
- Full documentation

✅ **Week 2 Initialized**:
- WEEK-2-START.md with overview and timeline
- week-2/vector_db.py with WeaviateStore class
- 5 tasks created in backlog (task-3 through task-7)
- Architecture diagram showing Week 2 improvements

---

## Day 1 Instructions: Vector DB Setup

### Prerequisites
```bash
# Docker must be installed and running
docker --version

# Check current branch
git branch -a
```

### Quick Start
```bash
# Option 1: Start Weaviate locally (requires Docker)
docker run -d -p 8080:8080 -p 50051:50051 \
  cr.weaviate.io/semitechnologies/weaviate:latest

# Verify running
curl http://localhost:8080/v1/.well-known/ready
# Expected: {"ready": true}

# Option 2: View the implementation
cat week-2/vector_db.py  # 250+ lines of production-grade code

# Option 3: Test the schema (after Weaviate is running)
pip install weaviate-client
python -m week-2.vector_db
```

### Acceptance Criteria for Day 1
- [ ] Weaviate running on localhost:8080
- [ ] WeaviateStore class can connect to Weaviate
- [ ] Schema creation works (DocumentChunk class)
- [ ] Can index documents with metadata
- [ ] Vector search returns results
- [ ] Metadata filtering works
- [ ] All tests pass

### Files to Work On
- `week-2/vector_db.py` - Main implementation (DONE - ready for testing)
- `week-2/test_vector_db.py` - Unit tests (TODO - create next)
- `week-2/integration_test.py` - End-to-end test (TODO - create next)

---

## Backlog Tasks

| ID | Title | Status | Priority |
|------|-------|--------|----------|
| task-3 | Week 2 Day 1: Vector DB Setup | To Do | High |
| task-4 | Week 2 Day 2: Chunking Strategies | To Do | High |
| task-5 | Week 2 Day 3: Reranking & Evaluation | To Do | High |
| task-6 | Week 2 Day 4: Caching & Performance | To Do | High |
| task-7 | Week 2 Day 5: Checkpoint & Report | To Do | High |

View full details:
```bash
cat backlog/tasks/task-3*
# ... etc for other tasks
```

---

## Week 2 Architecture Change

**Week 1 (In-Memory)**:
```
Document → Chunk → Embed → Dict[Vector] → Cosine Sim
                                ↓
                           Fast (in RAM)
                           Simple (no setup)
                           Limited (restart = loss)
```

**Week 2 (Persistent)**:
```
Document → Chunk → Embed → Weaviate → HNSW Index → Search → Rerank → Answer
                                ↓
                           Persistent (disk)
                           Scalable (100k+ docs)
                           HNSW (faster search)
                           Metadata filtering
```

---

## Success Metrics (Week 2)

| Metric | Target | Notes |
|--------|--------|-------|
| Weaviate uptime | 100% | Local development |
| Index latency | <50ms/doc | For 1k document test set |
| Search latency | <200ms | Including embedding |
| Retrieval MRR | >0.7 | With reranking |
| Cache hit rate | >30% | On typical queries |
| Disk usage | <500MB | For 1k documents |

---

## Commands Reference

```bash
# View Week 1 status
make check        # All tests pass

# View Week 2 status
ls week-2/        # Files created
git log -3        # Latest commits

# Check Docker
docker ps         # See running containers

# Weaviate health
curl http://localhost:8080/v1/.well-known/ready

# Run Week 2 vector_db tests (after Weaviate starts)
python -m week-2.vector_db
```

---

## Transition Plan

After completing Day 1:

1. **Move to new branch**:
   ```bash
   git checkout -b 002-week2-vector-db
   ```

2. **Continue Days 2-5**:
   - Day 2: Chunking strategies & metadata
   - Day 3: Reranking & evaluation metrics
   - Day 4: Query caching & performance
   - Day 5: Checkpoint & latency report

3. **Final deliverable**:
   ```
   WEEK-2-SUMMARY.md
   docs/chunking-analysis.md
   docs/retrieval-metrics.json
   docs/latency-analysis.md
   ```

---

## Getting Help

- Check `WEEK-2-START.md` for detailed learning goals
- Review `week-2/vector_db.py` for implementation patterns
- Run tests to verify setup: `python -m week-2.vector_db`
- Check backlog for task details: `cat backlog/tasks/task-3*`

---

**Ready to start? Follow the Quick Start section above.**
