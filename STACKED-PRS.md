# Stacked Pull Requests: 4-Week RAG Learning Path

All 4 weeks of work are organized as **stacked draft PRs on GitHub**, creating a clean dependency chain.

## Quick Summary

| Week | PR | Status | Base | Link |
|------|----|---------|----|------|
| 1 | #1 | 📝 DRAFT | main | [PR #1](https://github.com/jellydn/ai-architect-4-weeks/pull/1) |
| 2 | #2 | 📝 DRAFT | #1 | [PR #2](https://github.com/jellydn/ai-architect-4-weeks/pull/2) |
| 3 | #3 | 📝 DRAFT | #2 | [PR #3](https://github.com/jellydn/ai-architect-4-weeks/pull/3) |
| 4 | #4 | 📝 DRAFT | #3 | [PR #4](https://github.com/jellydn/ai-architect-4-weeks/pull/4) |

## Stacked Structure

```
PR #4 (Week 4) ──┐
                 ↓ depends on
             PR #3 (Week 3) ──┐
                              ↓ depends on
                          PR #2 (Week 2) ──┐
                                           ↓ depends on
                                       PR #1 (Week 1) ──┐
                                                        ↓ merges to
                                                       main
```

## PR Details

### PR #1: Week 1 - RAG Foundation ✅
- **Status**: 📝 DRAFT (Complete, ready for review)
- **Base**: `main`
- **Head**: `001-week1-rag-completion`
- **Progress**: ✅ COMPLETE
- **What**: Core RAG system (745 LOC, 27 commits, 8/8 tests)
- **Key Files**: ingestion.py, retrieval.py, generation.py, main.py, test_rag.py
- **Convert to ready**: `gh pr ready 1`

### PR #2: Week 2 - Production-Grade RAG 🚀
- **Status**: 📝 DRAFT (In progress)
- **Base**: `001-week1-rag-completion`
- **Head**: `002-week2-vector-db`
- **Progress**: 🚀 IN PROGRESS (Days 1-2 complete)
- **What**: Vector DB, Chunking, Reranking, Caching
- **Key Files**: vector_db.py, chunking_strategies.py, (reranking.py TODO), (caching.py TODO)
- **Convert to ready**: `gh pr ready 2`

### PR #3: Week 3 - Evaluation & Monitoring 🔍
- **Status**: 📝 DRAFT (Planned)
- **Base**: `002-week2-vector-db`
- **Head**: `003-week3-evals`
- **Progress**: 🔍 PLANNED (Jan 8 start)
- **What**: Metrics, Monitoring, Load Testing, Deployment Prep
- **Convert to ready**: `gh pr ready 3`

### PR #4: Week 4 - Production Deployment 🔧
- **Status**: 📝 DRAFT (Planned)
- **Base**: `003-week3-evals`
- **Head**: `004-week4-production`
- **Progress**: 🔧 PLANNED (Jan 15 start)
- **What**: Docker, Kubernetes, Security, Operations
- **Convert to ready**: `gh pr ready 4`

## Why Draft?

Draft PRs are ideal because:

- ✓ PRs exist and track progress
- ✓ Can be reviewed independently
- ✓ Not in active review queue
- ✓ Can be converted to "Ready for review" when complete
- ✓ Shows clear dependency chain on GitHub
- ✓ Team can see overall project progress

## Workflow

**For each week:**

1. Complete the week's work
2. Run tests: `make check`
3. Commit all changes
4. Convert PR from draft to ready:
   ```bash
   gh pr ready N  # N = PR number
   ```
5. Request reviewers and address feedback
6. Merge when approved:
   ```bash
   gh pr merge N --merge
   ```

**Timeline:**
- Week 1: Convert to ready now → Merge to main
- Week 2: Convert to ready Jan 7 → Merge to PR #1
- Week 3: Convert to ready Jan 12 → Merge to PR #2
- Week 4: Convert to ready Jan 19 → Merge to PR #3

## Commands Reference

### View PRs
```bash
# List all PRs with status
gh pr list

# View specific PR details
gh pr view 1
gh pr view 2

# View PR on GitHub in browser
gh pr view 1 --web

# Check PR status
gh pr status
```

### Convert Draft to Ready
```bash
# When Week 1 complete
gh pr ready 1

# When Week 2 complete
gh pr ready 2

# When Week 3 complete
gh pr ready 3

# When Week 4 complete
gh pr ready 4
```

### Convert Back to Draft
```bash
# If need to mark as draft again
gh pr ready 1 --undo
```

### Merge PR
```bash
# Merge with squash
gh pr merge 1 --squash

# Merge without squash
gh pr merge 1 --merge
```

### Checkout Branch
```bash
# Switch to Week 2 work
git checkout 002-week2-vector-db

# Update from origin
git pull origin 002-week2-vector-db
```

## Merge Sequence

When all weeks complete, final merge order:

1. **PR #1 → main**
   - Week 1 work becomes official main version

2. **PR #2 → 001-week1-rag-completion**
   - Week 1 branch now includes Week 2
   - PR #1 will be updated with Week 2 code
   - Can then merge PR #1 to main if desired

3. **PR #3 → 002-week2-vector-db**
   - Week 2 branch now includes Week 3
   - PR #2 will be updated with Week 3 code

4. **PR #4 → 003-week3-evals**
   - Week 3 branch now includes Week 4
   - PR #3 will be updated with Week 4 code

Final result: main has all 4 weeks integrated

## Branch Overview

```
origin/main
  └── origin/001-week1-rag-completion (Week 1 - PR #1)
      └── origin/002-week2-vector-db (Week 2 - PR #2)
          └── origin/003-week3-evals (Week 3 - PR #3)
              └── origin/004-week4-production (Week 4 - PR #4)
```

All branches exist locally and on GitHub.

## Documentation

- **PULL-REQUESTS.md** - Detailed PR documentation
- **WEEK-1-SUMMARY.md** - Week 1 outcomes
- **WEEK-1-CONCEPTS.md** - Learning concepts
- **WEEK-2-START.md** - Week 2 overview
- **NEXT-STEPS.md** - Week 2 instructions
- **READING-GUIDE.md** - How to learn Week 1

## Current Status

```
✅ Week 1: COMPLETE
   - 27 commits, 745 LOC
   - 8/8 tests passing
   - All documentation done
   - Ready: gh pr ready 1

🚀 Week 2: IN PROGRESS
   - 1 commit, 522 LOC
   - Vector DB + Chunking ready
   - Days 3-5 TODO
   - Ready in 5 days

🔍 Week 3: PLANNED
   - Branch created
   - Ready Jan 8

🔧 Week 4: PLANNED
   - Branch created
   - Ready Jan 15
```

## Next Steps

1. **Review Week 1** (PR #1)
   - Check implementation quality
   - Verify tests and docs
   - Approve when ready

2. **Continue Week 2** (PR #2)
   - Complete Days 3-5
   - Add reranking, caching, evaluation
   - Tests and documentation

3. **Complete Week 3** (PR #3)
   - Start Jan 8
   - Add evaluation & monitoring

4. **Complete Week 4** (PR #4)
   - Start Jan 15
   - Add deployment & operations

Then merge in sequence!

## Links

- **All PRs**: https://github.com/jellydn/ai-architect-4-weeks/pulls
- **PR #1**: https://github.com/jellydn/ai-architect-4-weeks/pull/1
- **PR #2**: https://github.com/jellydn/ai-architect-4-weeks/pull/2
- **PR #3**: https://github.com/jellydn/ai-architect-4-weeks/pull/3
- **PR #4**: https://github.com/jellydn/ai-architect-4-weeks/pull/4
- **Repository**: https://github.com/jellydn/ai-architect-4-weeks
