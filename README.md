# AI Architect: 4-Week Sprint

Production-grade AI system built in 4 weeks. Foundation to architecture.

From RAG basics → production vector DB → evaluation & monitoring → fine-tuning & design decisions.

## What This Is

A **systematic, execution-focused plan** to develop AI Architect-level skills and a portfolio artifact. Not a tutorial. Not a course. A 4-week sprint ending with deployable code + architecture documentation + public proof.

**Goal**: Credibly apply for AI Architect / Senior AI Engineer roles.

## Quick Start

```bash
# Clone
git clone https://github.com/jellydn/ai-architect-4-weeks
cd ai-architect-4-weeks

# Setup (Week 1+)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run (when code is ready)
docker-compose up

# Query API
curl -X POST http://localhost:8000/query \
  -d '{"query": "What is RAG?"}' \
  -H "Content-Type: application/json"
```

## The Plan

| Week | Focus | Deliverable | Checkpoint |
|------|-------|-------------|-----------|
| **1** | RAG Foundation | Running RAG API + architecture diagram | Can explain why RAG, not fine-tuning |
| **2** | Production RAG | Vector DB + reranking + caching + latency report | Can debug retrieval failures |
| **3** | Eval & Monitoring | Golden dataset + metrics dashboard + SLO | Can answer: "How do we know it's working?" |
| **4** | Architecture & Fine-Tuning | Fine-tuned model + system design v2 + public post | Can design an AI system from scratch |

## Documentation

- **[CONSTITUTION.md](CONSTITUTION.md)** — Core principles (architect-first, running code, evaluation-driven)
- **[WEEK-1-CHECKLIST.md](WEEK-1-CHECKLIST.md)** — Day-by-day (15.5 hours) to ship RAG foundation
- **[WEEK-2-CHECKLIST.md](WEEK-2-CHECKLIST.md)** — Day-by-day (15 hours) to production RAG
- **[WEEK-3-CHECKLIST.md](WEEK-3-CHECKLIST.md)** — Day-by-day (15 hours) to evaluation & monitoring
- **[WEEK-4-CHECKLIST.md](WEEK-4-CHECKLIST.md)** — Day-by-day (16 hours) to architecture & positioning

## Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Language | Python 3.10+ | Industry standard; typing support |
| API | FastAPI | Production-ready; async; auto OpenAPI docs |
| RAG | LangChain | Widest ecosystem; clear abstractions |
| LLM APIs | OpenAI + Claude | Cost-aware multi-model (Claude for reasoning, OpenAI for speed) |
| Vector DB | Weaviate | Built-in reranking; production features; local dev |
| Evaluation | MLflow + custom | Experiment tracking + domain-specific measures |
| Monitoring | OpenTelemetry | Vendor-agnostic observability |

## Key Principles

1. **Architect-First**: Every deliverable explains *why*, not just *how*
2. **Running Code**: Each week ends with deployable, measurable output
3. **Evaluation-Driven**: No iteration without data
4. **Single Repo**: All work lives here; clear weekly branches
5. **Written Artifacts**: Diagrams + trade-off analysis + decision records
6. **Public-Ready**: Code is clean; docs are clear; assume external audience

## Results (Target)

By end of Week 4:

- **Accuracy**: 85%+ on golden dataset (50 curated queries)
- **Latency**: P99 < 1600ms
- **Cost**: < $0.01/query
- **Architecture**: Documented, defended, diagram included
- **Fine-Tuning**: Comparison base vs fine-tuned (data-driven improvement)
- **Public Proof**: Blog post + video + GitHub (portfolio-ready)

## Portfolio Value

This repo demonstrates to AI Architect interview panel:

✅ **LLM System Design**: RAG vs fine-tuning decision framework  
✅ **Production Thinking**: Monitoring, cost, SLO definition  
✅ **Evaluation Rigor**: Golden dataset, metrics, data-driven iteration  
✅ **Architecture Skills**: Design decisions, trade-off analysis, scaling concerns  
✅ **Engineering Discipline**: Tests, logging, error handling, documentation  

## How to Use This

### For Self-Study
1. Read `CONSTITUTION.md` (understand principles)
2. Start `WEEK-1-CHECKLIST.md` (follow day-by-day)
3. Complete one week at a time (non-negotiable checkpoints)
4. Don't skip documentation (it's part of the deliverable)

### For Interviews
- **System Design**: "Here's my architecture v2; here's why I chose Weaviate"
- **Evaluation**: "My golden dataset has 50 queries; accuracy is 85%"
- **Trade-Offs**: "Fine-tuning vs RAG? Depends on [knowledge | style | task specificity]"
- **Production**: "P99 latency is 1600ms; cost is $0.005/query; I monitor both"

### For Your CV
- Designed + implemented production-grade RAG system (retrieval, generation, caching)
- Built evaluation framework measuring accuracy, latency, cost (golden dataset, metrics)
- Implemented fine-tuning pipeline; compared base vs fine-tuned models
- Achieved 85% accuracy, P99 < 1.6s latency, $0.005/query cost
- System architecture documented with trade-off analysis

## Progress Tracking

Use `git` branches for each week:
```bash
git checkout -b week-1   # Day 1 → Friday checkpoint
git checkout -b week-2   # Next cycle
...
```

Each branch should have:
- Working code
- Tests (pytest)
- README updates
- New documentation
- Metrics/measurements logged

## Next Steps (After Week 4)

1. **Polish**: Clean up code, add docstrings, finalize tests
2. **Public**: Publish blog post + GitHub link
3. **Applications**: Update CV, reach out to recruiters
4. **Interviews**: Use this repo + system design explanations

## License

MIT

## Author

**Dung Duc Huynh (Kaka)** | he/him  
[GitHub](https://github.com/jellydn) | [LinkedIn](https://www.linkedin.com/in/dung-huynh-duc/)

Lifelong learner. Building in public. Learning in public.

---

**Status**: Ready for Week 1 start  
**Last Updated**: 2025-12-28  
**Constitution Version**: 1.0

---

## Philosophy

This sprint embodies **#LearnInPublic** + **#BuildInPublic**:
- Every decision is documented, not hidden
- Progress is shared, not gatekept
- Code + writing are public artifacts from day 1
- The system itself becomes proof of capability

Not just learning AI. Learning *how to architect* AI systems. In public.

