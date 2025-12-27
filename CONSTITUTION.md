# AI Architect 4-Week Sprint Constitution

## Core Principles

### I. Architect-First, Not Tutorial-First
Every deliverable must explain **why**, not just **how**. Code without reasoning is incomplete. Every README, diagram, and artifact must answer: "Why this choice over alternatives?"

### II. Running Code + Production Thinking
Each week ends with deployable, measurable output. "Demo grade" is rejected. Every component must have latency, cost, and quality assertions in place by design.

### III. Evaluation Drives Architecture
Do not iterate blindly. By Week 3, all output is measured. Evaluation scripts exist before feature requests. If it cannot be measured, it cannot be optimized.

### IV. Single Flagship Repository
All work lives in one repo with clear weekly branches (`week-1`, `week-2`, etc.). This becomes the portfolio artifact. No scattered notebooks or throwaway scripts.

### V. Deliverables Are Written, Not Just Code
Each week includes:
- Architecture diagram (Mermaid or equivalent)
- Trade-off analysis (markdown)
- Decision record (why this, not that)
- Measurement report (latency, cost, accuracy)

### VI. Public-Ready Artifacts
By Week 4, all work is shaped for external communication. Code is clean. Diagrams are clear. Writing is concise. Assume someone will read this to assess your AI Architect capability.

## Execution Standards

### Weekly Rhythm
- **Monday**: Learning goals + hands-on spike
- **Tuesday–Thursday**: Build, measure, iterate
- **Friday**: Write-up, architecture review, checkpoint validation

### Checkpoint Validation (Non-Negotiable)
Each week's checkpoint must be demonstrated:
- **Week 1**: You explain RAG vs fine-tuning trade-offs + API works
- **Week 2**: Retrieval quality comparison + latency measured
- **Week 3**: Evaluation dashboard shows accuracy, cost, latency
- **Week 4**: You design a new AI system from scratch (no copying)

### Code Quality Rules
- Type hints required (Python)
- Tests for all retrieval/evaluation logic
- No hardcoded configs (use YAML/ENV)
- All external calls logged with latency
- README explains every directory

## Technology Stack (Fixed)

| Component | Choice | Reason |
|-----------|--------|--------|
| Language | Python 3.10+ | Industry standard for AI; typing support |
| API Framework | FastAPI | Production-ready; async; auto OpenAPI docs |
| RAG Library | LangChain | Widest ecosystem; clear abstractions |
| LLM APIs | OpenAI + Claude (Claude for reasoning, OpenAI for speed) | Cost-aware multi-model architecture |
| Vector DB | Weaviate (Week 2) | Built-in reranking; production features |
| Evaluation | MLflow + custom metrics | Experiment tracking + domain-specific measures |
| Monitoring | OpenTelemetry | Vendor-agnostic observability |

## Deliverables Checklist

### Week 1
- [ ] RAG API (FastAPI) accepting documents + queries
- [ ] Mermaid architecture diagram (ingestion → retrieval → generation)
- [ ] trade-offs.md (RAG vs fine-tuning vs prompting)
- [ ] README with setup + example request/response

### Week 2
- [ ] Vector DB integration (Weaviate)
- [ ] Metadata filtering implemented
- [ ] Reranking comparison (retrieval@1, MRR, NDCG)
- [ ] Latency report (P50, P99 for retrieval + generation)
- [ ] design-decisions.md (chunking strategy, DB choice, caching)

### Week 3
- [ ] Evaluation framework (accuracy, relevance, hallucination checks)
- [ ] Metrics dashboard (cost per request, token usage, latency)
- [ ] eval-results.json (baseline performance)
- [ ] monitoring.md ("How we know this is working")

### Week 4
- [ ] Fine-tuned model (LoRA on classification/summarization task)
- [ ] Comparison report (base vs fine-tuned on eval set)
- [ ] Architecture diagram v2 (with abstraction layer)
- [ ] One public artifact (blog post, video, or GitHub discussion)

## Governance

**This Constitution supersedes all other practices.** 
- Amendments require written justification + impact analysis
- All PRs/commits must reference this document
- Weekly checkpoints are mandatory gates; no exceptions

**Compliance Checks:**
- Code review verifies: Architecture decisions documented, measurements included, README updated
- Diagrams explain the system; prose explains why
- No "learning branches"—only production-ready work merges

**Version**: 1.0 | **Ratified**: 2025-12-28 | **Last Amended**: 2025-12-28
