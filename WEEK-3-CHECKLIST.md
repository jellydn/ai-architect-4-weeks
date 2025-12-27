# Week 3: Evaluation, Monitoring & GenAIOps — Daily Checklist

**Goal**: Become "production-ready AI engineer".  
**Deliverables**: Evaluation framework + metrics dashboard + monitoring setup  
**Checkpoint**: You can answer "How do we know this AI is working?"

---

## Day 1 (Monday): Evaluation Framework & Golden Dataset

**Theme**: Measure before you optimize.

### Learning (60 min)
- [ ] Read: [Evaluating RAG Systems](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/) — Metrics: faithfulness, relevance, answer similarity
- [ ] Read: [Golden Datasets](https://www.datarobot.com/wiki/golden-dataset/) — Curated test sets with known-good answers
- [ ] Read: [LLM-as-Judge Pattern](https://arxiv.org/abs/2306.05685) — Use LLM to evaluate other LLM outputs
- [ ] Question: "How do I know my RAG is better than baseline without manual labeling?"

### Build Golden Dataset (90 min)
- [ ] Create `data/golden-dataset.json`:
  ```json
  [
    {
      "query": "What is X?",
      "expected_answer": "X is...",
      "relevant_documents": ["doc1.txt", "doc2.txt"],
      "difficulty": "easy",
      "category": "definition"
    }
  ]
  ```
- [ ] Curate 50 test queries (mix: easy, medium, hard)
- [ ] For each query:
  - Hand-write expected answer
  - Identify relevant docs from your corpus
  - Assign difficulty + category
- [ ] Document: `docs/dataset-creation.md` (why these queries, how labeled)

### Build Evaluation Module (60 min)
- [ ] Create `week-3/evaluation.py`:
  ```python
  class RAGEvaluator:
      def evaluate_answer(self, prediction: str, reference: str) -> dict
      # Metrics: BLEU, ROUGE, semantic similarity (embedding-based)
      
      def evaluate_retrieval(self, retrieved_docs: List[str], relevant_docs: List[str]) -> dict
      # Metrics: precision, recall, F1
      
      def evaluate_query(self, query: str, result: dict) -> dict
      # Combines retrieval + answer evaluation
      
      def batch_evaluate(self, queries: List[dict]) -> dict
      # Returns: {"accuracy": float, "retrieval_precision": float, ...}
  ```
- [ ] Implement metrics:
  - **Answer metrics**: ROUGE-L, semantic similarity, token overlap
  - **Retrieval metrics**: Precision@3, Recall@10, MRR
  - **Hallucination detection**: Check if answer contradicts context

### Hands-On Deliverable
- [ ] `data/golden-dataset.json` with 50 test queries
- [ ] `evaluation.py` with 3+ evaluation metrics
- [ ] Test on 10 queries from golden dataset

### Success Criteria
- [ ] Golden dataset covers 3+ categories (definition, comparison, reasoning)
- [ ] Each query has hand-written reference answer
- [ ] Evaluation script runs without errors
- [ ] Baseline metrics calculated (Week 2 system on golden dataset)

**Estimated Time**: 3 hours  
**Blocker Risk**: Tedious golden dataset creation; do 50 queries, not 200

---

## Day 2 (Tuesday): Prompt Versioning & Experiment Tracking

**Theme**: Iterate systematically, not randomly.

### Learning (60 min)
- [ ] Read: [MLflow Tracking](https://mlflow.org/docs/latest/tracking/) — Log experiments, compare runs
- [ ] Read: [Prompt Versioning](https://www.promptingguide.ai/) — Template versioning, A/B testing prompts
- [ ] Question: "If I change a prompt, how do I know if it's better?"

### Build (120 min)
- [ ] Implement `prompting.py`:
  ```python
  class PromptTemplate:
      def __init__(self, name: str, version: str, template: str, description: str)
      def format(self, **kwargs) -> str
      def to_dict(self) -> dict
  
  class PromptRegistry:
      def __init__(self)
      def register(self, prompt: PromptTemplate) -> None
      def get(self, name: str, version: str = "latest") -> PromptTemplate
      def list_versions(self, name: str) -> List[str]
  ```
- [ ] Create `prompts/versions.yaml`:
  ```yaml
  rag_answer:
    v1.0:
      template: "Use the context to answer: {query}\nContext: {context}\nAnswer:"
      description: "Simple RAG prompt"
    v1.1:
      template: "Answer the question based ONLY on the context. Question: {query}\n\nContext:\n{context}\n\nIf the context doesn't contain the answer, say 'Not found in context'.\n\nAnswer:"
      description: "Added instruction to avoid hallucination"
  ```
- [ ] Set up MLflow:
  ```python
  import mlflow
  
  def log_experiment(prompt_version: str, metrics: dict, params: dict):
      with mlflow.start_run():
          mlflow.log_params(params)
          mlflow.log_metrics(metrics)
          mlflow.log_artifact("docs/results.json")
  ```
- [ ] Create experiment script:
  ```python
  def run_prompt_experiment(golden_dataset: List[dict], prompt_versions: List[str]):
      for prompt_v in prompt_versions:
          metrics = evaluate(golden_dataset, prompt_v)
          log_experiment(prompt_v, metrics, {"prompt": prompt_v})
  ```

### Hands-On Deliverable
- [ ] `prompting.py` with PromptRegistry
- [ ] `prompts/versions.yaml` with 3+ prompt versions
- [ ] MLflow running and logging experiments
- [ ] Comparison: v1.0 vs v1.1 on golden dataset

### Success Criteria
- [ ] Prompt versions tracked in YAML (not hardcoded)
- [ ] MLflow UI shows 2+ experiments with different prompts
- [ ] v1.1 shows improvement (even small) over v1.0
- [ ] Can replay any experiment by prompt version

**Estimated Time**: 3 hours  
**Blocker Risk**: MLflow setup complexity; use local backend initially

---

## Day 3 (Wednesday): Metrics, Dashboarding & Cost Tracking

**Theme**: Visibility = Control.

### Learning (60 min)
- [ ] Read: [OpenTelemetry Setup](https://opentelemetry.io/docs/instrumentation/python/getting-started/) — Structured logging, traces
- [ ] Read: [Token Cost Tracking](https://platform.openai.com/docs/guides/tokens) — Calculate cost per request
- [ ] Question: "How do I know if my optimization is saving money or just making it slower?"

### Build (120 min)
- [ ] Implement `metrics.py`:
  ```python
  class Metrics:
      def record_latency(self, component: str, latency_ms: float) -> None
      def record_tokens(self, model: str, input_tokens: int, output_tokens: int) -> None
      def record_accuracy(self, accuracy: float, metric_name: str) -> None
      
      def get_stats(self, component: str) -> dict
      # Returns: {"p50": X, "p90": Y, "p99": Z, "mean": A}
      
      def get_cost_breakdown(self) -> dict
      # Returns: {"retrieval_cost": X, "generation_cost": Y, "total": Z}
  ```
- [ ] Add structured logging to `main.py`:
  ```python
  import logging
  from pythonjsonlogger import jsonlogger
  
  logger = logging.getLogger()
  handler = logging.FileHandler("logs/requests.jsonl")
  handler.setFormatter(jsonlogger.JsonFormatter())
  logger.addHandler(handler)
  
  # Log each request:
  logger.info("query", extra={
      "query": query,
      "retrieval_latency_ms": 50,
      "generation_latency_ms": 1200,
      "tokens_used": 500,
      "cost_usd": 0.005
  })
  ```
- [ ] Create dashboard (simple HTML or use Grafana):
  ```python
  @app.get("/dashboard")
  def get_dashboard() -> dict:
      return {
          "accuracy": 0.85,
          "latency_p99_ms": 1500,
          "cost_per_query": 0.005,
          "cache_hit_rate": 0.35,
          "requests_last_hour": 120
      }
  ```
- [ ] Calculate cost model:
  ```python
  COSTS = {
      "text-embedding-3-small": 0.00002 / 1000,  # per token
      "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000}
  }
  
  def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
      if "embedding" in model:
          return input_tokens * COSTS[model]
      return input_tokens * COSTS[model]["input"] + output_tokens * COSTS[model]["output"]
  ```

### Hands-On Deliverable
- [ ] `metrics.py` with latency + token + cost tracking
- [ ] Structured logging to JSON file
- [ ] Dashboard endpoint returning key metrics
- [ ] Cost report: `docs/cost-analysis.md`

### Success Criteria
- [ ] Every request logged with latency + tokens + cost
- [ ] Latency percentiles (P50, P90, P99) computed
- [ ] Cost per query calculated and reported
- [ ] Dashboard shows accuracy, latency, cost side-by-side
- [ ] Can answer: "What's the cost impact of adding reranking?"

**Estimated Time**: 3 hours  
**Blocker Risk**: JSON logging complexity; use standard logging first, iterate

---

## Day 4 (Thursday): Monitoring & Alerting

**Theme**: Know when things break.

### Learning (60 min)
- [ ] Read: [LLM Observability](https://www.datadog.com/blog/llm-observability/) — What to monitor, red flags
- [ ] Read: [Latency Degradation](https://en.wikipedia.org/wiki/Service-level_objective) — SLO vs reality
- [ ] Question: "If latency suddenly doubles, how do I know? What caused it?"

### Build (120 min)
- [ ] Implement `monitoring.py`:
  ```python
  class Monitor:
      def check_latency_slo(self, latency_ms: float, threshold_ms: int = 2000) -> bool
      def check_accuracy_slo(self, accuracy: float, threshold: float = 0.8) -> bool
      def check_cost_anomaly(self, cost: float, baseline_cost: float) -> bool
      
      def get_alerts(self) -> List[str]:
          alerts = []
          if latency_p99 > SLO_LATENCY:
              alerts.append(f"P99 latency {latency_p99}ms exceeds SLO {SLO_LATENCY}ms")
          if accuracy < MIN_ACCURACY:
              alerts.append(f"Accuracy {accuracy} below minimum {MIN_ACCURACY}")
          return alerts
  ```
- [ ] Add health check endpoint:
  ```python
  @app.get("/health")
  def health_check() -> dict:
      alerts = monitor.get_alerts()
      status = "healthy" if not alerts else "degraded"
      return {
          "status": status,
          "alerts": alerts,
          "latency_p99_ms": metrics.p99_latency(),
          "accuracy": metrics.accuracy()
      }
  ```
- [ ] Create alert thresholds (SLO):
  - Latency: P99 < 2000ms
  - Accuracy: > 80%
  - Cost: < $0.01 per query
  - Cache hit rate: > 30%
- [ ] Write SLO document:
  ```markdown
  # Service Level Objectives
  
  - **Availability**: 99.5% (no more than 3.6 hours/month downtime)
  - **Latency**: P99 < 2s
  - **Accuracy**: > 80% on golden dataset
  - **Cost**: < $0.01 per query
  ```

### Hands-On Deliverable
- [ ] `monitoring.py` with SLO checks
- [ ] `/health` endpoint returning status + alerts
- [ ] `docs/slo.md` defining thresholds
- [ ] Manual alert test (simulate degradation, verify alert)

### Success Criteria
- [ ] Health endpoint shows current status
- [ ] Alerts trigger when SLO violated
- [ ] SLO document signed off (reasonable thresholds)
- [ ] Can explain: "If latency spikes, it's probably [retrieval | generation | DB]"

**Estimated Time**: 3 hours  
**Blocker Risk**: Finding right SLO thresholds (use conservative estimates)

---

## Day 5 (Friday): Documentation & GenAIOps Summary

**Theme**: Package Week 3. Document production knowledge.

### Write Evaluation Report (60 min)
- [ ] Create `docs/evaluation-report.md`:
  ```markdown
  # Evaluation Report: Week 3
  
  ## Golden Dataset
  - Size: 50 queries
  - Categories: definition (20), comparison (15), reasoning (15)
  - Creation process: [link to dataset-creation.md]
  
  ## Metrics (Week 2 baseline vs Week 3)
  | Metric | Week 2 | Week 3 | Change |
  |--------|--------|--------|--------|
  | Accuracy | 0.75 | 0.85 | +10% |
  | Retrieval Precision | 0.80 | 0.85 | +5% |
  | Latency P99 (ms) | 1800 | 1600 | -200ms |
  | Cost per query | $0.006 | $0.005 | -17% |
  
  ## Prompt Experiments
  - v1.0 (baseline): 75% accuracy
  - v1.1 (no hallucination instruction): 80% accuracy
  - v1.2 (with context validation): 85% accuracy ✓ (selected)
  
  ## Monitoring Setup
  - SLO thresholds: [link to slo.md]
  - Health check: GET /health
  - Alerts: Latency, accuracy, cost anomalies
  ```
- [ ] Create GenAIOps checklist:
  ```markdown
  # GenAIOps Checklist: Week 3
  
  - [x] Evaluation framework (ROUGE, BLEU, semantic similarity)
  - [x] Golden dataset (50 curated queries)
  - [x] Prompt versioning (3+ versions in YAML)
  - [x] Experiment tracking (MLflow)
  - [x] Metrics collection (latency, tokens, cost)
  - [x] Structured logging (JSON to file)
  - [x] Dashboard (metrics API endpoint)
  - [x] Health monitoring (SLO checks)
  - [x] Cost tracking (per-query cost calculated)
  - [x] SLO definition (latency, accuracy, cost)
  ```

### Checkpoint Validation (30 min)
- [ ] Can you answer: "How do we know this AI is working?"
  - Example answer: "We evaluate on 50 golden queries. Current accuracy: 85%. P99 latency: 1600ms. Cost: $0.005/query. All within SLO."
- [ ] Can you explain the prompt experiments? (Which version? Why?)
- [ ] Can you identify a degradation and root-cause it?
  - Example: "If accuracy drops to 75%, I'd check: [1] prompt version changed? [2] retrieval quality? [3] vector DB performance?"
- [ ] Do you have a production monitoring dashboard?

### Final Deliverables (30 min)
- [ ] README updated with Week 3 changes
- [ ] `docs/evaluation-report.md` complete
- [ ] `docs/slo.md` with thresholds
- [ ] MLflow experiments saved
- [ ] Commit to `week-3` branch

### Success Criteria
- [ ] 50-query golden dataset created
- [ ] Evaluation framework measures accuracy, retrieval quality, cost
- [ ] Prompt versioning in place; v1.2 beats v1.0
- [ ] Metrics dashboard + health endpoint
- [ ] Cost breakdown available
- [ ] You can justify improvements with data
- [ ] Production SLO defined

**Estimated Time**: 3 hours

---

## Week 3 Checkpoint Validation

**Before moving to Week 4, validate you can:**

1. **Measurement Literacy**: "What's your accuracy on the golden dataset? How was it measured?"
2. **Iteration Discipline**: "Show me how you improved from v1.0 to v1.2 prompt. What changed?"
3. **Production Monitoring**: "If accuracy drops to 70%, what do you check first?"
4. **Cost Awareness**: "Cost per query is $0.005. Can you justify it? Is it acceptable?"

**If you can't answer all 4, revisit evaluation or cost tracking.**

---

## Time Summary

| Day | Activity | Est. Time | Deliverable |
|-----|----------|-----------|-------------|
| Mon | Golden dataset + evaluation | 3h | 50 test queries + eval metrics |
| Tue | Prompt versioning + MLflow | 3h | 3+ prompt versions + experiments |
| Wed | Metrics + dashboarding | 3h | Latency/cost/accuracy tracking |
| Thu | Monitoring + SLO | 3h | Health checks + alerts |
| Fri | Docs + GenAIOps checklist | 3h | Evaluation report + summary |
| **Total** | | **15h** | **Production-Ready Monitoring** |

---

## Success Looks Like

By end of Week 3:
- [ ] 50-query golden dataset with hand-written answers
- [ ] Evaluation script measuring accuracy, retrieval quality, cost
- [ ] Prompt v1.2 outperforms v1.0 (data-driven improvement)
- [ ] Metrics dashboard showing accuracy, latency, cost
- [ ] Health endpoint returns status + alerts
- [ ] SLO document defines acceptable thresholds
- [ ] You can answer: "This system is 85% accurate, 1600ms P99, $0.005/query, within SLO"

**Ready for Week 4?** Commit `week-3`, create `week-4` branch, then focus on architecture + fine-tuning.
