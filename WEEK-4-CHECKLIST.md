# Week 4: Architecture, Fine-Tuning & Positioning — Daily Checklist

**Goal**: Architect mindset + public proof.  
**Deliverables**: Fine-tuned model + comparison results + system architecture v2 + public artifact  
**Checkpoint**: You can design an AI system, not just implement one.

---

## Day 1 (Monday): Fine-Tuning Foundations & Model Selection

**Theme**: When RAG isn't enough.

### Learning (90 min)
- [ ] Read: [Fine-Tuning vs RAG](https://www.anthropic.com/news/claude-3-5-haiku) — Decision framework
- [ ] Read: [LoRA / PEFT](https://huggingface.co/docs/peft/index) — Low-rank adaptation, memory efficiency
- [ ] Read: [Fine-Tuning API](https://platform.openai.com/docs/guides/fine-tuning) — OpenAI vs open-source models
- [ ] Read: [Cost vs Accuracy Trade-Off](https://arxiv.org/abs/2305.10350) — When is fine-tuning ROI positive?
- [ ] Question: "For my task, does fine-tuning beat RAG? What data do I need?"

### Model Selection (60 min)
- [ ] Evaluate fine-tuning targets:
  - **Option 1**: GPT-3.5-turbo via OpenAI (paid, fast, proprietary)
  - **Option 2**: Mistral-7B via Hugging Face (free, open-source, local)
  - **Option 3**: Llama-2-7B via Meta (free, open-source, local)
- [ ] Task selection (pick ONE):
  - **Classification**: Categorize support tickets or documents (easiest)
  - **Summarization**: Condense long texts to key points (medium)
  - **Q&A Refinement**: Improve RAG answers for specific domain (hardest)
- [ ] Decision matrix:
  ```markdown
  | Criterion | GPT-3.5 | Mistral | Llama |
  |-----------|---------|---------|-------|
  | Speed | Fast | Medium | Slow |
  | Cost | $$$ | Free | Free |
  | Accuracy | High | Medium | Medium |
  | Local? | No | Yes | Yes |
  | Effort | Low | High | High |
  ```
- [ ] **Recommendation**: Start with classification on Mistral-7B (free, clear task, achievable)

### Hands-On Spike (60 min)
- [ ] Set up fine-tuning environment:
  ```bash
  pip install transformers datasets peft torch accelerate
  ```
- [ ] Download Mistral-7B or Llama-2-7B (will take 30+ min, start in background)
- [ ] Sketch out task: "I want to fine-tune X to do Y"

### Success Criteria
- [ ] You've chosen fine-tuning task (classification or summarization)
- [ ] You've chosen model (Mistral-7B recommended)
- [ ] You understand: "RAG handles knowledge, fine-tuning handles style/task"
- [ ] Model weights downloaded (or queued)

**Estimated Time**: 3.5 hours  
**Blocker Risk**: Large model download time; run in background

---

## Day 2 (Tuesday): Fine-Tuning Data Preparation & Training

**Theme**: Data in, model out.

### Learning (45 min)
- [ ] Read: [Dataset Formatting](https://huggingface.co/docs/transformers/training) — JSON, CSV, format requirements
- [ ] Read: [Training Parameters](https://huggingface.co/docs/transformers/en/main_classes/trainer) — Learning rate, epochs, batch size
- [ ] Question: "How much data do I need? What if I only have 500 examples?"

### Build Fine-Tuning Dataset (90 min)
- [ ] Create `data/fine-tuning-dataset.json`:
  - **Classification**: 500+ examples of [text, label] pairs
  - **Summarization**: 500+ examples of [long_text, short_summary] pairs
- [ ] If using your golden dataset: Create pairs from Q&A
  ```json
  {
    "text": "...",
    "label": "..."
  }
  ```
- [ ] Split: 80% train, 10% val, 10% test
- [ ] Analyze: Class distribution, average length, token count
- [ ] Document: `docs/fine-tuning-dataset.md` (source, size, quality notes)

### Train Fine-Tuned Model (75 min)
- [ ] Create `week-4/fine_tune.py`:
  ```python
  from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
  from datasets import load_dataset
  
  def train_model(model_name: str, dataset_path: str, output_dir: str):
      tokenizer = AutoTokenizer.from_pretrained(model_name)
      model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=N_CLASSES)
      
      dataset = load_dataset("json", data_files=dataset_path)
      
      training_args = TrainingArguments(
          output_dir=output_dir,
          num_train_epochs=3,
          learning_rate=2e-5,
          per_device_train_batch_size=8,
          per_device_eval_batch_size=8,
          warmup_steps=100,
          save_steps=100,
          eval_strategy="steps"
      )
      
      trainer = Trainer(
          model=model,
          args=training_args,
          train_dataset=dataset["train"],
          eval_dataset=dataset["validation"]
      )
      
      trainer.train()
      model.save_pretrained(output_dir)
  ```
- [ ] Run training on test dataset (start small: 100 examples for 30 min validation)
- [ ] Log: Training loss, validation loss, final metrics

### Hands-On Deliverable
- [ ] Fine-tuning dataset with 500+ examples
- [ ] Training script complete
- [ ] Fine-tuned model checkpoint saved
- [ ] Training log showing loss curves

### Success Criteria
- [ ] Fine-tuning dataset created and validated
- [ ] Training runs without CUDA/memory errors (may be slow; that's OK)
- [ ] Validation loss decreases over epochs
- [ ] Model checkpoint saved to disk

**Estimated Time**: 3.5 hours  
**Blocker Risk**: VRAM issues (use smaller model or batch_size=4); slow training (expected)

---

## Day 3 (Wednesday): Comparison & Architecture Abstraction

**Theme**: Prove fine-tuning is better. Abstract the choice.

### Learning (45 min)
- [ ] Review: Week 2 evaluation metrics (your baseline)
- [ ] Read: [A/B Testing ML Models](https://microsoft.github.io/responsible-ai-dashboard/) — Statistical significance, sample size

### Evaluation (90 min)
- [ ] Create comparison script: `week-4/compare_models.py`
  ```python
  def compare_base_vs_finetuned(test_set: List[dict]):
      base_model_scores = evaluate_model(base_model, test_set)
      finetuned_model_scores = evaluate_model(finetuned_model, test_set)
      
      return {
          "base": base_model_scores,
          "finetuned": finetuned_model_scores,
          "improvement": {
              "accuracy": finetuned_model_scores["accuracy"] - base_model_scores["accuracy"],
              "f1": finetuned_model_scores["f1"] - base_model_scores["f1"]
          }
      }
  ```
- [ ] Run comparison on test set
- [ ] Measure: Accuracy, F1, latency, memory usage
- [ ] Create comparison table:
  ```markdown
  | Metric | Base Model | Fine-Tuned | Improvement |
  |--------|-----------|-----------|-------------|
  | Accuracy | 0.75 | 0.88 | +13% |
  | F1 Score | 0.70 | 0.85 | +15% |
  | Latency (ms) | 800 | 850 | -50ms (expected) |
  | Memory (MB) | 2000 | 2500 | +500MB |
  ```
- [ ] Document: `docs/fine-tuning-results.md`

### Architecture Abstraction (90 min)
- [ ] Create `week-4/model_factory.py`:
  ```python
  class ModelConfig:
      def __init__(self, model_type: str, model_name: str, **kwargs):
          self.model_type = model_type  # "base", "finetuned"
          self.model_name = model_name
          self.kwargs = kwargs
  
  class ModelFactory:
      def get_model(self, config: ModelConfig):
          if config.model_type == "base":
              return load_base_model(config.model_name)
          elif config.model_type == "finetuned":
              return load_finetuned_model(config.model_name, config.kwargs)
      
      def predict(self, model, text: str) -> str
  ```
- [ ] Create `config/model-config.yaml`:
  ```yaml
  models:
    production:
      model_type: finetuned  # or "base"
      model_name: ./models/mistral-finetuned
      quantization: int8  # optional
    
    baseline:
      model_type: base
      model_name: mistral-community/Mistral-7B-v0.1
  
  inference:
    batch_size: 8
    temperature: 0.7
    max_tokens: 512
  ```
- [ ] Update `main.py` to use config:
  ```python
  config = load_yaml("config/model-config.yaml")
  model = ModelFactory.get_model(config.models.production)
  ```

### Hands-On Deliverable
- [ ] Comparison table: base vs fine-tuned
- [ ] `model_factory.py` with abstraction
- [ ] `config/model-config.yaml` with production + baseline
- [ ] Test: Switch models via config (no code change)

### Success Criteria
- [ ] Fine-tuned model shows improvement (even small; e.g., +5% accuracy)
- [ ] Comparison documented with data
- [ ] Model selection is config-driven (not hardcoded)
- [ ] Can switch models by editing YAML

**Estimated Time**: 3 hours  
**Blocker Risk**: Fine-tuning may not show improvement (still OK; document why)

---

## Day 4 (Thursday): Architecture Diagram v2 & Design Patterns

**Theme**: You're an architect now. Design the whole system.

### Learning (45 min)
- [ ] Review: All previous architecture diagrams
- [ ] Read: [LLM Service Patterns](https://cloud.google.com/architecture/devops-patterns-for-llm-systems) — Common architectures
- [ ] Question: "What changes if we scale to 100k queries/day?"

### Design System Architecture v2 (90 min)
- [ ] Create `docs/architecture-v2.md` with comprehensive diagram:
  ```
  graph TB
    User["User / API Client"]
    
    subgraph "Request Processing"
      Cache["Query Cache"]
      Validator["Input Validator"]
    end
    
    subgraph "Retrieval"
      Query["Query Embedding"]
      VectorDB["Weaviate<br/>(Metadata Filtering)"]
      Reranker["Cross-Encoder<br/>Reranker"]
    end
    
    subgraph "Generation"
      ModelFactory["Model Factory<br/>(Config-Driven)"]
      BaseModel["Base Model<br/>(Mistral-7B)"]
      FineTuned["Fine-Tuned Model<br/>(Classification)"]
      Generator["Prompt Template<br/>+ Context Fusion"]
    end
    
    subgraph "Observability"
      Metrics["Metrics<br/>(Latency, Cost)"]
      Monitoring["Health Monitor<br/>(SLO Checks)"]
      Logging["Structured Logging"]
    end
    
    User --> Cache
    Cache --> Validator
    Validator --> Query
    Query --> VectorDB
    VectorDB --> Reranker
    Reranker --> ModelFactory
    ModelFactory --> BaseModel | FineTuned
    BaseModel --> Generator
    FineTuned --> Generator
    Generator --> Metrics
    Metrics --> Monitoring
    Monitoring --> Logging
    Logging --> User
  ```
- [ ] Annotate decision points:
  - Why Weaviate + reranking (vs Pinecone)?
  - Why config-driven models (vs hardcoded)?
  - Why this caching strategy?
  - Why these SLO thresholds?
- [ ] Include scaling notes:
  - 100k queries/day → what breaks?
  - Solution: async workers, multi-region, caching
  - Cost model at scale

### Design Trade-Off Matrix (45 min)
- [ ] Create `docs/system-design-decisions.md`:
  ```markdown
  # System Design Decisions
  
  ## Retrieval: Weaviate vs Pinecone vs Milvus
  | Factor | Weaviate | Pinecone | Milvus |
  |--------|----------|----------|--------|
  | Local dev | Yes | No | Yes |
  | Cost | Free | $$$ | Free |
  | Reranking | Built-in | No | No |
  | Production | ✓ Chosen | Expensive | Overkill |
  
  **Decision**: Weaviate (local dev + reranking)
  
  ## Model: Base vs Fine-Tuned
  | Scenario | Choice | Reason |
  |----------|--------|--------|
  | Knowledge Q&A | Base + RAG | RAG is more maintainable |
  | Domain classification | Fine-tuned | Task-specific accuracy |
  | **Our system** | Both | Classification + RAG for Q&A |
  
  **Decision**: Config-driven factory (switch at runtime)
  
  ## Deployment: Serverless vs Containers vs Local
  | Constraint | Serverless | Container | Local |
  |-----------|-----------|-----------|-------|
  | Scale to 0 | ✓ | ✗ | ✗ |
  | Cold start | Slow | Fast | Instant |
  | Cost at scale | $ | $$ | $$$ |
  | **Week 4** | POC | Docker | Dev only |
  
  **Decision**: Docker for Week 4, serverless when scaling
  ```

### Architecture Review Checklist (30 min)
- [ ] Can you explain the entire system in 5 minutes?
- [ ] Can you identify the critical path (slowest component)?
- [ ] Can you justify every major choice (no "because tutorial")?
- [ ] Can you describe scaling bottlenecks?

### Hands-On Deliverable
- [ ] `docs/architecture-v2.md` with comprehensive diagram
- [ ] `docs/system-design-decisions.md` with trade-off matrices
- [ ] Diagram is Mermaid (clickable + citations)

### Success Criteria
- [ ] Architecture diagram is clear, detailed, and well-annotated
- [ ] Every component has a reason (retrieval, generation, monitoring)
- [ ] Design decisions document explains trade-offs
- [ ] You can defend every choice

**Estimated Time**: 3 hours

---

## Day 5 (Friday): Public Artifact & Portfolio

**Theme**: Ship it. Share it. Own it.

### Create Public Artifact (120 min)
**Pick ONE:**

**Option A: Blog Post (Recommended)**
- [ ] Create `blog-post.md`:
  ```markdown
  # From RAG to Production AI: What I Built in 4 Weeks
  
  ## The Goal
  Build a production-ready AI system from scratch. Go from "hello world" to "here's my system architecture."
  
  ## Week 1: RAG Foundation
  [Brief recap of what RAG is, why I chose it]
  
  ## Week 2: Production Thinking
  [Vector DB, reranking, caching—why these choices]
  
  ## Week 3: Evaluation & Monitoring
  [How I know the system is working; metrics that matter]
  
  ## Week 4: Architecture & Fine-Tuning
  [When fine-tuning beats RAG; system design decisions]
  
  ## Key Learnings
  1. RAG is not one-size-fits-all
  2. Evaluation must come before optimization
  3. Architecture decisions are trade-offs, not absolutes
  4. Production thinking beats "clever code"
  
  ## Results
  - Accuracy: 85% on golden dataset
  - Latency: P99 < 1600ms
  - Cost: $0.005/query
  - All within SLO
  
  ## Code
  [Link to GitHub repo]
  ```
- [ ] Publish on Medium, Dev.to, or personal blog

**Option B: Video Walkthrough (3–5 min)**
- [ ] Script:
  - System demo (run a query)
  - Architecture diagram walkthrough
  - Key decisions (retrieval, evaluation, fine-tuning)
  - Results (metrics dashboard)
- [ ] Record with screen capture
- [ ] Upload to YouTube, share link in README

**Option C: GitHub Discussion or Tweet Thread**
- [ ] Write 5–10 tweets:
  1. "Built a production AI system in 4 weeks. Here's what I learned."
  2. [RAG vs fine-tuning]
  3. [Evaluation framework]
  4. [Monitoring setup]
  5. [Final system architecture]
  6. [Key results / metrics]
  7. [Open-source: link to repo]

### Final Documentation (60 min)
- [ ] Update root `README.md`:
  ```markdown
  # AI Architect 4-Week Sprint
  
  ## Overview
  Production-grade AI system built in 4 weeks. From RAG foundation to fine-tuning and monitoring.
  
  ## Architecture
  [Link to architecture-v2.md + diagram]
  
  ## Results
  - Accuracy: 85%
  - Latency: P99 < 1600ms
  - Cost: $0.005/query
  
  ## Key Components
  - **Week 1**: RAG foundation (ingestion, retrieval, generation)
  - **Week 2**: Production RAG (Weaviate, reranking, caching)
  - **Week 3**: Evaluation & monitoring (golden dataset, MLflow, SLO)
  - **Week 4**: Fine-tuning & architecture (LoRA, model factory, design decisions)
  
  ## Getting Started
  ```bash
  git clone https://github.com/[user]/ai-architect-4weeks
  cd ai-architect-4weeks
  
  # Setup
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  
  # Run
  docker-compose up
  
  # Query
  curl -X POST http://localhost:8000/query -d '{"query": "..."}' -H "Content-Type: application/json"
  ```
  
  ## Documentation
  - [Constitution](CONSTITUTION.md) — Core principles
  - [Week 1 Checklist](WEEK-1-CHECKLIST.md) — RAG foundation
  - [Week 2 Checklist](WEEK-2-CHECKLIST.md) — Production RAG
  - [Week 3 Checklist](WEEK-3-CHECKLIST.md) — Evaluation & monitoring
  - [Week 4 Checklist](WEEK-4-CHECKLIST.md) — Fine-tuning & architecture
  - [Architecture v2](docs/architecture-v2.md) — System design
  - [Design Decisions](docs/system-design-decisions.md) — Trade-off analysis
  
  ## Portfolio Value
  This repo demonstrates:
  - ✅ LLM system design (RAG, fine-tuning, evaluation)
  - ✅ Production thinking (monitoring, cost, SLO)
  - ✅ Architecture rigor (decision documentation, trade-off analysis)
  - ✅ Engineering discipline (testing, logging, metrics)
  
  Suitable for: AI Architect, Senior AI Engineer, ML Systems Engineer roles.
  
  ## Author
  [Your name] | [LinkedIn] | [Twitter]
  ```
- [ ] Create `CAREER.md`:
  ```markdown
  # How This Project Positions You for AI Architect Roles
  
  ## What Interviewers Will See
  - **System Design**: You explain architecture decisions, not just implement code
  - **Production Thinking**: You measure everything; optimization is data-driven
  - **Evaluation Rigor**: You have a golden dataset; you know your metrics
  - **Trade-Off Analysis**: You understand cost, latency, accuracy trade-offs
  - **Code Quality**: Tests, logging, error handling, documentation
  
  ## Interview Talking Points
  1. "I built a RAG system and measured when fine-tuning beats it."
  2. "My evaluation framework has 50 golden queries; accuracy is 85%."
  3. "Latency is P99 < 1600ms; cost is $0.005/query; all within SLO."
  4. "Here's my system architecture; here's why I chose Weaviate over Pinecone."
  5. "I can debug 'bad answers' by tracing retrieval, not guessing prompts."
  
  ## CV Bullet Points
  - Designed and implemented production-grade RAG system (Week 1–2)
  - Built evaluation framework measuring accuracy, latency, and cost (Week 3)
  - Evaluated fine-tuning vs RAG trade-offs; implemented model abstraction layer (Week 4)
  - Implemented vector database, reranking, caching, and monitoring (Weaviate, OpenTelemetry)
  - Achieved 85% accuracy on golden dataset with P99 latency < 1.6s and $0.005/query cost
  
  ## LinkedIn Messaging
  "Just shipped a production AI system in 4 weeks. Built RAG → optimized retrieval → added evaluation & monitoring → fine-tuned models. Focus: architecture rigor, not tutorial code. [Link to repo + blog]"
  ```

### Final Checkpoint (30 min)
- [ ] Can you design a new AI system from scratch? (Whiteboard test)
  - User says: "Build an AI system for X"
  - You: "I'd use [RAG | fine-tuning | retrieval | generation] because..."
  - Diagram the architecture, justify choices
- [ ] Can you pitch yourself as an AI Architect?
  - "I built this system in 4 weeks. Here's my approach to system design."
  - Point to architecture, evaluation, monitoring
- [ ] Is repo portfolio-ready?
  - Clean commits
  - Clear README + docs
  - No TODO comments or broken code
  - Public GitHub profile link

### Success Criteria
- [ ] Public artifact published (blog, video, or Twitter thread)
- [ ] GitHub repo is public and portfolio-ready
- [ ] README + docs are clear and comprehensive
- [ ] You can explain the entire system in 5 minutes
- [ ] You can justify every architectural choice

**Estimated Time**: 3 hours

---

## Week 4 Checkpoint Validation

**Before considering Week 4 complete, validate you can:**

1. **Design from Scratch**: "Design an AI system for customer support automation." (5 min design + justification)
2. **Explain Architecture**: Point to v2 diagram and trace a request (3 min)
3. **Defend Choices**: "Why Weaviate? Why this chunking? Why this SLO?" (answer with data)
4. **Own the Results**: "My system achieves 85% accuracy, P99 < 1600ms, $0.005/query—why is this good?"
5. **Pitch Yourself**: "I'm ready for AI Architect roles because..." (1 min pitch)

**If you can't answer all 5, spend extra time on architecture or public artifact.**

---

## Time Summary

| Day | Activity | Est. Time | Deliverable |
|-----|----------|-----------|-------------|
| Mon | Fine-tuning foundations | 3.5h | Model selection + setup |
| Tue | Data prep + training | 3.5h | Fine-tuned model checkpoint |
| Wed | Comparison + abstraction | 3h | Model factory + comparison results |
| Thu | Architecture v2 + design | 3h | System diagram + trade-offs |
| Fri | Public artifact + portfolio | 3h | Blog/video + GitHub-ready repo |
| **Total** | | **16h** | **Architect-Ready Portfolio** |

---

## Success Looks Like

By end of Week 4:
- [ ] Fine-tuned model trained and compared against base
- [ ] Comparison shows improvement (even small; e.g., +5%)
- [ ] Model selection is config-driven (factory pattern)
- [ ] Architecture v2 diagram is clear, detailed, annotated
- [ ] Design decisions document explains trade-offs
- [ ] Public artifact published (blog, video, or thread)
- [ ] GitHub repo is public, portfolio-ready, with comprehensive docs
- [ ] You can pitch yourself as an AI Architect (not just a coder)

---

## Post-Week 4: Applying for AI Architect Roles

**What you have:**
1. **Running system**: RAG + fine-tuning + monitoring
2. **Architect mindset**: Design decisions documented + trade-off analysis
3. **Production rigor**: Evaluation framework + metrics dashboard + SLO
4. **Public proof**: GitHub repo + blog/video demonstrating capability

**Next steps:**
1. Update CV with specific metrics (accuracy, latency, cost)
2. Reach out to recruiters with repo link
3. In interviews, discuss architecture decisions (not implementation details)
4. Answer "design an AI system" questions with confidence

**You are now credible for AI Architect / Senior AI Engineer roles.**
