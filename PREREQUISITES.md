# Prerequisites: AI Architect 4-Week Sprint

A self-paced prerequisite checklist. You don't need to master everything—just fill the gaps that matter.

**Given your background** (cloud engineer, Python intermediate, #LearnInPublic):

- You're strong on infrastructure, DevOps, systems thinking
- You need focused AI knowledge, not broad theory
- You learn best by building, not textbooks

**This guide helps you enter Week 1 with zero friction.**

---

## Quick Self-Assessment

Before diving into prep, answer these 5 questions honestly:

1. **Embeddings & Vector Search**: Do you know why cosine similarity is used for text, not Euclidean distance?
2. **RAG Pipeline**: Can you sketch the 4 stages (ingest → chunk → embed → retrieve → generate)?
3. **Tokens**: What's a token? Why does context window matter?
4. **Prompt Injection**: What is it? Why does it matter for RAG?
5. **MLOps**: Have you used experiment tracking (MLflow, Weights & Biases)?

**Your readiness path:**

- **5 YES**: You're ready. Skip prep, start Week 1 Monday. ✅
- **3–4 YES**: Do the **2-day blocker sprint** below. Smart move.
- **1–2 YES**: Do the **2–3 day blocker sprint** at relaxed pace. Don't rush.
- **0 YES**: Do the **full prerequisite checklist** section-by-section.

---

## Full Prerequisite Checklist

Reference guide. Skim, don't memorize.

### 1. Programming & Software Engineering (Review-Level)

**You already have this. Quick alignment only.**

Skills to verify:

- [ ] Python (write simple FastAPI endpoints)
- [ ] Virtual environments (`venv`)
- [ ] Dependency management (`pip`, `requirements.txt`)
- [ ] REST API design (GET/POST/response codes)
- [ ] JSON, YAML config patterns

If weak: [FastAPI docs core sections](https://fastapi.tiangolo.com/) (2 hours)

**Verdict**: You're good. Move on.

---

### 2. Data Handling & Text Processing

**Important for RAG quality. Practical focus.**

Knowledge to have:

- [ ] UTF-8 basics (what it is, why it matters)
- [ ] Tokenization (how text → tokens)
- [ ] Text normalization (lowercasing, removing punctuation, etc.)
- [ ] File formats: TXT, Markdown, PDF (conceptually)
- [ ] Data cleaning pipelines (why they matter)

Resources:

- [Hugging Face: Tokenizers Overview](https://huggingface.co/course/chapter2/4) (30 min)
- [LangChain: Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/) (skim, 30 min)

**Hands-on test**: Load a TXT file, split by sentences, count tokens. Done.

---

### 3. AI/ML Fundamentals (Conceptual Only, No Heavy Math)

**You need this for understanding fine-tuning later.**

Concepts to understand:

- [ ] Model vs parameters vs inference (what these mean)
- [ ] Training vs fine-tuning vs inference (when each happens)
- [ ] Overfitting (what it is, why it matters)
- [ ] Evaluation metrics: accuracy, precision, recall, F1 (high-level)

Resources:

- [Andrew Ng: Machine Learning (Ch. 1–2)](https://www.coursera.org/learn/machine-learning) (3 hours, optional)
- [Hugging Face Course (Ch. 1–3)](https://huggingface.co/course/chapter1) (2 hours, recommended)

**You can skip**: Math proofs, linear algebra, calculus. Not needed.

---

### 4. LLM-Specific Foundations (MANDATORY)

**This is where most engineers struggle. Don't skip.**

Knowledge you MUST have:

- [ ] Tokens & tokenization (how GPT counts inputs)
- [ ] Context window (why it's limited, implications)
- [ ] Prompt structure (system message, user, assistant roles)
- [ ] Temperature & top-p (what they control)
- [ ] Hallucinations (why LLMs make up things, how RAG helps)
- [ ] Prompt injection (what it is, why it's a security issue)

Resources:

- [OpenAI: How to work with tokens](https://platform.openai.com/docs/guides/tokens) (20 min, hands-on)
- [OpenAI: Tokenizer tool](https://platform.openai.com/tokenizer) (10 min, play with it)
- [Anthropic: Understanding Claude](https://www.anthropic.com/news/claude-3-5-haiku) (15 min)
- [Hugging Face: Transformers Course Chapter 1](https://huggingface.co/course/chapter1/1) (30 min)
- [OWASP Top 10 for LLM Apps](https://owasp.org/www-project-top-10-for-large-language-model-applications/) (skim, 20 min)

**Hands-on test**: Use OpenAI's tokenizer. Count tokens in 3 different texts. Understand why the count differs.

**Verdict**: Spend 2–3 hours here. Critical.

---

### 5. Embeddings & Vector Search (Core RAG Skill)

**This is RAG's foundation. Architect-level understanding required.**

Knowledge you MUST have:

- [ ] What embeddings are (text → vector of numbers)
- [ ] Why embeddings capture semantic meaning
- [ ] Cosine similarity vs dot product vs Euclidean distance (and why cosine for text)
- [ ] Vector databases (conceptually: what they do, not how they work internally)
- [ ] Approximate nearest neighbors (ANN) — fast search concept
- [ ] Chunking strategies (fixed size vs semantic vs sliding window)
- [ ] Metadata filtering (why it helps retrieval)

Resources:

- [Pinecone: Vector Databases 101](https://www.pinecone.io/learn/vector-database/) (45 min, excellent)
- [Weaviate: Vector Search Concepts](https://weaviate.io/blog/vector-search-explained) (30 min)
- [LlamaIndex: Node Parsers (Chunking)](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/) (30 min)

**Hands-on test**:

1. Generate embeddings for 3 texts using OpenAI API
2. Compute cosine similarity between them
3. Explain: "Why is similarity high for similar texts?"

**Verdict**: Spend 3 hours here. Core to RAG.

---

### 6. Retrieval-Augmented Generation (RAG)

**The main event. You'll learn hands-on in Week 1, but understand conceptually now.**

Knowledge you MUST have:

- [ ] RAG pipeline: ingest → chunk → embed → store → retrieve → generate
- [ ] Why RAG exists (knowledge cutoff, hallucinations, domain knowledge)
- [ ] RAG failure modes: bad chunking, bad retrieval, bad prompt
- [ ] RAG vs fine-tuning: when each is better
- [ ] Reranking: retrieving top-10, reranking to top-3 (why it helps)

Resources:

- [LlamaIndex: RAG Concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts/) (30 min)
- [Anthropic: RAG vs Fine-Tuning](https://www.anthropic.com/research/long-context) (20 min)
- ["Building RAG Systems" blog](https://blog.llamaindex.ai/) (skim 2–3 posts, 1 hour)

**Hands-on test**: Sketch RAG pipeline on paper. Label each stage. Explain why each matters.

**Verdict**: Spend 2 hours here. You'll build this in Week 1.

---

### 7. Model Fine-Tuning (Light, But Required)

**You'll do this in Week 4. Understand conceptually now, hands-on later.**

Knowledge you MUST have:

- [ ] Pretraining vs fine-tuning (what's the difference)
- [ ] LoRA / PEFT (parameter-efficient fine-tuning, why it matters)
- [ ] When fine-tuning beats RAG (style, domain-specific tasks)
- [ ] Training vs inference cost

Resources:

- [Hugging Face: Fine-tuning Transformers](https://huggingface.co/docs/transformers/training) (30 min, skim)
- [PEFT: LoRA Overview](https://huggingface.co/docs/peft/conceptual_guides/lora) (20 min)

**You can skip**: Math of backpropagation, optimizer details. Not needed.

**Verdict**: Spend 1 hour here. You'll revisit in Week 4.

---

### 8. MLOps / GenAIOps Fundamentals

**Critical differentiator at architect level. You'll implement this in Week 3.**

Knowledge you MUST have:

- [ ] Experiment tracking (MLflow, what it does)
- [ ] Model versioning (why versions matter)
- [ ] Prompt versioning (tracking prompt changes)
- [ ] Evaluation datasets (golden datasets, purpose)
- [ ] Metrics that matter (accuracy, latency, cost)
- [ ] Drift (model/data drift, conceptually)

Resources:

- [MLflow: Quickstart](https://mlflow.org/docs/latest/getting-started/quickstart/index.html) (20 min, hands-on)
- [LangSmith: Concepts](https://docs.smith.langchain.com/) (20 min, skim)
- [OpenTelemetry: Getting Started](https://opentelemetry.io/docs/getting-started/) (20 min, optional)

**Hands-on test**:

1. Log one experiment to MLflow
2. Compare 2 runs side-by-side
3. Export results

**Verdict**: Spend 1.5 hours here. Critical for Week 3.

---

### 9. Cloud & Deployment (You're Strong Here)

**Align your DevOps knowledge to AI workloads.**

Skills to verify:

- [ ] Docker (building images, running containers)
- [ ] Docker Compose (multi-service setup)
- [ ] Environment variables / secrets
- [ ] Cost control patterns
- [ ] Basic Kubernetes (optional)

Resources:

- [FastAPI: Deployment with Docker](https://fastapi.tiangolo.com/deployment/docker/) (20 min)
- [Docker Multi-Stage Builds for ML](https://docs.docker.com/build/building/multi-stage/) (15 min, optional)

**Verdict**: You're strong here. Light review only (30 min).

---

### 10. Responsible AI & Safety (Practical Only)

**Engineering controls, not policy. Week 1–4 minimal, but awareness matters.**

Knowledge to have:

- [ ] PII handling (don't log secrets, user data)
- [ ] Prompt injection defense (input validation)
- [ ] Output filtering (remove harmful content?)
- [ ] Access control (who can query the API?)

Resources:

- [OWASP Top 10 for LLM Apps](https://owasp.org/www-project-top-10-for-large-language-model-applications/) (skim, 30 min)
- [Anthropic: Constitutional AI](https://www.anthropic.com/news/constitutional-ai) (optional, 20 min)

**Verdict**: Spend 30 min here. Important but secondary.

---

### 11. Optional (Don't Block Week 1)

- [ ] LangGraph / agent frameworks
- [ ] ReAct (reasoning + acting)
- [ ] Tool calling / function calling
- [ ] Streaming responses
- [ ] Caching patterns

**These are Week 4+ territory. Ignore for now.**

---

## Lightweight 2–3 Day Blocker Sprint

**For people who answered 1–4 YES on the self-assessment.**

Do this at your own pace. One section per day, or spread across a week. No rush.

### Day 1: LLM Fundamentals + RAG Concepts (4 hours, flexible)

**Goal**: Understand tokens, embeddings, RAG pipeline.

**Learn (2 hours):**

1. Tokens: [OpenAI tokenizer overview](https://platform.openai.com/tokenizer) (20 min, hands-on)
   - Load 3 different prompts
   - Count tokens
   - Notice: longer text = more tokens, special chars = split differently
2. Context window: [Why it matters](https://www.anthropic.com/news/claude-3-5-haiku) (15 min)
   - Understand: fixed input size, cost per token
3. RAG pipeline: [LlamaIndex overview](https://docs.llamaindex.ai/en/stable/getting_started/concepts/) (30 min)
   - Ingest → Chunk → Embed → Store → Retrieve → Generate
   - Diagram it on paper
4. Embeddings 101: [Pinecone Vector Databases 101](https://www.pinecone.io/learn/vector-database/) (45 min)
   - What embeddings are
   - Why cosine similarity (not Euclidean)
   - Why vector databases exist

**Hands-On (2 hours):**

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

# Generate embeddings for 3 texts
texts = [
    "The cat sat on the mat",
    "A feline was on the rug",
    "Python is a programming language"
]

embeddings = []
for text in texts:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embeddings.append(resp.data[0].embedding)

# Cosine similarity
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"Sim(text0, text1): {cosine_sim(embeddings[0], embeddings[1]):.2f}")
print(f"Sim(text0, text2): {cosine_sim(embeddings[0], embeddings[2]):.2f}")
```

**Success Criteria:**

- [ ] You can explain: "Cosine similarity works because..."
- [ ] You understand: "Embeddings capture semantic meaning"
- [ ] You can name: "The 4 stages of RAG"

**Time**: ~4 hours, or split across 2 evenings.

---

### Day 2: Vector Search + Retrieval Patterns (3 hours, flexible)

**Goal**: Hands-on retrieval. Understand chunking + reranking.

**Learn (1.5 hours):**

1. Chunking: [LlamaIndex node parsers](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/) (30 min)
   - Fixed size: 512 tokens, 50 overlap
   - Semantic: chunk by sentence/paragraph
   - Trade-offs
2. Metadata filtering: [Vector Search Concepts](https://weaviate.io/blog/vector-search-explained) (20 min)
   - Filter before vector search
   - Reduce noise
3. Reranking: [Cross-encoders](https://www.sbert.net/examples/applications/cross-encoder/) (30 min)
    - Retrieve top-10, rerank to top-3
    - When it helps, cost

**Hands-On (1.5 hours):**

```python
# Simple in-memory retrieval
from openai import OpenAI
import numpy as np

client = OpenAI()

# Sample documents
docs = [
    {"id": "1", "text": "RAG is retrieval-augmented generation", "source": "blog"},
    {"id": "2", "text": "Fine-tuning adapts models to specific tasks", "source": "docs"},
    {"id": "3", "text": "Embeddings turn text into vectors", "source": "paper"},
]

# Embed all docs
for doc in docs:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc["text"]
    )
    doc["embedding"] = resp.data[0].embedding

# Query
query = "What is RAG?"
query_resp = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
)
query_emb = query_resp.data[0].embedding

# Retrieve top-3
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = [(doc["id"], cosine_sim(query_emb, doc["embedding"])) for doc in docs]
scores.sort(key=lambda x: x[1], reverse=True)

print("Top-3 results:")
for doc_id, score in scores[:3]:
    doc = next(d for d in docs if d["id"] == doc_id)
    print(f"  {doc['id']}: {doc['text']} (score: {score:.2f})")
```

**Success Criteria:**

- [ ] You indexed 10+ documents
- [ ] You retrieved top-3 for a query
- [ ] You understand: "Why metadata filtering helps"

**Time**: ~3 hours, or split across 2 evenings.

---

### Day 3: MLOps + Evaluation (3 hours, flexible)

**Goal**: Set up experiment tracking. Understand golden datasets.

**Learn (1 hour):**

1. MLflow: [Quickstart](https://mlflow.org/docs/latest/getting-started/quickstart/index.html) (20 min)
2. Golden datasets: [What they are](https://www.datarobot.com/wiki/golden-dataset/) (15 min)
3. RAG metrics: [Accuracy, precision, recall](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/) (20 min)

**Hands-On (2 hours):**

```bash
# Install MLflow
pip install mlflow

# Start UI
mlflow ui
```

```python
import mlflow

# Create a golden dataset (5 Q&A pairs)
golden_dataset = [
    {"query": "What is RAG?", "expected": "RAG is retrieval-augmented generation"},
    {"query": "What is fine-tuning?", "expected": "Fine-tuning adapts a model to a specific task"},
    # ... 3 more
]

# Simulate model responses
def simple_rag(query):
    # Placeholder: in Week 1, this will be real
    return f"Answer to: {query}"

# Log experiment
mlflow.set_experiment("week-0-baseline")

with mlflow.start_run():
    correct = 0
    for qa in golden_dataset:
        answer = simple_rag(qa["query"])
        if answer == qa["expected"]:
            correct += 1

    accuracy = correct / len(golden_dataset)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("model", "baseline")

print(f"Accuracy: {accuracy}")
```

**Success Criteria:**

- [ ] MLflow UI running (http://localhost:5000)
- [ ] You logged 1 experiment with 1 metric
- [ ] You understand: "Golden datasets are hand-curated test sets"

**Time**: ~3 hours, or split across 2 evenings.

---

## Minimal Resource List (No Overload)

**Pick and choose. Don't try to read everything.**

| Topic                | Resource                                                                                            | Time   | Format      | Priority     |
| -------------------- | --------------------------------------------------------------------------------------------------- | ------ | ----------- | ------------ |
| **Tokens**           | [OpenAI Tokenizer](https://platform.openai.com/tokenizer)                                           | 20 min | Interactive | MUST         |
| **Embeddings**       | [Pinecone 101](https://www.pinecone.io/learn/vector-database/)                                      | 45 min | Blog        | MUST         |
| **RAG Pipeline**     | [LlamaIndex Concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts/)               | 30 min | Docs        | MUST         |
| **Chunking**         | [LlamaIndex Node Parsers](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/) | 30 min | Docs        | SHOULD       |
| **Reranking**        | [SBERT Reranking](https://www.sbert.net/examples/applications/cross-encoder/)                      | 30 min | Blog        | SHOULD       |
| **MLflow**           | [Quickstart](https://mlflow.org/docs/latest/getting-started/quickstart/index.html)                  | 20 min | Docs        | MUST         |
| **Golden Datasets**  | [DataRobot Wiki](https://www.datarobot.com/wiki/golden-dataset/)                                    | 15 min | Wiki        | SHOULD       |
| **Prompt Injection** | [OWASP Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)         | 20 min | Docs        | NICE-TO-HAVE |

**Total reading + hands-on: ~5–6 hours.**

---

## Recommended Path for You

**Given: Cloud engineer background, Python intermediate, #LearnInPublic mindset**

### Path A: Skip Prep, Start Week 1 (Fast Track)

- Assume you know 3+ answers from self-assessment
- Week 1 Day 1 covers learning + building simultaneously
- You'll fill gaps just-in-time
- **When**: Monday, start Week 1
- **Risk**: May need to re-read things mid-implementation (acceptable)

### Path B: 1-Week Relaxed Prep (Smart)

- Spend 30 min–1 hour per day on blockers
- Hands-on sections on weekends
- Start Week 1 Monday of next week
- **When**: This week, 30 min daily + 2 hours weekend
- **Benefit**: Smoother Week 1, less stopping to learn

### Path C: 3-Day Focused Prep (Balanced)

- Do Days 1–3 of blocker sprint (concentrated)
- Start Week 1 Thursday or Friday
- **When**: Mon/Tue/Wed this week, 4–5 hours each day
- **Benefit**: Ready and confident

---

## How to Use This Doc

1. **Read** the self-assessment (5 min)
2. **Choose** your readiness path (A, B, or C)
3. **Do prep** at your pace (or skip it)
4. **Reference** this doc as needed during Week 1–4
5. **Update** this doc if you find better resources

---

## Check-In Questions

After prep (or after Day 1 of Week 1), ask yourself:

- [ ] Can I explain RAG in 2 minutes?
- [ ] Can I compute cosine similarity between two texts?
- [ ] Do I know why context window matters?
- [ ] Can I sketch the RAG pipeline?
- [ ] Do I know what MLflow does?

**If YES to all 5**: You're ready for Week 1. Go.
**If NO to any**: Spend 30 min on that topic before starting Week 1.

---

## FAQ

**Q: Do I need to finish all prerequisites before Week 1?**
A: No. Week 1 Day 1 covers learning. You can prep in parallel.

**Q: What if I don't understand embeddings after reading?**
A: Do the hands-on (generate embeddings, compute similarity). Understanding comes from doing.

**Q: Can I skip the MLOps section?**
A: Not recommended. Week 3 assumes you know MLflow. But you can learn it in Week 3 if needed.

**Q: What if I have stronger ML background and want to go deeper?**
A: Skip this guide. Start Week 1 immediately. Reference Week-specific links for depth.

**Q: Is there a video version of this?**
A: Not included here. Use the blog/doc resources; most have video options.

---

## Next Steps

1. **Answer the 5 self-assessment questions** (right now, 5 min)
2. **Choose your path** (A, B, or C)
3. **Do prep at your pace** (or skip it)
4. **Move to WEEK-1-CHECKLIST.md** when ready

You've got this. Build in public. 🚀

---

**Last Updated**: 2025-12-28
**Version**: 1.0
