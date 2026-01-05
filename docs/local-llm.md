# Running with Local/Free LLMs

This guide covers alternatives to OpenAI for running the RAG system locally or with free APIs.

## Option 1: Ollama (Recommended for Local)

[Ollama](https://ollama.ai) runs LLMs locally with an OpenAI-compatible API.

### Setup

```bash
# Install Ollama (macOS)
brew install ollama

# Or download from https://ollama.ai

# Start Ollama server
ollama serve

# Pull models
ollama pull llama3.2          # LLM (3B params, fast)
ollama pull nomic-embed-text  # Embeddings
```

### Configuration

```bash
# .env for Ollama
OPENAI_API_KEY=ollama          # Any value works
OPENAI_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2
EMBEDDING_MODEL=nomic-embed-text
```

### Code Changes

The OpenAI client supports custom base URLs:

```python
from openai import OpenAI

client = OpenAI(
    api_key="ollama",  # Required but unused
    base_url="http://localhost:11434/v1"
)
```

---

## Option 2: LM Studio

[LM Studio](https://lmstudio.ai) provides a GUI for running local LLMs with OpenAI-compatible API.

### Setup

1. Download from https://lmstudio.ai
2. Download a model (e.g., Llama 3.2, Mistral)
3. Start the local server (port 1234 by default)

### Configuration

```bash
# .env for LM Studio
OPENAI_API_KEY=lm-studio
OPENAI_BASE_URL=http://localhost:1234/v1
LLM_MODEL=local-model
```

---

## Option 3: Groq (Free Cloud API)

[Groq](https://groq.com) offers fast inference with a generous free tier.

### Setup

1. Sign up at https://console.groq.com
2. Create an API key

### Configuration

```bash
# .env for Groq
GROQ_API_KEY=gsk_...
LLM_MODEL=llama-3.3-70b-versatile
```

### Code Changes

```python
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}]
)
```

**Note**: Groq doesn't support embeddings. Use Ollama or OpenAI for embeddings.

---

## Option 4: Together AI (Free Credits)

[Together AI](https://together.ai) provides $5 free credits.

### Setup

1. Sign up at https://together.ai
2. Get API key from dashboard

### Configuration

```bash
# .env for Together AI
TOGETHER_API_KEY=...
OPENAI_BASE_URL=https://api.together.xyz/v1
LLM_MODEL=meta-llama/Llama-3.2-3B-Instruct-Turbo
EMBEDDING_MODEL=togethercomputer/m2-bert-80M-8k-retrieval
```

---

## Option 5: OpenRouter (Aggregator)

[OpenRouter](https://openrouter.ai) aggregates multiple providers with some free models.

### Free Models Available

- `meta-llama/llama-3.2-3b-instruct:free`
- `google/gemma-2-9b-it:free`
- `mistralai/mistral-7b-instruct:free`

### Configuration

```bash
# .env for OpenRouter
OPENAI_API_KEY=sk-or-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=meta-llama/llama-3.2-3b-instruct:free
```

---

## Local Embeddings with Sentence Transformers

For fully offline embeddings without any API:

### Setup

```bash
uv pip install sentence-transformers
```

### Code

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dims, fast
embeddings = model.encode(["text to embed"])
```

### Add to retrieval.py

```python
class LocalEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()
```

---

## Comparison

| Provider | Cost | Latency | Offline | Embeddings |
|----------|------|---------|---------|------------|
| **Ollama** | Free | ~1-5s | ✅ Yes | ✅ Yes |
| **LM Studio** | Free | ~1-5s | ✅ Yes | ❌ No |
| **Groq** | Free tier | ~200ms | ❌ No | ❌ No |
| **Together AI** | $5 free | ~500ms | ❌ No | ✅ Yes |
| **OpenRouter** | Some free | ~500ms | ❌ No | ❌ No |
| **OpenAI** | Paid | ~500ms | ❌ No | ✅ Yes |

---

## Recommended Setup for Learning

**Fully Local (No API costs)**:
```bash
# Install Ollama
brew install ollama
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text

# Configure
cat > .env << EOF
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2
EMBEDDING_MODEL=nomic-embed-text
EOF
```

**Hybrid (Free cloud LLM + Local embeddings)**:
```bash
# Use Groq for fast LLM, local for embeddings
GROQ_API_KEY=gsk_...
USE_LOCAL_EMBEDDINGS=true
```

---

## Hardware Requirements (Local)

| Model | RAM Required | GPU VRAM |
|-------|--------------|----------|
| Llama 3.2 3B | 4GB | 3GB |
| Llama 3.1 8B | 8GB | 6GB |
| Mistral 7B | 8GB | 6GB |
| Llama 3.1 70B | 48GB | 40GB |

For Apple Silicon Macs, Ollama uses Metal for GPU acceleration automatically.
