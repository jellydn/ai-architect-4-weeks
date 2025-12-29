# RAG Architecture - Week 1

**Status**: In Progress (Complete by Friday)

## System Diagram

```
User Query
    ↓
[Ingestion Pipeline]
    ├─ Load Document (TXT)
    ├─ Chunk (512 tokens, 50 overlap)
    ├─ Embed (OpenAI text-embedding-3-small)
    └─ Store (In-memory vector store)
    
[Retrieval]
    ├─ Embed Query
    ├─ Cosine Similarity Search
    └─ Return Top-3 Documents
    
[Generation]
    ├─ Format Prompt with Context
    ├─ Call LLM (gpt-3.5-turbo)
    └─ Return Answer + Sources
    
Response
```

## Components

### 1. DocumentIngester
- **Purpose**: Load and chunk documents
- **Input**: File path
- **Output**: List of chunks with metadata
- **Config**: `chunk_size=512`, `chunk_overlap=50`

### 2. RAGRetriever
- **Purpose**: Embed texts and retrieve similar documents
- **Input**: Query string
- **Output**: Ranked list of documents
- **Similarity Metric**: Cosine similarity
- **Caching**: Embeddings cached in memory

### 3. RAGGenerator
- **Purpose**: Generate answers using context
- **Input**: Query + retrieved context
- **Output**: Answer string with sources
- **Model**: gpt-3.5-turbo, temperature=0.7
- **Prompt**: LangChain PromptTemplate

### 4. FastAPI Server
- **Port**: 8000
- **Endpoints**: 
  - `POST /ingest` - Ingest documents
  - `POST /query` - Query RAG system
  - `GET /health` - Health check

## Data Flow Example

```
Input:  {"query": "What is RAG?", "top_k": 3}

1. Ingestion Phase (first time):
   - Load document from file
   - Split into ~10 chunks
   - Embed each chunk (API calls)
   - Store in memory: [id, text, embedding]

2. Retrieval Phase:
   - Embed query (API call)
   - Compute cosine similarity with all chunks
   - Return top-3 by score
   - Time: ~200ms

3. Generation Phase:
   - Format prompt: "Context: [chunk1]\n[chunk2]\n[chunk3]\n\nQuestion: What is RAG?"
   - Call LLM (gpt-3.5-turbo)
   - Parse response
   - Time: ~2000ms

Output: {"answer": "RAG is...", "sources": [...], "latency_ms": 2200}
```

## Metrics to Track

- **Ingestion**: Time to load + chunk + embed (ms)
- **Retrieval**: Time to embed query + search (ms)
- **Generation**: Time to call LLM (ms)
- **Total**: End-to-end latency (ms)
- **Cost**: Tokens × model price

## Trade-Offs & Design Decisions

(To be filled in trade-offs.md)

---

**Next**: Add example latency trace and update after implementation.
