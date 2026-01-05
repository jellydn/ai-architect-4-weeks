# Data Model: Week 1 RAG System

**Date**: 2025-12-29  
**Status**: Complete  
**Scope**: Document ingestion, embedding, retrieval, and answer generation

---

## Core Entities

### Document

Represents a source file with its metadata.

**Fields**:
- `filepath` (string, required): Absolute path to source file
- `content` (string, required): Raw text content from file
- `loaded_at` (ISO 8601 timestamp, auto): When document was loaded

**Validation**:
- `filepath` must exist and be readable (UTF-8)
- `content` length > 0 (skip empty files)

**Relationships**:
- Document → Chunk (1:many, cascading on delete)

**Example**:
```json
{
  "filepath": "data/sample.txt",
  "content": "Retrieval-Augmented Generation (RAG) is...",
  "loaded_at": "2025-12-29T10:30:00Z"
}
```

---

### Chunk

A fixed-size segment of a document with overlap for context preservation.

**Fields**:
- `id` (string, required, PK): Unique identifier `{filepath}:{chunk_index}`
- `text` (string, required): Chunk text (up to `chunk_size` characters)
- `source` (string, required): Filepath this chunk came from (FK → Document)
- `chunk_index` (integer, required): Sequential position in document (0-based)
- `start_char` (integer, optional): Starting character offset in original document
- `end_char` (integer, optional): Ending character offset in original document
- `created_at` (ISO 8601 timestamp, auto): When chunk was created

**Validation**:
- `text` length > 0 (skip empty chunks)
- `text` length ≤ `chunk_size` (default 512 characters)
- `chunk_index` ≥ 0
- `start_char` < `end_char` (if provided)

**Relationships**:
- Chunk → Document (many:1, FK on `source`)
- Chunk → Embedding (1:1, by chunk text)

**Example**:
```json
{
  "id": "data/sample.txt:0",
  "text": "Retrieval-Augmented Generation (RAG) is a technique...",
  "source": "data/sample.txt",
  "chunk_index": 0,
  "start_char": 0,
  "end_char": 512,
  "created_at": "2025-12-29T10:30:01Z"
}
```

---

### Embedding

Dense vector representation of chunk text for similarity search.

**Fields**:
- `chunk_id` (string, required, FK): Reference to Chunk.id
- `model` (string, required): Embedding model name (e.g., "text-embedding-3-small")
- `vector` (array[float], required): 1536-dimensional vector for text-embedding-3-small
- `created_at` (ISO 8601 timestamp, auto): When embedding was generated
- `cached` (boolean, optional): True if loaded from cache (not re-computed)

**Validation**:
- `vector` length = 1536 (for text-embedding-3-small)
- `vector[i]` is float ∈ [-1, 1] (typically normalized)
- `model` is known embedding model

**Relationships**:
- Embedding → Chunk (1:1, FK on chunk_id)

**Example**:
```json
{
  "chunk_id": "data/sample.txt:0",
  "model": "text-embedding-3-small",
  "vector": [0.123, -0.456, 0.789, ...],
  "created_at": "2025-12-29T10:30:02Z",
  "cached": false
}
```

---

### Query

User's natural language question for the RAG system.

**Fields**:
- `text` (string, required): The question
- `top_k` (integer, optional, default=3): Number of retrieved documents to consider
- `submitted_at` (ISO 8601 timestamp, auto): When query was submitted

**Validation**:
- `text` length > 0 and < 1000 characters
- `top_k` ∈ [1, 10] (reasonable bounds)

**Relationships**:
- Query → RetrievalResult (1:many)
- Query → Answer (1:1)

**Example**:
```json
{
  "text": "What is RAG and why is it useful?",
  "top_k": 3,
  "submitted_at": "2025-12-29T10:30:05Z"
}
```

---

### RetrievalResult

A ranked document chunk returned by similarity search.

**Fields**:
- `chunk_id` (string, required): Reference to Chunk.id
- `text` (string, required): Chunk text (from Chunk)
- `source` (string, required): Filepath (from Chunk.source)
- `similarity_score` (float, required): Cosine similarity to query embedding ∈ [0, 1]
- `rank` (integer, required): Position in ranked results (1-based, 1 = most similar)
- `retrieved_at` (ISO 8601 timestamp, auto): When this result was retrieved

**Validation**:
- `similarity_score` ∈ [0, 1] (cosine similarity is normalized)
- `rank` > 0
- `rank` ≤ query.top_k

**Relationships**:
- RetrievalResult → Chunk (many:1, FK on chunk_id)
- RetrievalResult → Query (many:1, implicit)

**Example**:
```json
{
  "chunk_id": "data/sample.txt:0",
  "text": "Retrieval-Augmented Generation (RAG) is...",
  "source": "data/sample.txt",
  "similarity_score": 0.87,
  "rank": 1,
  "retrieved_at": "2025-12-29T10:30:06Z"
}
```

---

### Answer

Generated response from the LLM with sources and metadata.

**Fields**:
- `query_text` (string, required): The original query
- `answer_text` (string, required): LLM-generated answer
- `sources` (array[string], required): List of source filepaths used
- `retrieved_chunks` (array[RetrievalResult], required): Full retrieval details
- `model` (string, required): LLM model name (e.g., "gpt-3.5-turbo")
- `tokens_used` (object, optional): `{"prompt_tokens": N, "completion_tokens": N}`
- `generation_latency_ms` (float, required): Time to generate answer (milliseconds)
- `retrieval_latency_ms` (float, required): Time to retrieve documents (milliseconds)
- `total_latency_ms` (float, required): Retrieval + generation time
- `generated_at` (ISO 8601 timestamp, auto): When answer was generated

**Validation**:
- `answer_text` length > 0
- `sources` is non-empty array of filepaths
- `generation_latency_ms` > 0
- `retrieval_latency_ms` > 0
- `total_latency_ms` = retrieval_latency_ms + generation_latency_ms ± 10ms (allows async overhead)
- `tokens_used.prompt_tokens` + `tokens_used.completion_tokens` = total tokens used

**Relationships**:
- Answer → Query (1:1)
- Answer → RetrievalResult (1:many, via retrieved_chunks)

**Example**:
```json
{
  "query_text": "What is RAG and why is it useful?",
  "answer_text": "RAG is retrieval-augmented generation, a technique that combines document retrieval with language model generation...",
  "sources": ["data/sample.txt"],
  "retrieved_chunks": [
    {
      "chunk_id": "data/sample.txt:0",
      "similarity_score": 0.87,
      "rank": 1
    }
  ],
  "model": "gpt-3.5-turbo",
  "tokens_used": {
    "prompt_tokens": 245,
    "completion_tokens": 156
  },
  "generation_latency_ms": 1234.5,
  "retrieval_latency_ms": 456.2,
  "total_latency_ms": 1690.7,
  "generated_at": "2025-12-29T10:30:08Z"
}
```

---

## Entity Relationships

```
Document (1:many) ←→ Chunk
         ↓
      created_at

Chunk (1:1) ←→ Embedding
   ↓
   chunk_id

Query (1:many) ←→ RetrievalResult
   ↓
   top_k

RetrievalResult (many:1) ←→ Chunk
   ↓
   chunk_id

Query (1:1) ←→ Answer
   ↓
   query_text

Answer (1:many) ← RetrievalResult (via retrieved_chunks)
```

---

## State Transitions

### Chunk Lifecycle

```
Created (from Document)
   ↓
Indexed (embedding generated)
   ↓
Queryable (available for retrieval)
   ↓
Cached (embedding loaded from cache on next access)
```

### Query Lifecycle

```
Submitted
   ↓
Retrieved (documents found via similarity search)
   ↓
Generated (LLM produces answer)
   ↓
Returned (to client)
```

### Answer Lifecycle

```
Generated (from Query + RetrievalResults)
   ↓
Stored (in memory or database)
   ↓
Returned (to API client)
   ↓
(Optional: Logged for evaluation Week 3)
```

---

## Validation Rules

### Cross-Entity Constraints

1. **Document-Chunk Consistency**: All chunks with source=X must have corresponding Document with filepath=X
2. **Chunk-Embedding Consistency**: Every Chunk must have exactly one Embedding with same chunk_id
3. **Chunk Size Bounds**: 0 < len(chunk.text) ≤ chunk_size (default 512)
4. **Similarity Score Bounds**: 0 ≤ similarity_score ≤ 1 (for cosine similarity)
5. **Retrieval Result Uniqueness**: Within one query, no duplicate chunk_ids in result set
6. **Answer Source Validation**: answer.sources ⊆ {chunk.source for chunk in answer.retrieved_chunks}
7. **Latency Non-Negative**: All latency fields > 0
8. **Timestamp Ordering**: retrieved_at ≥ submitted_at (for RetrievalResult)

---

## Week 1 Storage

### In-Memory Schema (Python)

```python
# Global in-memory store (simplified)
documents: dict[str, Document]  # filepath → Document
chunks: dict[str, Chunk]  # chunk_id → Chunk
embeddings: dict[str, np.ndarray]  # chunk_id → embedding vector
embedding_cache: dict[str, np.ndarray]  # text → cached embedding
```

### Future (Week 2): Vector Database Schema

Upgrade to Weaviate with these classes:
- `Document`: Weaviate class for metadata (filepath, loaded_at)
- `Chunk`: Weaviate class with vector property (text_embedding)
- Relationships via Weaviate `beamRef` (reference property)

---

## Checkpoint: Data Model Complete

✅ All entities defined with fields, validation, relationships  
✅ State transitions documented  
✅ In-memory storage sufficient for Week 1  
✅ Upgrade path to Weaviate Week 2 documented  

Ready for API contract generation (Phase 1 continued).
