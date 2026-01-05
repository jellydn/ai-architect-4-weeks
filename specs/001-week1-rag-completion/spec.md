# Feature Specification: Complete Week 1 RAG Foundation (Days 2-5)

**Feature Branch**: `001-week1-rag-completion`  
**Created**: 2025-12-29  
**Status**: Draft  
**Input**: Complete Week 1 RAG foundation - Days 2-5: ingestion testing, retrieval with embeddings, generation + full API, architecture and trade-offs documentation, checkpoint validation

## Overview

This feature completes Week 1 of the 4-week AI Architect sprint by implementing and validating the core RAG (Retrieval-Augmented Generation) pipeline. Starting from Day 1's foundational setup, this work builds, tests, and documents a fully functional RAG system capable of answering questions over documents without fine-tuning.

By end of Week 1, users should have:
- A working RAG API (FastAPI) serving Q&A queries
- Comprehensive architecture and trade-off documentation
- Understanding of RAG system design trade-offs
- Measured latency and performance baselines

## User Scenarios & Testing

### User Story 1 - Test Document Ingestion at Scale (Priority: P1)

A developer needs to verify that documents can be loaded, chunked consistently, and prepared for embedding without errors. This validates the first stage of the RAG pipeline and ensures the ingestion system works with real-world documents.

**Why this priority**: Ingestion is the foundational stage of RAG. Without working ingestion, retrieval cannot proceed. This is a critical path item.

**Independent Test**: Run `pytest test_rag.py::TestIngestion -v` and verify all chunks are created with correct metadata. Can test in isolation with sample documents before embedding/retrieval stages.

**Acceptance Scenarios**:

1. **Given** a text file with 10KB of content, **When** ingestion runs with chunk_size=512 and overlap=50, **Then** produces 15-25 chunks with metadata (id, text, source, chunk_index)
2. **Given** a sample document with mixed spacing, **When** chunking processes it, **Then** empty chunks are skipped and non-empty chunks meet size requirements
3. **Given** a missing file path, **When** ingestion runs, **Then** raises FileNotFoundError with clear message logged

---

### User Story 2 - Embed and Retrieve Relevant Documents (Priority: P1)

A developer needs to verify that document embeddings are generated correctly via OpenAI API, cached to reduce costs, and that similarity search returns documents ranked by relevance to a query.

**Why this priority**: Retrieval is the second critical stage. Without working retrieval, the system cannot find relevant context for answer generation.

**Independent Test**: Run `pytest test_rag.py::TestRetrieval -v` and verify `retrieve(query, top_k=3)` returns sorted results by similarity score. Can be tested independently with mock documents without calling generation stage.

**Acceptance Scenarios**:

1. **Given** 3 indexed documents on different topics, **When** query "What is RAG?" is retrieved, **Then** documents about RAG rank higher than unrelated documents
2. **Given** first embedding call, **When** embed() is called for same text twice, **Then** second call uses cache (API not called again)
3. **Given** empty index, **When** retrieve(query) is called, **Then** returns empty list with warning logged
4. **Given** query and top_k=3, **When** retrieve runs, **Then** returns results sorted by similarity_score in descending order

---

### User Story 3 - Generate Answers Using Retrieved Context (Priority: P1)

A developer needs to verify that the LLM generation stage takes retrieved documents and produces answers grounded in the context, with proper prompt templating to reduce hallucination.

**Why this priority**: Generation is the final critical stage. Together with retrieval and ingestion, it completes the RAG pipeline.

**Independent Test**: Verify `rag_answer(query, retriever)` returns dict with answer, sources, and latency_ms. Test full end-to-end pipeline with sample documents.

**Acceptance Scenarios**:

1. **Given** 3 retrieved documents on RAG, **When** rag_answer("What is RAG?") is called, **Then** returned answer is non-empty, cites retrieved documents, latency_ms is measured
2. **Given** empty retriever (no documents indexed), **When** rag_answer() is called, **Then** returns "No relevant documents found" instead of hallucinating

---

### User Story 4 - Serve Q&A Queries via FastAPI (Priority: P1)

A developer or API consumer needs HTTP endpoints to ingest documents and query the RAG system, enabling the system to work as a standalone service.

**Why this priority**: Without HTTP endpoints, the RAG system is not accessible as a service. API integration enables real-world usage patterns.

**Independent Test**: Start FastAPI server and test endpoints: `curl POST /ingest` creates chunks, `curl POST /query` returns answer+sources+latency.

**Acceptance Scenarios**:

1. **Given** FastAPI server running on localhost:8000, **When** GET /health is called, **Then** returns `{"status": "ok"}`
2. **Given** JSON request `{"file_paths": ["data/sample.txt"]}`, **When** POST /ingest is called, **Then** returns `{"status": "success", "chunks_created": N, "files_processed": 1}`
3. **Given** JSON request `{"query": "What is RAG?", "top_k": 3}`, **When** POST /query is called with documents indexed, **Then** returns `{"answer": "...", "sources": [...], "latency_ms": N}`
4. **Given** query called before /ingest, **When** POST /query is called, **Then** returns 400 error with clear message "No documents indexed. Call /ingest first"

---

### User Story 5 - Document and Validate Architecture (Priority: P2)

A developer or stakeholder needs clear documentation of system design, component interactions, and data flow to understand and communicate how the RAG system works.

**Why this priority**: Documentation enables knowledge sharing and future maintenance. Priority is P2 because functionality is more critical than documentation initially, but required before moving to Week 2.

**Independent Test**: `docs/architecture.md` contains Mermaid diagram, component descriptions, example data flow, and latency breakdown. Can be visually verified to ensure clarity.

**Acceptance Scenarios**:

1. **Given** architecture.md with Mermaid diagram, **When** reviewed, **Then** shows all 4 components (Ingester, Retriever, Generator, FastAPI) and data flow from query to response
2. **Given** architecture.md, **When** reviewed, **Then** includes example latency trace showing retrieval_ms + generation_ms + total_ms

---

### User Story 6 - Justify Design Trade-Offs (Priority: P2)

A developer needs clear documentation of key architectural decisions (RAG vs fine-tuning, embedding model choice, chunking strategy, etc.) with rationale to understand when to use this approach vs alternatives.

**Why this priority**: Understanding trade-offs enables making better decisions for Week 2+ extensions and future projects. Critical for learning outcomes of the sprint.

**Independent Test**: `docs/trade-offs.md` covers: RAG vs fine-tuning decision tree, embedding model choice, chunking strategy, vector store approach, hallucination mitigation. Each section includes "why this choice" + "when alternative wins".

**Acceptance Scenarios**:

1. **Given** trade-offs.md, **When** reviewed, **Then** explains why RAG chosen over fine-tuning with concrete cost/speed/flexibility comparisons
2. **Given** trade-offs.md, **When** reviewed, **Then** justifies text-embedding-3-small choice vs larger models with cost/quality trade-off quantified
3. **Given** trade-offs.md, **When** reviewed, **Then** identifies at least 2 failure modes of RAG (retrieval failure, hallucination) and one mitigation for each

---

### User Story 7 - Validate Week 1 Learning Outcomes (Priority: P2)

A developer needs to pass a checkpoint validation that confirms they understand RAG concepts deeply enough to explain design decisions, identify failure modes, and articulate trade-offs fluently.

**Why this priority**: Learning validation ensures readiness for Week 2 extensions. Prevents proceeding without foundational understanding.

**Independent Test**: Self-assessment checklist: Can explain RAG in 2 minutes? Can justify embedding choice? Can name 3 failure modes? Can explain latency breakdown to PM?

**Acceptance Scenarios**:

1. **Given** prompt "Why did you choose RAG over fine-tuning?", **When** developer answers, **Then** response demonstrates understanding of RAG fundamentals (retrieval + generation vs model parameter updates)
2. **Given** prompt "What are 3 failure modes of your RAG?", **When** developer answers, **Then** lists specific failure modes (retrieval failure, hallucination, latency, cost) with explanation

---

### Edge Cases

- What happens if document file is very large (>100MB)? → Should chunk and process without memory exhaustion
- What happens if query embedding API rate limit is hit? → Should fail gracefully with rate-limit error, not crash server
- What happens if top_k > number of indexed documents? → Should return all available documents (not error)
- What happens if retrieved context is too long for LLM context window? → Should truncate to fit (Week 2 extension: add reranking)
- What happens if OpenAI API key is invalid? → Should fail at initialization with clear error message

## Requirements

### Functional Requirements

- **FR-001**: System MUST load text documents from file paths and return raw content
- **FR-002**: System MUST split documents into overlapping chunks (configurable size and overlap)
- **FR-003**: System MUST generate embeddings for all chunks using OpenAI text-embedding-3-small model
- **FR-004**: System MUST cache embeddings in memory to avoid re-embedding identical texts
- **FR-005**: System MUST retrieve top-k documents using cosine similarity search on query embeddings
- **FR-006**: System MUST generate answers using LLM (gpt-3.5-turbo) with retrieved documents as context
- **FR-007**: System MUST use PromptTemplate to safely format prompts and reduce hallucination risk
- **FR-008**: System MUST provide FastAPI endpoint `POST /ingest` to load and index documents
- **FR-009**: System MUST provide FastAPI endpoint `POST /query` to retrieve and generate answers
- **FR-010**: System MUST provide FastAPI endpoint `GET /health` for health checks
- **FR-011**: System MUST log all operations (ingestion, retrieval, generation) with latency measurements
- **FR-012**: System MUST return structured responses with answer, sources, and latency_ms
- **FR-013**: System MUST validate that documents are indexed before allowing queries

### Key Entities

- **Document**: Source file with metadata (filepath, content)
- **Chunk**: Sub-section of document with fixed size, overlap, and unique ID
- **Embedding**: Dense vector representation of text (1536 dimensions for text-embedding-3-small)
- **Query**: User's natural language question
- **RetrievalResult**: Ranked document chunk with similarity score
- **Answer**: Generated response with sources and latency

## Success Criteria

### Measurable Outcomes

- **SC-001**: Ingestion latency for 10 documents (1MB total) is under 5 seconds (measured)
- **SC-002**: Retrieval latency (embed query + similarity search) is under 500ms for <100k documents (measured)
- **SC-003**: LLM generation latency is under 3 seconds for typical queries (measured)
- **SC-004**: Total end-to-end latency (ingest → retrieve → generate) is under 4 seconds per query
- **SC-005**: All pytest tests pass (ingestion, retrieval, generation, integration tests)
- **SC-006**: FastAPI endpoints respond with correct status codes and response shapes
- **SC-007**: Embeddings caching reduces API calls: 2nd query for same text uses cache (not API)
- **SC-008**: Architecture diagram and trade-offs documentation are complete and clear
- **SC-009**: Developer can articulate RAG vs fine-tuning decision in plain language within 2 minutes
- **SC-010**: Cost per query (embedding + LLM tokens) is quantified and documented

## Assumptions

- OpenAI API key is available and configured via .env
- Documents are text files (UTF-8 encoded)
- Context window of gpt-3.5-turbo (4K tokens) is sufficient for Week 1 (retrieval returns 3 documents)
- Vector store can remain in-memory for <10k documents
- Users can run Python 3.9+ with FastAPI/uvicorn locally

## Implementation Notes

**Day 2 (Tuesday) - Ingestion**:
- Test `DocumentIngester` with `pytest test_rag.py::TestIngestion -v`
- Run `python ingestion.py` to validate chunking with sample.txt
- Measure: Time to ingest 1MB of text

**Day 3 (Wednesday) - Retrieval**:
- Test `RAGRetriever` with `pytest test_rag.py::TestRetrieval -v`
- Run `python retrieval.py` to validate embeddings and similarity search
- Measure: Embedding latency, retrieval latency for top-3

**Day 4 (Thursday) - Generation & API**:
- Test `RAGGenerator` full pipeline with `pytest test_rag.py::test_rag_integration -v`
- Run `python main.py` to start FastAPI server on localhost:8000
- Test endpoints with curl commands from README
- Measure: Generation latency, total end-to-end latency

**Day 5 (Friday) - Docs & Checkpoint**:
- Complete `docs/architecture.md` with Mermaid diagram and data flow example
- Complete `docs/trade-offs.md` with design decision rationale
- Run checkpoint validation: explain RAG, identify failure modes, justify choices
- Commit week-1 branch with all deliverables

## Dependencies

- Week 1 Day 1 setup (environment, project structure, base modules)
- OpenAI API access (embeddings + LLM calls)
- Python 3.9+, FastAPI, uvicorn, LangChain, numpy, pytest

## Out of Scope (Week 2+)

- Vector database integration (Weaviate)
- Advanced retrieval (reranking, filtering, multi-hop)
- Evaluation metrics (retrieval quality, hallucination rate)
- Prompt optimization techniques
- Production deployment
