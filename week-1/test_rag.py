"""Tests for RAG components

Unit and integration tests for ingestion, retrieval, and generation.
"""

import pytest
import os
from pathlib import Path

from ingestion import DocumentIngester
from retrieval import RAGRetriever
from generation import RAGGenerator


# Fixtures
@pytest.fixture
def sample_document():
    """Create a sample test document."""
    content = """
    Retrieval-Augmented Generation (RAG) is a technique combining document retrieval 
    with language model generation. It works by retrieving relevant documents first, 
    then using them as context for the LLM to generate answers.
    
    RAG benefits include reduced hallucinations, up-to-date information, and no need 
    for fine-tuning. It's ideal for Q&A over document collections.
    
    Fine-tuning, by contrast, modifies model parameters. Use it when you need specific 
    writing style or domain-specific knowledge encoded permanently.
    """
    
    path = "test_sample.txt"
    with open(path, 'w') as f:
        f.write(content)
    
    yield path
    
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def ingester():
    """Create an ingester instance."""
    return DocumentIngester(chunk_size=256, chunk_overlap=25)


@pytest.fixture
def retriever():
    """Create a retriever instance."""
    return RAGRetriever()


# Test ingestion
class TestIngestion:
    
    def test_load_from_file(self, ingester, sample_document):
        """Test loading documents from file."""
        docs = ingester.load_from_file(sample_document)
        assert len(docs) > 0
        assert isinstance(docs[0], str)
    
    def test_chunking(self, ingester, sample_document):
        """Test document chunking."""
        docs = ingester.load_from_file(sample_document)
        chunks = ingester.chunk(docs)
        
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)
        assert all(len(c) <= ingester.chunk_size for c in chunks)
    
    def test_ingest_full_pipeline(self, ingester, sample_document):
        """Test full ingestion pipeline."""
        result = ingester.ingest(sample_document)
        
        assert len(result) > 0
        assert all('id' in chunk for chunk in result)
        assert all('text' in chunk for chunk in result)
        assert all('source' in chunk for chunk in result)


# Test retrieval
class TestRetrieval:
    
    @pytest.mark.asyncio
    async def test_embedding_caching(self, retriever):
        """Test that embeddings are cached."""
        text = "This is a test document."
        
        # First embedding (should call API)
        emb1 = retriever.embed([text])
        assert len(emb1) == 1
        assert len(retriever.embedding_cache) == 1
        
        # Second embedding (should use cache)
        emb2 = retriever.embed([text])
        assert len(emb2) == 1
        # Cache size should still be 1
        assert len(retriever.embedding_cache) == 1
    
    @pytest.mark.asyncio
    async def test_retrieval_empty(self, retriever):
        """Test retrieval on empty index."""
        results = retriever.retrieve("test query")
        assert results == []
    
    @pytest.mark.asyncio
    async def test_retrieval_ordering(self, retriever):
        """Test that retrieval returns results in similarity order."""
        # Create simple in-memory documents
        docs = [
            {"id": "1", "text": "RAG is great", "source": "test"},
            {"id": "2", "text": "RAG helps with retrieval", "source": "test"},
            {"id": "3", "text": "Machine learning is interesting", "source": "test"},
        ]
        
        retriever.index(docs)
        results = retriever.retrieve("What is RAG?", top_k=3)
        
        assert len(results) > 0
        # First result should have highest similarity
        assert results[0]['similarity_score'] >= results[-1]['similarity_score']


# Test generation (mock)
class TestGeneration:
    
    def test_prompt_template(self):
        """Test prompt template formatting."""
        generator = RAGGenerator()
        
        # Template should have correct variables
        assert "context" in generator.prompt_template.template
        assert "query" in generator.prompt_template.template


# Integration test
@pytest.mark.asyncio
async def test_rag_integration(sample_document):
    """Test full RAG pipeline (without calling actual LLM)."""
    # Setup
    ingester = DocumentIngester()
    retriever = RAGRetriever()
    
    # Ingest
    chunks = ingester.ingest(sample_document)
    assert len(chunks) > 0
    
    # Index
    retriever.index(chunks)
    assert len(retriever.documents) == len(chunks)
    
    # Retrieve
    results = retriever.retrieve("What is RAG?", top_k=2)
    assert len(results) > 0
    assert all('similarity_score' in r for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
