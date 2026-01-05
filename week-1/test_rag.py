"""Tests for RAG components

Unit and integration tests for ingestion, retrieval, and generation.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from ingestion import DocumentIngester


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
    with open(path, "w") as f:
        f.write(content)

    yield path

    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def ingester():
    """Create an ingester instance."""
    return DocumentIngester(chunk_size=256, chunk_overlap=25)


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = MagicMock()
    mock_embedding_response = MagicMock()
    mock_embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
    mock_client.embeddings.create.return_value = mock_embedding_response
    return mock_client


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
        assert all("id" in chunk for chunk in result)
        assert all("text" in chunk for chunk in result)
        assert all("source" in chunk for chunk in result)


class TestRetrieval:
    @patch("retrieval.OpenAI")
    def test_embedding_caching(self, mock_openai_class):
        """Test that embeddings are cached."""
        from retrieval import RAGRetriever

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        retriever = RAGRetriever()
        text = "This is a test document."

        emb1 = retriever.embed([text])
        assert len(emb1) == 1
        assert len(retriever.embedding_cache) == 1

        emb2 = retriever.embed([text])
        assert len(emb2) == 1
        assert len(retriever.embedding_cache) == 1

    @patch("retrieval.OpenAI")
    def test_retrieval_empty(self, mock_openai_class):
        """Test retrieval on empty index."""
        from retrieval import RAGRetriever

        mock_openai_class.return_value = MagicMock()
        retriever = RAGRetriever()
        results = retriever.retrieve("test query")
        assert results == []

    @patch("retrieval.OpenAI")
    def test_retrieval_ordering(self, mock_openai_class):
        """Test that retrieval returns results in similarity order."""
        import numpy as np
        from retrieval import RAGRetriever

        mock_client = MagicMock()

        def mock_embed(model, input):
            response = MagicMock()
            vec = np.random.rand(1536).tolist()
            response.data = [MagicMock(embedding=vec)]
            return response

        mock_client.embeddings.create.side_effect = mock_embed
        mock_openai_class.return_value = mock_client

        retriever = RAGRetriever()
        docs = [
            {"id": "1", "text": "RAG is great", "source": "test"},
            {"id": "2", "text": "RAG helps with retrieval", "source": "test"},
            {"id": "3", "text": "Machine learning is interesting", "source": "test"},
        ]

        retriever.index(docs)
        results = retriever.retrieve("What is RAG?", top_k=3)

        assert len(results) > 0
        assert results[0]["similarity_score"] >= results[-1]["similarity_score"]


class TestGeneration:
    @patch("generation.OpenAI")
    def test_prompt_template(self, mock_openai_class):
        """Test prompt template formatting."""
        from generation import RAGGenerator

        mock_openai_class.return_value = MagicMock()
        generator = RAGGenerator()

        assert "context" in generator.prompt_template.template
        assert "query" in generator.prompt_template.template


@patch("retrieval.OpenAI")
def test_rag_integration(mock_openai_class, sample_document):
    """Test full RAG pipeline (without calling actual LLM)."""
    import numpy as np
    from retrieval import RAGRetriever

    mock_client = MagicMock()

    def mock_embed(model, input):
        response = MagicMock()
        vec = np.random.rand(1536).tolist()
        response.data = [MagicMock(embedding=vec)]
        return response

    mock_client.embeddings.create.side_effect = mock_embed
    mock_openai_class.return_value = mock_client

    ingester = DocumentIngester()
    retriever = RAGRetriever()

    chunks = ingester.ingest(sample_document)
    assert len(chunks) > 0

    retriever.index(chunks)
    assert len(retriever.documents) == len(chunks)

    results = retriever.retrieve("What is RAG?", top_k=2)
    assert len(results) > 0
    assert all("similarity_score" in r for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
