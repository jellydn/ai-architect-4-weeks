"""Vector Retrieval & Embedding Module

Handles embedding generation and vector similarity search for RAG.
"""

import logging
import time
from typing import Dict, List

import numpy as np
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Manages embeddings and vector similarity search.

    Features:
    - OpenAI embedding generation
    - In-memory vector store with caching
    - Cosine similarity search
    - Latency tracking
    """

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize retriever with embedding model.

        Args:
            embedding_model: OpenAI embedding model name
        """
        self.embedding_model = embedding_model
        self.client = OpenAI()
        self.documents: List[Dict] = []
        self.embeddings: List[np.ndarray] = []
        self.embedding_cache: Dict[str, List[float]] = {}

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using OpenAI API.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (lists of floats)
        """
        start = time.time()
        embeddings = []
        cached_count = 0

        for text in texts:
            # Check cache
            if text in self.embedding_cache:
                embeddings.append(self.embedding_cache[text])
                cached_count += 1
            else:
                # Call OpenAI
                response = self.client.embeddings.create(model=self.embedding_model, input=text)
                embedding = response.data[0].embedding
                embeddings.append(embedding)
                self.embedding_cache[text] = embedding

        elapsed = time.time() - start
        logger.info(
            f"Embedded {len(texts)} texts in {elapsed:.2f}s "
            f"({cached_count} from cache, {len(texts) - cached_count} new)"
        )

        return embeddings

    def index(self, documents: List[Dict]) -> None:
        """
        Index documents by generating and caching embeddings.

        Args:
            documents: List of dicts with 'text' field (from ingestion)
        """
        start = time.time()

        self.documents = documents
        texts = [doc["text"] for doc in documents]

        # Generate embeddings
        embeddings = self.embed(texts)
        self.embeddings = [np.array(emb) for emb in embeddings]

        elapsed = time.time() - start
        logger.info(f"Indexed {len(documents)} documents in {elapsed:.2f}s")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve most similar documents for a query using cosine similarity.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of dicts with text, similarity score, and metadata
        """
        if not self.documents:
            logger.warning("No documents indexed. Call index() first.")
            return []

        start = time.time()

        # Embed query
        query_embedding = np.array(self.embed([query])[0])

        # Compute cosine similarity
        scores = []
        for i, doc_embedding in enumerate(self.embeddings):
            # Cosine similarity: dot(a,b) / (norm(a) * norm(b))
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-10
            )
            scores.append((i, similarity))

        # Sort by score and get top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scores[:top_k]]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            similarity = scores[[i for i, _ in scores].index(idx)][1]
            results.append(
                {
                    "id": doc.get("id"),
                    "text": doc.get("text"),
                    "source": doc.get("source"),
                    "similarity_score": float(similarity),
                }
            )

        elapsed = time.time() - start
        logger.info(f"Retrieved top-{top_k} in {elapsed:.2f}s")

        return results


# Example usage
if __name__ == "__main__":
    from ingestion import DocumentIngester

    # Load and index documents
    ingester = DocumentIngester()
    documents = ingester.ingest("data/sample.txt")

    # Create retriever and index
    retriever = RAGRetriever()
    retriever.index(documents)

    # Retrieve
    query = "What is RAG?"
    results = retriever.retrieve(query, top_k=2)

    print(f"\nTop results for '{query}':")
    for result in results:
        print(f"  Score: {result['similarity_score']:.3f}")
        print(f"  Text: {result['text'][:100]}...\n")
