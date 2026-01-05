"""Document Ingestion & Chunking Module

Handles loading, chunking, and processing documents for RAG pipeline.
"""

import logging
import time
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIngester:
    """
    Loads documents and chunks them for embedding and retrieval.

    Supports:
    - Text file loading
    - Configurable chunk size and overlap
    - Structured output with metadata
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize ingester with chunking parameters.

        Args:
            chunk_size: Number of characters per chunk
            chunk_overlap: Character overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_from_file(self, filepath: str) -> list[str]:
        """
        Load text from a file.

        Args:
            filepath: Path to text file

        Returns:
            List of raw document strings
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            logger.info(f"Loaded {len(content)} characters from {filepath}")
            return [content]
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading file {filepath}: {e}")
            raise

    def chunk(self, documents: list[str]) -> list[str]:
        """
        Split documents into overlapping chunks.

        Args:
            documents: List of document strings

        Returns:
            List of chunks (each chunk is a string)
        """
        chunks = []

        for doc in documents:
            # Simple sliding window chunking
            for i in range(0, len(doc), self.chunk_size - self.chunk_overlap):
                chunk = doc[i : i + self.chunk_size]
                if chunk.strip():  # Skip empty chunks
                    chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def ingest(self, filepath: str) -> list[Dict]:
        """
        Full ingestion pipeline: load → chunk → structure.

        Args:
            filepath: Path to document file

        Returns:
            List of dicts with id, text, source, and metadata
        """
        start = time.time()

        # Load
        documents = self.load_from_file(filepath)

        # Chunk
        chunks = self.chunk(documents)

        # Structure
        structured_chunks = [
            {
                "id": f"{filepath}:{i}",
                "text": chunk,
                "source": filepath,
                "chunk_index": i,
            }
            for i, chunk in enumerate(chunks)
        ]

        elapsed = time.time() - start
        logger.info(f"Ingestion complete in {elapsed:.2f}s. {len(structured_chunks)} chunks")

        return structured_chunks


# Example usage
if __name__ == "__main__":
    # Quick test
    ingester = DocumentIngester(chunk_size=512, chunk_overlap=50)

    # Create sample document for testing
    sample_text = """
    Retrieval-Augmented Generation (RAG) is a technique that combines document retrieval 
    with language model generation. It works by first retrieving relevant documents based on 
    a user query, then using those documents as context for the language model to generate 
    an answer. This approach helps reduce hallucinations and provides up-to-date information 
    without fine-tuning.
    
    RAG is particularly useful when you have a large corpus of documents that you want to 
    query without modifying the underlying language model. It's more cost-effective than 
    fine-tuning and allows for easy updates to the knowledge base.
    """

    # Write sample file
    sample_path = "data/sample.txt"
    import os

    os.makedirs("data", exist_ok=True)
    with open(sample_path, "w") as f:
        f.write(sample_text)

    # Ingest
    result = ingester.ingest(sample_path)
    print(f"\nIngested {len(result)} chunks:")
    for chunk in result[:2]:  # Show first 2
        print(f"  ID: {chunk['id']}")
        print(f"  Text: {chunk['text'][:100]}...")
