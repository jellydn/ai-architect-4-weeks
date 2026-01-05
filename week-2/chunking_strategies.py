"""
Week 2 Day 2: Chunking Strategies & Metadata Filtering

Implement multiple chunking strategies for comparison and evaluation.
Compare fixed-size vs semantic chunking approaches.

Learning Goals:
- Understand trade-offs between chunking strategies
- Implement semantic chunking using embeddings
- Extract and structure metadata
- Measure chunking effectiveness
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import re


@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    chunk_id: str
    source: str
    start_pos: int
    end_pos: int
    metadata: Optional[Dict[str, Any]] = None


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str, source: str) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: Document text to chunk
            source: Source document name
            
        Returns:
            List of chunks with metadata
        """
        pass


class FixedSizeChunker(ChunkingStrategy):
    """
    Split documents into fixed-size chunks with overlap.
    
    Strategy:
    - Split by token count (approximate using words)
    - Apply overlap to preserve context at boundaries
    - Maintain chunk continuity (don't split mid-sentence if possible)
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 100):
        """
        Initialize fixed-size chunker.
        
        Args:
            chunk_size: Target tokens per chunk (approximate via word count)
            overlap: Overlap between chunks in tokens
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Rough approximation: 1 word ≈ 1.3 tokens
        self.words_per_chunk = int(chunk_size / 1.3)
        self.words_overlap = int(overlap / 1.3)
    
    def chunk(self, text: str, source: str) -> List[Chunk]:
        """
        Split text into fixed-size overlapping chunks.
        """
        # Split into sentences for better boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        chunk_id = 0
        pos = 0
        
        current_chunk = []
        current_words = 0
        overlap_buffer = []
        
        for sentence in sentences:
            words = sentence.split()
            word_count = len(words)
            
            # Add to current chunk
            current_chunk.extend(words)
            current_words += word_count
            
            # Create chunk when size reached
            if current_words >= self.words_per_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=f"{source}-chunk-{chunk_id}",
                    source=source,
                    start_pos=pos,
                    end_pos=pos + len(chunk_text),
                    metadata={"strategy": "fixed-size", "size": self.chunk_size}
                )
                chunks.append(chunk)
                
                # Create overlap buffer (last N words for next chunk)
                overlap_buffer = current_chunk[-self.words_overlap:] if self.words_overlap > 0 else []
                current_chunk = overlap_buffer.copy()
                current_words = len(overlap_buffer)
                pos += len(chunk_text)
                chunk_id += 1
        
        # Handle remaining text
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"{source}-chunk-{chunk_id}",
                source=source,
                start_pos=pos,
                end_pos=pos + len(chunk_text),
                metadata={"strategy": "fixed-size", "size": self.chunk_size}
            )
            chunks.append(chunk)
        
        return chunks


class SemanticChunker(ChunkingStrategy):
    """
    Split documents at semantic boundaries using embeddings.
    
    Strategy:
    - Split by sentences first
    - Calculate embedding similarity between consecutive sentences
    - Merge sentences with high similarity (same topic)
    - Split when similarity drops (topic boundary)
    
    Note: Requires embedding model (Week 2 will integrate)
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize semantic chunker.
        
        Args:
            similarity_threshold: Threshold for semantic similarity
        """
        self.similarity_threshold = similarity_threshold
        self.embed_fn = None  # To be provided by retrieval module
    
    def set_embedding_fn(self, fn):
        """Set the embedding function for semantic similarity."""
        self.embed_fn = fn
    
    def chunk(self, text: str, source: str) -> List[Chunk]:
        """
        Split text at semantic boundaries.
        
        Note: This is a placeholder. Full implementation requires
        the embedding function from week-1/retrieval.py
        """
        # For now, fall back to fixed-size chunking
        # Week 2 Day 2 will integrate embeddings
        chunker = FixedSizeChunker(chunk_size=512, overlap=100)
        chunks = chunker.chunk(text, source)
        
        # Mark as semantic chunks
        for chunk in chunks:
            chunk.metadata = {"strategy": "semantic", "threshold": self.similarity_threshold}
        
        return chunks


def extract_metadata(text: str, source: str) -> Dict[str, Any]:
    """
    Extract metadata from document.
    
    Extracts:
    - Source document name
    - Approximate word/token count
    - Detected language
    - Presence of code blocks, lists, etc.
    """
    lines = text.split('\n')
    words = text.split()
    
    metadata = {
        "source": source,
        "word_count": len(words),
        "estimated_tokens": int(len(words) * 1.3),
        "line_count": len(lines),
        "has_code": "```" in text or "    " in text,
        "has_lists": any(line.strip().startswith(("- ", "* ", "• ")) for line in lines),
        "has_headers": any(line.strip().startswith("#") for line in lines),
    }
    
    return metadata


def compare_chunking_strategies(text: str, source: str) -> Dict[str, Any]:
    """
    Compare different chunking strategies on same text.
    
    Returns metrics for:
    - Fixed-size chunking (current)
    - Semantic chunking (Week 2)
    """
    # Fixed-size chunking
    fixed_chunker = FixedSizeChunker(chunk_size=512, overlap=100)
    fixed_chunks = fixed_chunker.chunk(text, source)
    
    # Semantic chunking (placeholder for now)
    semantic_chunker = SemanticChunker(similarity_threshold=0.7)
    semantic_chunks = semantic_chunker.chunk(text, source)
    
    # Calculate metrics
    def avg_chunk_size(chunks):
        if not chunks:
            return 0
        return sum(len(c.text.split()) for c in chunks) / len(chunks)
    
    comparison = {
        "document": source,
        "total_words": len(text.split()),
        "strategies": {
            "fixed-size": {
                "num_chunks": len(fixed_chunks),
                "avg_chunk_size": avg_chunk_size(fixed_chunks),
                "compression_ratio": len(text.split()) / len(fixed_chunks) if fixed_chunks else 0,
                "overlap": "100 tokens",
            },
            "semantic": {
                "num_chunks": len(semantic_chunks),
                "avg_chunk_size": avg_chunk_size(semantic_chunks),
                "compression_ratio": len(text.split()) / len(semantic_chunks) if semantic_chunks else 0,
                "note": "Placeholder - full implementation in Day 2",
            }
        }
    }
    
    return comparison


if __name__ == "__main__":
    # Example usage
    sample_text = """
    RAG combines document retrieval with LLM generation. Instead of relying solely on 
    the model's training data, RAG first retrieves relevant documents from your knowledge base.
    
    Vector embeddings represent text as dense vectors in semantic space. Similar texts 
    map to nearby points in this space. This allows mathematical comparison of meaning.
    
    Chunking is critical for RAG success. Good chunks preserve semantic units while 
    providing context. Overlap at boundaries prevents context loss.
    """
    
    print("Comparing chunking strategies:")
    print()
    comparison = compare_chunking_strategies(sample_text, "rag-guide.txt")
    print(f"Document: {comparison['document']}")
    print(f"Total words: {comparison['total_words']}")
    print()
    
    for strategy, metrics in comparison['strategies'].items():
        print(f"{strategy.upper()}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print()
