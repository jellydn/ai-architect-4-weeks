"""
Week 2: Weaviate Vector Database Integration

Persistent vector store for RAG system with metadata filtering and HNSW indexing.
Replaces Week 1's in-memory dict-based retrieval with production-grade Weaviate.

Learning Goals:
- Understand Weaviate schema design and HNSW indexing
- Implement metadata filtering patterns
- Compare persistence trade-offs (disk I/O vs memory)
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document chunk with metadata."""
    text: str
    source: str
    chunk_id: str
    embedding: List[float]
    metadata: Optional[dict] = None


@dataclass
class SearchResult:
    """Search result with relevance score."""
    text: str
    source: str
    chunk_id: str
    score: float
    metadata: Optional[dict] = None


class WeaviateStore:
    """
    Weaviate vector store with metadata filtering.
    
    Replaces Week 1's in-memory retrieval:
    - Persistent storage on disk
    - HNSW indexing for fast nearest neighbor search
    - Metadata filtering (e.g., filter by source document)
    - Scales to 100k+ documents
    """
    
    def __init__(
        self, 
        url: str = "http://localhost:8080",
        class_name: str = "DocumentChunk",
    ):
        """
        Initialize Weaviate client.
        
        Args:
            url: Weaviate instance URL
            class_name: Name of the document class in Weaviate
        """
        self.url = url
        self.class_name = class_name
        self.client = None
        self._connected = False
    
    async def connect(self) -> None:
        """Connect to Weaviate instance."""
        try:
            import weaviate
            self.client = weaviate.connect_to_local(
                host="localhost",
                port=8080,
                grpc_port=50051,
            )
            self._connected = True
            logger.info(f"Connected to Weaviate at {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise
    
    async def close(self) -> None:
        """Close Weaviate connection."""
        if self.client:
            self.client.close()
            self._connected = False
    
    async def create_class(self) -> None:
        """Create Weaviate class schema for document chunks."""
        if not self._connected:
            await self.connect()
        
        import weaviate
        
        # Delete class if it exists (for testing)
        try:
            self.client.collections.delete(self.class_name)
            logger.info(f"Deleted existing class {self.class_name}")
        except Exception:
            pass  # Class doesn't exist yet
        
        # Create new class
        try:
            self.client.collections.create(
                name=self.class_name,
                properties=[
                    weaviate.classes.config.Property(
                        name="text",
                        data_type=weaviate.classes.config.DataType.TEXT,
                        description="The chunk text",
                    ),
                    weaviate.classes.config.Property(
                        name="source",
                        data_type=weaviate.classes.config.DataType.TEXT,
                        description="Source document filename",
                    ),
                    weaviate.classes.config.Property(
                        name="chunk_id",
                        data_type=weaviate.classes.config.DataType.TEXT,
                        description="Unique chunk identifier",
                    ),
                    weaviate.classes.config.Property(
                        name="timestamp",
                        data_type=weaviate.classes.config.DataType.TEXT,
                        description="ISO timestamp when indexed",
                    ),
                    weaviate.classes.config.Property(
                        name="metadata_json",
                        data_type=weaviate.classes.config.DataType.TEXT,
                        description="JSON metadata field",
                    ),
                ],
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
                # Use none() because we provide pre-computed embeddings
            )
            logger.info(f"Created class {self.class_name}")
        except Exception as e:
            logger.error(f"Failed to create class: {e}")
            raise
    
    async def index_documents(self, documents: List[Document]) -> int:
        """
        Index documents in Weaviate.
        
        Args:
            documents: List of documents with embeddings
            
        Returns:
            Number of successfully indexed documents
        """
        if not self._connected:
            await self.connect()
        
        collection = self.client.collections.get(self.class_name)
        count = 0
        
        for doc in documents:
            try:
                # Prepare object for Weaviate
                obj = {
                    "text": doc.text,
                    "source": doc.source,
                    "chunk_id": doc.chunk_id,
                    "timestamp": doc.metadata.get("timestamp") if doc.metadata else "",
                    "metadata_json": json.dumps(doc.metadata or {}),
                }
                
                # Add object with vector
                uuid = collection.data.insert(
                    properties=obj,
                    vector=doc.embedding,
                )
                count += 1
                logger.debug(f"Indexed {doc.chunk_id} from {doc.source}")
            except Exception as e:
                logger.error(f"Failed to index {doc.chunk_id}: {e}")
                continue
        
        logger.info(f"Indexed {count}/{len(documents)} documents")
        return count
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where_filter: Optional[dict] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query vector (1536 dims)
            top_k: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            List of search results sorted by relevance
        """
        if not self._connected:
            await self.connect()
        
        collection = self.client.collections.get(self.class_name)
        
        try:
            import weaviate
            # Vector search with optional metadata filtering
            results = collection.query.near_vector(
                near_vector=query_embedding,
                limit=top_k,
                where=where_filter,
                return_metadata=weaviate.classes.query.MetadataQuery(
                    distance=True,
                ),
            )
            
            # Convert to SearchResult objects
            search_results = []
            for obj in results.objects:
                result = SearchResult(
                    text=obj.properties.get("text", ""),
                    source=obj.properties.get("source", ""),
                    chunk_id=obj.properties.get("chunk_id", ""),
                    score=1 - obj.metadata.distance,
                    metadata=json.loads(obj.properties.get("metadata_json", "{}")),
                )
                search_results.append(result)
            
            logger.debug(f"Found {len(search_results)} results")
            return search_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def delete_all(self) -> None:
        """Delete all documents (for testing/reset)."""
        if not self._connected:
            await self.connect()
        
        try:
            self.client.collections.delete(self.class_name)
            logger.info(f"Deleted all documents in {self.class_name}")
            # Recreate the class
            await self.create_class()
        except Exception as e:
            logger.error(f"Failed to delete all: {e}")
    
    async def get_stats(self) -> dict:
        """Get collection statistics."""
        if not self._connected:
            await self.connect()
        
        try:
            collection = self.client.collections.get(self.class_name)
            agg_result = collection.aggregate.over_all(total_count=True)
            
            return {
                "class_name": self.class_name,
                "document_count": agg_result.total_count,
                "status": "connected",
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"status": "error", "message": str(e)}


async def test_weaviate():
    """Test Weaviate integration."""
    store = WeaviateStore()
    
    print("\n=== Weaviate Integration Test ===\n")
    
    # Test 1: Connect
    print("1. Testing connection...")
    try:
        await store.connect()
        print("   ✓ Connected to Weaviate")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return
    
    # Test 2: Create schema
    print("2. Creating schema...")
    try:
        await store.create_class()
        print("   ✓ Schema created")
    except Exception as e:
        print(f"   ✗ Schema creation failed: {e}")
        await store.close()
        return
    
    # Test 3: Create sample documents
    print("3. Creating sample documents...")
    docs = [
        Document(
            text="RAG combines document retrieval with LLM generation.",
            source="rag-intro.txt",
            chunk_id="chunk-1",
            embedding=[0.1] * 1536,
            metadata={"section": "introduction", "timestamp": "2026-01-03T00:00:00"},
        ),
        Document(
            text="Vector databases use HNSW indexing for fast similarity search.",
            source="vector-db.txt",
            chunk_id="chunk-2",
            embedding=[0.2] * 1536,
            metadata={"section": "indexing", "timestamp": "2026-01-03T00:00:00"},
        ),
        Document(
            text="Weaviate provides built-in reranking and filtering capabilities.",
            source="weaviate-guide.txt",
            chunk_id="chunk-3",
            embedding=[0.15] * 1536,
            metadata={"section": "features", "timestamp": "2026-01-03T00:00:00"},
        ),
    ]
    print(f"   ✓ Created {len(docs)} test documents")
    
    # Test 4: Index documents
    print("4. Indexing documents...")
    try:
        count = await store.index_documents(docs)
        print(f"   ✓ Indexed {count} documents")
    except Exception as e:
        print(f"   ✗ Indexing failed: {e}")
        await store.close()
        return
    
    # Test 5: Verify stats
    print("5. Checking collection stats...")
    try:
        stats = await store.get_stats()
        print(f"   ✓ Collection: {stats.get('class_name')}")
        print(f"   ✓ Documents: {stats.get('document_count')}")
    except Exception as e:
        print(f"   ✗ Stats failed: {e}")
    
    # Test 6: Search
    print("6. Testing vector search...")
    try:
        query_embedding = [0.15] * 1536
        results = await store.search(query_embedding, top_k=2)
        print(f"   ✓ Found {len(results)} results")
        for result in results:
            print(f"      - {result.chunk_id}: {result.text[:50]}... (score: {result.score:.3f})")
    except Exception as e:
        print(f"   ✗ Search failed: {e}")
    
    await store.close()
    print("\n=== Tests Complete ===\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_weaviate())
