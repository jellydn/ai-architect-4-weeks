"""Tests for caching and performance profiling"""

import pytest
from week_2.caching import QueryCache, LatencyProfiler


class TestQueryCache:
    """Test query caching functionality"""
    
    @pytest.fixture
    def cache(self):
        return QueryCache(max_size=100, similarity_threshold=0.90)
    
    def test_cache_put_get(self, cache):
        """Test storing and retrieving from cache"""
        query = "What is RAG?"
        embedding = [0.1] * 1536
        results = [{"chunk_id": "1", "text": "RAG..."}]
        
        cache.put(query, embedding, results, latency_ms=1000)
        
        assert len(cache.cache) == 1
        assert query in cache.cache
    
    def test_cache_similarity_match(self, cache):
        """Test semantic similarity matching"""
        query1 = "What is RAG?"
        embedding1 = [0.1] * 1536
        results1 = [{"chunk_id": "1", "text": "RAG..."}]
        
        cache.put(query1, embedding1, results1, latency_ms=1000)
        
        # Very similar embedding
        embedding2 = [0.105] * 1536
        cached = cache.get_similar("What is RAG called?", embedding2)
        
        assert cached is not None
        assert cached.query == query1
    
    def test_cache_no_match_different_query(self, cache):
        """Test no match for very different query"""
        query1 = "What is RAG?"
        embedding1 = [0.1] * 1536
        results1 = [{"chunk_id": "1"}]
        
        cache.put(query1, embedding1, results1, latency_ms=1000)
        
        # Very different embedding
        embedding2 = [0.5] * 1536
        cached = cache.get_similar("Python programming", embedding2)
        
        assert cached is None
    
    def test_cache_hit_rate(self, cache):
        """Test hit rate calculation"""
        assert cache.hit_rate() == 0.0
        
        # Add to cache
        cache.put("Q1", [0.1] * 1536, [], 1000)
        
        # Simulate hits and misses
        cache.get_similar("Q1", [0.105] * 1536)  # Hit
        cache.get_similar("Q2", [0.5] * 1536)    # Miss
        cache.get_similar("Q1", [0.105] * 1536)  # Hit
        
        assert cache.stats["hits"] == 2
        assert cache.stats["misses"] == 1
        assert abs(cache.hit_rate() - 2/3) < 0.01
    
    def test_cache_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = QueryCache(max_size=2)
        
        cache.put("Q1", [0.1] * 1536, [], 1000)
        cache.put("Q2", [0.2] * 1536, [], 1000)
        cache.put("Q3", [0.3] * 1536, [], 1000)  # Should evict Q1
        
        assert len(cache.cache) == 2
        assert "Q3" in cache.cache
        assert "Q2" in cache.cache
        assert "Q1" not in cache.cache
        assert cache.stats["evictions"] == 1


class TestLatencyProfiler:
    """Test latency profiling"""
    
    @pytest.fixture
    def profiler(self):
        return LatencyProfiler()
    
    def test_profiler_record(self, profiler):
        """Test recording latencies"""
        profiler.record("embedding", 50.0)
        profiler.record("embedding", 55.0)
        profiler.record("search", 30.0)
        
        assert len(profiler.timings["embedding"]) == 2
        assert len(profiler.timings["search"]) == 1
    
    def test_profiler_stats(self, profiler):
        """Test stats calculation"""
        profiler.record("embedding", 50.0)
        profiler.record("embedding", 60.0)
        profiler.record("embedding", 40.0)
        
        stats = profiler.get_stats("embedding")
        
        assert stats["count"] == 3
        assert stats["min_ms"] == 40.0
        assert stats["max_ms"] == 60.0
        assert abs(stats["avg_ms"] - 50.0) < 0.1
    
    def test_profiler_context_manager(self, profiler):
        """Test context manager for profiling"""
        import time
        
        with profiler.profile("test_stage"):
            time.sleep(0.01)
        
        assert "test_stage" in profiler.timings
        assert len(profiler.timings["test_stage"]) == 1
        assert profiler.timings["test_stage"][0] > 5  # At least 5ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
