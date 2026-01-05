"""
Week 2 Day 4: Caching & Performance Tuning

Implement query result caching and latency profiling.
Goal: Reduce E2E latency with sub-10ms cache hits.

Learning Goals:
- Understand cache invalidation strategies
- Implement semantic similarity-based cache matching
- Measure cache effectiveness (hit rate, latency)
- Profile ML pipeline bottlenecks

Architecture:
Query Cache (LRU) → Semantic matching (query embeddings) → Latency profiling
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cached query result."""
    query: str
    embedding: List[float]
    results: List[Dict[str, Any]]
    latency_ms: float
    timestamp: float
    hits: int = 0


class QueryCache:
    """
    LRU cache for query results with semantic similarity matching.
    
    Strategy:
    - Store query embeddings + results
    - For new query: find semantically similar cached queries
    - Return cached result if similarity > threshold
    - Evict least-recently-used when full
    
    Trade-off:
    - Cache hit: ~1ms latency (dict lookup + embedding comparison)
    - Cache miss: ~1000ms latency (full retrieval)
    - Hit rate > 30% on typical workloads
    
    Example:
        >>> cache = QueryCache(max_size=1000, similarity_threshold=0.95)
        >>> 
        >>> # Check cache for similar query
        >>> cached = cache.get_similar("What is RAG?", query_embedding)
        >>> if cached:
        ...     return cached.results  # ~1ms
        >>> 
        >>> # Not in cache, retrieve normally
        >>> results = await retrieve(query)
        >>> cache.put(query, query_embedding, results, latency_ms)
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        similarity_threshold: float = 0.95,
    ):
        """
        Initialize query cache.
        
        Args:
            max_size: Maximum number of cached queries
            similarity_threshold: Similarity threshold for cache hits (0-1)
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_saved_latency_ms": 0.0,
        }
    
    def _similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        if not v1 or not v2:
            return 0.0
        
        a = np.array(v1)
        b = np.array(v2)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def get_similar(
        self,
        query: str,
        query_embedding: List[float],
    ) -> Optional[CacheEntry]:
        """
        Look up cache for semantically similar query.
        
        Args:
            query: Query string
            query_embedding: Query embedding vector
        
        Returns:
            Cached entry if found (similarity > threshold), else None
        """
        best_match = None
        best_similarity = 0.0
        
        for cached_query, entry in self.cache.items():
            similarity = self._similarity(query_embedding, entry.embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
        
        if best_similarity >= self.similarity_threshold:
            # Update hit count
            best_match.hits += 1
            
            # Move to end (LRU)
            self.cache.move_to_end(best_match.query)
            
            self.stats["hits"] += 1
            self.stats["total_saved_latency_ms"] += best_match.latency_ms
            
            logger.debug(
                f"Cache hit (similarity={best_similarity:.3f}): "
                f"'{query[:50]}' → '{best_match.query[:50]}'"
            )
            
            return best_match
        
        self.stats["misses"] += 1
        return None
    
    def put(
        self,
        query: str,
        query_embedding: List[float],
        results: List[Dict[str, Any]],
        latency_ms: float,
    ) -> None:
        """
        Store query result in cache.
        
        Args:
            query: Query string
            query_embedding: Query embedding
            results: Retrieval results
            latency_ms: Latency of retrieval
        """
        entry = CacheEntry(
            query=query,
            embedding=query_embedding,
            results=results,
            latency_ms=latency_ms,
            timestamp=time.time(),
        )
        
        # Add to cache
        self.cache[query] = entry
        self.cache.move_to_end(query)
        
        # Evict oldest if full
        if len(self.cache) > self.max_size:
            oldest_query, oldest_entry = self.cache.popitem(last=False)
            self.stats["evictions"] += 1
            logger.debug(f"Evicted cache entry: '{oldest_query[:50]}'")
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.stats["hits"] + self.stats["misses"]
        if total == 0:
            return 0.0
        return self.stats["hits"] / total
    
    def avg_latency_saved_per_hit(self) -> float:
        """Average latency saved per cache hit."""
        if self.stats["hits"] == 0:
            return 0.0
        return self.stats["total_saved_latency_ms"] / self.stats["hits"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": self.hit_rate(),
            "evictions": self.stats["evictions"],
            "total_saved_latency_ms": self.stats["total_saved_latency_ms"],
            "avg_latency_saved_per_hit": self.avg_latency_saved_per_hit(),
        }


class LatencyProfiler:
    """
    Profile latency of pipeline stages.
    
    Measures:
    - Embedding generation
    - Vector search
    - Reranking
    - LLM generation
    - Total E2E
    
    Example:
        >>> profiler = LatencyProfiler()
        >>> 
        >>> with profiler.profile("embedding"):
        ...     query_embedding = embed(query)
        >>> 
        >>> with profiler.profile("search"):
        ...     results = retrieve(query_embedding)
        >>> 
        >>> profiler.report()
        """
    
    def __init__(self):
        """Initialize profiler."""
        self.timings: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}
    
    def profile(self, stage: str):
        """Context manager for profiling a stage."""
        return _ProfileContext(self, stage)
    
    def record(self, stage: str, latency_ms: float) -> None:
        """Record latency for a stage."""
        if stage not in self.timings:
            self.timings[stage] = []
        self.timings[stage].append(latency_ms)
    
    def get_stats(self, stage: str) -> Dict[str, float]:
        """Get stats for a stage."""
        if stage not in self.timings or not self.timings[stage]:
            return {}
        
        times = self.timings[stage]
        return {
            "count": len(times),
            "total_ms": sum(times),
            "avg_ms": np.mean(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "p50_ms": np.percentile(times, 50),
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99),
        }
    
    def report(self) -> Dict[str, Dict]:
        """Generate latency report for all stages."""
        report = {}
        
        for stage in self.timings:
            stats = self.get_stats(stage)
            if stats:
                report[stage] = stats
        
        # Calculate E2E if we have all components
        if report:
            total = sum(stats["avg_ms"] for stats in report.values() if "avg_ms" in stats)
            report["e2e"] = {"avg_ms": total}
        
        return report
    
    def print_report(self) -> None:
        """Print formatted latency report."""
        report = self.report()
        
        print("\n=== Latency Profiling Report ===\n")
        
        for stage, stats in sorted(report.items()):
            print(f"{stage.upper()}:")
            for metric, value in stats.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.2f}ms")
                else:
                    print(f"  {metric}: {value}")
            print()


class _ProfileContext:
    """Context manager for profiling a stage."""
    
    def __init__(self, profiler: LatencyProfiler, stage: str):
        self.profiler = profiler
        self.stage = stage
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed_ms = (time.time() - self.start_time) * 1000
            self.profiler.record(self.stage, elapsed_ms)


class PipelineLatencyAnalyzer:
    """
    Analyze E2E latency and identify bottlenecks.
    
    Example:
        >>> analyzer = PipelineLatencyAnalyzer()
        >>> 
        >>> # Run pipeline with profiling
        >>> profiler = LatencyProfiler()
        >>> result = await pipeline(query, profiler)
        >>> 
        >>> analyzer.add_run(profiler)
        >>> analysis = analyzer.analyze()
        >>> print(f"Bottleneck: {analysis['bottleneck']}")
    """
    
    def __init__(self):
        """Initialize analyzer."""
        self.runs: List[LatencyProfiler] = []
    
    def add_run(self, profiler: LatencyProfiler) -> None:
        """Add profiling data from a pipeline run."""
        self.runs.append(profiler)
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze all runs to identify bottlenecks."""
        if not self.runs:
            return {}
        
        # Aggregate all stages
        all_stages = set()
        for profiler in self.runs:
            all_stages.update(profiler.timings.keys())
        
        stage_totals = {}
        for stage in all_stages:
            times = []
            for profiler in self.runs:
                if stage in profiler.timings:
                    times.extend(profiler.timings[stage])
            
            if times:
                stage_totals[stage] = {
                    "avg_ms": np.mean(times),
                    "total_ms": sum(times),
                }
        
        # Find bottleneck
        bottleneck = max(stage_totals.items(), key=lambda x: x[1]["avg_ms"])
        
        # Calculate total E2E
        total_e2e = sum(s["avg_ms"] for s in stage_totals.values())
        
        # Percentage breakdown
        breakdown = {}
        for stage, stats in stage_totals.items():
            percentage = (stats["avg_ms"] / total_e2e * 100) if total_e2e > 0 else 0
            breakdown[stage] = {
                "avg_ms": stats["avg_ms"],
                "percentage": percentage,
            }
        
        return {
            "num_runs": len(self.runs),
            "bottleneck": bottleneck[0],
            "bottleneck_latency_ms": bottleneck[1]["avg_ms"],
            "total_e2e_ms": total_e2e,
            "breakdown": breakdown,
        }


def test_caching():
    """Test caching functionality."""
    print("\n=== Caching Test ===\n")
    
    cache = QueryCache(max_size=100, similarity_threshold=0.90)
    
    # Simulate embeddings
    query1_embedding = [0.1] * 1536
    query2_embedding = [0.105] * 1536  # Very similar to query1
    query3_embedding = [0.5] * 1536    # Different from query1
    
    # Store first query
    results1 = [{"chunk_id": "1", "text": "RAG result"}]
    cache.put("What is RAG?", query1_embedding, results1, latency_ms=1200)
    
    # Try to retrieve similar query
    cached = cache.get_similar("What is RAG called?", query2_embedding)
    if cached:
        print("✓ Cache hit for similar query")
        print(f"  Original: '{cached.query}'")
        print(f"  Hit saved: {cached.latency_ms:.0f}ms")
    
    # Try to retrieve different query
    results3 = [{"chunk_id": "3", "text": "Python result"}]
    cache.put("How to use Python?", query3_embedding, results3, latency_ms=1100)
    
    cached = cache.get_similar("Is Python good?", [0.51] * 1536)
    if cached:
        print("✓ Cache hit for Python query")
    
    # Print stats
    print("\n" + "=" * 40)
    stats = cache.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")


def test_profiling():
    """Test latency profiling."""
    print("\n=== Latency Profiling Test ===\n")
    
    profiler = LatencyProfiler()
    
    # Simulate pipeline stages
    with profiler.profile("embedding"):
        time.sleep(0.05)
    
    with profiler.profile("search"):
        time.sleep(0.10)
    
    with profiler.profile("generation"):
        time.sleep(0.20)
    
    profiler.print_report()


if __name__ == "__main__":
    test_caching()
    test_profiling()
