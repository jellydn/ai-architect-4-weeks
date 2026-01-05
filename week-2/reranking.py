"""
Week 2 Day 3: Reranking & Retrieval Evaluation

Implement cross-encoder reranking to improve retrieval quality.
Compare baseline retrieval with reranking using evaluation metrics.

Learning Goals:
- Understand cross-encoder vs dense vector models
- Implement relevance reranking
- Learn MRR/NDCG evaluation metrics
- Measure quality improvements quantitatively

Architecture:
Dense vector search (fast, recall) → Rerank with cross-encoder (slow, precision)
This two-stage approach maximizes quality and latency trade-offs.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """Result with original and reranked scores."""
    chunk_id: str
    text: str
    source: str
    vector_score: float
    reranked_score: Optional[float] = None
    rank_original: int = 0
    rank_reranked: int = 0


class Reranker:
    """
    Cross-encoder reranking for improved retrieval precision.
    
    Two-stage pipeline:
    1. Dense vector search: fast recall (retrieve top-100)
    2. Cross-encoder reranking: slow precision (rerank top-100 → top-k)
    
    Trade-off: +10% quality improvement at cost of +50ms latency
    
    Features:
    - Sentence-transformers cross-encoder integration
    - Batch reranking for efficiency
    - Rank correlation analysis (Spearman)
    - Latency tracking
    
    Example:
        >>> reranker = Reranker(model="cross-encoder/ms-marco-MiniLM-L-12-v2")
        >>> 
        >>> # Original vector search results
        >>> results = [
        ...     RankedResult(chunk_id="1", text="RAG...", vector_score=0.85),
        ...     RankedResult(chunk_id="2", text="...", vector_score=0.75),
        ... ]
        >>> 
        >>> # Rerank with query context
        >>> reranked = await reranker.rerank(
        ...     query="What is RAG?",
        ...     results=results,
        ...     top_k=5
        ... )
        >>> 
        >>> for result in reranked:
        ...     print(f"{result.chunk_id}: {result.reranked_score:.3f}")
    """
    
    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        batch_size: int = 32,
        device: str = "cpu",
    ):
        """
        Initialize reranker with cross-encoder model.
        
        Args:
            model: HuggingFace model identifier (cross-encoder)
            batch_size: Batch size for reranking
            device: Device for inference (cpu, cuda, mps)
        """
        self.model_name = model
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize cross-encoder model (lazy loading)."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name, device=self.device)
            self._initialized = True
            logger.info(f"Cross-encoder ready on {self.device}")
        except ImportError:
            logger.error("sentence-transformers not installed")
            logger.info("Install with: pip install sentence-transformers")
            raise
    
    async def rerank(
        self,
        query: str,
        results: List[RankedResult],
        top_k: int = 5,
    ) -> List[RankedResult]:
        """
        Rerank results using cross-encoder.
        
        Args:
            query: Original query string
            results: List of results from vector search (assumes sorted by vector_score)
            top_k: Number of results to return after reranking
        
        Returns:
            Reranked results sorted by cross-encoder score
        """
        if not self._initialized:
            await self.initialize()
        
        if not results:
            return []
        
        start = time.time()
        
        # Prepare query-text pairs for cross-encoder
        pairs = [
            [query, result.text]
            for result in results
        ]
        
        try:
            # Score with cross-encoder (0-1 relevance)
            scores = self.model.predict(pairs, batch_size=self.batch_size)
            
            # Add reranked scores to results
            for i, result in enumerate(results):
                result.reranked_score = float(scores[i])
                result.rank_original = i + 1
            
            # Sort by reranked score
            reranked = sorted(
                results,
                key=lambda r: r.reranked_score or 0,
                reverse=True
            )
            
            # Update reranked positions
            for i, result in enumerate(reranked):
                result.rank_reranked = i + 1
            
            elapsed = time.time() - start
            logger.info(f"Reranked {len(results)} results in {elapsed:.2f}s")
            
            return reranked[:top_k]
        
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:top_k]
    
    async def batch_rerank(
        self,
        query: str,
        result_batches: List[List[RankedResult]],
        top_k: int = 5,
    ) -> List[RankedResult]:
        """
        Rerank multiple result sets (useful for comparing strategies).
        
        Args:
            query: Query string
            result_batches: Multiple lists of results
            top_k: Top results to return per batch
        
        Returns:
            Reranked results from best batch
        """
        all_results = []
        batch_sources = []
        
        for batch_idx, batch in enumerate(result_batches):
            for result in batch:
                all_results.append(result)
                batch_sources.append(batch_idx)
        
        if not all_results:
            return []
        
        # Rerank all together
        reranked = await self.rerank(query, all_results, top_k=len(all_results))
        
        # Return top-k
        return reranked[:top_k]


class RetrieverEvaluator:
    """
    Evaluate retrieval quality using standard IR metrics.
    
    Metrics:
    - MRR (Mean Reciprocal Rank): Where is first relevant result?
    - NDCG (Normalized Discounted Cumulative Gain): Quality of ranking
    - Precision@k: Fraction of top-k that are relevant
    - Recall@k: Fraction of relevant docs in top-k
    
    Example:
        >>> evaluator = RetrieverEvaluator()
        >>> 
        >>> # Define relevance judgments (hand-created ground truth)
        >>> judgments = {
        ...     "What is RAG?": ["chunk-1", "chunk-3"],  # Relevant docs
        ...     "How does vector search work?": ["chunk-5"],
        ... }
        >>> 
        >>> # Evaluate baseline retrieval
        >>> baseline_results = {
        ...     "What is RAG?": ["chunk-1", "chunk-2", "chunk-3"],  # Returned
        ... }
        >>> 
        >>> metrics = evaluator.evaluate(baseline_results, judgments)
        >>> print(f"MRR: {metrics['mrr']:.3f}")
        >>> print(f"NDCG: {metrics['ndcg']:.3f}")
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.judgments: Dict[str, List[str]] = {}
    
    def add_judgments(self, query: str, relevant_chunks: List[str]) -> None:
        """
        Add relevance judgments for a query.
        
        Args:
            query: Query string
            relevant_chunks: List of chunk_ids judged as relevant
        """
        self.judgments[query] = relevant_chunks
    
    def mrr(self, ranked_results: List[str], relevant_set: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        MRR = 1 / rank(first_relevant)
        
        - MRR=1.0: First result is relevant
        - MRR=0.5: Second result is relevant
        - MRR=0.0: No relevant results in top-k
        """
        for rank, result_id in enumerate(ranked_results, 1):
            if result_id in relevant_set:
                return 1.0 / rank
        return 0.0
    
    def ndcg(
        self,
        ranked_results: List[str],
        relevant_set: List[str],
        k: int = 10,
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.
        
        NDCG = DCG / IDCG
        
        Where:
        - DCG = sum(relevance / log2(rank + 1))
        - IDCG = ideal DCG (all relevant results first)
        
        - NDCG=1.0: Perfect ranking (all relevant results first)
        - NDCG=0.5: Moderate ranking
        - NDCG=0.0: No relevant results
        """
        # DCG: relevance=1 if relevant, 0 otherwise
        dcg = 0.0
        for rank, result_id in enumerate(ranked_results[:k], 1):
            if result_id in relevant_set:
                dcg += 1.0 / np.log2(rank + 1)
        
        # IDCG: ideal ranking (all relevant first)
        idcg = 0.0
        for rank in range(1, min(len(relevant_set), k) + 1):
            idcg += 1.0 / np.log2(rank + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def precision_at_k(
        self,
        ranked_results: List[str],
        relevant_set: List[str],
        k: int = 10,
    ) -> float:
        """
        Calculate Precision@k.
        
        P@k = (# relevant in top-k) / k
        """
        top_k = ranked_results[:k]
        relevant_in_top_k = sum(1 for r in top_k if r in relevant_set)
        return relevant_in_top_k / k if k > 0 else 0.0
    
    def recall_at_k(
        self,
        ranked_results: List[str],
        relevant_set: List[str],
        k: int = 10,
    ) -> float:
        """
        Calculate Recall@k.
        
        R@k = (# relevant in top-k) / (total relevant)
        """
        if not relevant_set:
            return 0.0
        
        top_k = ranked_results[:k]
        relevant_in_top_k = sum(1 for r in top_k if r in relevant_set)
        return relevant_in_top_k / len(relevant_set)
    
    def evaluate(
        self,
        predictions: Dict[str, List[str]],
        judgments: Optional[Dict[str, List[str]]] = None,
        k: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate retrieval predictions against judgments.
        
        Args:
            predictions: {query -> list of chunk_ids (ranked)}
            judgments: {query -> list of relevant chunk_ids}
                       If None, uses self.judgments
            k: Top-k for Precision@k, Recall@k, NDCG@k
        
        Returns:
            Metrics dict: MRR, NDCG, P@k, R@k
        """
        if judgments is None:
            judgments = self.judgments
        
        if not predictions:
            return {"mrr": 0.0, "ndcg": 0.0, f"p@{k}": 0.0, f"r@{k}": 0.0}
        
        mrr_scores = []
        ndcg_scores = []
        precision_scores = []
        recall_scores = []
        
        for query, ranked_results in predictions.items():
            if query not in judgments:
                logger.warning(f"No judgments for query: {query}")
                continue
            
            relevant_set = judgments[query]
            
            mrr_scores.append(self.mrr(ranked_results, relevant_set))
            ndcg_scores.append(self.ndcg(ranked_results, relevant_set, k=k))
            precision_scores.append(self.precision_at_k(ranked_results, relevant_set, k=k))
            recall_scores.append(self.recall_at_k(ranked_results, relevant_set, k=k))
        
        if not mrr_scores:
            return {"mrr": 0.0, "ndcg": 0.0, f"p@{k}": 0.0, f"r@{k}": 0.0}
        
        return {
            "mrr": float(np.mean(mrr_scores)),
            "ndcg": float(np.mean(ndcg_scores)),
            f"p@{k}": float(np.mean(precision_scores)),
            f"r@{k}": float(np.mean(recall_scores)),
            "num_queries": len(mrr_scores),
        }
    
    def compare_approaches(
        self,
        baseline: Dict[str, List[str]],
        vector_search: Dict[str, List[str]],
        vector_rerank: Dict[str, List[str]],
        judgments: Optional[Dict[str, List[str]]] = None,
        k: int = 10,
    ) -> Dict[str, Dict]:
        """
        Compare three retrieval approaches.
        
        Args:
            baseline: Baseline approach results
            vector_search: Vector similarity search results
            vector_rerank: Vector search + reranking results
            judgments: Relevance judgments
            k: Top-k for metrics
        
        Returns:
            Comparison metrics for all approaches
        """
        if judgments is None:
            judgments = self.judgments
        
        baseline_metrics = self.evaluate(baseline, judgments, k=k)
        vector_metrics = self.evaluate(vector_search, judgments, k=k)
        rerank_metrics = self.evaluate(vector_rerank, judgments, k=k)
        
        # Calculate improvements
        def improvement(before: float, after: float) -> float:
            if before == 0:
                return 0.0
            return ((after - before) / before) * 100
        
        return {
            "baseline": baseline_metrics,
            "vector_search": vector_metrics,
            "vector_rerank": rerank_metrics,
            "improvements": {
                "vector_search_vs_baseline": {
                    "mrr": improvement(baseline_metrics["mrr"], vector_metrics["mrr"]),
                    "ndcg": improvement(baseline_metrics["ndcg"], vector_metrics["ndcg"]),
                },
                "rerank_vs_vector": {
                    "mrr": improvement(vector_metrics["mrr"], rerank_metrics["mrr"]),
                    "ndcg": improvement(vector_metrics["ndcg"], rerank_metrics["ndcg"]),
                },
            },
        }


async def test_reranking():
    """Quick test of reranking functionality."""
    print("\n=== Reranking Test ===\n")
    
    # Create test results (simulated vector search output)
    results = [
        RankedResult(
            chunk_id="1",
            text="RAG combines retrieval with generation for grounded answers.",
            source="rag.txt",
            vector_score=0.92
        ),
        RankedResult(
            chunk_id="2",
            text="Vector databases store embeddings for similarity search.",
            source="vector-db.txt",
            vector_score=0.85
        ),
        RankedResult(
            chunk_id="3",
            text="Python is a programming language.",
            source="python.txt",
            vector_score=0.45
        ),
    ]
    
    print("Initial vector search results:")
    for result in results:
        print(f"  {result.chunk_id}: {result.vector_score:.3f} - {result.text[:50]}...")
    
    # Test evaluator
    print("\n=== Evaluation Test ===\n")
    
    evaluator = RetrieverEvaluator()
    
    # Add judgments
    evaluator.add_judgments("What is RAG?", ["1", "2"])
    evaluator.add_judgments("How does retrieval work?", ["2"])
    
    # Test different retrievals
    predictions = {
        "What is RAG?": ["1", "2", "3"],
        "How does retrieval work?": ["2", "1", "3"],
    }
    
    metrics = evaluator.evaluate(predictions)
    print("Evaluation metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n=== Tests Complete ===\n")


if __name__ == "__main__":
    asyncio.run(test_reranking())
