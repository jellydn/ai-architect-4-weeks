"""Tests for reranking and evaluation modules"""

import pytest
from week_2.reranking import RetrieverEvaluator, RankedResult


class TestRetrieverEvaluator:
    """Test evaluation metrics"""
    
    @pytest.fixture
    def evaluator(self):
        return RetrieverEvaluator()
    
    def test_mrr_first_relevant(self, evaluator):
        """MRR should be 1.0 when first result is relevant"""
        ranked = ["chunk-1", "chunk-2", "chunk-3"]
        relevant = ["chunk-1", "chunk-5"]
        
        assert evaluator.mrr(ranked, relevant) == 1.0
    
    def test_mrr_second_relevant(self, evaluator):
        """MRR should be 0.5 when second result is relevant"""
        ranked = ["chunk-1", "chunk-2", "chunk-3"]
        relevant = ["chunk-2", "chunk-5"]
        
        assert evaluator.mrr(ranked, relevant) == 0.5
    
    def test_mrr_no_relevant(self, evaluator):
        """MRR should be 0.0 when no relevant results"""
        ranked = ["chunk-1", "chunk-2", "chunk-3"]
        relevant = ["chunk-4", "chunk-5"]
        
        assert evaluator.mrr(ranked, relevant) == 0.0
    
    def test_ndcg_perfect_ranking(self, evaluator):
        """NDCG should be 1.0 for perfect ranking"""
        ranked = ["chunk-1", "chunk-2", "chunk-3"]
        relevant = ["chunk-1", "chunk-2"]
        
        ndcg_score = evaluator.ndcg(ranked, relevant, k=3)
        assert ndcg_score == 1.0
    
    def test_ndcg_no_relevant(self, evaluator):
        """NDCG should be 0.0 when no relevant results"""
        ranked = ["chunk-1", "chunk-2", "chunk-3"]
        relevant = []
        
        assert evaluator.ndcg(ranked, relevant, k=3) == 0.0
    
    def test_precision_at_k(self, evaluator):
        """Precision@3 should be 2/3 for 2 relevant in top-3"""
        ranked = ["chunk-1", "chunk-2", "chunk-3"]
        relevant = ["chunk-1", "chunk-3"]
        
        precision = evaluator.precision_at_k(ranked, relevant, k=3)
        assert abs(precision - 2/3) < 0.01
    
    def test_recall_at_k(self, evaluator):
        """Recall@3 should be 2/3 when 3 total relevant, 2 in top-3"""
        ranked = ["chunk-1", "chunk-2", "chunk-3"]
        relevant = ["chunk-1", "chunk-2", "chunk-4"]
        
        recall = evaluator.recall_at_k(ranked, relevant, k=3)
        assert abs(recall - 2/3) < 0.01
    
    def test_evaluate(self, evaluator):
        """Test evaluate() method"""
        predictions = {
            "query-1": ["chunk-1", "chunk-2", "chunk-3"],
            "query-2": ["chunk-4", "chunk-5"],
        }
        
        judgments = {
            "query-1": ["chunk-1", "chunk-3"],
            "query-2": ["chunk-4"],
        }
        
        metrics = evaluator.evaluate(predictions, judgments)
        
        assert "mrr" in metrics
        assert "ndcg" in metrics
        assert metrics["mrr"] > 0
        assert metrics["ndcg"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
