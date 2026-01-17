from __future__ import annotations
"""
Tests for evaluation metrics.
"""

import pytest

from eval.metrics import (
    normalize_answer,
    exact_match,
    f1_score,
    supporting_fact_f1,
    Metrics,
    SingleResult,
    LatencyMetrics,
    CostMetrics,
)


class TestNormalizeAnswer:
    """Tests for answer normalization."""
    
    def test_lowercase(self):
        """Test lowercasing."""
        assert normalize_answer("HELLO") == "hello"
    
    def test_remove_articles(self):
        """Test article removal."""
        assert normalize_answer("the quick brown fox") == "quick brown fox"
        assert normalize_answer("a dog") == "dog"
        assert normalize_answer("an apple") == "apple"
    
    def test_remove_punctuation(self):
        """Test punctuation removal."""
        assert normalize_answer("Hello, world!") == "hello world"
        assert normalize_answer("What's up?") == "whats up"
    
    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        assert normalize_answer("hello   world") == "hello world"
        assert normalize_answer("  hello  ") == "hello"


class TestExactMatch:
    """Tests for exact match metric."""
    
    def test_exact_match(self):
        """Test exact match."""
        assert exact_match("Paris", "Paris") == 1.0
        assert exact_match("paris", "PARIS") == 1.0  # Case insensitive
    
    def test_no_match(self):
        """Test no match."""
        assert exact_match("Paris", "London") == 0.0
    
    def test_article_handling(self):
        """Test articles are ignored."""
        assert exact_match("the Paris", "Paris") == 1.0
        assert exact_match("a dog", "dog") == 1.0


class TestF1Score:
    """Tests for F1 score metric."""
    
    def test_perfect_match(self):
        """Test perfect match gives F1=1."""
        assert f1_score("Paris is beautiful", "Paris is beautiful") == 1.0
    
    def test_partial_match(self):
        """Test partial match."""
        f1 = f1_score("Paris is beautiful", "Paris is great")
        assert 0 < f1 < 1  # Should be between 0 and 1
    
    def test_no_overlap(self):
        """Test no overlap gives F1=0."""
        assert f1_score("apple orange", "banana grape") == 0.0
    
    def test_empty_strings(self):
        """Test empty strings."""
        assert f1_score("", "") == 1.0
        assert f1_score("hello", "") == 0.0
        assert f1_score("", "hello") == 0.0


class TestSupportingFactF1:
    """Tests for supporting fact F1."""
    
    def test_perfect_match(self):
        """Test perfect match."""
        pred = ["Article A", "Article B"]
        gold = ["Article A", "Article B"]
        
        assert supporting_fact_f1(pred, gold) == 1.0
    
    def test_partial_match(self):
        """Test partial match."""
        pred = ["Article A", "Article C"]
        gold = ["Article A", "Article B"]
        
        f1 = supporting_fact_f1(pred, gold)
        assert 0 < f1 < 1
    
    def test_no_match(self):
        """Test no match."""
        pred = ["Article C", "Article D"]
        gold = ["Article A", "Article B"]
        
        assert supporting_fact_f1(pred, gold) == 0.0


class TestMetrics:
    """Tests for Metrics utility class."""
    
    def test_compute_accuracy(self):
        """Test computing accuracy metrics."""
        metrics = Metrics.compute_accuracy(
            prediction="Paris",
            ground_truth="Paris, France",
        )
        
        assert "exact_match" in metrics
        assert "f1" in metrics
        assert metrics["f1"] > 0  # Should have some overlap
    
    def test_aggregate_results(self):
        """Test aggregating results."""
        results = [
            SingleResult(
                example_id="1",
                question="Q1",
                prediction="A",
                ground_truth="A",
                exact_match=1.0,
                f1=1.0,
                latency=LatencyMetrics(total_ms=100),
                cost=CostMetrics(num_llm_calls=2),
            ),
            SingleResult(
                example_id="2",
                question="Q2",
                prediction="B",
                ground_truth="C",
                exact_match=0.0,
                f1=0.5,
                latency=LatencyMetrics(total_ms=200),
                cost=CostMetrics(num_llm_calls=3),
            ),
        ]
        
        aggregated = Metrics.aggregate_results("test_method", results)
        
        assert aggregated.method_name == "test_method"
        assert aggregated.num_examples == 2
        assert aggregated.avg_exact_match == 0.5
        assert aggregated.avg_f1 == 0.75
        assert aggregated.latency_mean_ms == 150.0
        assert aggregated.avg_llm_calls == 2.5
