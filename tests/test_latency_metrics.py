"""
Tests for Latency-Constrained Evaluation Metrics.

Tests Acc@T (conditional and strict) and AULC computation.
"""

import pytest
from eval.metrics import SingleResult, LatencyMetrics
from eval.latency_constrained import (
    LatencyConstrainedEvaluator,
    AccAtTResult,
    LatencyConstrainedResult,
)


def make_result(
    example_id: str,
    correct: bool,
    latency_ms: float,
) -> SingleResult:
    """Helper to create test results."""
    return SingleResult(
        example_id=example_id,
        question="test?",
        prediction="answer" if correct else "wrong",
        ground_truth="answer",
        exact_match=1.0 if correct else 0.0,
        f1=1.0 if correct else 0.0,
        latency=LatencyMetrics(total_ms=latency_ms),
    )


class TestAccAtT:
    """Tests for Acc@T computation."""
    
    def test_all_within_budget(self):
        """Test when all examples are within budget."""
        evaluator = LatencyConstrainedEvaluator()
        results = [
            make_result("1", True, 1000),   # 1s
            make_result("2", True, 1500),   # 1.5s
            make_result("3", False, 1800),  # 1.8s
        ]
        
        acc_at_t = evaluator.compute_acc_at_t(results, t_seconds=2.0)
        
        assert acc_at_t.n_within_budget == 3
        assert acc_at_t.coverage == 1.0
        assert acc_at_t.acc_conditional == 2/3  # 2 correct out of 3
        assert acc_at_t.acc_strict == 2/3
    
    def test_some_over_budget(self):
        """Test with some examples over budget."""
        evaluator = LatencyConstrainedEvaluator()
        results = [
            make_result("1", True, 1000),   # Within 2s
            make_result("2", True, 3000),   # Over 2s
            make_result("3", False, 1500),  # Within 2s
            make_result("4", True, 5000),   # Over 2s
        ]
        
        acc_at_t = evaluator.compute_acc_at_t(results, t_seconds=2.0)
        
        assert acc_at_t.n_within_budget == 2
        assert acc_at_t.n_total == 4
        assert acc_at_t.coverage == 0.5
        
        # Conditional: 1 correct out of 2 within budget
        assert acc_at_t.acc_conditional == 0.5
        
        # Strict: 1 correct out of 4 total
        assert acc_at_t.acc_strict == 0.25
    
    def test_none_within_budget(self):
        """Test when no examples are within budget."""
        evaluator = LatencyConstrainedEvaluator()
        results = [
            make_result("1", True, 5000),
            make_result("2", True, 6000),
        ]
        
        acc_at_t = evaluator.compute_acc_at_t(results, t_seconds=2.0)
        
        assert acc_at_t.n_within_budget == 0
        assert acc_at_t.coverage == 0.0
        assert acc_at_t.acc_conditional == 0.0
        assert acc_at_t.acc_strict == 0.0
    
    def test_empty_results(self):
        """Test with empty results."""
        evaluator = LatencyConstrainedEvaluator()
        
        acc_at_t = evaluator.compute_acc_at_t([], t_seconds=2.0)
        
        assert acc_at_t.n_total == 0
        assert acc_at_t.acc_conditional == 0.0


class TestAULC:
    """Tests for AULC computation."""
    
    def test_constant_accuracy(self):
        """Test AULC with constant accuracy."""
        evaluator = LatencyConstrainedEvaluator(t_values=[2, 4, 6, 8, 10])
        
        # Constant accuracy of 0.5 across all T
        acc_at_t = {2: 0.5, 4: 0.5, 6: 0.5, 8: 0.5, 10: 0.5}
        
        aulc = evaluator.compute_aulc(acc_at_t, t_min=2, t_max=10)
        
        assert abs(aulc - 0.5) < 0.01
    
    def test_increasing_accuracy(self):
        """Test AULC with increasing accuracy."""
        evaluator = LatencyConstrainedEvaluator()
        
        # Linear increase from 0 to 1
        acc_at_t = {2: 0.0, 4: 0.25, 6: 0.5, 8: 0.75, 10: 1.0}
        
        aulc = evaluator.compute_aulc(acc_at_t, t_min=2, t_max=10)
        
        # Area under line from (2,0) to (10,1) = 0.5
        assert abs(aulc - 0.5) < 0.01
    
    def test_empty_acc_at_t(self):
        """Test AULC with empty data."""
        evaluator = LatencyConstrainedEvaluator()
        
        aulc = evaluator.compute_aulc({})
        
        assert aulc == 0.0


class TestLatencyConstrainedEvaluator:
    """Integration tests for the full evaluator."""
    
    def test_full_evaluation(self):
        """Test complete evaluation pipeline."""
        evaluator = LatencyConstrainedEvaluator(t_values=[2, 4, 6])
        results = [
            make_result("1", True, 1000),   # 1s
            make_result("2", False, 2500),  # 2.5s
            make_result("3", True, 3500),   # 3.5s
            make_result("4", True, 5500),   # 5.5s
            make_result("5", False, 1500),  # 1.5s
        ]
        
        lc_result = evaluator.evaluate(results, method_name="test_method")
        
        assert lc_result.method == "test_method"
        assert 2 in lc_result.acc_at_t_conditional
        assert 4 in lc_result.acc_at_t_conditional
        assert 6 in lc_result.acc_at_t_conditional
        assert lc_result.aulc_conditional >= 0.0
        assert lc_result.aulc_strict >= 0.0
    
    def test_compare_methods(self):
        """Test method comparison."""
        evaluator = LatencyConstrainedEvaluator(t_values=[2, 4])
        
        fast_results = [
            make_result("1", True, 1000),
            make_result("2", True, 1500),
        ]
        slow_results = [
            make_result("1", True, 3000),
            make_result("2", True, 4500),
        ]
        
        comparison = evaluator.compare_methods({
            "fast": fast_results,
            "slow": slow_results,
        })
        
        assert "fast" in comparison
        assert "slow" in comparison
        
        # Fast should have better Acc@2s
        assert comparison["fast"].acc_at_t_conditional[2] > comparison["slow"].acc_at_t_conditional[2]
    
    def test_summary_table(self):
        """Test summary table generation."""
        evaluator = LatencyConstrainedEvaluator(t_values=[2, 4])
        results = [make_result("1", True, 1000)]
        
        lc_result = evaluator.evaluate(results, "test")
        table = lc_result.summary_table()
        
        assert "test" in table
        assert "AULC" in table
