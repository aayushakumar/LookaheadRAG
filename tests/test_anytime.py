"""
Tests for Anytime Optimizer.

Tests the dependency-constrained DP optimizer, quality function,
and Pareto curve generation.
"""

import pytest
from src.planner.schema import PlanNode, PlanGraph, OperatorType
from src.engine.anytime_optimizer import (
    AnytimeOptimizer,
    AnytimeConfig,
    OptimizationResult,
    ParetoPoint,
    BudgetedPrunerV2,
)


class TestAnytimeOptimizer:
    """Tests for AnytimeOptimizer."""
    
    def test_optimize_empty_plan(self):
        """Test optimizing an empty plan."""
        optimizer = AnytimeOptimizer()
        plan = PlanGraph(question="Test?", nodes=[])
        
        result, quality = optimizer.optimize(plan, budget=5)
        
        assert len(result.nodes) == 0
        assert quality == 0.0
    
    def test_optimize_under_budget(self):
        """Test that plans under budget are returned as-is."""
        optimizer = AnytimeOptimizer()
        plan = PlanGraph(
            question="Test?",
            nodes=[
                PlanNode(id="n1", query="query 1", confidence=0.8, budget_cost=1),
                PlanNode(id="n2", query="query 2", confidence=0.7, budget_cost=1),
            ],
        )
        
        result, quality = optimizer.optimize(plan, budget=10)
        
        assert len(result.nodes) == 2
        assert quality > 0.0
    
    def test_optimize_respects_budget(self):
        """Test that optimization respects budget constraint."""
        optimizer = AnytimeOptimizer()
        plan = PlanGraph(
            question="Test?",
            nodes=[
                PlanNode(id="n1", query="query 1", confidence=0.9, budget_cost=3),
                PlanNode(id="n2", query="query 2", confidence=0.8, budget_cost=3),
                PlanNode(id="n3", query="query 3", confidence=0.7, budget_cost=3),
            ],
        )
        
        result, quality = optimizer.optimize(plan, budget=5)
        
        # Should select at most 1 node (each costs 3)
        assert result.total_budget_cost() <= 5
        assert len(result.nodes) <= 1
    
    def test_optimize_respects_dependencies(self):
        """Test that optimization respects dependency constraints."""
        optimizer = AnytimeOptimizer()
        plan = PlanGraph(
            question="Test?",
            nodes=[
                PlanNode(id="n1", query="base query", confidence=0.5, budget_cost=1),
                PlanNode(
                    id="n2",
                    query="dependent query",
                    confidence=0.95,  # Higher confidence
                    budget_cost=1,
                    depends_on=["n1"],
                ),
            ],
        )
        
        # With budget=1, can only select one node
        result, quality = optimizer.optimize(plan, budget=1)
        
        # If n2 is selected, n1 must also be selected (but budget is 1)
        # So only n1 should be selected
        if "n2" in [n.id for n in result.nodes]:
            assert "n1" in [n.id for n in result.nodes]
    
    def test_optimize_prefers_higher_quality(self):
        """Test that optimizer prefers higher quality nodes."""
        optimizer = AnytimeOptimizer()
        plan = PlanGraph(
            question="Test?",
            nodes=[
                PlanNode(id="n1", query="low quality", confidence=0.3, budget_cost=1),
                PlanNode(id="n2", query="high quality", confidence=0.9, budget_cost=1),
            ],
        )
        
        result, quality = optimizer.optimize(plan, budget=1)
        
        # Should prefer n2 (higher confidence)
        assert len(result.nodes) == 1
        assert result.nodes[0].id == "n2"
    
    def test_topological_sort(self):
        """Test that topological sort orders dependencies correctly."""
        optimizer = AnytimeOptimizer()
        nodes = [
            PlanNode(id="n3", query="depends on n2", depends_on=["n2"]),
            PlanNode(id="n1", query="root"),
            PlanNode(id="n2", query="depends on n1", depends_on=["n1"]),
        ]
        
        sorted_nodes = optimizer._topological_sort(nodes)
        
        # n1 should come before n2, n2 before n3
        ids = [n.id for n in sorted_nodes]
        assert ids.index("n1") < ids.index("n2")
        assert ids.index("n2") < ids.index("n3")


class TestExpectedQuality:
    """Tests for quality estimation function."""
    
    def test_empty_nodes_zero_quality(self):
        """Test that empty node list gives zero quality."""
        optimizer = AnytimeOptimizer()
        plan = PlanGraph(question="Test?", nodes=[])
        
        quality = optimizer.expected_quality([], plan)
        
        assert quality == 0.0
    
    def test_high_confidence_high_quality(self):
        """Test that high confidence nodes give high quality."""
        optimizer = AnytimeOptimizer()
        high_conf_nodes = [
            PlanNode(id="n1", query="q1", confidence=0.95),
            PlanNode(id="n2", query="q2", confidence=0.90),
        ]
        low_conf_nodes = [
            PlanNode(id="n1", query="q1", confidence=0.3),
            PlanNode(id="n2", query="q2", confidence=0.2),
        ]
        plan = PlanGraph(question="Test?", nodes=[])
        
        high_quality = optimizer.expected_quality(high_conf_nodes, plan)
        low_quality = optimizer.expected_quality(low_conf_nodes, plan)
        
        assert high_quality > low_quality
    
    def test_diversity_bonus(self):
        """Test that diverse operators give quality bonus."""
        optimizer = AnytimeOptimizer()
        diverse_nodes = [
            PlanNode(id="n1", query="q1", op=OperatorType.LOOKUP, confidence=0.8),
            PlanNode(id="n2", query="q2", op=OperatorType.BRIDGE, confidence=0.8),
            PlanNode(id="n3", query="q3", op=OperatorType.COMPARE, confidence=0.8),
        ]
        same_op_nodes = [
            PlanNode(id="n1", query="q1", op=OperatorType.LOOKUP, confidence=0.8),
            PlanNode(id="n2", query="q2", op=OperatorType.LOOKUP, confidence=0.8),
            PlanNode(id="n3", query="q3", op=OperatorType.LOOKUP, confidence=0.8),
        ]
        plan = PlanGraph(question="Test?", nodes=[])
        
        diverse_quality = optimizer.expected_quality(diverse_nodes, plan)
        same_quality = optimizer.expected_quality(same_op_nodes, plan)
        
        # Diverse should have higher quality due to diversity bonus
        assert diverse_quality > same_quality


class TestParetoGeneration:
    """Tests for Pareto curve generation."""
    
    def test_generate_pareto_curve(self):
        """Test generating Pareto frontier points."""
        optimizer = AnytimeOptimizer()
        plan = PlanGraph(
            question="Test?",
            nodes=[
                PlanNode(id="n1", query="q1", confidence=0.8, budget_cost=1),
                PlanNode(id="n2", query="q2", confidence=0.7, budget_cost=2),
                PlanNode(id="n3", query="q3", confidence=0.6, budget_cost=3),
            ],
        )
        
        points = optimizer.generate_pareto_curve(plan, budget_levels=[1, 2, 3, 5])
        
        assert len(points) == 4
        assert all(isinstance(p, ParetoPoint) for p in points)
    
    def test_pareto_budget_increases_coverage(self):
        """Test that higher budget allows more nodes (coverage increases)."""
        optimizer = AnytimeOptimizer()
        plan = PlanGraph(
            question="Test?",
            nodes=[
                PlanNode(id="n1", query="q1", confidence=0.8, budget_cost=1),
                PlanNode(id="n2", query="q2", confidence=0.7, budget_cost=1),
                PlanNode(id="n3", query="q3", confidence=0.6, budget_cost=1),
            ],
        )
        
        points = optimizer.generate_pareto_curve(plan, budget_levels=[1, 2, 3])
        
        # Node count should be non-decreasing with budget
        for i in range(len(points) - 1):
            assert points[i + 1].num_nodes >= points[i].num_nodes


class TestBudgetedPrunerV2:
    """Tests for backward-compatible wrapper."""
    
    def test_prune_returns_pruning_result(self):
        """Test that wrapper returns PruningResult."""
        pruner = BudgetedPrunerV2()
        plan = PlanGraph(
            question="Test?",
            nodes=[
                PlanNode(id="n1", query="q1", confidence=0.8, budget_cost=2),
                PlanNode(id="n2", query="q2", confidence=0.7, budget_cost=2),
            ],
        )
        
        result = pruner.prune(plan, budget=3)
        
        from src.engine.pruning import PruningResult
        assert isinstance(result, PruningResult)
        assert result.budget_used <= 3
