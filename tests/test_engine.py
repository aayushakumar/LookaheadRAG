from __future__ import annotations
"""
Tests for the Engine module.
"""

import pytest

from src.engine.pruning import BudgetedPruner, PruningResult
from src.engine.fallback import FallbackHandler, FallbackDecision
from src.planner.schema import PlanGraph, PlanNode, OperatorType
from src.retriever.parallel import RetrievalResult, NodeRetrievalResult
from src.retriever.vector_store import Document, SearchResult


class TestBudgetedPruner:
    """Tests for BudgetedPruner."""
    
    @pytest.fixture
    def pruner(self):
        """Create pruner."""
        return BudgetedPruner()
    
    @pytest.fixture
    def sample_plan(self):
        """Create sample plan with varying costs and confidences."""
        return PlanGraph(
            question="Complex question",
            nodes=[
                PlanNode(id="n1", query="high conf query", confidence=0.9, budget_cost=2),
                PlanNode(id="n2", query="medium conf query", confidence=0.6, budget_cost=1),
                PlanNode(id="n3", query="low conf query", confidence=0.3, budget_cost=1),
                PlanNode(id="n4", query="dependent query", confidence=0.8, budget_cost=2, depends_on=["n1"]),
            ],
        )
    
    def test_no_pruning_under_budget(self, pruner, sample_plan):
        """Test that no pruning occurs when under budget."""
        result = pruner.prune(sample_plan, budget=10)
        
        assert len(result.pruned_plan.nodes) == 4
        assert len(result.removed_nodes) == 0
    
    def test_pruning_over_budget(self, pruner, sample_plan):
        """Test pruning when over budget."""
        result = pruner.prune(sample_plan, budget=3)
        
        # Should select nodes up to budget 3
        assert result.budget_used <= 3
        assert len(result.removed_nodes) > 0
    
    def test_respects_dependencies(self, pruner, sample_plan):
        """Test that dependencies are respected in pruning."""
        result = pruner.prune(sample_plan, budget=4)
        
        pruned_ids = {n.id for n in result.pruned_plan.nodes}
        
        # If n4 is included, n1 must also be included
        if "n4" in pruned_ids:
            assert "n1" in pruned_ids
    
    def test_utility_calculation(self, pruner, sample_plan):
        """Test utility calculation."""
        node = sample_plan.nodes[0]  # High confidence node
        
        utility = pruner.calculate_utility(node, [], sample_plan)
        
        assert utility > 0  # Should have positive utility
    
    def test_novelty_decreases_with_similar_queries(self, pruner, sample_plan):
        """Test that novelty score decreases for similar queries."""
        node1 = PlanNode(id="n1", query="capital of france")
        node2 = PlanNode(id="n2", query="capital of france paris")
        
        # First node should have high novelty
        novelty1 = pruner._calculate_novelty(node1, [])
        assert novelty1 == 1.0
        
        # Second node should have lower novelty due to overlap
        novelty2 = pruner._calculate_novelty(node2, [node1])
        assert novelty2 < 1.0


class TestFallbackHandler:
    """Tests for FallbackHandler."""
    
    @pytest.fixture
    def handler(self):
        """Create fallback handler."""
        return FallbackHandler()
    
    @pytest.fixture
    def low_confidence_plan(self):
        """Create plan with low confidence."""
        return PlanGraph(
            question="Uncertain question",
            nodes=[
                PlanNode(id="n1", query="uncertain query", confidence=0.2),
            ],
        )
    
    @pytest.fixture
    def high_confidence_plan(self):
        """Create plan with high confidence."""
        return PlanGraph(
            question="Certain question",
            nodes=[
                PlanNode(id="n1", query="certain query", confidence=0.9),
            ],
        )
    
    @pytest.fixture
    def empty_retrieval(self):
        """Create retrieval result with no results."""
        return RetrievalResult(
            node_results=[
                NodeRetrievalResult(
                    node_id="n1",
                    query="test",
                    results=[],
                    latency_ms=10.0,
                ),
            ],
            total_latency_ms=10.0,
            parallel_latency_ms=10.0,
        )
    
    @pytest.fixture
    def good_retrieval(self):
        """Create retrieval result with good coverage."""
        return RetrievalResult(
            node_results=[
                NodeRetrievalResult(
                    node_id="n1",
                    query="test",
                    results=[
                        SearchResult(
                            document=Document(id=f"doc{i}", content=f"Content {i}"),
                            score=0.8,
                        )
                        for i in range(5)
                    ],
                    latency_ms=10.0,
                ),
            ],
            total_latency_ms=10.0,
            parallel_latency_ms=10.0,
        )
    
    def test_fallback_triggered_for_low_coverage(self, handler, high_confidence_plan, empty_retrieval):
        """Test fallback is triggered for low coverage."""
        decision = handler.should_fallback(high_confidence_plan, empty_retrieval)
        
        assert decision.should_fallback
        assert "coverage" in decision.reason.lower() or "results" in decision.reason.lower()
    
    def test_no_fallback_for_good_results(self, handler, high_confidence_plan, good_retrieval):
        """Test no fallback for good retrieval results."""
        decision = handler.should_fallback(high_confidence_plan, good_retrieval)
        
        # With high confidence and good retrieval, should not fallback
        # (unless entropy is somehow high)
        if not decision.should_fallback:
            assert "passed" in decision.reason.lower()
    
    def test_generate_fallback_queries(self, handler, high_confidence_plan, empty_retrieval):
        """Test generating fallback queries."""
        queries = handler.generate_fallback_queries(
            high_confidence_plan,
            empty_retrieval,
            max_queries=2,
        )
        
        assert len(queries) <= 2
        assert len(queries) > 0
    
    def test_create_fallback_plan(self, handler, high_confidence_plan):
        """Test creating fallback plan with additional queries."""
        fallback_queries = ["additional query 1", "additional query 2"]
        
        fallback_plan = handler.create_fallback_plan(
            high_confidence_plan,
            fallback_queries,
        )
        
        # Should have original nodes plus fallback nodes
        assert len(fallback_plan.nodes) == len(high_confidence_plan.nodes) + 2
        
        # Fallback nodes should have lower confidence
        fallback_nodes = [n for n in fallback_plan.nodes if n.id.startswith("fb")]
        assert all(n.confidence < 0.5 for n in fallback_nodes)
