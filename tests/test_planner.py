from __future__ import annotations
"""
Tests for the Planner module.
"""

import json
import pytest

from src.planner.schema import PlanGraph, PlanNode, OperatorType, GlobalSettings


class TestPlanNode:
    """Tests for PlanNode model."""
    
    def test_create_basic_node(self):
        """Test creating a basic node."""
        node = PlanNode(
            id="n1",
            query="who wrote hamlet",
            op=OperatorType.LOOKUP,
            depends_on=[],
            confidence=0.8,
        )
        
        assert node.id == "n1"
        assert node.query == "who wrote hamlet"
        assert node.op == OperatorType.LOOKUP
        assert node.confidence == 0.8
        assert node.budget_cost == 1
    
    def test_node_with_dependencies(self):
        """Test node with dependencies."""
        node = PlanNode(
            id="n2",
            query="when was the author born",
            op=OperatorType.LOOKUP,
            depends_on=["n1"],
            confidence=0.7,
        )
        
        assert node.depends_on == ["n1"]
    
    def test_invalid_id_raises_error(self):
        """Test that invalid ID raises error."""
        with pytest.raises(ValueError):
            PlanNode(
                id="n1@invalid",
                query="test",
            )
    
    def test_duplicate_dependencies_raises_error(self):
        """Test that duplicate dependencies raise error."""
        with pytest.raises(ValueError):
            PlanNode(
                id="n1",
                query="test",
                depends_on=["n0", "n0"],
            )
    
    def test_confidence_bounds(self):
        """Test confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            PlanNode(id="n1", query="test", confidence=1.5)
        
        with pytest.raises(ValueError):
            PlanNode(id="n1", query="test", confidence=-0.1)
    
    def test_to_dict(self):
        """Test converting node to dictionary."""
        node = PlanNode(
            id="n1",
            query="test query",
            op=OperatorType.BRIDGE,
            depends_on=["n0"],
            confidence=0.75,
            budget_cost=2,
        )
        
        d = node.to_dict()
        
        assert d["id"] == "n1"
        assert d["query"] == "test query"
        assert d["op"] == "bridge"
        assert d["depends_on"] == ["n0"]
        assert d["confidence"] == 0.75
        assert d["budget_cost"] == 2


class TestPlanGraph:
    """Tests for PlanGraph model."""
    
    def test_create_empty_plan(self):
        """Test creating empty plan graph."""
        plan = PlanGraph(question="What is the capital of France?")
        
        assert plan.question == "What is the capital of France?"
        assert len(plan.nodes) == 0
    
    def test_create_plan_with_nodes(self):
        """Test creating plan with nodes."""
        plan = PlanGraph(
            question="Who wrote the book X?",
            nodes=[
                PlanNode(id="n1", query="book X author"),
                PlanNode(id="n2", query="author biography", depends_on=["n1"]),
            ],
        )
        
        assert len(plan.nodes) == 2
        assert plan.nodes[0].id == "n1"
        assert plan.nodes[1].id == "n2"
    
    def test_invalid_dependency_raises_error(self):
        """Test that referencing non-existent node raises error."""
        with pytest.raises(ValueError, match="unknown node"):
            PlanGraph(
                question="test",
                nodes=[
                    PlanNode(id="n1", query="test", depends_on=["n99"]),
                ],
            )
    
    def test_self_dependency_raises_error(self):
        """Test that self-dependency raises error."""
        with pytest.raises(ValueError, match="cannot depend on itself"):
            PlanGraph(
                question="test",
                nodes=[
                    PlanNode(id="n1", query="test", depends_on=["n1"]),
                ],
            )
    
    def test_cycle_detection(self):
        """Test that cycles are detected."""
        with pytest.raises(ValueError, match="cycle"):
            PlanGraph(
                question="test",
                nodes=[
                    PlanNode(id="n1", query="test1", depends_on=["n2"]),
                    PlanNode(id="n2", query="test2", depends_on=["n1"]),
                ],
            )
    
    def test_get_root_nodes(self):
        """Test getting root nodes (no dependencies)."""
        plan = PlanGraph(
            question="test",
            nodes=[
                PlanNode(id="n1", query="root1"),
                PlanNode(id="n2", query="root2"),
                PlanNode(id="n3", query="child", depends_on=["n1", "n2"]),
            ],
        )
        
        roots = plan.get_root_nodes()
        
        assert len(roots) == 2
        assert {r.id for r in roots} == {"n1", "n2"}
    
    def test_execution_order(self):
        """Test topological sort for execution order."""
        plan = PlanGraph(
            question="test",
            nodes=[
                PlanNode(id="n1", query="first"),
                PlanNode(id="n2", query="second", depends_on=["n1"]),
                PlanNode(id="n3", query="third", depends_on=["n2"]),
            ],
        )
        
        order = plan.get_execution_order()
        
        assert len(order) == 3  # 3 levels
        assert order[0][0].id == "n1"
        assert order[1][0].id == "n2"
        assert order[2][0].id == "n3"
    
    def test_execution_order_parallel(self):
        """Test execution order with parallel nodes."""
        plan = PlanGraph(
            question="test",
            nodes=[
                PlanNode(id="n1", query="parallel1"),
                PlanNode(id="n2", query="parallel2"),
                PlanNode(id="n3", query="after_both", depends_on=["n1", "n2"]),
            ],
        )
        
        order = plan.get_execution_order()
        
        assert len(order) == 2  # 2 levels
        assert len(order[0]) == 2  # First level has 2 parallel nodes
        assert len(order[1]) == 1  # Second level has 1 node
    
    def test_total_budget_cost(self):
        """Test calculating total budget cost."""
        plan = PlanGraph(
            question="test",
            nodes=[
                PlanNode(id="n1", query="test1", budget_cost=1),
                PlanNode(id="n2", query="test2", budget_cost=2),
                PlanNode(id="n3", query="test3", budget_cost=3),
            ],
        )
        
        assert plan.total_budget_cost() == 6
    
    def test_average_confidence(self):
        """Test calculating average confidence."""
        plan = PlanGraph(
            question="test",
            nodes=[
                PlanNode(id="n1", query="test1", confidence=0.5),
                PlanNode(id="n2", query="test2", confidence=0.7),
                PlanNode(id="n3", query="test3", confidence=0.9),
            ],
        )
        
        assert abs(plan.average_confidence() - 0.7) < 0.001
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        original = PlanGraph(
            question="test question",
            nodes=[
                PlanNode(id="n1", query="query1", confidence=0.8),
                PlanNode(id="n2", query="query2", depends_on=["n1"]),
            ],
        )
        
        # Serialize
        data = original.to_json()
        json_str = json.dumps(data)
        
        # Deserialize
        loaded_data = json.loads(json_str)
        restored = PlanGraph.from_json(loaded_data)
        
        assert restored.question == original.question
        assert len(restored.nodes) == len(original.nodes)
        assert restored.nodes[0].id == original.nodes[0].id
        assert restored.nodes[1].depends_on == original.nodes[1].depends_on


class TestOperatorType:
    """Tests for OperatorType enum."""
    
    def test_all_operators_defined(self):
        """Test all expected operators are defined."""
        expected = {"lookup", "bridge", "filter", "compare", "aggregate", "verify"}
        actual = {op.value for op in OperatorType}
        
        assert actual == expected
    
    def test_operator_from_string(self):
        """Test creating operator from string."""
        assert OperatorType("lookup") == OperatorType.LOOKUP
        assert OperatorType("bridge") == OperatorType.BRIDGE
