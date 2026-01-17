from __future__ import annotations
"""
Integration tests for LookaheadRAG.

These tests require LLM access and are marked as slow.
"""

import pytest

from src.planner.schema import PlanGraph, PlanNode, OperatorType
from src.engine.lookahead import LookaheadRAG
from src.baselines import StandardRAG, MultiQueryRAG, AgenticRAG
from src.config import Config


@pytest.fixture
def config():
    """Create test config."""
    return Config()


class TestPlanGraphIntegration:
    """Integration tests for PlanGraph operations."""
    
    def test_complex_dag_execution_order(self):
        """Test execution order for complex DAG."""
        plan = PlanGraph(
            question="Complex multi-hop question",
            nodes=[
                PlanNode(id="a1", query="first entity"),
                PlanNode(id="a2", query="second entity"),
                PlanNode(id="b1", query="bridge a1 a2", depends_on=["a1", "a2"]),
                PlanNode(id="c1", query="verify", depends_on=["b1"]),
            ],
        )
        
        order = plan.get_execution_order()
        
        # Level 0: a1, a2 (parallel)
        assert len(order[0]) == 2
        assert {n.id for n in order[0]} == {"a1", "a2"}
        
        # Level 1: b1
        assert len(order[1]) == 1
        assert order[1][0].id == "b1"
        
        # Level 2: c1
        assert len(order[2]) == 1
        assert order[2][0].id == "c1"
    
    def test_diamond_dependency(self):
        """Test diamond dependency pattern."""
        plan = PlanGraph(
            question="Diamond pattern",
            nodes=[
                PlanNode(id="top", query="start"),
                PlanNode(id="left", query="left path", depends_on=["top"]),
                PlanNode(id="right", query="right path", depends_on=["top"]),
                PlanNode(id="bottom", query="merge", depends_on=["left", "right"]),
            ],
        )
        
        order = plan.get_execution_order()
        
        assert len(order) == 3
        assert order[0][0].id == "top"
        assert {n.id for n in order[1]} == {"left", "right"}
        assert order[2][0].id == "bottom"


@pytest.mark.slow
@pytest.mark.integration
class TestLookaheadRAGIntegration:
    """Integration tests that require LLM access."""
    
    def test_end_to_end_with_mock(self, config):
        """Test end-to-end pipeline (mocked)."""
        # This test would require actual LLM access
        # Skipping actual execution, just testing initialization
        engine = LookaheadRAG(config)
        
        assert engine.planner is not None
        assert engine.retriever is not None
        assert engine.synthesizer is not None
    
    def test_baselines_initialization(self, config):
        """Test baseline initialization."""
        standard = StandardRAG(config)
        multiquery = MultiQueryRAG(config)
        agentic = AgenticRAG(config)
        
        assert standard.vector_store is not None
        assert multiquery.vector_store is not None
        assert agentic.vector_store is not None


class TestConfigIntegration:
    """Integration tests for configuration."""
    
    def test_default_config(self):
        """Test default configuration loads correctly."""
        config = Config()
        
        assert config.seed == 42
        assert config.llm.provider == "ollama"
        assert config.retrieval.top_k == 5
    
    def test_config_modification(self):
        """Test config can be modified."""
        config = Config(seed=123)
        
        assert config.seed == 123
