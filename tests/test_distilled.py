"""
Tests for Distilled Planner Module.

Tests distillation data generation, dataset handling, and planner interface.
Note: LoRA training tests require optional dependencies.
"""

import pytest
import json
from pathlib import Path
from src.planner.schema import PlanNode, PlanGraph, OperatorType
from src.planner.distilled import (
    DistillationExample,
    DistillationDataset,
    DistilledPlanner,
    LoRAConfig,
    PlanQualityMetrics,
)


class TestDistillationExample:
    """Tests for DistillationExample."""
    
    def test_to_dict(self):
        """Test serialization."""
        example = DistillationExample(
            question="Who directed Inception?",
            teacher_output='{"nodes": []}',
            num_nodes=3,
            has_bindings=True,
            teacher_confidence=0.85,
        )
        
        data = example.to_dict()
        
        assert data["question"] == "Who directed Inception?"
        assert data["num_nodes"] == 3
        assert data["has_bindings"] is True
    
    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "question": "Test?",
            "teacher_output": "{}",
            "num_nodes": 2,
            "has_bindings": False,
            "teacher_confidence": 0.7,
        }
        
        example = DistillationExample.from_dict(data)
        
        assert example.question == "Test?"
        assert example.num_nodes == 2


class TestDistillationDataset:
    """Tests for DistillationDataset."""
    
    def test_empty_dataset(self):
        """Test empty dataset."""
        dataset = DistillationDataset()
        
        assert len(dataset) == 0
    
    def test_filter_quality(self):
        """Test quality filtering."""
        dataset = DistillationDataset(
            examples=[
                DistillationExample(
                    question="q1",
                    teacher_output="{}",
                    num_nodes=3,
                    has_bindings=True,
                    teacher_confidence=0.9,
                ),
                DistillationExample(
                    question="q2",
                    teacher_output="{}",
                    num_nodes=1,
                    has_bindings=False,
                    teacher_confidence=0.3,
                ),
            ]
        )
        
        filtered = dataset.filter_quality(min_nodes=2, min_confidence=0.5)
        
        assert len(filtered) == 1
        assert filtered.examples[0].question == "q1"
    
    def test_to_training_format(self):
        """Test conversion to training format."""
        dataset = DistillationDataset(
            examples=[
                DistillationExample(
                    question="Who is Einstein?",
                    teacher_output='{"nodes": [{"id": "n0"}]}',
                ),
            ]
        )
        
        training = dataset.to_training_format()
        
        assert len(training) == 1
        assert "Einstein" in training[0]["input"]
        assert "nodes" in training[0]["output"]
    
    def test_save_and_load(self, tmp_path):
        """Test save and load round-trip."""
        dataset = DistillationDataset(
            examples=[
                DistillationExample(
                    question="Test?",
                    teacher_output="{}",
                    num_nodes=2,
                )
            ],
            teacher_model="test-model",
        )
        
        path = tmp_path / "dataset.json"
        dataset.save(path)
        
        loaded = DistillationDataset.load(path)
        
        assert len(loaded) == 1
        assert loaded.teacher_model == "test-model"
        assert loaded.examples[0].question == "Test?"


class TestLoRAConfig:
    """Tests for LoRAConfig."""
    
    def test_defaults(self):
        """Test default values."""
        config = LoRAConfig()
        
        assert config.r == 8
        assert config.alpha == 16
        assert config.base_model == "google/flan-t5-small"
    
    def test_custom_values(self):
        """Test custom configuration."""
        config = LoRAConfig(
            r=16,
            learning_rate=5e-5,
            base_model="google/flan-t5-base",
        )
        
        assert config.r == 16
        assert config.learning_rate == 5e-5


class TestDistilledPlanner:
    """Tests for DistilledPlanner interface."""
    
    def test_init(self):
        """Test initialization without loading model."""
        planner = DistilledPlanner(use_cpu=True)
        
        assert not planner._is_loaded
    
    def test_fallback_plan(self):
        """Test fallback plan generation."""
        planner = DistilledPlanner(use_cpu=True)
        
        plan = planner._fallback_plan("What is AI?", 0.1)
        
        assert plan.question == "What is AI?"
        assert len(plan.nodes) == 1
        assert plan.nodes[0].op == OperatorType.LOOKUP
    
    def test_parse_plan(self):
        """Test plan parsing."""
        planner = DistilledPlanner(use_cpu=True)
        data = {
            "nodes": [
                {"id": "n0", "query": "What is AI?", "op": "lookup", "confidence": 0.8},
                {"id": "n1", "query": "History of AI", "depends_on": ["n0"]},
            ]
        }
        
        plan = planner._parse_plan(data, "What is AI?", 0.05)
        
        assert len(plan.nodes) == 2
        assert plan.nodes[0].confidence == 0.8
        assert plan.nodes[1].depends_on == ["n0"]


class TestPlanQualityMetrics:
    """Tests for plan quality metrics."""
    
    def test_to_dict(self):
        """Test serialization."""
        metrics = PlanQualityMetrics(
            num_examples=100,
            avg_node_count_diff=0.5,
            exact_structure_match=0.8,
            speedup=5.0,
        )
        
        data = metrics.to_dict()
        
        assert data["num_examples"] == 100
        assert data["speedup"] == 5.0
