"""
Tests for Reliability Module.

Tests feature extraction, calibration, and selective prediction.
"""

import pytest
import numpy as np
from src.planner.schema import PlanNode, PlanGraph, OperatorType
from src.retriever.parallel import RetrievalResult, NodeRetrievalResult
from src.retriever.vector_store import SearchResult, Document
from src.synthesizer.evidence_verifier import VerificationResult, VerificationStatus
from src.planner.reliability import (
    ReliabilityClassifier,
    ReliabilityScore,
    ReliabilityFeatures,
    RecommendedAction,
    SelectivePredictionMetrics,
)


def make_plan(confidences: list[float]) -> PlanGraph:
    """Helper to create plans with given confidences."""
    return PlanGraph(
        question="Test?",
        nodes=[
            PlanNode(id=f"n{i}", query=f"query {i}", confidence=c)
            for i, c in enumerate(confidences)
        ],
    )


def make_retrieval(node_ids: list[str], chunks_per_node: int = 2) -> RetrievalResult:
    """Helper to create retrieval results."""
    return RetrievalResult(
        node_results=[
            NodeRetrievalResult(
                node_id=nid,
                query=f"query {nid}",
                results=[
                    SearchResult(
                        document=Document(id=f"doc_{nid}_{i}", content=f"chunk {i}"),
                        score=0.9,
                    )
                    for i in range(chunks_per_node)
                ],
                latency_ms=100,
            )
            for nid in node_ids
        ],
        total_latency_ms=100 * len(node_ids),
        parallel_latency_ms=100,
    )


def make_verification(
    coverage: float = 1.0,
    status: VerificationStatus = VerificationStatus.SUFFICIENT,
    contradictions: int = 0,
) -> VerificationResult:
    """Helper to create verification results."""
    return VerificationResult(
        status=status,
        total_hops=2,
        covered_hops=int(2 * coverage),
        coverage_ratio=coverage,
        contradictions=[None] * contradictions if contradictions else [],  # type: ignore
    )


class TestFeatureExtraction:
    """Tests for feature extraction."""
    
    def test_extract_plan_features(self):
        """Test extracting features from plan only."""
        classifier = ReliabilityClassifier()
        plan = make_plan([0.8, 0.9, 0.7])
        
        features = classifier.extract_features(plan)
        
        assert features.plan_confidence_mean == pytest.approx(0.8, rel=0.01)
        assert features.plan_confidence_min == 0.7
        assert features.nodes_total == 3
    
    def test_extract_retrieval_features(self):
        """Test extracting features from retrieval."""
        classifier = ReliabilityClassifier()
        plan = make_plan([0.8, 0.9])
        retrieval = make_retrieval(["n0", "n1"], chunks_per_node=3)
        
        features = classifier.extract_features(plan, retrieval)
        
        assert features.num_chunks_total == 6
        assert features.nodes_with_results == 2
        assert features.retrieval_coverage_mean == 1.0
    
    def test_extract_verification_features(self):
        """Test extracting features from verification."""
        classifier = ReliabilityClassifier()
        plan = make_plan([0.8])
        verification = make_verification(coverage=0.75, status=VerificationStatus.SUFFICIENT)
        
        features = classifier.extract_features(plan, verification=verification)
        
        assert features.verification_coverage == 0.75
        assert features.verification_status == "sufficient"


class TestRawScore:
    """Tests for raw score computation."""
    
    def test_high_confidence_high_score(self):
        """Test that high confidence gives high score."""
        classifier = ReliabilityClassifier()
        
        high_features = ReliabilityFeatures(
            plan_confidence_mean=0.9,
            plan_confidence_geom=0.88,
            plan_confidence_min=0.8,
            retrieval_coverage_mean=1.0,
            binding_success_rate=1.0,
            verification_coverage=1.0,
            verification_status="sufficient",
        )
        low_features = ReliabilityFeatures(
            plan_confidence_mean=0.3,
            plan_confidence_geom=0.25,
            plan_confidence_min=0.2,
            retrieval_coverage_mean=0.5,
            binding_success_rate=0.5,
            verification_coverage=0.5,
            verification_status="insufficient",
        )
        
        high_score = classifier.compute_raw_score(high_features)
        low_score = classifier.compute_raw_score(low_features)
        
        assert high_score > low_score
        assert high_score > 0.8
        assert low_score < 0.5
    
    def test_contradiction_penalty(self):
        """Test that contradictions reduce score."""
        classifier = ReliabilityClassifier()
        
        no_contradiction = ReliabilityFeatures(
            plan_confidence_mean=0.8,
            plan_confidence_geom=0.8,
            plan_confidence_min=0.8,
            verification_coverage=1.0,
            verification_status="sufficient",
            num_contradictions=0,
        )
        with_contradiction = ReliabilityFeatures(
            plan_confidence_mean=0.8,
            plan_confidence_geom=0.8,
            plan_confidence_min=0.8,
            verification_coverage=1.0,
            verification_status="contradictory",
            num_contradictions=2,
        )
        
        score_no_c = classifier.compute_raw_score(no_contradiction)
        score_with_c = classifier.compute_raw_score(with_contradiction)
        
        assert score_no_c > score_with_c


class TestReliabilityAssessment:
    """Tests for full reliability assessment."""
    
    def test_proceed_on_high_confidence(self):
        """Test that high confidence leads to PROCEED."""
        classifier = ReliabilityClassifier(proceed_threshold=0.7)
        plan = make_plan([0.9, 0.95, 0.85])
        retrieval = make_retrieval(["n0", "n1", "n2"])
        verification = make_verification(coverage=1.0)
        
        result = classifier.assess(plan, retrieval, verification)
        
        assert result.recommended_action == RecommendedAction.PROCEED
        assert result.will_succeed_prob > 0.7
    
    def test_fallback_on_low_confidence(self):
        """Test that low confidence leads to FALLBACK."""
        classifier = ReliabilityClassifier(fallback_threshold=0.3)
        plan = make_plan([0.2, 0.15, 0.1])
        # No retrieval, no verification - very low signal
        
        result = classifier.assess(plan)
        
        assert result.recommended_action == RecommendedAction.FALLBACK
        assert result.will_succeed_prob < 0.3
    
    def test_expand_on_medium_confidence(self):
        """Test that medium confidence leads to EXPAND."""
        classifier = ReliabilityClassifier(
            proceed_threshold=0.7,
            fallback_threshold=0.3,
        )
        plan = make_plan([0.5, 0.55, 0.6])
        retrieval = make_retrieval(["n0"])  # Partial retrieval
        
        result = classifier.assess(plan, retrieval)
        
        assert result.recommended_action == RecommendedAction.EXPAND


class TestSelectivePredictionMetrics:
    """Tests for selective prediction metrics."""
    
    def test_from_predictions(self):
        """Test computing metrics from predictions."""
        predictions = [
            (ReliabilityScore(
                will_succeed_prob=0.9,
                recommended_action=RecommendedAction.PROCEED,
                features=ReliabilityFeatures(),
            ), True),
            (ReliabilityScore(
                will_succeed_prob=0.8,
                recommended_action=RecommendedAction.PROCEED,
                features=ReliabilityFeatures(),
            ), True),
            (ReliabilityScore(
                will_succeed_prob=0.5,
                recommended_action=RecommendedAction.EXPAND,
                features=ReliabilityFeatures(),
            ), False),
            (ReliabilityScore(
                will_succeed_prob=0.2,
                recommended_action=RecommendedAction.FALLBACK,
                features=ReliabilityFeatures(),
            ), False),
        ]
        
        metrics = SelectivePredictionMetrics.from_predictions(predictions)
        
        assert metrics.total_examples == 4
        assert metrics.proceed_count == 2
        assert metrics.expand_count == 1
        assert metrics.fallback_count == 1
        assert metrics.accuracy_if_proceed == 1.0  # Both proceed were correct
        assert metrics.coverage == 0.5  # 2/4 proceeded
