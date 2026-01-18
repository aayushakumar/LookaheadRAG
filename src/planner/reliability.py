"""
Reliability Module.

Implements calibrated selective prediction for plan success.
Outputs: proceed / expand / fallback decisions.

Key feature: Uses isotonic regression calibration on dev logs
to produce well-calibrated success probability estimates.

Features used (cheap and defendable):
- plan_confidence: mean, geom_mean, min of node confidences
- retrieval_coverage: #chunks above threshold per node
- binding_success: #required_inputs resolved / total
- verification_status: from EvidenceVerifier

Reference: Selective prediction / risk-coverage curves.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from src.config import Config, get_config
from src.planner.schema import PlanGraph, PlanNode

# Avoid circular import - only used for type hints
if TYPE_CHECKING:
    from src.retriever.parallel import RetrievalResult
    from src.synthesizer.evidence_verifier import VerificationResult

logger = logging.getLogger(__name__)


class RecommendedAction(str, Enum):
    """Recommended action based on reliability assessment."""
    PROCEED = "proceed"          # High confidence, continue to synthesis
    EXPAND = "expand"            # Medium confidence, request more evidence
    FALLBACK = "fallback"        # Low confidence, use fallback strategy


@dataclass
class ReliabilityFeatures:
    """Features for reliability prediction."""
    
    # Plan confidence features
    plan_confidence_mean: float = 0.0
    plan_confidence_geom: float = 0.0
    plan_confidence_min: float = 0.0
    
    # Retrieval coverage features
    retrieval_coverage_mean: float = 0.0
    num_chunks_total: int = 0
    nodes_with_results: int = 0
    nodes_total: int = 0
    
    # Binding features (if applicable)
    binding_success_rate: float = 1.0  # 1.0 if no bindings needed
    required_inputs_resolved: int = 0
    required_inputs_total: int = 0
    
    # Verification features (from EvidenceVerifier)
    verification_coverage: float = 0.0
    verification_status: str = "unknown"
    num_contradictions: int = 0
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for model."""
        return np.array([
            self.plan_confidence_mean,
            self.plan_confidence_geom,
            self.plan_confidence_min,
            self.retrieval_coverage_mean,
            self.nodes_with_results / max(self.nodes_total, 1),
            self.binding_success_rate,
            self.verification_coverage,
            1.0 if self.verification_status == "sufficient" else 0.0,
            1.0 if self.num_contradictions == 0 else 0.0,
        ])
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_confidence_mean": self.plan_confidence_mean,
            "plan_confidence_geom": self.plan_confidence_geom,
            "plan_confidence_min": self.plan_confidence_min,
            "retrieval_coverage_mean": self.retrieval_coverage_mean,
            "nodes_with_results_ratio": self.nodes_with_results / max(self.nodes_total, 1),
            "binding_success_rate": self.binding_success_rate,
            "verification_coverage": self.verification_coverage,
            "verification_status": self.verification_status,
            "num_contradictions": self.num_contradictions,
        }


@dataclass
class ReliabilityScore:
    """Result of reliability assessment."""
    
    will_succeed_prob: float
    recommended_action: RecommendedAction
    features: ReliabilityFeatures
    
    # Thresholds used for decision
    proceed_threshold: float = 0.7
    fallback_threshold: float = 0.3
    
    # Raw score before calibration
    raw_score: float = 0.0
    is_calibrated: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "will_succeed_prob": self.will_succeed_prob,
            "recommended_action": self.recommended_action.value,
            "raw_score": self.raw_score,
            "is_calibrated": self.is_calibrated,
            "features": self.features.to_dict(),
        }


class ReliabilityClassifier:
    """
    Calibrated selective prediction for plan success.
    
    Training:
    1. Collect dev logs: (features, correctness) pairs
    2. Fit isotonic regression on raw_score -> P(correct)
    
    Inference:
    1. Extract features from plan + retrieval + verification
    2. Compute raw score (weighted combination)
    3. Apply calibrator to get P(success)
    4. Decide: proceed / expand / fallback
    """
    
    def __init__(
        self,
        proceed_threshold: float = 0.7,
        fallback_threshold: float = 0.3,
        calibrator_path: Path | None = None,
    ):
        self.proceed_threshold = proceed_threshold
        self.fallback_threshold = fallback_threshold
        
        # Calibrator (fitted isotonic regression)
        self._calibrator = None
        self._is_calibrated = False
        
        # Feature weights for raw score
        self._weights = {
            "plan_confidence": 0.30,
            "retrieval_coverage": 0.25,
            "binding_success": 0.15,
            "verification": 0.30,
        }
        
        if calibrator_path and calibrator_path.exists():
            self.load_calibrator(calibrator_path)
    
    def extract_features(
        self,
        plan: PlanGraph,
        retrieval: RetrievalResult | None = None,
        verification: VerificationResult | None = None,
    ) -> ReliabilityFeatures:
        """
        Extract features from plan, retrieval, and verification.
        
        All features are cheap to compute and interpretable.
        """
        features = ReliabilityFeatures()
        
        # == Plan confidence features ==
        if plan.nodes:
            confidences = [n.confidence for n in plan.nodes]
            features.plan_confidence_mean = np.mean(confidences)
            features.plan_confidence_geom = float(np.prod(confidences) ** (1/len(confidences)))
            features.plan_confidence_min = min(confidences)
            features.nodes_total = len(plan.nodes)
        
        # == Retrieval coverage features ==
        if retrieval:
            features.num_chunks_total = retrieval.total_chunks
            features.nodes_with_results = sum(
                1 for nr in retrieval.node_results if nr.results
            )
            if features.nodes_total > 0:
                features.retrieval_coverage_mean = (
                    features.nodes_with_results / features.nodes_total
                )
        
        # == Binding features ==
        required_total = sum(len(n.required_inputs) for n in plan.nodes)
        if required_total > 0:
            # Count how many nodes have bound queries
            resolved = sum(
                1 for n in plan.nodes
                if n.bound_query is not None or not n.required_inputs
            )
            features.binding_success_rate = resolved / len(plan.nodes)
            features.required_inputs_total = required_total
        
        # == Verification features ==
        if verification:
            features.verification_coverage = verification.coverage_ratio
            features.verification_status = verification.status.value
            features.num_contradictions = len(verification.contradictions)
        
        return features
    
    def compute_raw_score(self, features: ReliabilityFeatures) -> float:
        """
        Compute raw reliability score from features.
        
        This is a weighted combination - not calibrated.
        """
        # Plan confidence component
        plan_score = (
            0.5 * features.plan_confidence_geom +
            0.3 * features.plan_confidence_mean +
            0.2 * features.plan_confidence_min
        )
        
        # Retrieval component
        retrieval_score = features.retrieval_coverage_mean
        
        # Binding component
        binding_score = features.binding_success_rate
        
        # Verification component
        verif_score = features.verification_coverage
        if features.verification_status == "contradictory":
            verif_score *= 0.5  # Penalize contradictions
        if features.num_contradictions > 0:
            verif_score *= 0.8 ** features.num_contradictions
        
        # Weighted combination
        raw = (
            self._weights["plan_confidence"] * plan_score +
            self._weights["retrieval_coverage"] * retrieval_score +
            self._weights["binding_success"] * binding_score +
            self._weights["verification"] * verif_score
        )
        
        return float(np.clip(raw, 0.0, 1.0))
    
    def assess(
        self,
        plan: PlanGraph,
        retrieval: RetrievalResult | None = None,
        verification: VerificationResult | None = None,
    ) -> ReliabilityScore:
        """
        Assess reliability and recommend action.
        
        Returns calibrated probability if calibrator is available,
        otherwise returns raw score.
        """
        features = self.extract_features(plan, retrieval, verification)
        raw_score = self.compute_raw_score(features)
        
        # Apply calibration if available
        if self._is_calibrated and self._calibrator is not None:
            prob = float(self._calibrator.transform([raw_score])[0])
        else:
            prob = raw_score
        
        # Decide action
        if prob >= self.proceed_threshold:
            action = RecommendedAction.PROCEED
        elif prob <= self.fallback_threshold:
            action = RecommendedAction.FALLBACK
        else:
            action = RecommendedAction.EXPAND
        
        return ReliabilityScore(
            will_succeed_prob=prob,
            recommended_action=action,
            features=features,
            proceed_threshold=self.proceed_threshold,
            fallback_threshold=self.fallback_threshold,
            raw_score=raw_score,
            is_calibrated=self._is_calibrated,
        )
    
    def fit_calibrator(
        self,
        dev_logs: list[tuple[ReliabilityFeatures, bool]],
    ) -> None:
        """
        Fit isotonic regression calibrator from dev logs.
        
        Args:
            dev_logs: List of (features, correctness) pairs
        """
        from sklearn.isotonic import IsotonicRegression
        
        if len(dev_logs) < 10:
            logger.warning(f"Only {len(dev_logs)} examples for calibration, may be unreliable")
        
        # Compute raw scores
        raw_scores = [self.compute_raw_score(f) for f, _ in dev_logs]
        labels = [int(c) for _, c in dev_logs]
        
        # Fit isotonic regression
        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(raw_scores, labels)
        self._is_calibrated = True
        
        logger.info(f"Fitted calibrator on {len(dev_logs)} examples")
    
    def save_calibrator(self, path: Path) -> None:
        """Save calibrator to disk."""
        if self._calibrator is None:
            raise ValueError("No calibrator to save")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "calibrator": self._calibrator,
                "weights": self._weights,
                "thresholds": (self.proceed_threshold, self.fallback_threshold),
            }, f)
        logger.info(f"Saved calibrator to {path}")
    
    def load_calibrator(self, path: Path) -> None:
        """Load calibrator from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self._calibrator = data["calibrator"]
        self._weights = data.get("weights", self._weights)
        thresholds = data.get("thresholds", (0.7, 0.3))
        self.proceed_threshold, self.fallback_threshold = thresholds
        self._is_calibrated = True
        
        logger.info(f"Loaded calibrator from {path}")


# === Evaluation Metrics ===

@dataclass
class SelectivePredictionMetrics:
    """Metrics for selective prediction / risk-coverage analysis."""
    
    total_examples: int = 0
    
    # Action distribution
    proceed_count: int = 0
    expand_count: int = 0
    fallback_count: int = 0
    
    # Accuracy by action
    accuracy_if_proceed: float = 0.0
    accuracy_if_expand: float = 0.0
    accuracy_if_fallback: float = 0.0
    
    # Risk-coverage
    coverage: float = 0.0  # Fraction proceeding (not abstaining)
    risk: float = 0.0       # Error rate among proceeding
    
    # Calibration
    avg_predicted_prob: float = 0.0
    actual_accuracy: float = 0.0
    calibration_error: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_examples": self.total_examples,
            "proceed_rate": self.proceed_count / max(self.total_examples, 1),
            "expand_rate": self.expand_count / max(self.total_examples, 1),
            "fallback_rate": self.fallback_count / max(self.total_examples, 1),
            "accuracy_if_proceed": self.accuracy_if_proceed,
            "coverage": self.coverage,
            "risk": self.risk,
            "calibration_error": self.calibration_error,
        }
    
    @classmethod
    def from_predictions(
        cls,
        predictions: list[tuple[ReliabilityScore, bool]],
    ) -> "SelectivePredictionMetrics":
        """Compute metrics from (prediction, actual_correct) pairs."""
        if not predictions:
            return cls()
        
        n = len(predictions)
        
        # Count by action
        proceed = [(p, c) for p, c in predictions if p.recommended_action == RecommendedAction.PROCEED]
        expand = [(p, c) for p, c in predictions if p.recommended_action == RecommendedAction.EXPAND]
        fallback = [(p, c) for p, c in predictions if p.recommended_action == RecommendedAction.FALLBACK]
        
        def accuracy(pairs):
            if not pairs:
                return 0.0
            return sum(c for _, c in pairs) / len(pairs)
        
        # Coverage = fraction proceeding
        coverage = len(proceed) / n
        
        # Risk = error rate among proceeding
        risk = 1.0 - accuracy(proceed) if proceed else 0.0
        
        # Calibration
        avg_prob = np.mean([p.will_succeed_prob for p, _ in predictions])
        actual_acc = sum(c for _, c in predictions) / n
        
        return cls(
            total_examples=n,
            proceed_count=len(proceed),
            expand_count=len(expand),
            fallback_count=len(fallback),
            accuracy_if_proceed=accuracy(proceed),
            accuracy_if_expand=accuracy(expand),
            accuracy_if_fallback=accuracy(fallback),
            coverage=coverage,
            risk=risk,
            avg_predicted_prob=float(avg_prob),
            actual_accuracy=actual_acc,
            calibration_error=abs(avg_prob - actual_acc),
        )
