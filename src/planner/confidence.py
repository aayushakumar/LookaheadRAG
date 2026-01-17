from __future__ import annotations
"""
Confidence Estimation for Planner.

Implements methods to estimate confidence in generated plans:
1. Self-consistency sampling
2. Token-level entropy (when available)
3. Heuristic-based estimation
"""

import logging
from collections import Counter
from typing import Any

from src.config import Config, get_config
from src.planner.schema import PlanGraph, PlanNode

logger = logging.getLogger(__name__)


class ConfidenceEstimator:
    """Estimates confidence scores for plan nodes."""
    
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.sc_config = self.config.planner.self_consistency
    
    def estimate_from_self_consistency(
        self,
        plans: list[PlanGraph],
    ) -> dict[str, float]:
        """
        Estimate confidence using self-consistency across multiple plan samples.
        
        Args:
            plans: Multiple sampled plans for the same question
            
        Returns:
            Dictionary mapping node queries to confidence scores
        """
        if len(plans) < 2:
            logger.warning("Self-consistency requires at least 2 samples")
            return {}
        
        # Collect all queries across plans
        all_queries: list[str] = []
        for plan in plans:
            for node in plan.nodes:
                all_queries.append(self._normalize_query(node.query))
        
        # Count query frequencies
        query_counts = Counter(all_queries)
        num_samples = len(plans)
        
        # Calculate confidence as agreement rate
        confidence_scores = {
            query: count / num_samples
            for query, count in query_counts.items()
        }
        
        return confidence_scores
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for comparison."""
        return query.lower().strip()
    
    def merge_plans_with_confidence(
        self,
        plans: list[PlanGraph],
    ) -> PlanGraph:
        """
        Merge multiple sampled plans into one with updated confidence scores.
        
        Nodes that appear in more samples get higher confidence.
        """
        if not plans:
            raise ValueError("At least one plan required")
        
        if len(plans) == 1:
            return plans[0]
        
        # Get confidence from self-consistency
        confidence_scores = self.estimate_from_self_consistency(plans)
        
        # Use the first plan as base
        base_plan = plans[0]
        
        # Update node confidence based on self-consistency
        merged_nodes = []
        for node in base_plan.nodes:
            normalized_query = self._normalize_query(node.query)
            new_confidence = confidence_scores.get(
                normalized_query,
                node.confidence
            )
            
            # Blend with original confidence
            blended_confidence = (node.confidence + new_confidence) / 2
            
            merged_nodes.append(
                PlanNode(
                    id=node.id,
                    query=node.query,
                    op=node.op,
                    depends_on=node.depends_on,
                    confidence=blended_confidence,
                    budget_cost=node.budget_cost,
                    reasoning=node.reasoning,
                    expected_evidence_type=node.expected_evidence_type,
                )
            )
        
        return PlanGraph(
            question=base_plan.question,
            nodes=merged_nodes,
            global_settings=base_plan.global_settings,
            planner_model=base_plan.planner_model,
            generation_time_ms=base_plan.generation_time_ms,
        )
    
    def estimate_plan_entropy(self, plan: PlanGraph) -> float:
        """
        Estimate overall plan entropy based on node confidences.
        
        Higher entropy indicates less certain plan.
        """
        if not plan.nodes:
            return 0.0
        
        import math
        
        total_entropy = 0.0
        for node in plan.nodes:
            p = node.confidence
            if 0 < p < 1:
                # Binary entropy for each node
                entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
                total_entropy += entropy
        
        # Normalize by number of nodes
        return total_entropy / len(plan.nodes)
    
    def should_trigger_fallback(self, plan: PlanGraph) -> tuple[bool, str]:
        """
        Determine if fallback should be triggered based on plan confidence.
        
        Returns:
            Tuple of (should_fallback, reason)
        """
        # Check entropy threshold
        entropy = self.estimate_plan_entropy(plan)
        if entropy > self.config.fallback.triggers.high_entropy_threshold:
            return True, f"High plan entropy: {entropy:.2f}"
        
        # Check if any node has very low confidence
        min_confidence = min(
            (node.confidence for node in plan.nodes),
            default=1.0
        )
        if min_confidence < self.config.planner.confidence_threshold:
            return True, f"Low minimum confidence: {min_confidence:.2f}"
        
        # Check average confidence
        avg_confidence = plan.average_confidence()
        if avg_confidence < 0.4:
            return True, f"Low average confidence: {avg_confidence:.2f}"
        
        return False, "Confidence OK"


def calibrate_confidence(
    predicted_confidence: float,
    actual_success: bool,
    history: list[tuple[float, bool]],
) -> float:
    """
    Calibrate confidence based on historical accuracy.
    
    This is a simple binning-based calibration.
    
    Args:
        predicted_confidence: The predicted confidence score
        actual_success: Whether the prediction was successful
        history: List of (confidence, success) tuples
        
    Returns:
        Calibrated confidence score
    """
    if not history:
        return predicted_confidence
    
    # Bin confidences
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_counts: dict[int, list[bool]] = {i: [] for i in range(len(bins) - 1)}
    
    for conf, success in history:
        for i in range(len(bins) - 1):
            if bins[i] <= conf < bins[i + 1]:
                bin_counts[i].append(success)
                break
    
    # Find the bin for predicted confidence
    pred_bin = 0
    for i in range(len(bins) - 1):
        if bins[i] <= predicted_confidence < bins[i + 1]:
            pred_bin = i
            break
    
    # Calculate empirical accuracy for that bin
    if bin_counts[pred_bin]:
        empirical_accuracy = sum(bin_counts[pred_bin]) / len(bin_counts[pred_bin])
        # Blend predicted and empirical
        return (predicted_confidence + empirical_accuracy) / 2
    
    return predicted_confidence
