from __future__ import annotations
"""
Fallback Handler.

Implements robust fallback mechanism when speculative retrieval fails.
"""

import logging
from dataclasses import dataclass
from typing import Any

from src.config import Config, get_config
from src.planner.confidence import ConfidenceEstimator
from src.planner.schema import PlanGraph, PlanNode, OperatorType
from src.retriever.parallel import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class FallbackDecision:
    """Decision about whether to trigger fallback."""
    
    should_fallback: bool
    reason: str
    severity: str  # "low", "medium", "high"
    recommended_action: str


@dataclass
class FallbackResult:
    """Result from fallback execution."""
    
    additional_queries: list[str]
    additional_results: RetrievalResult | None
    steps_taken: int
    improved: bool


class FallbackHandler:
    """
    Handles fallback when speculative retrieval is insufficient.
    
    Triggers fallback when:
    1. Low retrieval coverage (< X relevant chunks)
    2. High planner uncertainty (entropy > τ)
    3. Contradictory or insufficient evidence
    
    Fallback policy: bounded ≤2 additional retrieval steps
    """
    
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.confidence_estimator = ConfidenceEstimator(config)
    
    def should_fallback(
        self,
        plan: PlanGraph,
        retrieval_result: RetrievalResult,
    ) -> FallbackDecision:
        """
        Determine if fallback should be triggered.
        
        Args:
            plan: The retrieval plan
            retrieval_result: Results from parallel retrieval
            
        Returns:
            FallbackDecision with recommendation
        """
        if not self.config.fallback.enabled:
            return FallbackDecision(
                should_fallback=False,
                reason="Fallback disabled",
                severity="low",
                recommended_action="continue",
            )
        
        triggers = self.config.fallback.triggers
        reasons = []
        severity = "low"
        
        # Check 1: Planner uncertainty
        entropy = self.confidence_estimator.estimate_plan_entropy(plan)
        if entropy > triggers.high_entropy_threshold:
            reasons.append(f"High plan entropy: {entropy:.2f}")
            severity = "medium"
        
        # Check 2: Low retrieval coverage
        total_results = len(retrieval_result.all_results)
        if total_results < triggers.low_coverage_threshold:
            reasons.append(f"Low coverage: only {total_results} results")
            severity = "high" if severity == "medium" else "medium"
        
        # Check 3: Many failed nodes
        failed_nodes = len(retrieval_result.node_results) - retrieval_result.successful_nodes
        if failed_nodes > 0:
            reasons.append(f"{failed_nodes} nodes failed retrieval")
            severity = "high"
        
        # Check 4: Low average confidence
        avg_confidence = plan.average_confidence()
        if avg_confidence < 0.4:
            reasons.append(f"Low average confidence: {avg_confidence:.2f}")
            if severity == "low":
                severity = "medium"
        
        if reasons:
            reason = "; ".join(reasons)
            recommended_action = self._get_recommended_action(
                plan, retrieval_result, severity
            )
            return FallbackDecision(
                should_fallback=True,
                reason=reason,
                severity=severity,
                recommended_action=recommended_action,
            )
        
        return FallbackDecision(
            should_fallback=False,
            reason="All checks passed",
            severity="low",
            recommended_action="continue",
        )
    
    def _get_recommended_action(
        self,
        plan: PlanGraph,
        retrieval_result: RetrievalResult,
        severity: str,
    ) -> str:
        """Get recommended fallback action."""
        if severity == "high":
            return "full_replan"  # Re-generate plan with more context
        elif severity == "medium":
            return "extend_queries"  # Add more diverse queries
        else:
            return "rerank_only"  # Just rerank existing results
    
    def generate_fallback_queries(
        self,
        plan: PlanGraph,
        retrieval_result: RetrievalResult,
        max_queries: int = 2,
    ) -> list[str]:
        """
        Generate additional queries for fallback.
        
        Strategies:
        1. Query reformulation (rephrase low-result nodes)
        2. Bridge queries (connect retrieved entities)
        3. Broader queries (remove constraints)
        """
        additional_queries = []
        
        # Strategy 1: Retry failed or low-result nodes
        for node_result in retrieval_result.node_results:
            if len(node_result.results) < 2:
                # Reformulate query
                reformulated = self._reformulate_query(node_result.query)
                if reformulated and reformulated not in additional_queries:
                    additional_queries.append(reformulated)
                    if len(additional_queries) >= max_queries:
                        break
        
        # Strategy 2: Generate bridge queries
        if len(additional_queries) < max_queries:
            bridge_query = self._generate_bridge_query(plan, retrieval_result)
            if bridge_query and bridge_query not in additional_queries:
                additional_queries.append(bridge_query)
        
        # Strategy 3: Broader question-based query
        if len(additional_queries) < max_queries:
            broader = self._broaden_query(plan.question)
            if broader not in additional_queries:
                additional_queries.append(broader)
        
        return additional_queries[:max_queries]
    
    def _reformulate_query(self, query: str) -> str | None:
        """Reformulate a query to be more search-friendly."""
        # Simple reformulation: remove question words
        question_words = {"what", "who", "when", "where", "why", "how", "which"}
        words = query.lower().split()
        
        filtered = [w for w in words if w not in question_words and len(w) > 2]
        
        if len(filtered) >= 2:
            return " ".join(filtered)
        return None
    
    def _generate_bridge_query(
        self,
        plan: PlanGraph,
        retrieval_result: RetrievalResult,
    ) -> str | None:
        """Generate a query to bridge between retrieved entities."""
        # Find nodes with bridge operator
        bridge_nodes = [n for n in plan.nodes if n.op == OperatorType.BRIDGE]
        
        if bridge_nodes:
            # Use the first bridge node's query
            return bridge_nodes[0].query
        
        # If no explicit bridge, combine first two node queries
        if len(plan.nodes) >= 2:
            return f"{plan.nodes[0].query} {plan.nodes[1].query}"
        
        return None
    
    def _broaden_query(self, question: str) -> str:
        """Create a broader query from the original question."""
        # Extract key nouns/entities (simple approach)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "what", "who",
            "when", "where", "why", "how", "which", "and", "or", "of", "in",
            "to", "for", "with", "on", "at", "by", "that", "this", "it",
        }
        
        words = question.lower().replace("?", "").replace(".", "").split()
        key_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        return " ".join(key_words[:5])  # Top 5 key words
    
    def create_fallback_plan(
        self,
        original_plan: PlanGraph,
        fallback_queries: list[str],
    ) -> PlanGraph:
        """Create a new plan with fallback queries added."""
        # Add new nodes with lower confidence (they're fallback attempts)
        new_nodes = list(original_plan.nodes)
        
        for i, query in enumerate(fallback_queries):
            new_node = PlanNode(
                id=f"fb{i+1}",
                query=query,
                op=OperatorType.LOOKUP,
                depends_on=[],
                confidence=0.4,  # Lower confidence for fallback queries
                budget_cost=1,
                reasoning="Fallback query",
            )
            new_nodes.append(new_node)
        
        return PlanGraph(
            question=original_plan.question,
            nodes=new_nodes,
            planner_model=original_plan.planner_model,
            generation_time_ms=original_plan.generation_time_ms,
        )
