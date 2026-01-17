from __future__ import annotations
"""
Budgeted Graph Pruning.

Implements principled pruning of plan nodes based on budget constraints
and utility estimation.
"""

import logging
from dataclasses import dataclass

from src.config import Config, get_config
from src.planner.schema import PlanGraph, PlanNode

logger = logging.getLogger(__name__)


@dataclass
class PruningResult:
    """Result from budgeted pruning."""
    
    pruned_plan: PlanGraph
    removed_nodes: list[str]
    total_utility: float
    budget_used: int
    budget_limit: int


class BudgetedPruner:
    """
    Selects optimal nodes from a plan graph under budget constraints.
    
    Utility function:
        utility = confidence × novelty × hop_coverage
    
    Uses greedy selection subject to Σ budget_cost ≤ B
    """
    
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.weights = self.config.pruning.utility_weights
    
    def calculate_utility(
        self,
        node: PlanNode,
        selected_nodes: list[PlanNode],
        plan: PlanGraph,
    ) -> float:
        """
        Calculate utility of adding a node.
        
        Args:
            node: Node to evaluate
            selected_nodes: Already selected nodes
            plan: Full plan graph
            
        Returns:
            Utility score (higher is better)
        """
        # Confidence score (directly from planner)
        confidence_score = node.confidence
        
        # Novelty score (lower if similar queries already selected)
        novelty_score = self._calculate_novelty(node, selected_nodes)
        
        # Hop coverage score (prefer nodes that complete multi-hop chains)
        hop_score = self._calculate_hop_coverage(node, selected_nodes, plan)
        
        # Weighted combination
        utility = (
            self.weights.confidence * confidence_score +
            self.weights.novelty * novelty_score +
            self.weights.hop_coverage * hop_score
        )
        
        return utility
    
    def _calculate_novelty(
        self,
        node: PlanNode,
        selected_nodes: list[PlanNode],
    ) -> float:
        """Calculate novelty based on query diversity."""
        if not selected_nodes:
            return 1.0
        
        node_words = set(node.query.lower().split())
        
        max_overlap = 0.0
        for selected in selected_nodes:
            selected_words = set(selected.query.lower().split())
            
            if not node_words or not selected_words:
                continue
            
            intersection = len(node_words & selected_words)
            union = len(node_words | selected_words)
            overlap = intersection / union if union > 0 else 0
            max_overlap = max(max_overlap, overlap)
        
        # Novelty is inverse of max overlap
        return 1.0 - max_overlap
    
    def _calculate_hop_coverage(
        self,
        node: PlanNode,
        selected_nodes: list[PlanNode],
        plan: PlanGraph,
    ) -> float:
        """
        Calculate hop coverage score.
        
        Prefers nodes that:
        1. Complete dependency chains
        2. Are bridges in multi-hop reasoning
        """
        selected_ids = {n.id for n in selected_nodes}
        
        # Check if this node's dependencies are satisfied
        deps_satisfied = all(dep in selected_ids for dep in node.depends_on)
        
        # Check if this node is a dependency for other nodes
        is_dependency = any(
            node.id in other.depends_on
            for other in plan.nodes
            if other.id != node.id
        )
        
        # Check if this node is a bridge (has both deps and dependents)
        is_bridge = bool(node.depends_on) and is_dependency
        
        score = 0.5  # Base score
        
        if deps_satisfied:
            score += 0.2
        
        if is_dependency and not all(
            dep in selected_ids for dep in node.depends_on
        ):
            # This node enables others but its deps aren't met
            score += 0.1
        
        if is_bridge:
            score += 0.2
        
        return min(score, 1.0)
    
    def prune(
        self,
        plan: PlanGraph,
        budget: int | None = None,
    ) -> PruningResult:
        """
        Prune plan to fit within budget using greedy selection.
        
        Args:
            plan: Full plan graph
            budget: Maximum budget (default from config)
            
        Returns:
            PruningResult with pruned plan
        """
        budget = budget or self.config.pruning.max_budget
        
        if not plan.nodes:
            return PruningResult(
                pruned_plan=plan,
                removed_nodes=[],
                total_utility=0.0,
                budget_used=0,
                budget_limit=budget,
            )
        
        # If already under budget, return as-is
        total_cost = plan.total_budget_cost()
        if total_cost <= budget:
            return PruningResult(
                pruned_plan=plan,
                removed_nodes=[],
                total_utility=sum(n.confidence for n in plan.nodes),
                budget_used=total_cost,
                budget_limit=budget,
            )
        
        # Greedy selection
        selected_nodes: list[PlanNode] = []
        remaining_nodes = list(plan.nodes)
        current_budget = 0
        total_utility = 0.0
        
        while remaining_nodes and current_budget < budget:
            best_node = None
            best_utility = -1.0
            best_index = -1
            
            for i, node in enumerate(remaining_nodes):
                # Skip if would exceed budget
                if current_budget + node.budget_cost > budget:
                    continue
                
                # Skip if dependencies not satisfied
                selected_ids = {n.id for n in selected_nodes}
                if not all(dep in selected_ids for dep in node.depends_on):
                    continue
                
                utility = self.calculate_utility(node, selected_nodes, plan)
                
                if utility > best_utility:
                    best_utility = utility
                    best_node = node
                    best_index = i
            
            if best_node is None:
                break
            
            selected_nodes.append(best_node)
            remaining_nodes.pop(best_index)
            current_budget += best_node.budget_cost
            total_utility += best_utility
        
        # Create pruned plan
        removed_ids = {n.id for n in plan.nodes} - {n.id for n in selected_nodes}
        
        pruned_plan = PlanGraph(
            question=plan.question,
            nodes=selected_nodes,
            planner_model=plan.planner_model,
            generation_time_ms=plan.generation_time_ms,
        )
        
        return PruningResult(
            pruned_plan=pruned_plan,
            removed_nodes=list(removed_ids),
            total_utility=total_utility,
            budget_used=current_budget,
            budget_limit=budget,
        )
