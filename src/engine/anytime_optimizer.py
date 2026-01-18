"""
Anytime Optimizer Module.

Implements principled budgeted node selection via dependency-constrained
dynamic programming. This replaces the heuristic greedy approach with
an optimization objective that can be validated empirically.

Key insight: Treat as a dependency-constrained knapsack problem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config import Config, get_config
from src.planner.schema import PlanGraph, PlanNode, OperatorType

logger = logging.getLogger(__name__)


@dataclass
class AnytimeConfig:
    """Configuration for anytime optimization."""
    
    budget_levels: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 7, 10])
    quality_model: str = "confidence"  # "confidence" | "learned" | "oracle"
    
    # Quality function weights
    confidence_weight: float = 0.60
    chain_coverage_weight: float = 0.25
    diversity_weight: float = 0.15


@dataclass
class OptimizationResult:
    """Result from anytime optimization."""
    
    optimized_plan: PlanGraph
    selected_node_ids: list[str]
    removed_node_ids: list[str]
    expected_quality: float
    budget_used: int
    budget_limit: int
    
    # Comparison metrics
    original_node_count: int = 0
    optimization_method: str = "dp"  # "dp" | "greedy"


@dataclass
class ParetoPoint:
    """A single point on the Pareto frontier."""
    
    budget: int
    accuracy: float
    latency_ms: float
    num_nodes: int


class AnytimeOptimizer:
    """
    Budgeted node selection via dependency-constrained dynamic programming.
    
    Objective: max E[quality(answer)] s.t. cost ≤ B, deps satisfied
    
    This creates an "anytime algorithm" property where quality improves
    monotonically with budget, enabling Pareto frontier analysis.
    
    NOTE: Quality function is heuristic but validated empirically.
    We do NOT claim "provable" optimality - just empirically better
    than greedy selection.
    """
    
    def __init__(self, config: AnytimeConfig | None = None):
        self.config = config or AnytimeConfig()
    
    def _topological_sort(self, nodes: list[PlanNode]) -> list[PlanNode]:
        """
        Sort nodes in topological order (dependencies first).
        
        This is CRITICAL for DP correctness - we must process
        dependencies before nodes that depend on them.
        """
        node_map = {n.id: n for n in nodes}
        in_degree = {n.id: len(n.depends_on) for n in nodes}
        
        # Start with nodes that have no dependencies
        queue = [n for n in nodes if in_degree[n.id] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # Reduce in-degree for dependent nodes
            for other in nodes:
                if node.id in other.depends_on:
                    in_degree[other.id] -= 1
                    if in_degree[other.id] == 0:
                        queue.append(other)
        
        return result
    
    def expected_quality(
        self,
        nodes: list[PlanNode],
        plan: PlanGraph,
    ) -> float:
        """
        Estimate expected answer quality from selected nodes.
        
        This is a heuristic function - we validate it empirically
        rather than claiming theoretical guarantees.
        
        Components:
        1. Confidence: Geometric mean of node confidences
        2. Chain coverage: Fraction of dependency chains completed
        3. Diversity: Operator type diversity
        """
        if not nodes:
            return 0.0
        
        # 1. Confidence score (geometric mean for multiplicative combination)
        confidences = [n.confidence for n in nodes]
        conf_score = np.prod(confidences) ** (1 / len(confidences))
        
        # 2. Dependency chain coverage
        chain_score = self._dependency_coverage(nodes, plan)
        
        # 3. Operator diversity
        unique_ops = len(set(n.op for n in nodes))
        total_ops = len(OperatorType)
        diversity_score = unique_ops / total_ops
        
        # Weighted combination
        quality = (
            self.config.confidence_weight * conf_score +
            self.config.chain_coverage_weight * chain_score +
            self.config.diversity_weight * diversity_score
        )
        
        return quality
    
    def _dependency_coverage(
        self,
        selected: list[PlanNode],
        plan: PlanGraph,
    ) -> float:
        """
        Calculate what fraction of dependency chains are complete.
        
        A chain is complete if: for every selected node with dependencies,
        all its dependencies are also selected.
        """
        if not selected:
            return 0.0
        
        selected_ids = {n.id for n in selected}
        
        # Count nodes with satisfied dependencies
        satisfied = 0
        nodes_with_deps = 0
        
        for node in selected:
            if node.depends_on:
                nodes_with_deps += 1
                if all(dep in selected_ids for dep in node.depends_on):
                    satisfied += 1
        
        if nodes_with_deps == 0:
            # All root nodes - perfect coverage
            return 1.0
        
        return satisfied / nodes_with_deps
    
    def _deps_satisfied(
        self,
        node: PlanNode,
        selected_indices: list[int],
        nodes: list[PlanNode],
    ) -> bool:
        """Check if all dependencies of node are in selected set."""
        if not node.depends_on:
            return True
        
        selected_ids = {nodes[i].id for i in selected_indices}
        return all(dep in selected_ids for dep in node.depends_on)
    
    def optimize(
        self,
        plan: PlanGraph,
        budget: int,
    ) -> tuple[PlanGraph, float]:
        """
        Select optimal nodes under budget using DP.
        
        Algorithm: Dependency-constrained knapsack
        - Process nodes in topological order
        - Only add node if all its dependencies are already selected
        - DP state: (node_index, remaining_budget) -> (quality, selected_set)
        
        Args:
            plan: Full plan graph
            budget: Maximum budget
            
        Returns:
            Tuple of (optimized PlanGraph, expected quality)
        """
        if not plan.nodes:
            return plan, 0.0
        
        # If plan already under budget, return as-is
        if plan.total_budget_cost() <= budget:
            quality = self.expected_quality(plan.nodes, plan)
            return plan, quality
        
        # Sort nodes topologically for valid DP ordering
        nodes = self._topological_sort(plan.nodes)
        n = len(nodes)
        
        # DP approach: iterate through nodes, tracking best selection
        # for each budget level
        #
        # dp[b] = (best_quality, selected_indices) for budget b
        dp: dict[int, tuple[float, list[int]]] = {
            b: (0.0, []) for b in range(budget + 1)
        }
        
        for i, node in enumerate(nodes):
            cost = node.budget_cost
            
            # Process budgets in reverse to avoid using same node twice
            for b in range(budget, cost - 1, -1):
                prev_quality, prev_selected = dp[b - cost]
                
                # Check if we can add this node (deps satisfied)
                if self._deps_satisfied(node, prev_selected, nodes):
                    new_selected = prev_selected + [i]
                    new_quality = self.expected_quality(
                        [nodes[j] for j in new_selected], plan
                    )
                    
                    if new_quality > dp[b][0]:
                        dp[b] = (new_quality, new_selected)
        
        # Find best result across all budget levels
        best_quality, best_indices = max(
            (dp[b] for b in range(budget + 1)),
            key=lambda x: x[0]
        )
        
        # Build optimized plan
        selected_nodes = [nodes[i] for i in best_indices]
        
        optimized_plan = PlanGraph(
            question=plan.question,
            nodes=selected_nodes,
            planner_model=plan.planner_model,
            generation_time_ms=plan.generation_time_ms,
        )
        
        return optimized_plan, best_quality
    
    def optimize_with_result(
        self,
        plan: PlanGraph,
        budget: int,
    ) -> OptimizationResult:
        """Optimize and return detailed result."""
        optimized, quality = self.optimize(plan, budget)
        
        selected_ids = {n.id for n in optimized.nodes}
        removed_ids = [n.id for n in plan.nodes if n.id not in selected_ids]
        
        return OptimizationResult(
            optimized_plan=optimized,
            selected_node_ids=[n.id for n in optimized.nodes],
            removed_node_ids=removed_ids,
            expected_quality=quality,
            budget_used=optimized.total_budget_cost(),
            budget_limit=budget,
            original_node_count=len(plan.nodes),
            optimization_method="dp",
        )
    
    def generate_pareto_curve(
        self,
        plan: PlanGraph,
        budget_levels: list[int] | None = None,
    ) -> list[ParetoPoint]:
        """
        Generate Pareto frontier points for budget analysis.
        
        Returns (budget, quality, estimated_latency) tuples
        for plotting accuracy vs budget curves.
        """
        budgets = budget_levels or self.config.budget_levels
        points = []
        
        for budget in budgets:
            optimized, quality = self.optimize(plan, budget)
            
            # Estimate latency (rough: ~1s per node retrieval + fixed overhead)
            num_nodes = len(optimized.nodes)
            est_latency = 2000 + (num_nodes * 500)  # 2s base + 0.5s per node
            
            points.append(ParetoPoint(
                budget=budget,
                accuracy=quality,
                latency_ms=est_latency,
                num_nodes=num_nodes,
            ))
        
        return points


class BudgetedPrunerV2:
    """
    Wrapper around AnytimeOptimizer for backward compatibility.
    
    Provides same interface as legacy BudgetedPruner but uses
    the DP-based optimizer internally.
    """
    
    def __init__(
        self,
        config: Config | None = None,
        use_legacy: bool = False,
    ):
        self.config = config or get_config()
        self.use_legacy = use_legacy
        self.optimizer = AnytimeOptimizer()
    
    def prune(
        self,
        plan: PlanGraph,
        budget: int | None = None,
    ):
        """
        Prune plan to fit within budget.
        
        Uses DP optimization by default, falls back to greedy if specified.
        """
        from src.engine.pruning import PruningResult, BudgetedPruner
        
        budget = budget or self.config.pruning.max_budget
        
        if self.use_legacy:
            # Use original greedy pruner
            legacy = BudgetedPruner(self.config)
            return legacy.prune(plan, budget)
        
        # Use DP optimizer
        result = self.optimizer.optimize_with_result(plan, budget)
        
        return PruningResult(
            pruned_plan=result.optimized_plan,
            removed_nodes=result.removed_node_ids,
            total_utility=result.expected_quality,
            budget_used=result.budget_used,
            budget_limit=result.budget_limit,
        )
