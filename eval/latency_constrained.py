"""
Latency-Constrained Evaluation Metrics.

Implements Acc@T (Accuracy under Latency Budget) and AULC (Area Under
Latency Curve) for fair comparison of RAG methods.

Two variants:
- Acc@T (conditional): Accuracy among examples that completed within budget
- Acc@T (strict): Over-budget examples count as failures

Reference: Similar to "Pareto Optimal" evaluation in retrieval systems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eval.metrics import SingleResult, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class AccAtTResult:
    """Accuracy at a specific time budget T."""
    
    t_seconds: float
    
    # Conditional: accuracy among those within budget
    acc_conditional: float
    n_within_budget: int
    
    # Strict: count over-budget as failures
    acc_strict: float
    n_total: int
    
    # Additional info
    coverage: float  # Fraction of examples within budget
    
    def __repr__(self) -> str:
        return (
            f"Acc@{self.t_seconds}s: "
            f"conditional={self.acc_conditional:.3f} ({self.n_within_budget}/{self.n_total}), "
            f"strict={self.acc_strict:.3f}"
        )


@dataclass
class LatencyConstrainedResult:
    """Complete latency-constrained evaluation result."""
    
    method: str
    
    # Acc@T for multiple thresholds
    acc_at_t_conditional: dict[float, float] = field(default_factory=dict)
    acc_at_t_strict: dict[float, float] = field(default_factory=dict)
    coverage_at_t: dict[float, float] = field(default_factory=dict)
    
    # AULC (Area Under Latency Curve)
    aulc_conditional: float = 0.0
    aulc_strict: float = 0.0
    
    # Raw data
    t_values: list[float] = field(default_factory=list)
    results_by_t: dict[float, AccAtTResult] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "method": self.method,
            "acc_at_t_conditional": self.acc_at_t_conditional,
            "acc_at_t_strict": self.acc_at_t_strict,
            "coverage_at_t": self.coverage_at_t,
            "aulc_conditional": self.aulc_conditional,
            "aulc_strict": self.aulc_strict,
        }
    
    def summary_table(self) -> str:
        """Generate summary table."""
        lines = [
            f"=== Latency-Constrained Evaluation: {self.method} ===",
            f"{'T (s)':<10} {'Acc (cond)':<12} {'Acc (strict)':<12} {'Coverage':<10}",
            "-" * 44,
        ]
        for t in sorted(self.t_values):
            cond = self.acc_at_t_conditional.get(t, 0)
            strict = self.acc_at_t_strict.get(t, 0)
            cov = self.coverage_at_t.get(t, 0)
            lines.append(f"{t:<10.1f} {cond:<12.3f} {strict:<12.3f} {cov:<10.2%}")
        
        lines.append("-" * 44)
        lines.append(f"AULC (conditional): {self.aulc_conditional:.4f}")
        lines.append(f"AULC (strict):      {self.aulc_strict:.4f}")
        
        return "\n".join(lines)


class LatencyConstrainedEvaluator:
    """
    Evaluator for Acc@T and AULC metrics.
    
    Acc@T = P(correct | latency ≤ T)  [conditional]
    Acc@T = P(correct AND latency ≤ T) / P(all)  [strict]
    
    AULC = ∫ Acc@T dT (piecewise linear interpolation)
    """
    
    def __init__(
        self,
        t_values: list[float] | None = None,
        t_min: float = 2.0,
        t_max: float = 10.0,
    ):
        """
        Initialize evaluator.
        
        Args:
            t_values: Specific time thresholds in seconds
            t_min: Minimum T for AULC integration
            t_max: Maximum T for AULC integration
        """
        self.t_values = t_values or [2.0, 4.0, 6.0, 8.0, 10.0]
        self.t_min = t_min
        self.t_max = t_max
    
    def compute_acc_at_t(
        self,
        results: list[SingleResult],
        t_seconds: float,
    ) -> AccAtTResult:
        """
        Compute Acc@T for a single time threshold.
        
        Args:
            results: List of evaluation results with latency
            t_seconds: Time budget in seconds
            
        Returns:
            AccAtTResult with both conditional and strict metrics
        """
        t_ms = t_seconds * 1000
        n_total = len(results)
        
        if n_total == 0:
            return AccAtTResult(
                t_seconds=t_seconds,
                acc_conditional=0.0,
                n_within_budget=0,
                acc_strict=0.0,
                n_total=0,
                coverage=0.0,
            )
        
        # Filter to examples within budget
        within_budget = [
            r for r in results
            if r.latency and r.latency.total_ms <= t_ms
        ]
        n_within = len(within_budget)
        
        # Conditional accuracy: among those within budget
        if n_within > 0:
            acc_cond = sum(r.exact_match for r in within_budget) / n_within
        else:
            acc_cond = 0.0
        
        # Strict accuracy: over-budget counts as failure
        n_correct_within = sum(r.exact_match for r in within_budget)
        acc_strict = n_correct_within / n_total
        
        # Coverage: fraction within budget
        coverage = n_within / n_total
        
        return AccAtTResult(
            t_seconds=t_seconds,
            acc_conditional=acc_cond,
            n_within_budget=n_within,
            acc_strict=acc_strict,
            n_total=n_total,
            coverage=coverage,
        )
    
    def compute_aulc(
        self,
        acc_at_t: dict[float, float],
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> float:
        """
        Compute Area Under Latency Curve via piecewise linear interpolation.
        
        AULC = (1/(t_max - t_min)) ∫_{t_min}^{t_max} Acc@T dT
        
        Normalized to [0, 1] range.
        """
        t_min = t_min or self.t_min
        t_max = t_max or self.t_max
        
        if not acc_at_t:
            return 0.0
        
        # Sort by T
        sorted_t = sorted(acc_at_t.keys())
        
        # Filter to range
        ts = [t for t in sorted_t if t_min <= t <= t_max]
        if len(ts) < 2:
            return acc_at_t.get(sorted_t[0], 0.0) if sorted_t else 0.0
        
        # Piecewise linear integration (trapezoidal rule)
        area = 0.0
        for i in range(len(ts) - 1):
            t1, t2 = ts[i], ts[i + 1]
            acc1, acc2 = acc_at_t[t1], acc_at_t[t2]
            # Trapezoid area
            area += (t2 - t1) * (acc1 + acc2) / 2
        
        # Normalize by range
        aulc = area / (t_max - t_min)
        
        return aulc
    
    def evaluate(
        self,
        results: list[SingleResult],
        method_name: str = "unknown",
    ) -> LatencyConstrainedResult:
        """
        Full latency-constrained evaluation.
        
        Args:
            results: List of SingleResult with latency info
            method_name: Name of the method being evaluated
            
        Returns:
            LatencyConstrainedResult with Acc@T and AULC
        """
        if not results:
            return LatencyConstrainedResult(method=method_name)
        
        # Compute Acc@T for each threshold
        results_by_t = {}
        acc_cond = {}
        acc_strict = {}
        coverage = {}
        
        for t in self.t_values:
            result = self.compute_acc_at_t(results, t)
            results_by_t[t] = result
            acc_cond[t] = result.acc_conditional
            acc_strict[t] = result.acc_strict
            coverage[t] = result.coverage
        
        # Compute AULC
        aulc_cond = self.compute_aulc(acc_cond)
        aulc_strict = self.compute_aulc(acc_strict)
        
        return LatencyConstrainedResult(
            method=method_name,
            acc_at_t_conditional=acc_cond,
            acc_at_t_strict=acc_strict,
            coverage_at_t=coverage,
            aulc_conditional=aulc_cond,
            aulc_strict=aulc_strict,
            t_values=list(self.t_values),
            results_by_t=results_by_t,
        )
    
    def compare_methods(
        self,
        method_results: dict[str, list[SingleResult]],
    ) -> dict[str, LatencyConstrainedResult]:
        """
        Compare multiple methods on latency-constrained metrics.
        
        Args:
            method_results: Mapping of method name -> results
            
        Returns:
            Mapping of method name -> LatencyConstrainedResult
        """
        return {
            name: self.evaluate(results, name)
            for name, results in method_results.items()
        }
    
    @staticmethod
    def print_comparison_table(
        results: dict[str, LatencyConstrainedResult],
    ) -> str:
        """Generate comparison table across methods."""
        if not results:
            return "No results to compare"
        
        methods = sorted(results.keys())
        t_values = sorted(set(
            t for r in results.values() for t in r.t_values
        ))
        
        lines = [
            "=== Latency-Constrained Comparison ===",
            "",
            "Acc@T (conditional):",
            f"{'Method':<20} " + " ".join(f"{'@'+str(int(t))+'s':<8}" for t in t_values) + " AULC",
            "-" * (20 + 9 * len(t_values) + 8),
        ]
        
        for method in methods:
            r = results[method]
            values = [f"{r.acc_at_t_conditional.get(t, 0):.3f}" for t in t_values]
            lines.append(
                f"{method:<20} " + " ".join(f"{v:<8}" for v in values) + 
                f" {r.aulc_conditional:.4f}"
            )
        
        lines.extend([
            "",
            "Acc@T (strict):",
            f"{'Method':<20} " + " ".join(f"{'@'+str(int(t))+'s':<8}" for t in t_values) + " AULC",
            "-" * (20 + 9 * len(t_values) + 8),
        ])
        
        for method in methods:
            r = results[method]
            values = [f"{r.acc_at_t_strict.get(t, 0):.3f}" for t in t_values]
            lines.append(
                f"{method:<20} " + " ".join(f"{v:<8}" for v in values) + 
                f" {r.aulc_strict:.4f}"
            )
        
        return "\n".join(lines)
