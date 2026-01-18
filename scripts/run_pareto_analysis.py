"""
Pareto Analysis Script.

Generates Accuracy vs Budget and Latency vs Budget curves
for comparing LookaheadRAG against baselines.

Usage:
    python scripts/run_pareto_analysis.py --methods lookahead multiquery_rag --subset 50
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from src.engine.anytime_optimizer import AnytimeOptimizer, ParetoPoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MethodResult:
    """Results for a single method at various budget levels."""
    
    method: str
    budget_levels: list[int]
    accuracy_at_budget: list[float]
    latency_at_budget: list[float]
    nodes_at_budget: list[int]


def analyze_pareto_single_plan(
    optimizer: AnytimeOptimizer,
    question: str,
    budget_levels: list[int] = [1, 2, 3, 5, 7, 10],
) -> list[ParetoPoint]:
    """Generate Pareto curve for a single question."""
    from src.planner import LLMPlanner
    
    planner = LLMPlanner()
    plan = planner.generate_plan_sync(question)
    
    return optimizer.generate_pareto_curve(plan, budget_levels)


def run_pareto_comparison(
    questions: list[str],
    budget_levels: list[int] = [1, 2, 3, 5, 7, 10],
) -> dict[int, dict[str, float]]:
    """
    Run Pareto analysis across multiple questions.
    
    Returns:
        Mapping of budget -> {avg_quality, avg_nodes, avg_latency}
    """
    optimizer = AnytimeOptimizer()
    
    # Aggregate results by budget
    results_by_budget: dict[int, list[ParetoPoint]] = {b: [] for b in budget_levels}
    
    for i, question in enumerate(questions):
        logger.info(f"Processing question {i+1}/{len(questions)}")
        try:
            points = analyze_pareto_single_plan(optimizer, question, budget_levels)
            for point in points:
                results_by_budget[point.budget].append(point)
        except Exception as e:
            logger.warning(f"Failed to process question: {e}")
    
    # Compute averages
    summary = {}
    for budget, points in results_by_budget.items():
        if points:
            summary[budget] = {
                "avg_quality": sum(p.accuracy for p in points) / len(points),
                "avg_nodes": sum(p.num_nodes for p in points) / len(points),
                "avg_latency_ms": sum(p.latency_ms for p in points) / len(points),
                "num_samples": len(points),
            }
    
    return summary


def plot_pareto_curves(
    results: dict[int, dict[str, float]],
    output_path: str = "docs/pareto_analysis.png",
):
    """Generate Pareto frontier plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot generation")
        return
    
    budgets = sorted(results.keys())
    qualities = [results[b]["avg_quality"] for b in budgets]
    latencies = [results[b]["avg_latency_ms"] for b in budgets]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Quality vs Budget
    axes[0].plot(budgets, qualities, 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel("Budget", fontsize=12)
    axes[0].set_ylabel("Expected Quality", fontsize=12)
    axes[0].set_title("Quality vs Budget (LookaheadRAG)", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Latency vs Budget
    axes[1].plot(budgets, [l/1000 for l in latencies], 'r-s', linewidth=2, markersize=8)
    axes[1].set_xlabel("Budget", fontsize=12)
    axes[1].set_ylabel("Estimated Latency (s)", fontsize=12)
    axes[1].set_title("Latency vs Budget (LookaheadRAG)", fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Pareto analysis")
    parser.add_argument(
        "--questions-file",
        type=str,
        default=None,
        help="Path to file with questions (one per line)",
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="1,2,3,5,7,10",
        help="Comma-separated budget levels",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/pareto_analysis.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots",
    )
    args = parser.parse_args()
    
    budget_levels = [int(b) for b in args.budgets.split(",")]
    
    # Default sample questions if no file provided
    if args.questions_file:
        with open(args.questions_file) as f:
            questions = [line.strip() for line in f if line.strip()]
    else:
        questions = [
            "Who directed the film starring the actress who won the Oscar for La La Land?",
            "What is the capital of the country where the Eiffel Tower is located?",
            "When was the author of Harry Potter born?",
            "Which company did the founder of Tesla also create for space exploration?",
            "What is the population of the city where the Olympics were held in 2008?",
        ]
    
    logger.info(f"Running Pareto analysis on {len(questions)} questions")
    logger.info(f"Budget levels: {budget_levels}")
    
    results = run_pareto_comparison(questions, budget_levels)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_path}")
    
    # Print summary
    print("\n=== Pareto Analysis Summary ===")
    print(f"{'Budget':<10} {'Quality':<10} {'Nodes':<10} {'Latency(s)':<12}")
    print("-" * 42)
    for budget in sorted(results.keys()):
        r = results[budget]
        print(f"{budget:<10} {r['avg_quality']:<10.3f} {r['avg_nodes']:<10.1f} {r['avg_latency_ms']/1000:<12.2f}")
    
    if args.plot:
        plot_pareto_curves(results)


if __name__ == "__main__":
    main()
