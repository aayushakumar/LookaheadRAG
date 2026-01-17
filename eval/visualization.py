from __future__ import annotations
"""
Visualization Utilities.

Generates plots for evaluation results, including Pareto frontiers.
"""

import logging
from pathlib import Path
from typing import Any

from eval.metrics import EvaluationResult

logger = logging.getLogger(__name__)


def plot_pareto_frontier(
    results: dict[str, EvaluationResult],
    x_metric: str = "latency_p50_ms",
    y_metric: str = "avg_f1",
    output_path: str | Path | None = None,
    title: str = "Latency vs Accuracy Pareto Frontier",
) -> Any:
    """
    Plot Pareto frontier for latency vs accuracy.
    
    Args:
        results: Dictionary of method name to EvaluationResult
        x_metric: X-axis metric (latency)
        y_metric: Y-axis metric (accuracy)
        output_path: Path to save the plot
        title: Plot title
        
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each method
    colors = {
        "lookahead": "#2ecc71",      # Green
        "standard_rag": "#3498db",   # Blue
        "multiquery_rag": "#9b59b6", # Purple
        "agentic_rag": "#e74c3c",    # Red
    }
    
    markers = {
        "lookahead": "o",
        "standard_rag": "s",
        "multiquery_rag": "^",
        "agentic_rag": "D",
    }
    
    points = []
    for method, result in results.items():
        x = getattr(result, x_metric)
        y = getattr(result, y_metric)
        
        color = colors.get(method, "#95a5a6")
        marker = markers.get(method, "o")
        
        ax.scatter(
            x, y,
            s=150,
            c=color,
            marker=marker,
            label=method,
            edgecolors="white",
            linewidths=2,
            zorder=5,
        )
        
        points.append((x, y, method))
    
    # Calculate and plot Pareto frontier
    pareto_points = _compute_pareto_frontier(points, minimize_x=True)
    if len(pareto_points) > 1:
        pareto_x = [p[0] for p in pareto_points]
        pareto_y = [p[1] for p in pareto_points]
        
        # Sort by x for line
        sorted_idx = np.argsort(pareto_x)
        pareto_x = [pareto_x[i] for i in sorted_idx]
        pareto_y = [pareto_y[i] for i in sorted_idx]
        
        ax.plot(
            pareto_x, pareto_y,
            "--",
            color="#2c3e50",
            alpha=0.5,
            linewidth=2,
            label="Pareto Frontier",
            zorder=1,
        )
    
    # Styling
    ax.set_xlabel("Latency (ms, p50)", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits with padding
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
    
    return fig


def _compute_pareto_frontier(
    points: list[tuple[float, float, str]],
    minimize_x: bool = True,
) -> list[tuple[float, float, str]]:
    """
    Compute Pareto frontier points.
    
    For RAG evaluation:
    - We want to minimize latency (x)
    - We want to maximize accuracy (y)
    """
    if not points:
        return []
    
    # Sort by accuracy (descending) then by latency (ascending)
    sorted_points = sorted(
        points,
        key=lambda p: (-p[1], p[0] if minimize_x else -p[0]),
    )
    
    pareto = []
    best_x = float("inf") if minimize_x else float("-inf")
    
    for point in sorted_points:
        x, y, name = point
        
        if minimize_x:
            if x < best_x:
                pareto.append(point)
                best_x = x
        else:
            if x > best_x:
                pareto.append(point)
                best_x = x
    
    return pareto


def plot_ablation_comparison(
    results: dict[str, EvaluationResult],
    output_path: str | Path | None = None,
) -> Any:
    """
    Plot ablation study results as bar chart.
    
    Args:
        results: Dictionary of ablation name to EvaluationResult
        output_path: Path to save the plot
        
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    methods = list(results.keys())
    em_scores = [results[m].avg_exact_match for m in methods]
    f1_scores = [results[m].avg_f1 for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, em_scores, width, label="Exact Match", color="#3498db")
    bars2 = ax.bar(x + width/2, f1_scores, width, label="F1", color="#2ecc71")
    
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Ablation Study Results", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved ablation plot to {output_path}")
    
    return fig


def generate_results_table(
    results: dict[str, EvaluationResult],
    format: str = "markdown",
) -> str:
    """
    Generate a results table.
    
    Args:
        results: Dictionary of method name to EvaluationResult
        format: Output format (markdown, latex, csv)
        
    Returns:
        Formatted table string
    """
    if format == "markdown":
        lines = [
            "| Method | EM | F1 | Latency (p50) | Latency (p95) | LLM Calls |",
            "|--------|:---:|:---:|:-------------:|:-------------:|:---------:|",
        ]
        for method, result in results.items():
            lines.append(
                f"| {method} | {result.avg_exact_match:.3f} | "
                f"{result.avg_f1:.3f} | {result.latency_p50_ms:.0f}ms | "
                f"{result.latency_p95_ms:.0f}ms | "
                f"{result.avg_llm_calls:.1f} |"
            )
        return "\n".join(lines)
    
    elif format == "latex":
        lines = [
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Method & EM & F1 & Latency (p50) & Latency (p95) & LLM Calls \\",
            r"\midrule",
        ]
        for method, result in results.items():
            lines.append(
                f"{method} & {result.avg_exact_match:.3f} & "
                f"{result.avg_f1:.3f} & {result.latency_p50_ms:.0f}ms & "
                f"{result.latency_p95_ms:.0f}ms & "
                f"{result.avg_llm_calls:.1f} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return "\n".join(lines)
    
    elif format == "csv":
        lines = ["method,em,f1,latency_p50_ms,latency_p95_ms,llm_calls"]
        for method, result in results.items():
            lines.append(
                f"{method},{result.avg_exact_match:.3f},"
                f"{result.avg_f1:.3f},{result.latency_p50_ms:.0f},"
                f"{result.latency_p95_ms:.0f},{result.avg_llm_calls:.1f}"
            )
        return "\n".join(lines)
    
    else:
        raise ValueError(f"Unknown format: {format}")
