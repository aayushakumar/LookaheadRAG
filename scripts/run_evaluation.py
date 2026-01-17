from __future__ import annotations
#!/usr/bin/env python3
"""
Run Evaluation.

Main evaluation script for LookaheadRAG and baselines.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_evaluation(
    methods: list[str],
    dataset_name: str,
    subset_size: int,
    output_dir: Path,
    plot: bool = True,
) -> None:
    """
    Run evaluation for specified methods.
    
    Args:
        methods: List of methods to evaluate
        dataset_name: Dataset name (hotpotqa)
        subset_size: Number of examples to evaluate
        output_dir: Directory to save results
        plot: Whether to generate plots
    """
    from eval.datasets import get_dataset
    from eval.runner import EvaluationRunner
    from eval.visualization import plot_pareto_frontier, generate_results_table
    from src.config import get_config
    
    console.print(f"[bold blue]LookaheadRAG Evaluation[/]")
    console.print(f"  Methods: {', '.join(methods)}")
    console.print(f"  Dataset: {dataset_name}")
    console.print(f"  Subset size: {subset_size}")
    console.print()
    
    # Load dataset
    console.print("[yellow]Loading dataset...[/]")
    dataset = get_dataset(dataset_name, subset_size=subset_size)
    
    # Initialize runner
    config = get_config()
    runner = EvaluationRunner(config)
    
    # Run evaluation for each method
    results = {}
    for method in methods:
        console.print(f"\n[bold]Evaluating {method}...[/]")
        try:
            result = runner.run(method, dataset)
            results[method] = result
            console.print(f"[green]✓ {result.summary()}[/]")
        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/]")
            logger.exception(f"Error evaluating {method}")
    
    # Print comparison table
    console.print("\n[bold]Results Summary:[/]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Method")
    table.add_column("EM", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Latency (p50)", justify="right")
    table.add_column("Latency (p95)", justify="right")
    table.add_column("LLM Calls", justify="right")
    
    for method, result in results.items():
        table.add_row(
            method,
            f"{result.avg_exact_match:.3f}",
            f"{result.avg_f1:.3f}",
            f"{result.latency_p50_ms:.0f}ms",
            f"{result.latency_p95_ms:.0f}ms",
            f"{result.avg_llm_calls:.1f}",
        )
    
    console.print(table)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_path = output_dir / f"results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(
            {method: result.to_dict() for method, result in results.items()},
            f,
            indent=2,
        )
    console.print(f"\n[green]✓ Results saved to {results_path}[/]")
    
    # Generate markdown table
    md_table = generate_results_table(results, format="markdown")
    md_path = output_dir / f"results_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(f"# LookaheadRAG Evaluation Results\n\n")
        f.write(f"**Date:** {datetime.now().isoformat()}\n")
        f.write(f"**Dataset:** {dataset_name} (n={subset_size})\n\n")
        f.write(md_table)
    console.print(f"[green]✓ Markdown table saved to {md_path}[/]")
    
    # Generate plot
    if plot and len(results) >= 2:
        plot_path = output_dir / f"pareto_{timestamp}.png"
        try:
            plot_pareto_frontier(results, output_path=plot_path)
            console.print(f"[green]✓ Pareto plot saved to {plot_path}[/]")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not generate plot: {e}[/]")


def main():
    parser = argparse.ArgumentParser(description="Run LookaheadRAG evaluation")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["lookahead", "standard_rag", "multiquery_rag", "agentic_rag"],
        help="Methods to evaluate",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hotpotqa",
        help="Dataset name",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=100,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results"),
        help="Output directory",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation",
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        methods=args.methods,
        dataset_name=args.dataset,
        subset_size=args.subset,
        output_dir=args.output_dir,
        plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
