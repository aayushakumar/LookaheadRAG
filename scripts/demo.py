from __future__ import annotations
#!/usr/bin/env python3
"""
Demo script for LookaheadRAG.

Interactive demo showing the pipeline in action.
"""

import argparse
import asyncio
import logging
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()
logging.basicConfig(level=logging.WARNING)


def demo_planner(question: str) -> None:
    """Demo the planner module."""
    from src.planner import LLMPlanner
    from src.config import get_config
    
    console.print(Panel(f"[bold]Question:[/] {question}", title="LookaheadRAG Demo"))
    
    console.print("\n[bold blue]Step 1: Planning[/]")
    console.print("Generating retrieval plan...\n")
    
    planner = LLMPlanner()
    
    try:
        plan = planner.generate_plan_sync(question)
        
        # Display plan
        console.print("[green]✓ Plan generated![/]\n")
        
        table = Table(title="Retrieval Plan", show_header=True)
        table.add_column("ID")
        table.add_column("Query")
        table.add_column("Op")
        table.add_column("Depends On")
        table.add_column("Confidence")
        
        for node in plan.nodes:
            table.add_row(
                node.id,
                node.query[:50] + "..." if len(node.query) > 50 else node.query,
                node.op.value,
                ", ".join(node.depends_on) or "-",
                f"{node.confidence:.2f}",
            )
        
        console.print(table)
        console.print(f"\n[dim]Plan generated in {plan.generation_time_ms:.0f}ms[/]")
        
        # Show execution order
        console.print("\n[bold]Execution Order:[/]")
        for i, level in enumerate(plan.get_execution_order()):
            node_ids = [n.id for n in level]
            console.print(f"  Level {i}: {', '.join(node_ids)} (parallel)")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        console.print("\n[yellow]Make sure Ollama is running:[/]")
        console.print("  ollama serve")
        console.print("  ollama pull llama3.2:3b")


def demo_full_pipeline(question: str) -> None:
    """Demo the full pipeline."""
    from src.engine import LookaheadRAG
    from src.config import get_config
    
    console.print(Panel(f"[bold]Question:[/] {question}", title="LookaheadRAG Full Pipeline"))
    
    engine = LookaheadRAG()
    
    try:
        result = asyncio.run(engine.run(question))
        
        # Display results
        console.print("\n[bold green]Answer:[/]")
        console.print(Panel(result.answer))
        
        # Display metrics
        console.print("\n[bold blue]Metrics:[/]")
        table = Table(show_header=False)
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        
        table.add_row("Total Latency", f"{result.latency.total_ms:.0f}ms")
        table.add_row("Planning", f"{result.latency.planning_ms:.0f}ms")
        table.add_row("Retrieval", f"{result.latency.retrieval_ms:.0f}ms")
        table.add_row("Synthesis", f"{result.latency.synthesis_ms:.0f}ms")
        table.add_row("LLM Calls", str(result.num_llm_calls))
        table.add_row("Retrieved Chunks", str(result.context.total_chunks))
        table.add_row("Fallback Triggered", str(result.fallback_triggered))
        
        console.print(table)
        
        # Show citations if any
        if result.synthesis_result.citations:
            console.print("\n[bold]Citations:[/]")
            for cite in result.synthesis_result.citations:
                console.print(f"  {cite.raw}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise


def interactive_mode() -> None:
    """Run interactive demo mode."""
    console.print(Panel(
        "[bold]LookaheadRAG Interactive Demo[/]\n\n"
        "Enter questions to see the pipeline in action.\n"
        "Type 'quit' to exit.",
        title="Welcome",
    ))
    
    while True:
        console.print()
        question = console.input("[bold blue]Enter question:[/] ")
        
        if question.lower() in ["quit", "exit", "q"]:
            console.print("[yellow]Goodbye![/]")
            break
        
        if not question.strip():
            continue
        
        demo_planner(question)


def main():
    parser = argparse.ArgumentParser(description="LookaheadRAG Demo")
    parser.add_argument(
        "--question",
        type=str,
        help="Question to answer (uses interactive mode if not provided)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["planner", "full"],
        default="planner",
        help="Demo mode: planner only or full pipeline",
    )
    
    args = parser.parse_args()
    
    if args.question:
        if args.mode == "planner":
            demo_planner(args.question)
        else:
            demo_full_pipeline(args.question)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
