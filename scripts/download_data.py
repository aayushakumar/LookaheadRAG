from __future__ import annotations
#!/usr/bin/env python3
"""
Download and prepare HotpotQA dataset.

This script downloads the HotpotQA dataset and prepares it for evaluation.
"""

import argparse
import logging
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.progress import Progress

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_hotpotqa(
    output_dir: Path,
    split: str = "validation",
    subset_size: int | None = None,
) -> None:
    """
    Download HotpotQA dataset.
    
    Args:
        output_dir: Directory to save the data
        split: Dataset split (train, validation, test)
        subset_size: Optional subset size for quick testing
    """
    console.print(f"[bold blue]Downloading HotpotQA {split} split...[/]")
    
    # Load dataset
    dataset = load_dataset("hotpot_qa", "fullwiki", split=split)
    
    if subset_size:
        dataset = dataset.select(range(min(subset_size, len(dataset))))
        console.print(f"[yellow]Using subset of {len(dataset)} examples[/]")
    
    console.print(f"[green]Loaded {len(dataset)} examples[/]")
    
    # Save to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"hotpotqa_{split}.jsonl"
    
    with Progress() as progress:
        task = progress.add_task("Saving...", total=len(dataset))
        
        with open(output_path, "w") as f:
            for item in dataset:
                import json
                f.write(json.dumps(item) + "\n")
                progress.update(task, advance=1)
    
    console.print(f"[bold green]✓ Saved to {output_path}[/]")
    
    # Print statistics
    console.print("\n[bold]Dataset Statistics:[/]")
    console.print(f"  Total examples: {len(dataset)}")
    
    # Count by type
    types = {}
    levels = {}
    for item in dataset:
        q_type = item.get("type", "unknown")
        level = item.get("level", "unknown")
        types[q_type] = types.get(q_type, 0) + 1
        levels[level] = levels.get(level, 0) + 1
    
    console.print("  By type:")
    for t, count in sorted(types.items()):
        console.print(f"    {t}: {count}")
    
    console.print("  By level:")
    for l, count in sorted(levels.items()):
        console.print(f"    {l}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Download HotpotQA dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data"),
        help="Output directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Dataset split",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Subset size for quick testing",
    )
    
    args = parser.parse_args()
    
    download_hotpotqa(
        output_dir=args.output_dir,
        split=args.split,
        subset_size=args.subset,
    )


if __name__ == "__main__":
    main()
