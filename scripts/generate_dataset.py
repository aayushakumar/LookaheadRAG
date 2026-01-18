#!/usr/bin/env python
"""
High-Quality PlanGraph Dataset Generator.

Generates publication-quality PlanGraphs from HotpotQA or custom questions.
Includes quality filtering, validation, and diversity sampling.

Usage:
    # Generate from HotpotQA
    python scripts/generate_dataset.py --source hotpotqa --split hard --limit 1000 --output data/plangraphs
    
    # Generate from custom questions
    python scripts/generate_dataset.py --source custom --questions data/my_questions.txt --output data/plangraphs
    
    # With quality filtering
    python scripts/generate_dataset.py --source hotpotqa --min-nodes 2 --require-bindings --min-confidence 0.6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generates high-quality PlanGraph datasets."""
    
    def __init__(self, config=None):
        from src.config import get_config
        from src.planner import LLMPlanner
        
        self.config = config or get_config()
        self.planner = LLMPlanner(self.config)
        
        # Quality thresholds
        self.min_nodes = 2
        self.max_nodes = 7
        self.min_confidence = 0.5
        self.require_bindings = False
        self.require_dag = True
    
    async def generate_single(self, question: str, question_id: str = "") -> dict | None:
        """Generate single PlanGraph with quality checks."""
        try:
            start = time.time()
            plan = await self.planner.generate_plan(question)
            gen_time = time.time() - start
            
            # Quality checks
            if not self._passes_quality(plan):
                return None
            
            # Extract features
            has_bindings = any(
                n.produces or n.bindings
                for n in plan.nodes
            )
            
            has_dependencies = any(
                n.depends_on for n in plan.nodes
            )
            
            # Operator diversity
            ops = set(n.op.value for n in plan.nodes)
            
            return {
                "id": question_id or f"q_{hash(question) % 100000}",
                "question": question,
                "plan": {
                    "nodes": [n.to_dict() for n in plan.nodes],
                },
                "metadata": {
                    "num_nodes": len(plan.nodes),
                    "has_bindings": has_bindings,
                    "has_dependencies": has_dependencies,
                    "operators": list(ops),
                    "avg_confidence": plan.average_confidence(),
                    "min_confidence": min(n.confidence for n in plan.nodes),
                    "planner_model": plan.planner_model,
                    "generation_time_ms": gen_time * 1000,
                },
            }
        except Exception as e:
            logger.warning(f"Failed to generate plan for: {question[:50]}... - {e}")
            return None
    
    def _passes_quality(self, plan) -> bool:
        """Check if plan meets quality thresholds."""
        # Node count
        if len(plan.nodes) < self.min_nodes or len(plan.nodes) > self.max_nodes:
            return False
        
        # Confidence
        if plan.average_confidence() < self.min_confidence:
            return False
        
        # Bindings requirement
        if self.require_bindings:
            has_bindings = any(
                n.produces or n.bindings
                for n in plan.nodes
            )
            if not has_bindings:
                return False
        
        # DAG validation
        if self.require_dag:
            try:
                plan.get_execution_order()
            except ValueError:
                return False
        
        return True
    
    async def generate_batch(
        self,
        questions: list[tuple[str, str]],  # (question, id) pairs
        batch_size: int = 5,
        delay: float = 0.2,
    ) -> list[dict]:
        """Generate PlanGraphs in batches with rate limiting."""
        results = []
        total = len(questions)
        
        for i in range(0, total, batch_size):
            batch = questions[i:i + batch_size]
            
            # Generate batch concurrently
            tasks = [
                self.generate_single(q, qid)
                for q, qid in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            
            # Filter None results
            valid = [r for r in batch_results if r is not None]
            results.extend(valid)
            
            # Progress
            pct = ((i + len(batch)) / total) * 100
            logger.info(
                f"Progress: {i + len(batch)}/{total} ({pct:.1f}%) - "
                f"Valid: {len(results)}"
            )
            
            # Rate limiting
            await asyncio.sleep(delay)
        
        return results
    
    def compute_statistics(self, results: list[dict]) -> dict:
        """Compute dataset statistics."""
        if not results:
            return {}
        
        n = len(results)
        
        node_counts = [r["metadata"]["num_nodes"] for r in results]
        confidences = [r["metadata"]["avg_confidence"] for r in results]
        
        # Operator distribution
        op_counts = {}
        for r in results:
            for op in r["metadata"]["operators"]:
                op_counts[op] = op_counts.get(op, 0) + 1
        
        return {
            "total_examples": n,
            "with_bindings": sum(1 for r in results if r["metadata"]["has_bindings"]),
            "with_dependencies": sum(1 for r in results if r["metadata"]["has_dependencies"]),
            "avg_nodes": sum(node_counts) / n,
            "min_nodes": min(node_counts),
            "max_nodes": max(node_counts),
            "avg_confidence": sum(confidences) / n,
            "operator_distribution": op_counts,
        }


def load_hotpotqa_questions(split: str = "hard", limit: int | None = None) -> list[tuple[str, str]]:
    """Load questions from HotpotQA dataset."""
    data_path = Path("data/hotpotqa") if Path("data/hotpotqa").exists() else Path("data")
    
    # Try different file names
    possible_files = [
        data_path / f"hotpotqa_{split}.json",
        data_path / f"hotpotqa_{split}.jsonl",
        data_path / f"{split}.json",
        data_path / "hotpotqa_validation.jsonl",
        data_path / "dev.json",
    ]
    
    for path in possible_files:
        if path.exists():
            logger.info(f"Loading from {path}")
            
            # Handle JSONL vs JSON
            if str(path).endswith(".jsonl"):
                questions = []
                with open(path) as f:
                    for i, line in enumerate(f):
                        if line.strip():
                            d = json.loads(line)
                            questions.append((d["question"], d.get("_id", str(i))))
            else:
                with open(path) as f:
                    data = json.load(f)
                
                # Handle different formats
                if isinstance(data, list):
                    questions = [(d["question"], d.get("_id", str(i))) for i, d in enumerate(data)]
                else:
                    questions = [(d["question"], k) for k, d in data.items()]
            
            if limit:
                # Sample for diversity
                if len(questions) > limit:
                    questions = random.sample(questions, limit)
            
            return questions
    
    raise FileNotFoundError(f"No HotpotQA data found in {data_path}")


def load_custom_questions(path: str) -> list[tuple[str, str]]:
    """Load questions from custom file (one per line or JSON)."""
    file_path = Path(path)
    
    if path.endswith(".json"):
        with open(file_path) as f:
            data = json.load(f)
        return [(d["question"], d.get("id", str(i))) for i, d in enumerate(data)]
    else:
        with open(file_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        return [(line, f"q_{i}") for i, line in enumerate(lines)]


def save_dataset(results: list[dict], stats: dict, output_dir: Path, name: str = "plangraphs"):
    """Save dataset with metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Main dataset
    data_path = output_dir / f"{name}_{timestamp}.json"
    with open(data_path, "w") as f:
        json.dump({
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "statistics": stats,
            },
            "examples": results,
        }, f, indent=2)
    
    logger.info(f"Saved dataset to {data_path}")
    
    # Also save as release format
    release_dir = output_dir / "release"
    release_dir.mkdir(exist_ok=True)
    
    # Export in release format (no metadata per example)
    release_data = [
        {
            "id": r["id"],
            "question": r["question"],
            "plan_json": r["plan"],
        }
        for r in results
    ]
    
    release_path = release_dir / "plangraphs.json"
    with open(release_path, "w") as f:
        json.dump(release_data, f, indent=2)
    
    # README for release
    readme = f"""# PlanGraph Dataset

Generated: {datetime.now().isoformat()}

## Statistics
- Total examples: {stats.get('total_examples', 0)}
- With bindings: {stats.get('with_bindings', 0)}
- With dependencies: {stats.get('with_dependencies', 0)}
- Average nodes: {stats.get('avg_nodes', 0):.2f}

## Usage

```python
import json
with open('plangraphs.json') as f:
    data = json.load(f)
    
for example in data:
    print(example['question'])
    print(example['plan_json'])
```
"""
    
    with open(release_dir / "README.md", "w") as f:
        f.write(readme)
    
    logger.info(f"Saved release format to {release_dir}")
    
    return data_path


async def main():
    parser = argparse.ArgumentParser(description="Generate high-quality PlanGraph dataset")
    
    # Source selection
    parser.add_argument("--source", choices=["hotpotqa", "custom"], default="hotpotqa")
    parser.add_argument("--split", default="hard", help="HotpotQA split (hard, medium, easy)")
    parser.add_argument("--questions", help="Path to custom questions file")
    
    # Output
    parser.add_argument("--output", default="data/plangraphs", help="Output directory")
    parser.add_argument("--name", default="plangraphs", help="Dataset name prefix")
    
    # Generation options
    parser.add_argument("--limit", type=int, default=100, help="Max questions to process")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for generation")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between batches")
    
    # Quality filters
    parser.add_argument("--min-nodes", type=int, default=2, help="Minimum nodes per plan")
    parser.add_argument("--max-nodes", type=int, default=7, help="Maximum nodes per plan")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum average confidence")
    parser.add_argument("--require-bindings", action="store_true", help="Require binding annotations")
    
    # Seed for reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    
    # Load questions
    if args.source == "hotpotqa":
        questions = load_hotpotqa_questions(args.split, args.limit)
    else:
        if not args.questions:
            raise ValueError("--questions is required for custom source")
        questions = load_custom_questions(args.questions)
        if args.limit:
            questions = questions[:args.limit]
    
    logger.info(f"Loaded {len(questions)} questions")
    
    # Initialize generator
    generator = DatasetGenerator()
    generator.min_nodes = args.min_nodes
    generator.max_nodes = args.max_nodes
    generator.min_confidence = args.min_confidence
    generator.require_bindings = args.require_bindings
    
    # Generate
    logger.info("Starting generation...")
    start_time = time.time()
    
    results = await generator.generate_batch(
        questions,
        batch_size=args.batch_size,
        delay=args.delay,
    )
    
    elapsed = time.time() - start_time
    
    # Statistics
    stats = generator.compute_statistics(results)
    stats["generation_time_seconds"] = elapsed
    stats["questions_processed"] = len(questions)
    stats["acceptance_rate"] = len(results) / len(questions) if questions else 0
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Generation complete in {elapsed:.1f}s")
    logger.info(f"Valid plans: {len(results)}/{len(questions)} ({stats['acceptance_rate']:.1%})")
    logger.info(f"With bindings: {stats.get('with_bindings', 0)}")
    logger.info(f"Avg nodes: {stats.get('avg_nodes', 0):.2f}")
    
    # Save
    output_path = save_dataset(
        results,
        stats,
        Path(args.output),
        args.name,
    )
    
    logger.info(f"\nDataset saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
