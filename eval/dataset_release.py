"""
PlanGraph Dataset Release Module.

Tools for releasing PlanGraph dataset and evaluation harness:
1. Export PlanGraphs with metadata (no copyrighted text)
2. Scripts to regenerate indexes from public sources
3. Standardized evaluation harness

Reference: Enables reproducibility and community extensions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from typing import Any

from src.planner.schema import PlanGraph, PlanNode

logger = logging.getLogger(__name__)


@dataclass
class PlanGraphEntry:
    """Single entry in the PlanGraph dataset."""
    
    id: str
    question_id: str
    question: str
    
    # Plan structure (no retrieved text)
    plan_json: dict[str, Any]
    
    # Metadata
    source_dataset: str = "hotpotqa"
    difficulty: str = "hard"  # HotpotQA difficulty
    num_nodes: int = 0
    has_bindings: bool = False
    
    # Generation info
    planner_model: str = ""
    generation_time_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_plan(
        cls,
        plan: PlanGraph,
        question_id: str,
        source_dataset: str = "hotpotqa",
        difficulty: str = "hard",
    ) -> "PlanGraphEntry":
        """Create entry from PlanGraph."""
        has_bindings = any(
            n.produces or n.bindings
            for n in plan.nodes
        )
        
        return cls(
            id=f"{source_dataset}_{question_id}",
            question_id=question_id,
            question=plan.question,
            plan_json={
                "nodes": [n.to_dict() for n in plan.nodes],
            },
            source_dataset=source_dataset,
            difficulty=difficulty,
            num_nodes=len(plan.nodes),
            has_bindings=has_bindings,
            planner_model=plan.planner_model,
            generation_time_ms=plan.generation_time_ms,
        )


@dataclass
class PlanGraphDataset:
    """
    PlanGraph dataset for release.
    
    Contains:
    - Question IDs (to retrieve from public sources)
    - PlanGraph structures (our contribution)
    - No copyrighted retrieved text
    """
    
    entries: list[PlanGraphEntry] = field(default_factory=list)
    
    # Metadata
    version: str = "1.0.0"
    created_at: str = ""
    description: str = ""
    license: str = "MIT"
    
    # Statistics
    num_entries: int = 0
    source_datasets: list[str] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def add(self, entry: PlanGraphEntry) -> None:
        """Add entry to dataset."""
        self.entries.append(entry)
        self.num_entries = len(self.entries)
        if entry.source_dataset not in self.source_datasets:
            self.source_datasets.append(entry.source_dataset)
    
    def save(self, output_dir: Path) -> None:
        """
        Save dataset in release format.
        
        Creates:
        - plangraphs.json: Main dataset file
        - metadata.json: Dataset info
        - README.md: Usage instructions
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Main data
        with open(output_dir / "plangraphs.json", "w") as f:
            json.dump(
                [e.to_dict() for e in self.entries],
                f,
                indent=2,
            )
        
        # Metadata
        metadata = {
            "version": self.version,
            "created_at": self.created_at or datetime.now().isoformat(),
            "description": self.description,
            "license": self.license,
            "num_entries": len(self.entries),
            "source_datasets": self.source_datasets,
            "statistics": self._compute_stats(),
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # README
        readme = self._generate_readme()
        with open(output_dir / "README.md", "w") as f:
            f.write(readme)
        
        logger.info(f"Saved dataset with {len(self.entries)} entries to {output_dir}")
    
    @classmethod
    def load(cls, path: Path) -> "PlanGraphDataset":
        """Load dataset from file."""
        with open(path / "plangraphs.json") as f:
            data = json.load(f)
        
        entries = []
        for item in data:
            entries.append(PlanGraphEntry(
                id=item["id"],
                question_id=item["question_id"],
                question=item["question"],
                plan_json=item["plan_json"],
                source_dataset=item.get("source_dataset", "unknown"),
                difficulty=item.get("difficulty", "unknown"),
                num_nodes=item.get("num_nodes", 0),
                has_bindings=item.get("has_bindings", False),
                planner_model=item.get("planner_model", ""),
                generation_time_ms=item.get("generation_time_ms", 0.0),
            ))
        
        # Load metadata
        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        return cls(
            entries=entries,
            version=metadata.get("version", "1.0.0"),
            created_at=metadata.get("created_at", ""),
            description=metadata.get("description", ""),
        )
    
    def _compute_stats(self) -> dict[str, Any]:
        """Compute dataset statistics."""
        if not self.entries:
            return {}
        
        nodes = [e.num_nodes for e in self.entries]
        bindings = [e for e in self.entries if e.has_bindings]
        
        return {
            "total_entries": len(self.entries),
            "avg_nodes": sum(nodes) / len(nodes),
            "max_nodes": max(nodes),
            "with_bindings": len(bindings),
            "binding_rate": len(bindings) / len(self.entries),
            "by_source": {
                ds: sum(1 for e in self.entries if e.source_dataset == ds)
                for ds in self.source_datasets
            },
        }
    
    def _generate_readme(self) -> str:
        """Generate README for dataset release."""
        stats = self._compute_stats()
        
        return f"""# LookaheadRAG PlanGraph Dataset

## Overview

This dataset contains **{len(self.entries):,}** PlanGraphs generated by LookaheadRAG.
Each PlanGraph represents a retrieval dependency graph for multi-hop questions.

## Contents

- `plangraphs.json`: Main dataset file
- `metadata.json`: Dataset statistics and info
- `README.md`: This file

## Statistics

- Total entries: {stats.get('total_entries', 0):,}
- Avg nodes per plan: {stats.get('avg_nodes', 0):.1f}
- Plans with bindings: {stats.get('binding_rate', 0):.1%}

## Usage

```python
from eval.dataset_release import PlanGraphDataset

# Load dataset
dataset = PlanGraphDataset.load(Path("path/to/dataset"))

# Iterate
for entry in dataset.entries:
    print(f"Q: {{entry.question}}")
    print(f"Nodes: {{entry.num_nodes}}")
```

## Regenerating Indexes

To regenerate the vector index from source data:

```bash
# Download HotpotQA
python scripts/download_data.py hotpotqa

# Build index
python scripts/build_index.py --dataset hotpotqa
```

## License

{self.license}

## Citation

```bibtex
@article{{lookaheadrag2024,
    title={{LookaheadRAG: Speculative Retrieval with Dependency Graphs}},
    author={{...}},
    year={{2024}}
}}
```
"""


class DatasetExporter:
    """Exports PlanGraphs to release format."""
    
    def __init__(self, config=None):
        self.config = config
    
    async def export_from_evaluation(
        self,
        eval_results: list[dict],
        output_dir: Path,
        description: str = "",
    ) -> PlanGraphDataset:
        """
        Export PlanGraphs collected during evaluation.
        
        Args:
            eval_results: List of evaluation results with plan info
            output_dir: Output directory
            description: Dataset description
        """
        dataset = PlanGraphDataset(
            description=description,
            created_at=datetime.now().isoformat(),
        )
        
        for result in eval_results:
            if "plan" not in result:
                continue
            
            plan = result["plan"]
            if isinstance(plan, dict):
                # Reconstruct from dict
                nodes = [
                    PlanNode(**n) for n in plan.get("nodes", [])
                ]
                plan = PlanGraph(
                    question=plan.get("question", result.get("question", "")),
                    nodes=nodes,
                    planner_model=plan.get("planner_model", "unknown"),
                    generation_time_ms=plan.get("generation_time_ms", 0.0),
                )
            
            entry = PlanGraphEntry.from_plan(
                plan=plan,
                question_id=result.get("id", result.get("example_id", "")),
                source_dataset=result.get("dataset", "hotpotqa"),
                difficulty=result.get("difficulty", "hard"),
            )
            dataset.add(entry)
        
        dataset.save(output_dir)
        return dataset


# === Benchmark Harness ===

@dataclass
class BenchmarkConfig:
    """Configuration for running benchmarks."""
    
    dataset_path: Path = Path("data/plangraphs")
    methods: list[str] = field(default_factory=lambda: [
        "lookahead", "planrag", "multiquery", "agentic"
    ])
    budget_levels: list[int] = field(default_factory=lambda: [3, 5, 10])
    time_limits: list[float] = field(default_factory=lambda: [2.0, 4.0, 6.0])
    output_dir: Path = Path("results/benchmark")


class BenchmarkHarness:
    """
    Standardized benchmark harness for comparing methods.
    
    Runs all methods on the same questions with:
    - Budget-constrained evaluation
    - Latency-constrained evaluation
    - Cross-method comparison
    """
    
    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config or BenchmarkConfig()
    
    async def run_benchmark(self) -> dict[str, Any]:
        """Run full benchmark suite."""
        from src.engine import LookaheadRAG
        from src.baselines import PlanRAGBaseline, MultiQueryRAG, AgenticRAG
        from eval.latency_constrained import LatencyConstrainedEvaluator
        
        # Load dataset
        dataset = PlanGraphDataset.load(self.config.dataset_path)
        questions = [e.question for e in dataset.entries]
        
        logger.info(f"Running benchmark on {len(questions)} questions")
        
        results = {}
        
        # Initialize methods
        methods = {
            "lookahead": LookaheadRAG(),
            "planrag": PlanRAGBaseline(),
        }
        
        for method_name, method in methods.items():
            if method_name not in self.config.methods:
                continue
            
            logger.info(f"Evaluating: {method_name}")
            
            method_results = []
            for question in questions[:50]:  # Limit for demo
                try:
                    result = await method.run(question)
                    method_results.append({
                        "question": question,
                        "answer": result.answer if hasattr(result, "answer") else str(result),
                        "latency_ms": result.total_latency_ms if hasattr(result, "total_latency_ms") else 0,
                    })
                except Exception as e:
                    logger.warning(f"Error on {question}: {e}")
            
            results[method_name] = method_results
        
        # Save results
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
