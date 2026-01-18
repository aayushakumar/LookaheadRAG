from __future__ import annotations
"""
Evaluation module for LookaheadRAG.

Provides dataset loaders, metrics, evaluation runner, and visualization.
"""

from eval.datasets import HotpotQADataset, DatasetExample
from eval.metrics import Metrics, EvaluationResult, SingleResult, LatencyMetrics
from eval.runner import EvaluationRunner
from eval.visualization import plot_pareto_frontier
from eval.latency_constrained import (
    LatencyConstrainedEvaluator,
    AccAtTResult,
    LatencyConstrainedResult,
)
from eval.domain_eval import (
    DomainExample,
    DomainDataset,
    DomainEvaluator,
    QASPERDataset,
    PubHealthDataset,
    CustomDataset,
)
from eval.dataset_release import (
    PlanGraphDataset,
    PlanGraphEntry,
    DatasetExporter,
    BenchmarkHarness,
)

__all__ = [
    "HotpotQADataset",
    "DatasetExample",
    "Metrics",
    "EvaluationResult",
    "SingleResult",
    "LatencyMetrics",
    "EvaluationRunner",
    "plot_pareto_frontier",
    "LatencyConstrainedEvaluator",
    "AccAtTResult",
    "LatencyConstrainedResult",
    # Domain eval
    "DomainExample",
    "DomainDataset",
    "DomainEvaluator",
    "QASPERDataset",
    "PubHealthDataset",
    "CustomDataset",
    # Dataset release
    "PlanGraphDataset",
    "PlanGraphEntry",
    "DatasetExporter",
    "BenchmarkHarness",
]


