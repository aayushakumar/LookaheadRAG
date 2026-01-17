from __future__ import annotations
"""
Evaluation module for LookaheadRAG.

Provides dataset loaders, metrics, evaluation runner, and visualization.
"""

from eval.datasets import HotpotQADataset, DatasetExample
from eval.metrics import Metrics, EvaluationResult
from eval.runner import EvaluationRunner
from eval.visualization import plot_pareto_frontier

__all__ = [
    "HotpotQADataset",
    "DatasetExample",
    "Metrics",
    "EvaluationResult",
    "EvaluationRunner",
    "plot_pareto_frontier",
]
