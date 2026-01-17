from __future__ import annotations
"""
Evaluation Metrics.

Implements standard QA metrics: Exact Match, F1, etc.
Also includes latency and cost metrics.
"""

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


def normalize_answer(s: str) -> str:
    """
    Normalize answer for evaluation.
    
    - Lowercase
    - Remove punctuation
    - Remove articles
    - Remove extra whitespace
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())
    
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text: str) -> str:
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, ground_truth: str) -> float:
    """Calculate Exact Match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def supporting_fact_f1(
    predicted_titles: list[str],
    gold_titles: list[str],
) -> float:
    """Calculate F1 for supporting fact prediction."""
    if not predicted_titles or not gold_titles:
        return float(predicted_titles == gold_titles)
    
    pred_set = set(normalize_answer(t) for t in predicted_titles)
    gold_set = set(normalize_answer(t) for t in gold_titles)
    
    common = pred_set & gold_set
    
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_set)
    recall = len(common) / len(gold_set)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


@dataclass
class LatencyMetrics:
    """Latency metrics for a single run."""
    
    total_ms: float
    planning_ms: float = 0.0
    retrieval_ms: float = 0.0
    synthesis_ms: float = 0.0


@dataclass
class CostMetrics:
    """Cost metrics for a single run."""
    
    num_llm_calls: int
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class SingleResult:
    """Result for a single example."""
    
    example_id: str
    question: str
    prediction: str
    ground_truth: str
    
    # Accuracy metrics
    exact_match: float
    f1: float
    supporting_fact_f1: float = 0.0
    
    # Latency & cost
    latency: LatencyMetrics | None = None
    cost: CostMetrics | None = None
    
    # Method-specific metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Aggregated evaluation results."""
    
    method_name: str
    num_examples: int
    
    # Accuracy metrics (averaged)
    avg_exact_match: float
    avg_f1: float
    avg_supporting_fact_f1: float = 0.0
    
    # Latency metrics (percentiles)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_mean_ms: float = 0.0
    
    # Cost metrics (totals)
    total_llm_calls: int = 0
    total_tokens: int = 0
    avg_llm_calls: float = 0.0
    
    # Individual results
    results: list[SingleResult] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "method": self.method_name,
            "num_examples": self.num_examples,
            "exact_match": self.avg_exact_match,
            "f1": self.avg_f1,
            "supporting_fact_f1": self.avg_supporting_fact_f1,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_mean_ms": self.latency_mean_ms,
            "avg_llm_calls": self.avg_llm_calls,
            "total_tokens": self.total_tokens,
        }
    
    def summary(self) -> str:
        """Generate a summary string."""
        return (
            f"{self.method_name}: "
            f"EM={self.avg_exact_match:.3f}, "
            f"F1={self.avg_f1:.3f}, "
            f"Latency(p50)={self.latency_p50_ms:.0f}ms, "
            f"LLM calls={self.avg_llm_calls:.1f}"
        )


class Metrics:
    """Utility class for computing metrics."""
    
    @staticmethod
    def compute_accuracy(prediction: str, ground_truth: str) -> dict[str, float]:
        """Compute accuracy metrics for a single example."""
        return {
            "exact_match": exact_match(prediction, ground_truth),
            "f1": f1_score(prediction, ground_truth),
        }
    
    @staticmethod
    def aggregate_results(
        method_name: str,
        results: list[SingleResult],
    ) -> EvaluationResult:
        """Aggregate individual results into EvaluationResult."""
        if not results:
            return EvaluationResult(
                method_name=method_name,
                num_examples=0,
                avg_exact_match=0.0,
                avg_f1=0.0,
            )
        
        # Accuracy
        avg_em = sum(r.exact_match for r in results) / len(results)
        avg_f1 = sum(r.f1 for r in results) / len(results)
        avg_sf_f1 = sum(r.supporting_fact_f1 for r in results) / len(results)
        
        # Latency
        latencies = [r.latency.total_ms for r in results if r.latency]
        if latencies:
            latencies.sort()
            latency_p50 = latencies[len(latencies) // 2]
            latency_p95 = latencies[int(len(latencies) * 0.95)]
            latency_mean = sum(latencies) / len(latencies)
        else:
            latency_p50 = latency_p95 = latency_mean = 0.0
        
        # Cost
        total_llm_calls = sum(r.cost.num_llm_calls for r in results if r.cost)
        total_tokens = sum(r.cost.total_tokens for r in results if r.cost)
        avg_llm_calls = total_llm_calls / len(results) if results else 0.0
        
        return EvaluationResult(
            method_name=method_name,
            num_examples=len(results),
            avg_exact_match=avg_em,
            avg_f1=avg_f1,
            avg_supporting_fact_f1=avg_sf_f1,
            latency_p50_ms=latency_p50,
            latency_p95_ms=latency_p95,
            latency_mean_ms=latency_mean,
            total_llm_calls=total_llm_calls,
            total_tokens=total_tokens,
            avg_llm_calls=avg_llm_calls,
            results=results,
        )
