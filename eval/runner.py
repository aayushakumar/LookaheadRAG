from __future__ import annotations
"""
Evaluation Runner.

Runs evaluation across methods and datasets.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

from tqdm import tqdm

from eval.datasets import DatasetExample, HotpotQADataset
from eval.metrics import (
    EvaluationResult,
    LatencyMetrics,
    CostMetrics,
    Metrics,
    SingleResult,
)
from src.config import Config, get_config
from src.engine import LookaheadRAG
from src.baselines import StandardRAG, MultiQueryRAG, AgenticRAG
from src.retriever import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for an evaluation run."""
    
    method: str  # lookahead, standard_rag, multiquery_rag, agentic_rag
    dataset: str
    subset_size: int
    save_results: bool = True
    output_dir: str = "./results"


class EvaluationRunner:
    """
    Runs evaluation for different methods.
    
    Supports:
    - LookaheadRAG
    - StandardRAG (baseline)
    - MultiQueryRAG (baseline)
    - AgenticRAG (baseline)
    """
    
    def __init__(
        self,
        config: Config | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.config = config or get_config()
        self.vector_store = vector_store or VectorStore(self.config)
        
        # Initialize methods
        self.methods = {
            "lookahead": lambda: LookaheadRAG(self.config, self.vector_store),
            "standard_rag": lambda: StandardRAG(self.config, self.vector_store),
            "multiquery_rag": lambda: MultiQueryRAG(self.config, self.vector_store),
            "agentic_rag": lambda: AgenticRAG(self.config, self.vector_store),
        }
    
    def evaluate_example(
        self,
        method_name: str,
        example: DatasetExample,
    ) -> SingleResult:
        """Evaluate a single example with a method."""
        method = self.methods[method_name]()
        
        try:
            if method_name == "lookahead":
                result = asyncio.run(method.run(example.question))
                prediction = result.answer
                latency = LatencyMetrics(
                    total_ms=result.latency.total_ms,
                    planning_ms=result.latency.planning_ms,
                    retrieval_ms=result.latency.retrieval_ms,
                    synthesis_ms=result.latency.synthesis_ms,
                )
                cost = CostMetrics(
                    num_llm_calls=result.num_llm_calls,
                    total_tokens=result.total_tokens,
                )
                metadata = result.to_dict()
            
            elif method_name == "standard_rag":
                result = method.run(example.question)
                prediction = result.answer
                latency = LatencyMetrics(
                    total_ms=result.total_latency_ms,
                    retrieval_ms=result.retrieval_latency_ms,
                    synthesis_ms=result.synthesis_latency_ms,
                )
                cost = CostMetrics(num_llm_calls=result.num_llm_calls)
                metadata = result.to_dict()
            
            elif method_name == "multiquery_rag":
                result = method.run(example.question)
                prediction = result.answer
                latency = LatencyMetrics(
                    total_ms=result.total_latency_ms,
                    retrieval_ms=result.retrieval_latency_ms,
                    synthesis_ms=result.synthesis_latency_ms,
                )
                cost = CostMetrics(num_llm_calls=result.num_llm_calls)
                metadata = result.to_dict()
            
            elif method_name == "agentic_rag":
                result = method.run(example.question)
                prediction = result.answer
                latency = LatencyMetrics(total_ms=result.total_latency_ms)
                cost = CostMetrics(num_llm_calls=result.num_llm_calls)
                metadata = result.to_dict()
            
            else:
                raise ValueError(f"Unknown method: {method_name}")
            
            # Compute metrics
            accuracy = Metrics.compute_accuracy(prediction, example.answer)
            
            return SingleResult(
                example_id=example.id,
                question=example.question,
                prediction=prediction,
                ground_truth=example.answer,
                exact_match=accuracy["exact_match"],
                f1=accuracy["f1"],
                latency=latency,
                cost=cost,
                metadata=metadata,
            )
        
        except Exception as e:
            logger.error(f"Error evaluating example {example.id}: {e}")
            return SingleResult(
                example_id=example.id,
                question=example.question,
                prediction=f"ERROR: {str(e)}",
                ground_truth=example.answer,
                exact_match=0.0,
                f1=0.0,
                metadata={"error": str(e)},
            )
    
    def run(
        self,
        method_name: str,
        dataset: HotpotQADataset,
        show_progress: bool = True,
    ) -> EvaluationResult:
        """
        Run evaluation for a method on a dataset.
        
        Args:
            method_name: Name of the method to evaluate
            dataset: Dataset to evaluate on
            show_progress: Whether to show progress bar
            
        Returns:
            EvaluationResult with aggregated metrics
        """
        logger.info(f"Running evaluation for {method_name} on {len(dataset)} examples")
        
        results = []
        examples = list(dataset)
        
        iterator = tqdm(examples, desc=method_name) if show_progress else examples
        
        for example in iterator:
            result = self.evaluate_example(method_name, example)
            results.append(result)
            
            if show_progress:
                # Update progress bar with current metrics
                current_em = sum(r.exact_match for r in results) / len(results)
                current_f1 = sum(r.f1 for r in results) / len(results)
                iterator.set_postfix(EM=f"{current_em:.3f}", F1=f"{current_f1:.3f}")
        
        return Metrics.aggregate_results(method_name, results)
    
    def run_all(
        self,
        dataset: HotpotQADataset,
        methods: list[str] | None = None,
    ) -> dict[str, EvaluationResult]:
        """Run evaluation for all methods."""
        methods = methods or list(self.methods.keys())
        
        results = {}
        for method in methods:
            results[method] = self.run(method, dataset)
            logger.info(results[method].summary())
        
        return results
    
    def compare(
        self,
        results: dict[str, EvaluationResult],
    ) -> str:
        """Generate comparison table."""
        lines = [
            "| Method | EM | F1 | Latency (p50) | LLM Calls |",
            "|--------|----|----|---------------|-----------|",
        ]
        
        for method, result in results.items():
            lines.append(
                f"| {method} | {result.avg_exact_match:.3f} | "
                f"{result.avg_f1:.3f} | {result.latency_p50_ms:.0f}ms | "
                f"{result.avg_llm_calls:.1f} |"
            )
        
        return "\n".join(lines)
