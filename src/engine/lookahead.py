from __future__ import annotations
"""
LookaheadRAG Main Engine.

The main pipeline orchestrator that combines:
1. Planner (PlanGraph generation)
2. Budgeted Pruning
3. Parallel Retrieval
4. Context Assembly
5. Synthesis
6. Fallback Handling
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.config import Config, get_config
from src.engine.fallback import FallbackHandler, FallbackDecision
from src.engine.pruning import BudgetedPruner, PruningResult
from src.planner import LLMPlanner, PlanGraph
from src.planner.confidence import ConfidenceEstimator
from src.retriever import ParallelRetriever, VectorStore, Reranker, RetrievalResult
from src.synthesizer import ContextAssembler, Synthesizer, SynthesisResult, AssembledContext

logger = logging.getLogger(__name__)


@dataclass
class LatencyBreakdown:
    """Detailed latency breakdown for analysis."""
    
    planning_ms: float = 0.0
    pruning_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    context_assembly_ms: float = 0.0
    synthesis_ms: float = 0.0
    fallback_ms: float = 0.0
    total_ms: float = 0.0
    
    def to_dict(self) -> dict[str, float]:
        return {
            "planning_ms": self.planning_ms,
            "pruning_ms": self.pruning_ms,
            "retrieval_ms": self.retrieval_ms,
            "reranking_ms": self.reranking_ms,
            "context_assembly_ms": self.context_assembly_ms,
            "synthesis_ms": self.synthesis_ms,
            "fallback_ms": self.fallback_ms,
            "total_ms": self.total_ms,
        }


@dataclass
class LookaheadResult:
    """Complete result from LookaheadRAG pipeline."""
    
    question: str
    answer: str
    
    # Pipeline outputs
    plan: PlanGraph
    pruning_result: PruningResult | None
    retrieval_result: RetrievalResult
    context: AssembledContext
    synthesis_result: SynthesisResult
    
    # Fallback info
    fallback_triggered: bool = False
    fallback_decision: FallbackDecision | None = None
    
    # Metrics
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    num_llm_calls: int = 2  # planning + synthesis (minimum)
    total_tokens: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/analysis."""
        return {
            "question": self.question,
            "answer": self.answer,
            "num_nodes": len(self.plan.nodes),
            "num_retrieved_chunks": self.context.total_chunks,
            "fallback_triggered": self.fallback_triggered,
            "latency": self.latency.to_dict(),
            "num_llm_calls": self.num_llm_calls,
            "total_tokens": self.total_tokens,
        }


class LookaheadRAG:
    """
    Main LookaheadRAG pipeline orchestrator.
    
    Pipeline stages:
    1. Planning: Generate PlanGraph from question
    2. Pruning: Apply budgeted pruning (optional)
    3. Retrieval: Parallel retrieval based on plan
    4. Reranking: Cross-encoder reranking (optional)
    5. Context Assembly: Build structured context
    6. Synthesis: Generate final answer
    7. Fallback: Retry with additional queries if needed
    """
    
    def __init__(
        self,
        config: Config | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.config = config or get_config()
        
        # Initialize components
        self.planner = LLMPlanner(self.config)
        self.pruner = BudgetedPruner(self.config)
        self.vector_store = vector_store or VectorStore(self.config)
        self.retriever = ParallelRetriever(self.vector_store, self.config)
        self.reranker = Reranker(self.config)
        self.context_assembler = ContextAssembler(self.config)
        self.synthesizer = Synthesizer(self.config)
        self.fallback_handler = FallbackHandler(self.config)
        self.confidence_estimator = ConfidenceEstimator(self.config)
        
        # Pre-load models to avoid race conditions in parallel execution
        self._preload_models()
    
    def _preload_models(self):
        """Pre-load embedding and reranker models to avoid race conditions."""
        try:
            # Pre-load embedding model by triggering a dummy search
            _ = self.vector_store._get_embedding_model()
            logger.info("Pre-loaded embedding model")
        except Exception as e:
            logger.warning(f"Could not pre-load embedding model: {e}")
        
        try:
            # Pre-load reranker model
            _ = self.reranker._get_model()
            logger.info("Pre-loaded reranker model")
        except Exception as e:
            logger.warning(f"Could not pre-load reranker model: {e}")
    
    async def run(
        self,
        question: str,
        enable_pruning: bool = True,
        enable_reranking: bool = True,
        enable_fallback: bool = True,
    ) -> LookaheadResult:
        """
        Run the complete LookaheadRAG pipeline.
        
        Args:
            question: The question to answer
            enable_pruning: Whether to apply budgeted pruning
            enable_reranking: Whether to apply cross-encoder reranking
            enable_fallback: Whether to enable fallback mechanism
            
        Returns:
            LookaheadResult with answer and metadata
        """
        start_time = time.time()
        latency = LatencyBreakdown()
        num_llm_calls = 0
        total_tokens = 0
        
        # Stage 1: Planning
        logger.info(f"Stage 1: Planning for question: {question[:100]}...")
        plan_start = time.time()
        plan = await self.planner.generate_plan(question)
        latency.planning_ms = (time.time() - plan_start) * 1000
        num_llm_calls += 1
        logger.info(f"Generated plan with {len(plan.nodes)} nodes")
        
        # Stage 2: Pruning (optional)
        pruning_result = None
        if enable_pruning and self.config.pruning.enabled:
            logger.info("Stage 2: Budgeted pruning...")
            prune_start = time.time()
            pruning_result = self.pruner.prune(plan)
            plan = pruning_result.pruned_plan
            latency.pruning_ms = (time.time() - prune_start) * 1000
            logger.info(f"Pruned to {len(plan.nodes)} nodes")
        
        # Stage 3: Parallel Retrieval
        logger.info("Stage 3: Parallel retrieval...")
        retrieval_result = await self.retriever.retrieve(plan)
        latency.retrieval_ms = retrieval_result.parallel_latency_ms
        logger.info(f"Retrieved {len(retrieval_result.all_results)} total chunks")
        
        # Stage 4: Reranking (optional)
        if enable_reranking:
            logger.info("Stage 4: Reranking...")
            rerank_start = time.time()
            retrieval_result = self._apply_reranking(plan, retrieval_result)
            latency.reranking_ms = (time.time() - rerank_start) * 1000
        
        # Stage 5: Check for fallback
        fallback_triggered = False
        fallback_decision = None
        
        if enable_fallback and self.config.fallback.enabled:
            fallback_decision = self.fallback_handler.should_fallback(plan, retrieval_result)
            
            if fallback_decision.should_fallback:
                logger.info(f"Stage 5: Fallback triggered - {fallback_decision.reason}")
                fallback_start = time.time()
                
                # Generate additional queries
                additional_queries = self.fallback_handler.generate_fallback_queries(
                    plan, retrieval_result,
                    max_queries=self.config.fallback.max_additional_steps,
                )
                
                if additional_queries:
                    # Create extended plan and re-retrieve
                    extended_plan = self.fallback_handler.create_fallback_plan(
                        plan, additional_queries
                    )
                    
                    # Only retrieve for new nodes
                    new_nodes = [n for n in extended_plan.nodes if n.id.startswith("fb")]
                    if new_nodes:
                        from src.planner.schema import PlanGraph as PG
                        fallback_plan = PG(
                            question=question,
                            nodes=new_nodes,
                        )
                        fallback_retrieval = await self.retriever.retrieve(fallback_plan)
                        
                        # Merge results
                        retrieval_result.node_results.extend(fallback_retrieval.node_results)
                        plan = extended_plan
                
                latency.fallback_ms = (time.time() - fallback_start) * 1000
                fallback_triggered = True
        
        # Stage 6: Context Assembly
        logger.info("Stage 6: Context assembly...")
        assembly_start = time.time()
        context = self.context_assembler.assemble(plan, retrieval_result)
        latency.context_assembly_ms = (time.time() - assembly_start) * 1000
        
        # Stage 7: Synthesis
        logger.info("Stage 7: Synthesis...")
        synthesis_result = self.synthesizer.synthesize(context)
        latency.synthesis_ms = synthesis_result.latency_ms
        num_llm_calls += 1
        total_tokens = synthesis_result.prompt_tokens + synthesis_result.completion_tokens
        
        # Calculate total latency
        latency.total_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Complete! Answer: {synthesis_result.answer[:200]}...")
        
        return LookaheadResult(
            question=question,
            answer=synthesis_result.answer,
            plan=plan,
            pruning_result=pruning_result,
            retrieval_result=retrieval_result,
            context=context,
            synthesis_result=synthesis_result,
            fallback_triggered=fallback_triggered,
            fallback_decision=fallback_decision,
            latency=latency,
            num_llm_calls=num_llm_calls,
            total_tokens=total_tokens,
        )
    
    def run_sync(
        self,
        question: str,
        **kwargs,
    ) -> LookaheadResult:
        """Synchronous version of run."""
        return asyncio.run(self.run(question, **kwargs))
    
    def _apply_reranking(
        self,
        plan: PlanGraph,
        retrieval_result: RetrievalResult,
    ) -> RetrievalResult:
        """Apply reranking to retrieval results."""
        from src.retriever.parallel import NodeRetrievalResult
        
        reranked_node_results = []
        
        for node_result in retrieval_result.node_results:
            if node_result.results:
                rerank_result = self.reranker.rerank(
                    plan.question,  # Use original question for reranking
                    node_result.results,
                )
                
                reranked_node_results.append(NodeRetrievalResult(
                    node_id=node_result.node_id,
                    query=node_result.query,
                    results=rerank_result.results,
                    latency_ms=node_result.latency_ms,
                    error=node_result.error,
                ))
            else:
                reranked_node_results.append(node_result)
        
        return RetrievalResult(
            node_results=reranked_node_results,
            total_latency_ms=retrieval_result.total_latency_ms,
            parallel_latency_ms=retrieval_result.parallel_latency_ms,
        )


async def run_lookahead(
    question: str,
    config: Config | None = None,
    **kwargs,
) -> LookaheadResult:
    """Convenience function to run LookaheadRAG."""
    engine = LookaheadRAG(config)
    return await engine.run(question, **kwargs)
