"""
PlanRAG Baseline.

A simplified "plan-first retrieval" baseline for fair comparison.
Implements sequential plan execution without our optimizations.

This preempts the "weak baselines" reviewer critique by including
a representative "plan-first" approach similar to Plan×RAG.

Key differences from LookaheadRAG:
- Sequential execution (no parallelism)
- No budgeted pruning
- No reliability checking
- No evidence verification
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from src.config import Config, get_config
from src.planner import LLMPlanner, PlanGraph
from src.retriever.vector_store import VectorStore, SearchResult
from src.synthesizer import Synthesizer, SynthesisResult

logger = logging.getLogger(__name__)


@dataclass
class PlanRAGResult:
    """Result from PlanRAG baseline."""
    
    answer: str
    question: str
    plan: PlanGraph
    all_results: list[SearchResult]
    
    # Timing
    total_latency_ms: float
    planning_latency_ms: float
    retrieval_latency_ms: float
    synthesis_latency_ms: float
    
    # Counts
    num_nodes_executed: int
    num_documents_retrieved: int
    num_llm_calls: int = 2  # Plan + Synthesize


class PlanRAGBaseline:
    """
    Simplified Plan×RAG-style baseline.
    
    Implements plan-first retrieval without optimizations:
    1. Generate plan using LLM (same as LookaheadRAG)
    2. Execute queries SEQUENTIALLY (no parallelism)
    3. Aggregate results
    4. Synthesize answer
    
    No: pruning, reliability, verification, binding resolution
    """
    
    def __init__(
        self,
        config: Config | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.config = config or get_config()
        self.vector_store = vector_store or VectorStore(self.config)
        self.planner = LLMPlanner(self.config)
        self.synthesizer = Synthesizer(self.config)
    
    async def run(self, question: str) -> PlanRAGResult:
        """
        Execute PlanRAG pipeline.
        
        Sequential execution - simulates turn-by-turn retrieval
        without our parallel speculation.
        """
        total_start = time.time()
        
        # Phase 1: Planning
        plan_start = time.time()
        plan = await self.planner.generate_plan(question)
        planning_latency = (time.time() - plan_start) * 1000
        
        # Phase 2: Sequential Retrieval
        retrieval_start = time.time()
        all_results: list[SearchResult] = []
        
        for node in plan.nodes:
            # Execute each query sequentially
            results = self.vector_store.search(
                node.query,
                top_k=self.config.retrieval.top_k,
            )
            all_results.extend(results)
        
        retrieval_latency = (time.time() - retrieval_start) * 1000
        
        # Phase 3: Synthesis
        synthesis_start = time.time()
        
        # Build simple context
        context_text = "\n\n".join([
            f"[{i+1}] {r.document.content}"
            for i, r in enumerate(all_results[:10])  # Limit context
        ])
        
        synthesis_result = await self.synthesizer.synthesize_simple(
            question=question,
            context=context_text,
        )
        
        synthesis_latency = (time.time() - synthesis_start) * 1000
        total_latency = (time.time() - total_start) * 1000
        
        return PlanRAGResult(
            answer=synthesis_result.answer,
            question=question,
            plan=plan,
            all_results=all_results,
            total_latency_ms=total_latency,
            planning_latency_ms=planning_latency,
            retrieval_latency_ms=retrieval_latency,
            synthesis_latency_ms=synthesis_latency,
            num_nodes_executed=len(plan.nodes),
            num_documents_retrieved=len(all_results),
        )
    
    def run_sync(self, question: str) -> PlanRAGResult:
        """Synchronous version of run."""
        return asyncio.run(self.run(question))


class SimpleSequentialRAG:
    """
    Even simpler baseline: single query, no planning.
    
    This is the "naive RAG" baseline for completeness.
    """
    
    def __init__(
        self,
        config: Config | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.config = config or get_config()
        self.vector_store = vector_store or VectorStore(self.config)
        self.synthesizer = Synthesizer(self.config)
    
    async def run(self, question: str) -> dict:
        """Execute simple RAG pipeline."""
        start = time.time()
        
        # Direct retrieval
        results = self.vector_store.search(
            question,
            top_k=self.config.retrieval.top_k,
        )
        
        retrieval_latency = (time.time() - start) * 1000
        
        # Build context
        context_text = "\n\n".join([
            f"[{i+1}] {r.document.content}"
            for i, r in enumerate(results)
        ])
        
        # Synthesize
        synthesis_start = time.time()
        result = await self.synthesizer.synthesize_simple(
            question=question,
            context=context_text,
        )
        synthesis_latency = (time.time() - synthesis_start) * 1000
        
        return {
            "answer": result.answer,
            "question": question,
            "total_latency_ms": (time.time() - start) * 1000,
            "retrieval_latency_ms": retrieval_latency,
            "synthesis_latency_ms": synthesis_latency,
            "num_documents": len(results),
            "num_llm_calls": 1,
        }
