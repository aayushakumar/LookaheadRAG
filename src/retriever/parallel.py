from __future__ import annotations
"""
Parallel Retrieval Executor.

Executes multiple retrieval queries concurrently and aggregates results.
Includes latency instrumentation for evaluation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.config import Config, get_config
from src.planner.schema import PlanGraph, PlanNode
from src.retriever.vector_store import VectorStore, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class NodeRetrievalResult:
    """Results from retrieving for a single node."""
    
    node_id: str
    query: str
    results: list[SearchResult]
    latency_ms: float
    error: str | None = None


@dataclass
class RetrievalResult:
    """Aggregated results from parallel retrieval."""
    
    node_results: list[NodeRetrievalResult]
    total_latency_ms: float
    parallel_latency_ms: float  # Wall-clock time (parallel execution)
    
    @property
    def all_results(self) -> list[SearchResult]:
        """Get all search results across all nodes."""
        results = []
        for node_result in self.node_results:
            results.extend(node_result.results)
        return results
    
    @property
    def successful_nodes(self) -> int:
        """Count of nodes that retrieved successfully."""
        return sum(1 for r in self.node_results if r.error is None)
    
    def get_results_by_node(self) -> dict[str, list[SearchResult]]:
        """Get results grouped by node ID."""
        return {
            nr.node_id: nr.results
            for nr in self.node_results
        }
    
    def get_provenance(self) -> dict[str, list[str]]:
        """Get provenance mapping: node_id -> document_ids."""
        return {
            nr.node_id: [r.document.id for r in nr.results]
            for nr in self.node_results
        }


class ParallelRetriever:
    """
    Executes parallel retrieval based on a PlanGraph.
    
    For nodes without dependencies, retrieval happens in parallel.
    For nodes with dependencies, retrieval follows the DAG order.
    """
    
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        config: Config | None = None,
    ):
        self.config = config or get_config()
        self.vector_store = vector_store or VectorStore(self.config)
    
    async def _retrieve_node(
        self,
        node: PlanNode,
        top_k: int,
    ) -> NodeRetrievalResult:
        """Retrieve documents for a single node."""
        start_time = time.time()
        
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.vector_store.search(node.query, top_k=top_k),
            )
            
            latency = (time.time() - start_time) * 1000
            
            return NodeRetrievalResult(
                node_id=node.id,
                query=node.query,
                results=results,
                latency_ms=latency,
            )
        
        except Exception as e:
            logger.error(f"Error retrieving for node {node.id}: {e}")
            latency = (time.time() - start_time) * 1000
            
            return NodeRetrievalResult(
                node_id=node.id,
                query=node.query,
                results=[],
                latency_ms=latency,
                error=str(e),
            )
    
    async def retrieve(
        self,
        plan: PlanGraph,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """
        Execute retrieval based on plan graph.
        
        Nodes are executed in parallel within each level of the DAG.
        """
        start_time = time.time()
        top_k = top_k or self.config.retrieval.top_k
        
        all_node_results: list[NodeRetrievalResult] = []
        total_latency = 0.0
        
        # Get execution order (list of parallel batches)
        execution_order = plan.get_execution_order()
        
        for level_nodes in execution_order:
            # Limit parallelism
            max_parallel = self.config.retrieval.max_parallel_queries
            
            # Process nodes in batches
            for i in range(0, len(level_nodes), max_parallel):
                batch = level_nodes[i:i + max_parallel]
                
                # Execute batch in parallel
                tasks = [
                    self._retrieve_node(node, top_k)
                    for node in batch
                ]
                
                batch_results = await asyncio.gather(*tasks)
                
                all_node_results.extend(batch_results)
                total_latency += sum(r.latency_ms for r in batch_results)
        
        parallel_latency = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            node_results=all_node_results,
            total_latency_ms=total_latency,
            parallel_latency_ms=parallel_latency,
        )
    
    def retrieve_sync(
        self,
        plan: PlanGraph,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Synchronous version of retrieve."""
        return asyncio.run(self.retrieve(plan, top_k))
    
    def retrieve_single(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Retrieve for a single query (no plan needed)."""
        top_k = top_k or self.config.retrieval.top_k
        return self.vector_store.search(query, top_k=top_k)


class SequentialRetriever(ParallelRetriever):
    """
    Sequential retrieval for baseline comparison.
    
    Executes retrieval one node at a time (simulates agentic behavior).
    """
    
    async def retrieve(
        self,
        plan: PlanGraph,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Execute retrieval sequentially."""
        start_time = time.time()
        top_k = top_k or self.config.retrieval.top_k
        
        all_node_results: list[NodeRetrievalResult] = []
        total_latency = 0.0
        
        for node in plan.nodes:
            result = await self._retrieve_node(node, top_k)
            all_node_results.append(result)
            total_latency += result.latency_ms
        
        parallel_latency = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            node_results=all_node_results,
            total_latency_ms=total_latency,
            parallel_latency_ms=parallel_latency,
        )
