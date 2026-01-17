from __future__ import annotations
"""
Reranker Module.

Cross-encoder reranker for improving retrieval quality.
"""

import logging
from dataclasses import dataclass

from src.config import Config, get_config
from src.retriever.vector_store import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result from reranking."""
    
    results: list[SearchResult]
    original_scores: list[float]
    rerank_scores: list[float]


class Reranker:
    """
    Cross-encoder reranker for improving retrieval precision.
    
    Uses a small cross-encoder model (ms-marco-MiniLM) to rerank
    search results based on query-document relevance.
    """
    
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._model = None
    
    def _get_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            
            self._model = CrossEncoder(
                self.config.reranker.model,
                max_length=512,
            )
            logger.info(f"Loaded reranker model: {self.config.reranker.model}")
        
        return self._model
    
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> RerankResult:
        """
        Rerank search results using cross-encoder.
        
        Args:
            query: The search query
            results: List of search results to rerank
            top_k: Number of results to return after reranking
            
        Returns:
            RerankResult with reranked results
        """
        if not results:
            return RerankResult(results=[], original_scores=[], rerank_scores=[])
        
        top_k = top_k or self.config.reranker.top_k
        model = self._get_model()
        
        # Prepare pairs for cross-encoder
        pairs = [(query, result.document.content) for result in results]
        
        # Get rerank scores
        scores = model.predict(pairs)
        
        # Store original scores
        original_scores = [r.score for r in results]
        
        # Create new results with rerank scores
        reranked = []
        for result, score in zip(results, scores):
            new_result = SearchResult(
                document=result.document,
                score=float(score),
            )
            reranked.append(new_result)
        
        # Sort by rerank score
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        # Apply threshold
        threshold = self.config.reranker.threshold
        filtered = [r for r in reranked if r.score >= threshold]
        
        # Take top-k
        final_results = filtered[:top_k] if filtered else reranked[:top_k]
        
        return RerankResult(
            results=final_results,
            original_scores=original_scores,
            rerank_scores=list(scores),
        )
    
    def rerank_batch(
        self,
        queries: list[str],
        results_list: list[list[SearchResult]],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """
        Rerank multiple query-results pairs.
        
        More efficient than calling rerank() multiple times.
        """
        return [
            self.rerank(query, results, top_k)
            for query, results in zip(queries, results_list)
        ]
