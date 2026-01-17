from __future__ import annotations
"""
Standard RAG Baseline.

Single query, top-k retrieval, direct synthesis.
This is the simplest baseline for comparison.
"""

import logging
import time
from dataclasses import dataclass

from src.config import Config, get_config
from src.retriever import VectorStore, SearchResult
from src.synthesizer import Synthesizer, SynthesisResult

logger = logging.getLogger(__name__)


@dataclass
class StandardRAGResult:
    """Result from Standard RAG."""
    
    question: str
    answer: str
    retrieved_chunks: list[SearchResult]
    synthesis_result: SynthesisResult
    
    # Metrics
    retrieval_latency_ms: float
    synthesis_latency_ms: float
    total_latency_ms: float
    num_llm_calls: int = 1
    
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "num_chunks": len(self.retrieved_chunks),
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "synthesis_latency_ms": self.synthesis_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "num_llm_calls": self.num_llm_calls,
        }


class StandardRAG:
    """
    Standard RAG baseline.
    
    Pipeline:
    1. Use question as query
    2. Retrieve top-k chunks
    3. Synthesize answer
    """
    
    def __init__(
        self,
        config: Config | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.config = config or get_config()
        self.vector_store = vector_store or VectorStore(self.config)
        self.synthesizer = Synthesizer(self.config)
    
    def run(
        self,
        question: str,
        top_k: int | None = None,
    ) -> StandardRAGResult:
        """
        Run Standard RAG pipeline.
        
        Args:
            question: The question to answer
            top_k: Number of chunks to retrieve
            
        Returns:
            StandardRAGResult
        """
        start_time = time.time()
        top_k = top_k or self.config.retrieval.top_k
        
        # Step 1: Retrieve
        logger.info(f"Standard RAG: Retrieving for question: {question[:100]}...")
        retrieval_start = time.time()
        chunks = self.vector_store.search(question, top_k=top_k)
        retrieval_latency = (time.time() - retrieval_start) * 1000
        
        # Step 2: Build context
        context = "\n\n".join([
            f"[{i+1}] {chunk.document.content}"
            for i, chunk in enumerate(chunks)
        ])
        
        # Step 3: Synthesize
        logger.info("Standard RAG: Synthesizing answer...")
        synthesis_result = self.synthesizer.synthesize_simple(question, context)
        
        total_latency = (time.time() - start_time) * 1000
        
        return StandardRAGResult(
            question=question,
            answer=synthesis_result.answer,
            retrieved_chunks=chunks,
            synthesis_result=synthesis_result,
            retrieval_latency_ms=retrieval_latency,
            synthesis_latency_ms=synthesis_result.latency_ms,
            total_latency_ms=total_latency,
            num_llm_calls=1,
        )
