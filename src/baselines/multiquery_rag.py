from __future__ import annotations
"""
Multi-Query RAG Baseline.

Uses LLM to generate multiple search queries from the question,
retrieves for each query in parallel, then synthesizes.
"""

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

from src.config import Config, get_config
from src.retriever import VectorStore, SearchResult
from src.synthesizer import Synthesizer, SynthesisResult

logger = logging.getLogger(__name__)


QUERY_EXPANSION_PROMPT = """Given a question, generate {num_queries} diverse search queries that would help find evidence to answer it.

Question: {question}

Return the queries as a JSON array of strings. Example: ["query 1", "query 2", "query 3"]

Queries:"""


@dataclass
class MultiQueryRAGResult:
    """Result from Multi-Query RAG."""
    
    question: str
    answer: str
    expanded_queries: list[str]
    retrieved_chunks: list[SearchResult]
    synthesis_result: SynthesisResult
    
    # Metrics
    expansion_latency_ms: float
    retrieval_latency_ms: float
    synthesis_latency_ms: float
    total_latency_ms: float
    num_llm_calls: int = 2  # expansion + synthesis
    
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "num_queries": len(self.expanded_queries),
            "num_chunks": len(self.retrieved_chunks),
            "expansion_latency_ms": self.expansion_latency_ms,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "synthesis_latency_ms": self.synthesis_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "num_llm_calls": self.num_llm_calls,
        }


class MultiQueryRAG:
    """
    Multi-Query RAG baseline.
    
    Pipeline:
    1. Generate multiple queries from question using LLM
    2. Retrieve for each query in parallel
    3. Deduplicate and merge results
    4. Synthesize answer
    """
    
    def __init__(
        self,
        config: Config | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.config = config or get_config()
        self.vector_store = vector_store or VectorStore(self.config)
        self.synthesizer = Synthesizer(self.config)
    
    def _expand_queries(
        self,
        question: str,
        num_queries: int = 3,
    ) -> tuple[list[str], float]:
        """Generate multiple queries using LLM."""
        import json
        
        start_time = time.time()
        prompt = QUERY_EXPANSION_PROMPT.format(
            question=question,
            num_queries=num_queries,
        )
        
        if self.config.llm.provider == "ollama":
            response = self._call_ollama(prompt)
        elif self.config.llm.provider == "groq":
            response = self._call_groq(prompt)
        else:
            # Fallback: simple split
            queries = [question]
            return queries, (time.time() - start_time) * 1000
        
        latency = (time.time() - start_time) * 1000
        
        # Parse JSON array
        try:
            # Find JSON array in response
            response = response.strip()
            if "[" in response:
                start = response.index("[")
                end = response.rindex("]") + 1
                queries = json.loads(response[start:end])
            else:
                queries = [question]
        except (json.JSONDecodeError, ValueError):
            queries = [question]
        
        # Always include original question
        if question not in queries:
            queries.insert(0, question)
        
        return queries, latency
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        url = f"{self.config.llm.ollama.host}/api/generate"
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                url,
                json={
                    "model": self.config.llm.ollama.planner_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 256,
                    },
                },
            )
            response.raise_for_status()
            return response.json()["response"]
    
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.config.llm.groq.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.llm.groq.planner_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 256,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    
    async def _retrieve_parallel(
        self,
        queries: list[str],
        top_k: int,
    ) -> tuple[list[SearchResult], float]:
        """Retrieve for multiple queries in parallel."""
        start_time = time.time()
        
        async def retrieve_one(query: str) -> list[SearchResult]:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.vector_store.search(query, top_k=top_k),
            )
        
        results = await asyncio.gather(*[retrieve_one(q) for q in queries])
        
        # Flatten and deduplicate
        seen_ids = set()
        unique_results = []
        for result_list in results:
            for result in result_list:
                if result.document.id not in seen_ids:
                    seen_ids.add(result.document.id)
                    unique_results.append(result)
        
        # Sort by score
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        latency = (time.time() - start_time) * 1000
        return unique_results, latency
    
    def run(
        self,
        question: str,
        num_queries: int = 3,
        top_k_per_query: int = 3,
    ) -> MultiQueryRAGResult:
        """
        Run Multi-Query RAG pipeline.
        
        Args:
            question: The question to answer
            num_queries: Number of queries to generate
            top_k_per_query: Chunks to retrieve per query
            
        Returns:
            MultiQueryRAGResult
        """
        start_time = time.time()
        
        # Step 1: Query expansion
        logger.info(f"Multi-Query RAG: Expanding queries for: {question[:100]}...")
        queries, expansion_latency = self._expand_queries(question, num_queries)
        logger.info(f"Generated {len(queries)} queries")
        
        # Step 2: Parallel retrieval
        logger.info("Multi-Query RAG: Parallel retrieval...")
        chunks, retrieval_latency = asyncio.run(
            self._retrieve_parallel(queries, top_k_per_query)
        )
        logger.info(f"Retrieved {len(chunks)} unique chunks")
        
        # Step 3: Build context
        context = "\n\n".join([
            f"[{i+1}] {chunk.document.content}"
            for i, chunk in enumerate(chunks[:10])  # Limit to top 10
        ])
        
        # Step 4: Synthesize
        logger.info("Multi-Query RAG: Synthesizing answer...")
        synthesis_result = self.synthesizer.synthesize_simple(question, context)
        
        total_latency = (time.time() - start_time) * 1000
        
        return MultiQueryRAGResult(
            question=question,
            answer=synthesis_result.answer,
            expanded_queries=queries,
            retrieved_chunks=chunks,
            synthesis_result=synthesis_result,
            expansion_latency_ms=expansion_latency,
            retrieval_latency_ms=retrieval_latency,
            synthesis_latency_ms=synthesis_result.latency_ms,
            total_latency_ms=total_latency,
            num_llm_calls=2,
        )
