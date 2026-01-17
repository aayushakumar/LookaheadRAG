from __future__ import annotations
"""
Baselines module for LookaheadRAG.

Implements baseline methods for comparison:
1. Standard RAG (single query)
2. Multi-query RAG (query expansion)
3. Agentic RAG (ReAct-style iterative)
"""

from src.baselines.standard_rag import StandardRAG, StandardRAGResult
from src.baselines.multiquery_rag import MultiQueryRAG, MultiQueryRAGResult
from src.baselines.agentic_rag import AgenticRAG, AgenticRAGResult

__all__ = [
    "StandardRAG",
    "StandardRAGResult",
    "MultiQueryRAG",
    "MultiQueryRAGResult",
    "AgenticRAG",
    "AgenticRAGResult",
]
