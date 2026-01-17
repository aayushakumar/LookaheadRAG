from __future__ import annotations
"""
LookaheadRAG: Speculative Retrieval Planning for Low-Latency Multi-Hop QA

This package implements a novel RAG system that uses a small planner model to
generate a retrieval dependency graph, enabling parallel retrieval and one-shot
synthesis for near-agentic accuracy with RAG-like latency.
"""

__version__ = "0.1.0"
__author__ = "Aayush Kumar"

from src.config import Config, get_config

__all__ = ["Config", "get_config", "__version__"]
