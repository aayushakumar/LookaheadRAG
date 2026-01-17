from __future__ import annotations
"""
Synthesizer module for LookaheadRAG.

This module handles context assembly and final answer synthesis.
"""

from src.synthesizer.context import ContextAssembler, AssembledContext
from src.synthesizer.prompts import SynthesisPromptBuilder
from src.synthesizer.synthesizer import Synthesizer, SynthesisResult

__all__ = [
    "ContextAssembler",
    "AssembledContext",
    "SynthesisPromptBuilder",
    "Synthesizer",
    "SynthesisResult",
]
