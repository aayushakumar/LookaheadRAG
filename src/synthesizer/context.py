from __future__ import annotations
"""
Context Assembly Module.

Assembles retrieved documents into a structured context for synthesis.
Handles deduplication, token budgeting, and provenance tracking.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from src.config import Config, get_config
from src.planner.schema import PlanGraph
from src.retriever.parallel import RetrievalResult
from src.retriever.vector_store import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class EvidenceChunk:
    """A single piece of evidence with provenance."""
    
    content: str
    document_id: str
    node_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssembledContext:
    """Assembled context ready for synthesis."""
    
    question: str
    plan_summary: str
    evidence_by_node: dict[str, list[EvidenceChunk]]
    total_chunks: int
    total_tokens: int  # Approximate
    provenance: dict[str, list[str]]  # node_id -> document_ids
    
    def to_structured_string(self) -> str:
        """Convert to structured string for synthesis prompt."""
        lines = []
        
        for node_id, chunks in self.evidence_by_node.items():
            if chunks:
                lines.append(f"\n## Evidence for {node_id}")
                for i, chunk in enumerate(chunks, 1):
                    lines.append(f"\n[{node_id}.{i}] {chunk.content}")
        
        return "\n".join(lines)
    
    def to_flat_string(self) -> str:
        """Convert to flat string (no structure)."""
        all_chunks = []
        for chunks in self.evidence_by_node.values():
            all_chunks.extend(chunks)
        
        # Deduplicate by content
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk.content not in seen:
                seen.add(chunk.content)
                unique_chunks.append(chunk)
        
        return "\n\n".join(chunk.content for chunk in unique_chunks)


class ContextAssembler:
    """
    Assembles retrieved documents into context for synthesis.
    
    Key responsibilities:
    1. Deduplication (by document ID and content overlap)
    2. Token budgeting (allocate budget proportional to confidence)
    3. Provenance tracking (node -> documents)
    4. Optional compression for low-confidence evidence
    """
    
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
    
    def assemble(
        self,
        plan: PlanGraph,
        retrieval_result: RetrievalResult,
    ) -> AssembledContext:
        """
        Assemble context from retrieval results.
        
        Args:
            plan: The plan graph with confidence scores
            retrieval_result: Results from parallel retrieval
            
        Returns:
            AssembledContext ready for synthesis
        """
        evidence_by_node: dict[str, list[EvidenceChunk]] = {}
        provenance: dict[str, list[str]] = {}
        seen_doc_ids: set[str] = set()
        seen_contents: set[str] = set()
        
        total_chunks = 0
        total_tokens = 0
        
        # Calculate budget per node based on confidence
        confidence_sum = sum(
            node.confidence
            for node in plan.nodes
        )
        
        for node_result in retrieval_result.node_results:
            node_id = node_result.node_id
            node = plan.get_node(node_id)
            
            if node is None:
                continue
            
            # Calculate token budget for this node
            if confidence_sum > 0:
                node_budget_ratio = node.confidence / confidence_sum
            else:
                node_budget_ratio = 1.0 / len(plan.nodes)
            
            node_budget = int(self.config.context.max_tokens * node_budget_ratio)
            current_tokens = 0
            
            evidence_by_node[node_id] = []
            provenance[node_id] = []
            
            for result in node_result.results:
                doc = result.document
                
                # Skip duplicates by document ID
                if doc.id in seen_doc_ids:
                    continue
                
                # Skip near-duplicate content
                content_hash = self._content_hash(doc.content)
                if content_hash in seen_contents:
                    continue
                
                # Check token budget
                content_tokens = self._estimate_tokens(doc.content)
                if current_tokens + content_tokens > node_budget:
                    # Try to compress if compression is enabled
                    if self.config.context.compression.enabled:
                        compressed = self._compress_content(
                            doc.content,
                            node_budget - current_tokens,
                        )
                        if compressed:
                            content_tokens = self._estimate_tokens(compressed)
                            chunk = EvidenceChunk(
                                content=compressed,
                                document_id=doc.id,
                                node_id=node_id,
                                score=result.score,
                                metadata=doc.metadata,
                            )
                            evidence_by_node[node_id].append(chunk)
                            provenance[node_id].append(doc.id)
                            current_tokens += content_tokens
                            total_chunks += 1
                    continue
                
                chunk = EvidenceChunk(
                    content=doc.content,
                    document_id=doc.id,
                    node_id=node_id,
                    score=result.score,
                    metadata=doc.metadata,
                )
                
                evidence_by_node[node_id].append(chunk)
                provenance[node_id].append(doc.id)
                seen_doc_ids.add(doc.id)
                seen_contents.add(content_hash)
                current_tokens += content_tokens
                total_chunks += 1
            
            total_tokens += current_tokens
        
        return AssembledContext(
            question=plan.question,
            plan_summary=plan.summary(),
            evidence_by_node=evidence_by_node,
            total_chunks=total_chunks,
            total_tokens=total_tokens,
            provenance=provenance,
        )
    
    def _content_hash(self, content: str) -> str:
        """Create a simple hash for near-duplicate detection."""
        # Use first 100 chars + last 100 chars
        key = (content[:100] + content[-100:]).lower()
        return key
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Average ~4 chars per token for English
        return len(text) // 4
    
    def _compress_content(self, content: str, max_tokens: int) -> str | None:
        """
        Compress content to fit within token budget.
        
        Simple extractive approach: take first and last parts.
        """
        if max_tokens <= 50:  # Too small
            return None
        
        max_chars = max_tokens * 4
        
        if len(content) <= max_chars:
            return content
        
        # Take first 60% and last 40%
        first_part = int(max_chars * 0.6)
        last_part = max_chars - first_part - 5  # 5 for "..."
        
        compressed = content[:first_part] + " ... " + content[-last_part:]
        return compressed
    
    def deduplicate_results(
        self,
        results: list[SearchResult],
        threshold: float | None = None,
    ) -> list[SearchResult]:
        """
        Deduplicate search results by content similarity.
        
        Uses Jaccard similarity on word sets.
        """
        threshold = threshold or self.config.context.dedup_threshold
        
        deduplicated = []
        seen_word_sets: list[set[str]] = []
        
        for result in results:
            words = set(result.document.content.lower().split())
            
            is_duplicate = False
            for seen_words in seen_word_sets:
                # Jaccard similarity
                if not words or not seen_words:
                    continue
                    
                intersection = len(words & seen_words)
                union = len(words | seen_words)
                similarity = intersection / union if union > 0 else 0
                
                if similarity > threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(result)
                seen_word_sets.append(words)
        
        return deduplicated
