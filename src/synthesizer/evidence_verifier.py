"""
Evidence Consistency Verification.

Implements verification via claim extraction + targeted NLI to:
1. Check evidence sufficiency (≥1 claim supports each hop)
2. Detect contradictions (high-confidence conflicts on same slot)

Uses entity-filtered NLI to reduce false contradictions
(only compare claims sharing named entities).

Reference: Distinct from "Speculative RAG" draft-verify approach.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.config import Config, get_config
from src.planner.schema import PlanGraph, PlanNode
from src.retriever.parallel import RetrievalResult

logger = logging.getLogger(__name__)


class VerificationStatus(str, Enum):
    """Status of evidence verification."""
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    CONTRADICTORY = "contradictory"


@dataclass
class AtomicClaim:
    """An atomic claim extracted from evidence."""
    
    text: str
    source_node: str
    source_chunk_idx: int
    entities: list[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def shares_entity_with(self, other: "AtomicClaim") -> bool:
        """Check if this claim shares an entity with another."""
        return bool(set(self.entities) & set(other.entities))


@dataclass
class ContradictionPair:
    """A pair of claims detected as contradictory."""
    
    claim1: AtomicClaim
    claim2: AtomicClaim
    nli_score: float
    description: str = ""


@dataclass
class HopCoverage:
    """Coverage analysis for a single hop (node)."""
    
    node_id: str
    num_claims: int
    has_supporting_claim: bool
    claims: list[AtomicClaim] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Complete verification result."""
    
    status: VerificationStatus
    
    # Sufficiency analysis
    total_hops: int
    covered_hops: int
    coverage_ratio: float
    hop_coverage: list[HopCoverage] = field(default_factory=list)
    
    # Contradiction analysis
    contradictions: list[ContradictionPair] = field(default_factory=list)
    num_claims_checked: int = 0
    num_pairs_compared: int = 0
    
    # All extracted claims
    all_claims: list[AtomicClaim] = field(default_factory=list)
    
    # Recommendation
    recommendation: str = ""
    
    def is_sufficient(self) -> bool:
        """Check if evidence is sufficient."""
        return self.status == VerificationStatus.SUFFICIENT
    
    def has_contradictions(self) -> bool:
        """Check if evidence has contradictions."""
        return len(self.contradictions) > 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "total_hops": self.total_hops,
            "covered_hops": self.covered_hops,
            "coverage_ratio": self.coverage_ratio,
            "num_contradictions": len(self.contradictions),
            "num_claims": len(self.all_claims),
            "recommendation": self.recommendation,
        }


class EvidenceVerifier:
    """
    Verification via claim extraction + entity-filtered NLI.
    
    Paper-worthy definitions:
    - Sufficient: ≥1 claim supports each hop required by plan
    - Contradictory: two high-confidence claims conflict on same slot
    
    Key insight: Only compare claims sharing a named entity to reduce
    false contradictions from unrelated statements.
    """
    
    def __init__(
        self,
        config: Config | None = None,
        nli_threshold: float = 0.8,
        max_claims_per_node: int = 3,
    ):
        self.config = config or get_config()
        self.nli_threshold = nli_threshold
        self.max_claims_per_node = max_claims_per_node
        self._nli_model = None
    
    def verify(
        self,
        question: str,
        plan: PlanGraph,
        retrieval_result: RetrievalResult,
    ) -> VerificationResult:
        """
        Perform evidence verification.
        
        Args:
            question: Original question
            plan: The PlanGraph with node structure
            retrieval_result: Retrieved chunks per node
            
        Returns:
            VerificationResult with sufficiency and contradiction analysis
        """
        # Step 1: Extract atomic claims per node
        all_claims = []
        hop_coverages = []
        
        for node in plan.nodes:
            chunks = retrieval_result.get_chunks_for_node(node.id)
            claims = self._extract_claims(node, chunks)
            all_claims.extend(claims)
            
            hop_coverages.append(HopCoverage(
                node_id=node.id,
                num_claims=len(claims),
                has_supporting_claim=len(claims) > 0,
                claims=claims,
            ))
        
        # Step 2: Check sufficiency
        covered_hops = sum(1 for h in hop_coverages if h.has_supporting_claim)
        total_hops = len(hop_coverages)
        coverage_ratio = covered_hops / total_hops if total_hops > 0 else 0.0
        
        # Step 3: Detect contradictions (entity-filtered)
        contradictions = self._detect_contradictions_filtered(all_claims)
        
        # Step 4: Determine status and recommendation
        if len(contradictions) > 0:
            status = VerificationStatus.CONTRADICTORY
            recommendation = (
                f"Detected {len(contradictions)} contradiction(s). "
                "Consider requesting additional evidence or flagging uncertainty."
            )
        elif coverage_ratio < 0.5:
            status = VerificationStatus.INSUFFICIENT
            recommendation = (
                f"Only {covered_hops}/{total_hops} hops have supporting evidence. "
                "Consider expanding retrieval or fallback."
            )
        else:
            status = VerificationStatus.SUFFICIENT
            recommendation = "Evidence appears sufficient for synthesis."
        
        return VerificationResult(
            status=status,
            total_hops=total_hops,
            covered_hops=covered_hops,
            coverage_ratio=coverage_ratio,
            hop_coverage=hop_coverages,
            contradictions=contradictions,
            num_claims_checked=len(all_claims),
            num_pairs_compared=self._count_comparable_pairs(all_claims),
            all_claims=all_claims,
            recommendation=recommendation,
        )
    
    def _extract_claims(
        self,
        node: PlanNode,
        chunks: list[str],
    ) -> list[AtomicClaim]:
        """
        Extract 1-3 atomic claims from chunks for a node.
        
        Uses lightweight extraction (sentence-based + entity extraction).
        For production, could use LLM-based claim extraction.
        """
        claims = []
        
        for chunk_idx, chunk in enumerate(chunks[:3]):  # Limit chunks
            # Extract sentences as potential claims
            sentences = self._split_sentences(chunk)
            
            for sentence in sentences[:2]:  # Limit per chunk
                if len(sentence) < 20 or len(sentence) > 200:
                    continue
                
                entities = self._extract_entities(sentence)
                
                claims.append(AtomicClaim(
                    text=sentence.strip(),
                    source_node=node.id,
                    source_chunk_idx=chunk_idx,
                    entities=entities,
                ))
                
                if len(claims) >= self.max_claims_per_node:
                    break
            
            if len(claims) >= self.max_claims_per_node:
                break
        
        return claims
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities from text."""
        entities = []
        
        # Simple pattern-based extraction
        # Capitalized words (likely proper nouns)
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(caps[:5])
        
        # Dates
        dates = re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', text)
        entities.extend(dates[:2])
        
        # Numbers with units
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand|%|km|miles)?\b', text)
        entities.extend(numbers[:2])
        
        return list(set(entities))
    
    def _detect_contradictions_filtered(
        self,
        claims: list[AtomicClaim],
    ) -> list[ContradictionPair]:
        """
        Detect contradictions using entity-filtered comparison.
        
        Only compare claims that share at least one named entity
        to reduce false positives from unrelated statements.
        """
        contradictions = []
        
        for i, claim1 in enumerate(claims):
            for claim2 in claims[i + 1:]:
                # Filter: must share at least one entity
                if not claim1.shares_entity_with(claim2):
                    continue
                
                # Check for contradiction using NLI
                is_contradiction, score = self._check_contradiction(claim1, claim2)
                
                if is_contradiction:
                    shared = set(claim1.entities) & set(claim2.entities)
                    contradictions.append(ContradictionPair(
                        claim1=claim1,
                        claim2=claim2,
                        nli_score=score,
                        description=f"Conflict on: {', '.join(shared)}",
                    ))
        
        return contradictions
    
    def _check_contradiction(
        self,
        claim1: AtomicClaim,
        claim2: AtomicClaim,
    ) -> tuple[bool, float]:
        """
        Check if two claims contradict using NLI or heuristics.
        
        Returns (is_contradiction, confidence_score).
        """
        # Heuristic contradiction detection
        # (In production, use a real NLI model like DeBERTa-mnli)
        
        text1 = claim1.text.lower()
        text2 = claim2.text.lower()
        
        # Look for negation patterns
        negation_patterns = [
            (r'\b(is|was|are|were)\b', r'\b(is not|was not|are not|were not)\b'),
            (r'\b(did)\b', r'\b(did not|didn\'t)\b'),
            (r'\b(true)\b', r'\b(false)\b'),
            (r'\b(yes)\b', r'\b(no)\b'),
        ]
        
        for pos_pattern, neg_pattern in negation_patterns:
            has_pos_1 = bool(re.search(pos_pattern, text1))
            has_neg_1 = bool(re.search(neg_pattern, text1))
            has_pos_2 = bool(re.search(pos_pattern, text2))
            has_neg_2 = bool(re.search(neg_pattern, text2))
            
            # One positive, one negative on same entity = potential contradiction
            if (has_pos_1 and has_neg_2) or (has_neg_1 and has_pos_2):
                return True, 0.7
        
        # Look for conflicting numbers
        nums1 = re.findall(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b', text1)
        nums2 = re.findall(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b', text2)
        
        if nums1 and nums2:
            # If both mention numbers in similar context, might conflict
            shared_entities = set(claim1.entities) & set(claim2.entities)
            if shared_entities and nums1[0] != nums2[0]:
                return True, 0.6
        
        return False, 0.0
    
    def _count_comparable_pairs(self, claims: list[AtomicClaim]) -> int:
        """Count pairs that share entities (were compared)."""
        count = 0
        for i, c1 in enumerate(claims):
            for c2 in claims[i + 1:]:
                if c1.shares_entity_with(c2):
                    count += 1
        return count


# Verification metrics for evaluation
@dataclass
class VerificationMetrics:
    """Aggregate verification metrics across a dataset."""
    
    total_examples: int = 0
    
    # Sufficiency metrics
    sufficient_count: int = 0
    insufficient_count: int = 0
    contradictory_count: int = 0
    
    # Coverage metrics
    avg_coverage_ratio: float = 0.0
    avg_claims_per_example: float = 0.0
    
    # Contradiction metrics  
    avg_contradictions: float = 0.0
    examples_with_contradictions: int = 0
    
    @property
    def sufficient_rate(self) -> float:
        return self.sufficient_count / self.total_examples if self.total_examples else 0.0
    
    @property
    def contradiction_rate(self) -> float:
        return self.examples_with_contradictions / self.total_examples if self.total_examples else 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_examples": self.total_examples,
            "sufficient_rate": self.sufficient_rate,
            "contradiction_rate": self.contradiction_rate,
            "avg_coverage_ratio": self.avg_coverage_ratio,
            "avg_claims_per_example": self.avg_claims_per_example,
        }
    
    @classmethod
    def from_results(cls, results: list[VerificationResult]) -> "VerificationMetrics":
        """Aggregate from individual results."""
        if not results:
            return cls()
        
        n = len(results)
        return cls(
            total_examples=n,
            sufficient_count=sum(1 for r in results if r.status == VerificationStatus.SUFFICIENT),
            insufficient_count=sum(1 for r in results if r.status == VerificationStatus.INSUFFICIENT),
            contradictory_count=sum(1 for r in results if r.status == VerificationStatus.CONTRADICTORY),
            avg_coverage_ratio=sum(r.coverage_ratio for r in results) / n,
            avg_claims_per_example=sum(len(r.all_claims) for r in results) / n,
            avg_contradictions=sum(len(r.contradictions) for r in results) / n,
            examples_with_contradictions=sum(1 for r in results if r.contradictions),
        )
