"""
Tests for Evidence Verification.

Tests claim extraction, entity filtering, contradiction detection,
and sufficiency analysis.
"""

import pytest
from src.planner.schema import PlanNode, PlanGraph, OperatorType
from src.retriever.parallel import RetrievalResult, NodeRetrievalResult
from src.retriever.vector_store import SearchResult, Document
from src.synthesizer.evidence_verifier import (
    EvidenceVerifier,
    AtomicClaim,
    ContradictionPair,
    VerificationResult,
    VerificationStatus,
    VerificationMetrics,
)


def make_search_result(content: str, doc_id: str = "doc1") -> SearchResult:
    """Helper to create search results."""
    return SearchResult(
        document=Document(id=doc_id, content=content),
        score=0.9,
    )


def make_node_result(node_id: str, chunks: list[str]) -> NodeRetrievalResult:
    """Helper to create node retrieval results."""
    return NodeRetrievalResult(
        node_id=node_id,
        query="test query",
        results=[make_search_result(c, f"doc_{i}") for i, c in enumerate(chunks)],
        latency_ms=100,
    )


class TestAtomicClaim:
    """Tests for AtomicClaim class."""
    
    def test_shares_entity_same(self):
        """Test entity sharing detection."""
        claim1 = AtomicClaim(
            text="Einstein was born in 1879.",
            source_node="n1",
            source_chunk_idx=0,
            entities=["Einstein", "1879"],
        )
        claim2 = AtomicClaim(
            text="Einstein developed relativity.",
            source_node="n2",
            source_chunk_idx=0,
            entities=["Einstein"],
        )
        
        assert claim1.shares_entity_with(claim2)
    
    def test_shares_entity_none(self):
        """Test when claims don't share entities."""
        claim1 = AtomicClaim(
            text="Einstein was born.",
            source_node="n1",
            source_chunk_idx=0,
            entities=["Einstein"],
        )
        claim2 = AtomicClaim(
            text="Newton invented calculus.",
            source_node="n2",
            source_chunk_idx=0,
            entities=["Newton"],
        )
        
        assert not claim1.shares_entity_with(claim2)


class TestClaimExtraction:
    """Tests for claim extraction."""
    
    def test_extract_claims(self):
        """Test basic claim extraction."""
        verifier = EvidenceVerifier()
        node = PlanNode(id="n1", query="Einstein birth")
        chunks = [
            "Albert Einstein was born in 1879 in Germany. He later moved to the US.",
        ]
        
        claims = verifier._extract_claims(node, chunks)
        
        assert len(claims) > 0
        assert all(c.source_node == "n1" for c in claims)
    
    def test_extract_entities(self):
        """Test entity extraction from text."""
        verifier = EvidenceVerifier()
        
        entities = verifier._extract_entities(
            "Albert Einstein was born in 1879 in Germany."
        )
        
        assert "Albert Einstein" in entities or "Einstein" in entities
        assert "1879" in entities


class TestContradictionDetection:
    """Tests for contradiction detection."""
    
    def test_detect_negation_contradiction(self):
        """Test detecting negation-based contradictions."""
        verifier = EvidenceVerifier()
        
        claim1 = AtomicClaim(
            text="Einstein was born in Germany.",
            source_node="n1",
            source_chunk_idx=0,
            entities=["Einstein", "Germany"],
        )
        claim2 = AtomicClaim(
            text="Einstein was not born in Germany.",
            source_node="n2",
            source_chunk_idx=0,
            entities=["Einstein", "Germany"],
        )
        
        is_contradiction, score = verifier._check_contradiction(claim1, claim2)
        
        assert is_contradiction
        assert score > 0.5
    
    def test_no_contradiction_different_facts(self):
        """Test that different facts don't contradict."""
        verifier = EvidenceVerifier()
        
        claim1 = AtomicClaim(
            text="Einstein was born in 1879.",
            source_node="n1",
            source_chunk_idx=0,
            entities=["Einstein", "1879"],
        )
        claim2 = AtomicClaim(
            text="Einstein won the Nobel Prize.",
            source_node="n2",
            source_chunk_idx=0,
            entities=["Einstein", "Nobel Prize"],
        )
        
        is_contradiction, score = verifier._check_contradiction(claim1, claim2)
        
        assert not is_contradiction
    
    def test_entity_filtered_comparison(self):
        """Test that only claims with shared entities are compared."""
        verifier = EvidenceVerifier()
        
        claims = [
            AtomicClaim(
                text="Einstein was born in 1879.",
                source_node="n1",
                source_chunk_idx=0,
                entities=["Einstein"],
            ),
            AtomicClaim(
                text="Newton was born in 1642.",
                source_node="n2",
                source_chunk_idx=0,
                entities=["Newton"],
            ),
        ]
        
        contradictions = verifier._detect_contradictions_filtered(claims)
        
        # No contradictions because entities don't overlap
        assert len(contradictions) == 0


class TestFullVerification:
    """Integration tests for full verification."""
    
    def test_verify_sufficient(self):
        """Test verification with sufficient evidence."""
        verifier = EvidenceVerifier()
        plan = PlanGraph(
            question="When was Einstein born?",
            nodes=[
                PlanNode(id="n1", query="Einstein birth"),
            ],
        )
        retrieval_result = RetrievalResult(
            node_results=[
                make_node_result("n1", ["Einstein was born in 1879."]),
            ],
            total_latency_ms=100,
            parallel_latency_ms=100,
        )
        
        result = verifier.verify("When was Einstein born?", plan, retrieval_result)
        
        assert result.status == VerificationStatus.SUFFICIENT
        assert result.coverage_ratio == 1.0
    
    def test_verify_insufficient(self):
        """Test verification with empty evidence."""
        verifier = EvidenceVerifier()
        plan = PlanGraph(
            question="Test?",
            nodes=[
                PlanNode(id="n1", query="q1"),
                PlanNode(id="n2", query="q2"),
            ],
        )
        retrieval_result = RetrievalResult(
            node_results=[
                make_node_result("n1", []),  # Empty
                make_node_result("n2", []),  # Empty
            ],
            total_latency_ms=100,
            parallel_latency_ms=100,
        )
        
        result = verifier.verify("Test?", plan, retrieval_result)
        
        assert result.status == VerificationStatus.INSUFFICIENT
        assert result.coverage_ratio == 0.0


class TestVerificationMetrics:
    """Tests for aggregate metrics."""
    
    def test_from_results(self):
        """Test aggregating verification results."""
        results = [
            VerificationResult(
                status=VerificationStatus.SUFFICIENT,
                total_hops=2,
                covered_hops=2,
                coverage_ratio=1.0,
            ),
            VerificationResult(
                status=VerificationStatus.INSUFFICIENT,
                total_hops=2,
                covered_hops=1,
                coverage_ratio=0.5,
            ),
        ]
        
        metrics = VerificationMetrics.from_results(results)
        
        assert metrics.total_examples == 2
        assert metrics.sufficient_count == 1
        assert metrics.insufficient_count == 1
        assert metrics.avg_coverage_ratio == 0.75
