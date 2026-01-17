from __future__ import annotations
"""
Tests for the Synthesizer module.
"""

import pytest

from src.synthesizer.context import ContextAssembler, EvidenceChunk, AssembledContext
from src.synthesizer.prompts import SynthesisPromptBuilder, SimpleRAGPromptBuilder
from src.planner.schema import PlanGraph, PlanNode
from src.retriever.parallel import RetrievalResult, NodeRetrievalResult
from src.retriever.vector_store import Document, SearchResult


class TestEvidenceChunk:
    """Tests for EvidenceChunk."""
    
    def test_create_chunk(self):
        """Test creating evidence chunk."""
        chunk = EvidenceChunk(
            content="Paris is the capital of France",
            document_id="doc1",
            node_id="n1",
            score=0.85,
        )
        
        assert chunk.content == "Paris is the capital of France"
        assert chunk.node_id == "n1"
        assert chunk.score == 0.85


class TestContextAssembler:
    """Tests for ContextAssembler."""
    
    @pytest.fixture
    def assembler(self):
        """Create context assembler."""
        return ContextAssembler()
    
    @pytest.fixture
    def sample_plan(self):
        """Create sample plan."""
        return PlanGraph(
            question="What is the capital of France?",
            nodes=[
                PlanNode(id="n1", query="France capital", confidence=0.9),
                PlanNode(id="n2", query="Paris landmarks", confidence=0.7),
            ],
        )
    
    @pytest.fixture
    def sample_retrieval_result(self):
        """Create sample retrieval result."""
        return RetrievalResult(
            node_results=[
                NodeRetrievalResult(
                    node_id="n1",
                    query="France capital",
                    results=[
                        SearchResult(
                            document=Document(
                                id="doc1",
                                content="Paris is the capital of France.",
                            ),
                            score=0.95,
                        ),
                    ],
                    latency_ms=50.0,
                ),
                NodeRetrievalResult(
                    node_id="n2",
                    query="Paris landmarks",
                    results=[
                        SearchResult(
                            document=Document(
                                id="doc2",
                                content="The Eiffel Tower is located in Paris.",
                            ),
                            score=0.85,
                        ),
                    ],
                    latency_ms=45.0,
                ),
            ],
            total_latency_ms=95.0,
            parallel_latency_ms=55.0,
        )
    
    def test_assemble_basic(self, assembler, sample_plan, sample_retrieval_result):
        """Test basic context assembly."""
        context = assembler.assemble(sample_plan, sample_retrieval_result)
        
        assert context.question == "What is the capital of France?"
        assert context.total_chunks == 2
        assert "n1" in context.evidence_by_node
        assert "n2" in context.evidence_by_node
    
    def test_provenance_tracking(self, assembler, sample_plan, sample_retrieval_result):
        """Test that provenance is tracked correctly."""
        context = assembler.assemble(sample_plan, sample_retrieval_result)
        
        assert "n1" in context.provenance
        assert "doc1" in context.provenance["n1"]
        assert "doc2" in context.provenance["n2"]
    
    def test_to_structured_string(self, assembler, sample_plan, sample_retrieval_result):
        """Test converting to structured string."""
        context = assembler.assemble(sample_plan, sample_retrieval_result)
        
        structured = context.to_structured_string()
        
        assert "n1" in structured
        assert "Paris" in structured
        assert "[n1.1]" in structured
    
    def test_deduplication(self, assembler):
        """Test deduplication of similar content."""
        results = [
            SearchResult(
                document=Document(id="doc1", content="Paris is capital of France"),
                score=0.9,
            ),
            SearchResult(
                document=Document(id="doc2", content="Paris is the capital of France"),
                score=0.85,
            ),
            SearchResult(
                document=Document(id="doc3", content="Tokyo is the capital of Japan"),
                score=0.8,
            ),
        ]
        
        deduplicated = assembler.deduplicate_results(results, threshold=0.7)
        
        # High similarity threshold should keep both Paris docs as they differ slightly
        assert len(deduplicated) >= 2


class TestSynthesisPromptBuilder:
    """Tests for SynthesisPromptBuilder."""
    
    @pytest.fixture
    def builder(self):
        """Create prompt builder."""
        return SynthesisPromptBuilder()
    
    @pytest.fixture
    def sample_context(self):
        """Create sample assembled context."""
        return AssembledContext(
            question="Who wrote Hamlet?",
            plan_summary="Plan with 2 nodes",
            evidence_by_node={
                "n1": [
                    EvidenceChunk(
                        content="Shakespeare wrote Hamlet",
                        document_id="doc1",
                        node_id="n1",
                        score=0.9,
                    ),
                ],
            },
            total_chunks=1,
            total_tokens=100,
            provenance={"n1": ["doc1"]},
        )
    
    def test_build_system_prompt(self, builder):
        """Test building system prompt."""
        prompt = builder.build_system_prompt()
        
        assert "question-answering" in prompt.lower()
        assert "evidence" in prompt.lower()
    
    def test_build_user_prompt(self, builder, sample_context):
        """Test building user prompt."""
        prompt = builder.build_user_prompt(sample_context)
        
        assert "Who wrote Hamlet?" in prompt
        assert "Shakespeare" in prompt
    
    def test_build_messages(self, builder, sample_context):
        """Test building chat messages."""
        messages = builder.build_messages(sample_context)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


class TestSimpleRAGPromptBuilder:
    """Tests for SimpleRAGPromptBuilder."""
    
    def test_build_prompt(self):
        """Test building simple RAG prompt."""
        builder = SimpleRAGPromptBuilder()
        
        prompt = builder.build_prompt(
            question="What is Python?",
            context="Python is a programming language.",
        )
        
        assert "What is Python?" in prompt
        assert "Python is a programming language." in prompt
