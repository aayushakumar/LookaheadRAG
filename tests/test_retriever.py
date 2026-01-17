from __future__ import annotations
"""
Tests for the Retriever module.
"""

import pytest

from src.retriever.vector_store import Document, SearchResult, VectorStore
from src.config import Config


class TestDocument:
    """Tests for Document class."""
    
    def test_create_document(self):
        """Test creating a document."""
        doc = Document(
            id="doc1",
            content="This is test content",
            metadata={"source": "test"},
        )
        
        assert doc.id == "doc1"
        assert doc.content == "This is test content"
        assert doc.metadata["source"] == "test"
    
    def test_from_text(self):
        """Test creating document from text."""
        doc = Document.from_text(
            content="Some content here",
            source="wikipedia",
            title="Test Article",
        )
        
        assert doc.content == "Some content here"
        assert doc.metadata["source"] == "wikipedia"
        assert doc.metadata["title"] == "Test Article"
        assert len(doc.id) == 12  # MD5 hash truncated
    
    def test_from_text_generates_unique_ids(self):
        """Test that different content generates different IDs."""
        doc1 = Document.from_text("content one")
        doc2 = Document.from_text("content two")
        
        assert doc1.id != doc2.id


class TestSearchResult:
    """Tests for SearchResult class."""
    
    def test_comparison(self):
        """Test search result comparison by score."""
        doc = Document(id="doc1", content="test")
        
        result1 = SearchResult(document=doc, score=0.9)
        result2 = SearchResult(document=doc, score=0.8)
        
        assert result2 < result1  # Higher score is "greater"


class TestVectorStore:
    """Tests for VectorStore class."""
    
    @pytest.fixture
    def config(self, tmp_path):
        """Create test config with temp directory."""
        return Config(
            vector_store=Config.model_fields["vector_store"].default.model_copy(
                update={"persist_dir": str(tmp_path / "chroma")}
            ),
        )
    
    @pytest.fixture
    def vector_store(self, config):
        """Create test vector store."""
        store = VectorStore(config, collection_name="test_collection")
        yield store
        # Cleanup
        store.delete_collection()
    
    @pytest.mark.slow
    def test_add_and_search(self, vector_store):
        """Test adding documents and searching."""
        docs = [
            Document.from_text("The capital of France is Paris", title="France"),
            Document.from_text("The Eiffel Tower is in Paris", title="Landmarks"),
            Document.from_text("Tokyo is the capital of Japan", title="Japan"),
        ]
        
        vector_store.add_documents(docs)
        
        results = vector_store.search("What is the capital of France?", top_k=2)
        
        assert len(results) == 2
        assert any("Paris" in r.document.content for r in results)
    
    @pytest.mark.slow
    def test_count(self, vector_store):
        """Test document count."""
        docs = [
            Document.from_text(f"Document {i}") 
            for i in range(5)
        ]
        
        vector_store.add_documents(docs)
        
        assert vector_store.count() == 5
    
    @pytest.mark.slow
    def test_clear(self, vector_store):
        """Test clearing collection."""
        docs = [Document.from_text("test")]
        vector_store.add_documents(docs)
        
        assert vector_store.count() == 1
        
        vector_store.clear()
        
        assert vector_store.count() == 0
