from __future__ import annotations
"""
Vector Store Interface.

Provides a unified interface for vector databases (ChromaDB, FAISS).
Handles document storage, embedding, and semantic search.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import Config, get_config

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document chunk in the vector store."""
    
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    
    @classmethod
    def from_text(
        cls,
        content: str,
        source: str = "",
        title: str = "",
        **metadata: Any,
    ) -> "Document":
        """Create a document from text with auto-generated ID."""
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
        return cls(
            id=doc_id,
            content=content,
            metadata={
                "source": source,
                "title": title,
                **metadata,
            },
        )


@dataclass
class SearchResult:
    """Result from a vector search."""
    
    document: Document
    score: float  # Similarity score (higher is better)
    
    def __lt__(self, other: "SearchResult") -> bool:
        return self.score < other.score


class VectorStore:
    """
    Unified interface for vector storage and retrieval.
    
    Supports ChromaDB as the primary backend.
    """
    
    def __init__(
        self,
        config: Config | None = None,
        collection_name: str | None = None,
    ):
        self.config = config or get_config()
        self.collection_name = collection_name or self.config.vector_store.collection_name
        
        self._embedding_model = None
        self._client = None
        self._collection = None
    
    def _get_embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            
            self._embedding_model = SentenceTransformer(
                self.config.embedding.model
            )
            logger.info(f"Loaded embedding model: {self.config.embedding.model}")
        
        return self._embedding_model
    
    def _get_client(self):
        """Get or create ChromaDB client."""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings
            
            persist_dir = Path(self.config.vector_store.persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                ),
            )
            logger.info(f"Created ChromaDB client at: {persist_dir}")
        
        return self._client
    
    def _get_collection(self):
        """Get or create the collection."""
        if self._collection is None:
            client = self._get_client()
            
            # Create embedding function wrapper for ChromaDB
            embedding_model = self._get_embedding_model()
            model_name = self.config.embedding.model
            
            class EmbeddingFunction:
                def __init__(self, model, name):
                    self._model = model
                    self._name = name
                
                def __call__(self, input: list[str]) -> list[list[float]]:
                    embeddings = self._model.encode(
                        input,
                        normalize_embeddings=True,
                    )
                    return embeddings.tolist()
                
                def embed_query(self, input) -> list[float]:
                    """Embed a single query (required by ChromaDB 1.4.x)."""
                    # ChromaDB 1.4.x may pass a list or a string
                    if isinstance(input, list):
                        texts = input
                    else:
                        texts = [input]
                    
                    embeddings = self._model.encode(
                        texts,
                        normalize_embeddings=True,
                    )
                    # Return first embedding if single query
                    if not isinstance(input, list):
                        return embeddings[0].tolist()
                    return embeddings.tolist()
                
                def embed_documents(self, input: list[str]) -> list[list[float]]:
                    """Embed documents (required by ChromaDB 1.4.x)."""
                    embeddings = self._model.encode(
                        input,
                        normalize_embeddings=True,
                    )
                    return embeddings.tolist()
                
                def name(self) -> str:
                    """Return the name of the embedding function (required by ChromaDB)."""
                    return self._name
            
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=EmbeddingFunction(embedding_model, model_name),
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Got collection: {self.collection_name}")
        
        return self._collection
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        model = self._get_embedding_model()
        embeddings = model.encode(
            texts,
            batch_size=self.config.embedding.batch_size,
            normalize_embeddings=self.config.embedding.normalize,
        )
        return embeddings.tolist()
    
    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store."""
        if not documents:
            return
        
        collection = self._get_collection()
        
        # Prepare data for ChromaDB
        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Add in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_contents = contents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            collection.add(
                ids=batch_ids,
                documents=batch_contents,
                metadatas=batch_metadatas,
            )
        
        logger.info(f"Added {len(documents)} documents to collection")
    
    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of SearchResult sorted by relevance
        """
        collection = self._get_collection()
        top_k = top_k or self.config.retrieval.top_k
        
        # Build query parameters
        query_params = {
            "query_texts": [query],
            "n_results": top_k,
        }
        
        if filter_metadata:
            query_params["where"] = filter_metadata
        
        results = collection.query(**query_params)
        
        # Convert to SearchResult objects
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                doc = Document(
                    id=doc_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                )
                # ChromaDB returns distances, convert to similarity
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # Cosine distance to similarity
                
                search_results.append(SearchResult(document=doc, score=score))
        
        return sorted(search_results, key=lambda x: x.score, reverse=True)
    
    def count(self) -> int:
        """Get the number of documents in the collection."""
        collection = self._get_collection()
        return collection.count()
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
            self._collection = None
            logger.info(f"Deleted collection: {self.collection_name}")
        except ValueError:
            logger.warning(f"Collection not found: {self.collection_name}")
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.delete_collection()
        # Re-create empty collection
        self._get_collection()
