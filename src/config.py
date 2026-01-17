from __future__ import annotations
"""
Configuration management for LookaheadRAG.

Supports loading from YAML files, environment variables, and programmatic overrides.
Uses Pydantic for validation and type safety.
"""

import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class OllamaConfig(BaseModel):
    """Ollama LLM configuration."""
    host: str = Field(default="http://localhost:11434")
    planner_model: str = Field(default="llama3.2:3b")
    synthesizer_model: str = Field(default="llama3.1:8b")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)


class GroqConfig(BaseModel):
    """Groq API configuration."""
    api_key: str = Field(default="")
    planner_model: str = Field(default="llama-3.2-3b-preview")
    synthesizer_model: str = Field(default="llama-3.1-8b-instant")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)


class GoogleConfig(BaseModel):
    """Google AI configuration."""
    api_key: str = Field(default="")
    model: str = Field(default="gemini-1.5-flash")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: Literal["ollama", "groq", "google"] = Field(default="ollama")
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    groq: GroqConfig = Field(default_factory=GroqConfig)
    google: GoogleConfig = Field(default_factory=GoogleConfig)


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    model: str = Field(default="all-MiniLM-L6-v2")
    batch_size: int = Field(default=32, gt=0)
    normalize: bool = Field(default=True)


class RerankerConfig(BaseModel):
    """Reranker configuration."""
    model: str = Field(default="cross-encoder/ms-marco-MiniLM-L6-v2")
    top_k: int = Field(default=10, gt=0)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    provider: Literal["chromadb", "faiss"] = Field(default="chromadb")
    persist_dir: str = Field(default="./data/chroma_db")
    collection_name: str = Field(default="hotpotqa_wiki")


class SelfConsistencyConfig(BaseModel):
    """Self-consistency configuration for confidence estimation."""
    enabled: bool = Field(default=True)
    num_samples: int = Field(default=3, ge=1)
    agreement_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class PlannerConfig(BaseModel):
    """Planner model configuration."""
    max_nodes: int = Field(default=5, ge=1)
    min_nodes: int = Field(default=2, ge=1)
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    self_consistency: SelfConsistencyConfig = Field(default_factory=SelfConsistencyConfig)


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    top_k: int = Field(default=5, ge=1)
    max_parallel_queries: int = Field(default=5, ge=1)
    timeout_seconds: float = Field(default=30.0, gt=0)
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)


class CompressionConfig(BaseModel):
    """Context compression configuration."""
    enabled: bool = Field(default=True)
    target_ratio: float = Field(default=0.5, ge=0.0, le=1.0)


class ContextConfig(BaseModel):
    """Context assembly configuration."""
    max_tokens: int = Field(default=4096, gt=0)
    dedup_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)


class UtilityWeights(BaseModel):
    """Weights for utility function in budgeted pruning."""
    confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    novelty: float = Field(default=0.3, ge=0.0, le=1.0)
    hop_coverage: float = Field(default=0.3, ge=0.0, le=1.0)


class PruningConfig(BaseModel):
    """Budgeted pruning configuration."""
    enabled: bool = Field(default=True)
    max_budget: int = Field(default=10, ge=1)
    utility_weights: UtilityWeights = Field(default_factory=UtilityWeights)


class FallbackTriggers(BaseModel):
    """Fallback trigger thresholds."""
    low_coverage_threshold: int = Field(default=2, ge=0)
    high_entropy_threshold: float = Field(default=1.5, ge=0.0)


class FallbackConfig(BaseModel):
    """Fallback configuration."""
    enabled: bool = Field(default=True)
    triggers: FallbackTriggers = Field(default_factory=FallbackTriggers)
    max_additional_steps: int = Field(default=2, ge=0)


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    datasets: list[str] = Field(default=["hotpotqa"])
    metrics: list[str] = Field(
        default=["exact_match", "f1", "latency_p50", "latency_p95", "num_llm_calls"]
    )
    subset_size: int = Field(default=500, ge=1)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file: str = Field(default="./logs/lookahead_rag.log")


class Config(BaseModel):
    """Main configuration class for LookaheadRAG."""
    seed: int = Field(default=42)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    pruning: PruningConfig = Field(default_factory=PruningConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    def set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        # Note: torch seed should be set if using PyTorch models
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        # Expand environment variables
        data = cls._expand_env_vars(data)
        
        return cls(**data)
    
    @classmethod
    def _expand_env_vars(cls, data: dict) -> dict:
        """Recursively expand environment variables in config values."""
        if isinstance(data, dict):
            return {k: cls._expand_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls._expand_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and "}" in data:
            # Parse ${VAR:default} format
            var_part = data[2:data.index("}")]
            if ":" in var_part:
                var_name, default = var_part.split(":", 1)
            else:
                var_name, default = var_part, ""
            return os.environ.get(var_name, default)
        return data


# Global config instance
_config: Config | None = None


def get_config(config_path: str | Path | None = None) -> Config:
    """Get or create global configuration instance."""
    global _config
    
    if _config is None:
        if config_path is None:
            # Default config path
            config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        
        if Path(config_path).exists():
            _config = Config.from_yaml(config_path)
        else:
            _config = Config()
        
        _config.set_seed()
    
    return _config


def reset_config() -> None:
    """Reset global configuration (useful for testing)."""
    global _config
    _config = None
