"""
Binding Resolver Module.

Resolves variable bindings in PlanGraph nodes using LLM-based
entity extraction with citations.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.config import Config, get_config
from src.planner.schema import PlanGraph, PlanNode, ProducedVariable, EntityType

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """An entity extracted from retrieved text."""
    value: str
    entity_type: EntityType
    citation: str  # e.g., "[n1.2]" indicating node 1, chunk 2
    confidence: float = 1.0


@dataclass
class BindingContext:
    """Context for variable binding across nodes."""
    
    # Mapping: "node_id.var_name" -> ExtractedEntity
    resolved: dict[str, ExtractedEntity] = field(default_factory=dict)
    
    def get_value(self, source: str) -> str | None:
        """Get resolved value for a source like 'n1.person'."""
        if source in self.resolved:
            return self.resolved[source].value
        return None
    
    def get_citation(self, source: str) -> str | None:
        """Get citation for a resolved binding."""
        if source in self.resolved:
            return self.resolved[source].citation
        return None
    
    def to_context_dict(self) -> dict[str, str]:
        """Convert to simple string mapping for resolve_bindings()."""
        return {k: v.value for k, v in self.resolved.items()}


class BindingResolver:
    """
    Resolves variable bindings in PlanGraph nodes.
    
    Uses LLM-based extraction for accurate entity identification
    with citations back to source chunks.
    
    Execution rule: A node with placeholders CANNOT execute until
    all required_inputs are resolved from upstream produces.
    """
    
    def __init__(
        self,
        config: Config | None = None,
        extractor: str = "llm",
    ):
        """
        Initialize the binding resolver.
        
        Args:
            config: Configuration object
            extractor: Extraction method - "llm" (default), "ner", or "hybrid"
        """
        self.config = config or get_config()
        self.extractor = extractor
        self._llm_client = None
    
    def extract_produces(
        self,
        node: PlanNode,
        chunks: list[str],
    ) -> dict[str, ExtractedEntity]:
        """
        Extract values for node.produces from retrieved chunks.
        
        Args:
            node: The PlanNode with produces definitions
            chunks: Retrieved text chunks for this node
            
        Returns:
            Mapping of var_name -> ExtractedEntity
        """
        if not node.produces:
            return {}
        
        if not chunks:
            logger.warning(f"No chunks to extract from for node {node.id}")
            return {}
        
        if self.extractor == "llm":
            return self._llm_extract(node, chunks)
        elif self.extractor == "ner":
            return self._ner_extract(node, chunks)
        else:  # hybrid
            result = self._ner_extract(node, chunks)
            # Fall back to LLM for missing produces
            missing = [p for p in node.produces if p.var not in result]
            if missing:
                llm_result = self._llm_extract(node, chunks, target_vars=missing)
                result.update(llm_result)
            return result
    
    def _llm_extract(
        self,
        node: PlanNode,
        chunks: list[str],
        target_vars: list[ProducedVariable] | None = None,
    ) -> dict[str, ExtractedEntity]:
        """Use LLM to extract typed entities with citation."""
        target = target_vars or node.produces
        
        if not target:
            return {}
        
        # Build prompt
        var_descriptions = "\n".join([
            f"- {p.var} ({p.type.value}): {p.description or 'extract this entity'}"
            for p in target
        ])
        
        chunks_text = "\n\n".join([
            f"[Chunk {i}]: {chunk[:500]}"  # Truncate long chunks
            for i, chunk in enumerate(chunks)
        ])
        
        prompt = f"""Extract entities from the evidence below.

Query context: {node.query}

Evidence:
{chunks_text}

Extract these variables (return JSON):
{var_descriptions}

Return format:
{{"var_name": {{"value": "extracted value", "chunk_idx": 0}}}}

Only include variables you can find with high confidence. Return valid JSON only."""

        try:
            response = self._call_llm(prompt)
            return self._parse_llm_response(response, target, len(chunks))
        except Exception as e:
            logger.error(f"LLM extraction failed for node {node.id}: {e}")
            return {}
    
    def _parse_llm_response(
        self,
        response: str,
        target_vars: list[ProducedVariable],
        num_chunks: int,
    ) -> dict[str, ExtractedEntity]:
        """Parse LLM JSON response into ExtractedEntity objects."""
        result = {}
        
        # Extract JSON from response
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response as JSON: {response[:200]}")
            return {}
        
        # Map var names to their types
        var_types = {p.var: p.type for p in target_vars}
        
        for var_name, info in data.items():
            if var_name not in var_types:
                continue
            
            if isinstance(info, dict):
                value = info.get("value", "")
                chunk_idx = info.get("chunk_idx", 0)
            else:
                value = str(info)
                chunk_idx = 0
            
            if value:
                result[var_name] = ExtractedEntity(
                    value=value,
                    entity_type=var_types[var_name],
                    citation=f"[chunk_{chunk_idx}]",
                    confidence=0.9,  # LLM extraction assumed high confidence
                )
        
        return result
    
    def _ner_extract(
        self,
        node: PlanNode,
        chunks: list[str],
    ) -> dict[str, ExtractedEntity]:
        """Use NER for fast extraction (fallback method)."""
        # Simplified NER-based extraction using regex patterns
        result = {}
        text = " ".join(chunks)
        
        for prod in node.produces:
            extracted = self._extract_by_type(text, prod.type)
            if extracted:
                result[prod.var] = ExtractedEntity(
                    value=extracted,
                    entity_type=prod.type,
                    citation="[ner]",
                    confidence=0.7,
                )
        
        return result
    
    def _extract_by_type(self, text: str, entity_type: EntityType) -> str | None:
        """Extract first entity of given type using patterns."""
        patterns = {
            EntityType.DATE: r'\b(\d{4}|\d{1,2}/\d{1,2}/\d{2,4})\b',
            EntityType.NUMBER: r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b',
        }
        
        if entity_type in patterns:
            match = re.search(patterns[entity_type], text)
            if match:
                return match.group(1)
        
        # For other types, would need actual NER
        return None
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM for extraction."""
        import httpx
        
        provider = self.config.llm.provider
        
        if provider == "groq":
            api_key = self.config.llm.groq.api_key
            if not api_key:
                raise ValueError("GROQ_API_KEY not set")
            
            response = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 256,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        
        elif provider == "ollama":
            host = self.config.llm.ollama.host
            response = httpx.post(
                f"{host}/api/generate",
                json={
                    "model": self.config.llm.ollama.planner_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["response"]
        
        else:
            raise ValueError(f"Unsupported provider for binding: {provider}")
    
    def resolve_node(
        self,
        node: PlanNode,
        context: BindingContext,
    ) -> bool:
        """
        Resolve bindings for a single node using the context.
        
        Args:
            node: Node to resolve
            context: Current binding context
            
        Returns:
            True if all bindings resolved successfully
        """
        if not node.required_inputs:
            return True
        
        context_dict = context.to_context_dict()
        
        # Check if all required inputs are available
        for var in node.required_inputs:
            source = node.bindings.get(var)
            if not source or source not in context_dict:
                logger.warning(
                    f"Node {node.id} missing required input '{var}' "
                    f"(source: {source})"
                )
                return False
        
        # Resolve the query
        node.resolve_bindings(context_dict)
        return True
    
    def resolve_plan(
        self,
        plan: PlanGraph,
        retrieval_results: dict[str, list[str]],
    ) -> tuple[PlanGraph, BindingContext]:
        """
        Resolve all bindings in a plan in execution order.
        
        Args:
            plan: The PlanGraph to resolve
            retrieval_results: Mapping of node_id -> list of chunk texts
            
        Returns:
            Tuple of (resolved plan, binding context)
        """
        context = BindingContext()
        
        for level in plan.get_execution_order():
            for node in level:
                # First, extract any produces from this node's retrieval
                if node.id in retrieval_results and node.produces:
                    extracted = self.extract_produces(
                        node, retrieval_results[node.id]
                    )
                    # Add to context with node_id prefix
                    for var_name, entity in extracted.items():
                        key = f"{node.id}.{var_name}"
                        context.resolved[key] = entity
                        logger.debug(f"Extracted {key} = {entity.value}")
                
                # Then resolve any bindings this node needs
                if node.required_inputs:
                    success = self.resolve_node(node, context)
                    if not success:
                        logger.warning(
                            f"Failed to resolve bindings for node {node.id}"
                        )
        
        return plan, context
