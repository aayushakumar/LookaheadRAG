from __future__ import annotations
"""
PlanGraph Schema Definition.

Defines the JSON schema for retrieval plans using Pydantic models.
A PlanGraph is a DAG where nodes represent retrieval sub-queries with
dependencies, operators, and confidence scores.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class OperatorType(str, Enum):
    """Types of retrieval/reasoning operators."""
    LOOKUP = "lookup"       # Entity/property fetch
    BRIDGE = "bridge"       # Find connecting entity/relation
    FILTER = "filter"       # Apply constraints (date/type)
    COMPARE = "compare"     # A vs B comparison
    AGGREGATE = "aggregate" # Count/list operations
    VERIFY = "verify"       # Support/refute with evidence


class PlanNode(BaseModel):
    """A single node in the retrieval plan graph."""
    
    id: str = Field(
        ...,
        description="Unique identifier for this node (e.g., 'n1', 'n2')",
        min_length=1,
        max_length=10,
    )
    query: str = Field(
        ...,
        description="The search query or sub-question for this node",
        min_length=1,
        max_length=500,
    )
    op: OperatorType = Field(
        default=OperatorType.LOOKUP,
        description="The type of retrieval/reasoning operation",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of node IDs this node depends on",
    )
    confidence: float = Field(
        default=0.5,
        description="Confidence score for this node (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    budget_cost: int = Field(
        default=1,
        description="Cost units for budgeted pruning",
        ge=1,
    )
    
    # Optional metadata for analysis
    reasoning: str | None = Field(
        default=None,
        description="Brief reasoning for why this node is needed",
    )
    expected_evidence_type: str | None = Field(
        default=None,
        description="Expected type of evidence (e.g., 'entity', 'date', 'relation')",
    )
    
    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Ensure ID is alphanumeric."""
        if not v.replace("_", "").isalnum():
            raise ValueError(f"Node ID must be alphanumeric: {v}")
        return v
    
    @field_validator("depends_on")
    @classmethod
    def validate_depends_on(cls, v: list[str]) -> list[str]:
        """Ensure no duplicate dependencies."""
        if len(v) != len(set(v)):
            raise ValueError("Duplicate dependencies not allowed")
        return v
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "query": self.query,
            "op": self.op.value,
            "depends_on": self.depends_on,
            "confidence": self.confidence,
            "budget_cost": self.budget_cost,
        }


class GlobalSettings(BaseModel):
    """Global settings for plan execution."""
    
    max_nodes: int = Field(
        default=5,
        description="Maximum number of nodes to execute",
        ge=1,
        le=10,
    )
    max_parallel_queries: int = Field(
        default=5,
        description="Maximum parallel retrieval queries",
        ge=1,
        le=10,
    )
    fallback_allowed: bool = Field(
        default=True,
        description="Whether fallback to iterative retrieval is allowed",
    )


class PlanGraph(BaseModel):
    """
    A retrieval plan represented as a Directed Acyclic Graph (DAG).
    
    The planner generates this structure from a question, specifying
    what sub-queries to execute and their dependencies.
    """
    
    question: str = Field(
        ...,
        description="The original question to answer",
        min_length=1,
    )
    nodes: list[PlanNode] = Field(
        default_factory=list,
        description="List of retrieval nodes",
    )
    global_settings: GlobalSettings = Field(
        default_factory=GlobalSettings,
        alias="global",
    )
    
    # Metadata
    planner_model: str | None = Field(
        default=None,
        description="Model used to generate this plan",
    )
    generation_time_ms: float | None = Field(
        default=None,
        description="Time taken to generate this plan in milliseconds",
    )
    
    @model_validator(mode="after")
    def validate_dag(self) -> "PlanGraph":
        """Validate that the graph is a valid DAG (no cycles)."""
        node_ids = {node.id for node in self.nodes}
        
        # Check all dependencies reference valid nodes
        for node in self.nodes:
            for dep in node.depends_on:
                if dep not in node_ids:
                    raise ValueError(f"Node {node.id} depends on unknown node: {dep}")
            
            # Check no self-dependency
            if node.id in node.depends_on:
                raise ValueError(f"Node {node.id} cannot depend on itself")
        
        # Check for cycles using DFS
        if self._has_cycle():
            raise ValueError("Plan graph contains a cycle - must be a DAG")
        
        return self
    
    def _has_cycle(self) -> bool:
        """Check if the graph has a cycle using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            node = self.get_node(node_id)
            if node:
                for dep in node.depends_on:
                    if dep not in visited:
                        if dfs(dep):
                            return True
                    elif dep in rec_stack:
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        for node in self.nodes:
            if node.id not in visited:
                if dfs(node.id):
                    return True
        
        return False
    
    def get_node(self, node_id: str) -> PlanNode | None:
        """Get a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_root_nodes(self) -> list[PlanNode]:
        """Get nodes with no dependencies (can be executed first)."""
        return [node for node in self.nodes if not node.depends_on]
    
    def get_execution_order(self) -> list[list[PlanNode]]:
        """
        Get nodes grouped by execution level (topological sort).
        
        Returns a list of lists, where each inner list contains nodes
        that can be executed in parallel at that level.
        """
        if not self.nodes:
            return []
        
        # Calculate in-degree for each node
        in_degree = {node.id: len(node.depends_on) for node in self.nodes}
        
        # Start with root nodes
        result = []
        ready = [node for node in self.nodes if in_degree[node.id] == 0]
        
        while ready:
            result.append(ready)
            next_ready = []
            
            for node in ready:
                # Find nodes that depend on this one
                for other in self.nodes:
                    if node.id in other.depends_on:
                        in_degree[other.id] -= 1
                        if in_degree[other.id] == 0:
                            next_ready.append(other)
            
            ready = next_ready
        
        return result
    
    def total_budget_cost(self) -> int:
        """Calculate total budget cost of all nodes."""
        return sum(node.budget_cost for node in self.nodes)
    
    def average_confidence(self) -> float:
        """Calculate average confidence across all nodes."""
        if not self.nodes:
            return 0.0
        return sum(node.confidence for node in self.nodes) / len(self.nodes)
    
    def to_json(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "question": self.question,
            "nodes": [node.to_dict() for node in self.nodes],
            "global": self.global_settings.model_dump(),
        }
    
    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "PlanGraph":
        """Create PlanGraph from JSON dictionary."""
        return cls(
            question=data["question"],
            nodes=[PlanNode(**node) for node in data.get("nodes", [])],
            global_settings=GlobalSettings(**data.get("global", {})),
        )
    
    def summary(self) -> str:
        """Generate a human-readable summary of the plan."""
        lines = [
            f"Question: {self.question}",
            f"Nodes: {len(self.nodes)}",
            f"Total cost: {self.total_budget_cost()}",
            f"Avg confidence: {self.average_confidence():.2f}",
            "",
            "Execution order:",
        ]
        
        for level, nodes in enumerate(self.get_execution_order()):
            node_strs = [f"{n.id}({n.op.value})" for n in nodes]
            lines.append(f"  Level {level}: {', '.join(node_strs)}")
        
        return "\n".join(lines)
