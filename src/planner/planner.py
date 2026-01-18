from __future__ import annotations
"""
Planner Model Implementation.

Provides LLM-based planners that generate PlanGraphs from questions.
Supports multiple backends: Ollama (local), Groq (free tier), and manual prompts.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import Config, get_config
from src.planner.schema import PlanGraph, PlanNode, OperatorType, ProducedVariable, EntityType

logger = logging.getLogger(__name__)


# Planner system prompt with variable binding support
PLANNER_SYSTEM_PROMPT = """You are a retrieval planner for a multi-hop question answering system.

Given a question, you must generate a retrieval plan as a JSON object. The plan specifies what sub-queries to execute to gather evidence for answering the question.

## Output Format
Return ONLY valid JSON with this structure:
{
  "question": "<original question>",
  "nodes": [
    {
      "id": "n1",
      "query": "<search query>",
      "op": "<operator type>",
      "depends_on": [],
      "confidence": <0.0-1.0>,
      "budget_cost": 1,
      "produces": [{"var": "<name>", "type": "<entity_type>", "description": "<what this extracts>"}],
      "bindings": {"<var>": "<source_node.var>"},
      "required_inputs": ["<var>"]
    }
  ],
  "global": {"max_nodes": 5, "max_parallel_queries": 5, "fallback_allowed": true}
}

## Operator Types
- "lookup": Fetch entity or property information
- "bridge": Find connecting entity or relation between concepts
- "filter": Apply constraints (dates, types, categories)
- "compare": Compare two or more items
- "aggregate": Count, list, or summarize
- "verify": Support or refute with evidence

## Variable Binding (for multi-hop queries)
- Use {placeholder} in query for values filled from upstream results
- "produces": List entities this node extracts (var, type, description)
- "bindings": Map placeholders to sources, e.g., {"person": "n1.person"}
- "required_inputs": Placeholders that MUST be resolved before execution
- Entity types: person, organization, location, date, work_of_art, number, event, other

## Rules
1. Generate 2-5 diverse search queries
2. Each query must be distinct and non-overlapping
3. Use dependencies + bindings to model multi-hop reasoning
4. Assign confidence 0.6-0.9 for straightforward queries, 0.3-0.6 for uncertain ones
5. For multi-hop questions, use produces/bindings to chain entity extraction
6. Keep queries concise and searchable

## Example 1 (Simple)
Question: "When was Albert Einstein born?"
{
  "nodes": [
    {"id": "n1", "query": "Albert Einstein birth date", "op": "lookup", "depends_on": [], "confidence": 0.9, "budget_cost": 1}
  ]
}

## Example 2 (Multi-hop with Bindings)
Question: "Who directed films starring the actress who won the Oscar for La La Land?"
{
  "nodes": [
    {
      "id": "n1",
      "query": "La La Land Oscar winner Best Actress",
      "op": "lookup",
      "depends_on": [],
      "confidence": 0.85,
      "budget_cost": 1,
      "produces": [{"var": "actress", "type": "person", "description": "Oscar-winning actress from La La Land"}]
    },
    {
      "id": "n2",
      "query": "{actress} filmography movies",
      "op": "bridge",
      "depends_on": ["n1"],
      "confidence": 0.75,
      "budget_cost": 1,
      "bindings": {"actress": "n1.actress"},
      "required_inputs": ["actress"],
      "produces": [{"var": "films", "type": "work_of_art", "description": "Films starring the actress"}]
    },
    {
      "id": "n3",
      "query": "{actress} films director",
      "op": "lookup",
      "depends_on": ["n1"],
      "confidence": 0.80,
      "budget_cost": 1,
      "bindings": {"actress": "n1.actress"},
      "required_inputs": ["actress"]
    }
  ]
}

Now generate a plan for the following question. Return ONLY the JSON, no explanation."""



class BasePlanner(ABC):
    """Abstract base class for planners."""
    
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
    
    @abstractmethod
    async def generate_plan(self, question: str) -> PlanGraph:
        """Generate a retrieval plan for the given question."""
        pass
    
    @abstractmethod
    def generate_plan_sync(self, question: str) -> PlanGraph:
        """Synchronous version of generate_plan."""
        pass


class LLMPlanner(BasePlanner):
    """Planner using LLM to generate retrieval plans."""
    
    def __init__(self, config: Config | None = None, provider: str | None = None):
        super().__init__(config)
        self.provider = provider or self.config.llm.provider
        self._client: httpx.AsyncClient | None = None
    
    def _get_model_name(self) -> str:
        """Get the planner model name based on provider."""
        if self.provider == "ollama":
            return self.config.llm.ollama.planner_model
        elif self.provider == "groq":
            return self.config.llm.groq.planner_model
        elif self.provider == "google":
            return self.config.llm.google.model
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _get_temperature(self) -> float:
        """Get temperature based on provider."""
        if self.provider == "ollama":
            return self.config.llm.ollama.temperature
        elif self.provider == "groq":
            return self.config.llm.groq.temperature
        elif self.provider == "google":
            return self.config.llm.google.temperature
        return 0.1
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        url = f"{self.config.llm.ollama.host}/api/generate"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                json={
                    "model": self._get_model_name(),
                    "prompt": prompt,
                    "system": PLANNER_SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": self._get_temperature(),
                        "num_predict": self.config.llm.ollama.max_tokens,
                    },
                },
            )
            response.raise_for_status()
            return response.json()["response"]
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=30))
    async def _call_groq(self, prompt: str) -> str:
        """Call Groq API with rate limit handling."""
        import asyncio
        
        # Add delay to avoid rate limits (30 req/min = 2 sec between)
        await asyncio.sleep(2.0)
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.config.llm.groq.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._get_model_name(),
                    "messages": [
                        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self._get_temperature(),
                    "max_tokens": self.config.llm.groq.max_tokens,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    
    def _call_ollama_sync(self, prompt: str) -> str:
        """Synchronous Ollama call."""
        url = f"{self.config.llm.ollama.host}/api/generate"
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                url,
                json={
                    "model": self._get_model_name(),
                    "prompt": prompt,
                    "system": PLANNER_SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": self._get_temperature(),
                        "num_predict": self.config.llm.ollama.max_tokens,
                    },
                },
            )
            response.raise_for_status()
            return response.json()["response"]
    
    def _call_groq_sync(self, prompt: str) -> str:
        """Synchronous Groq call with rate limit handling."""
        import time as t
        
        # Add delay to avoid rate limits
        t.sleep(2.0)
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.config.llm.groq.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._get_model_name(),
                    "messages": [
                        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self._get_temperature(),
                    "max_tokens": self.config.llm.groq.max_tokens,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    
    def _parse_response(self, response: str, question: str, generation_time: float) -> PlanGraph:
        """Parse LLM response into PlanGraph."""
        # Try to extract JSON from response
        response = response.strip()
        
        # Handle markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            # Find start and end of code block
            start = 1
            end = len(lines) - 1
            for i, line in enumerate(lines):
                if line.startswith("```") and i > 0:
                    end = i
                    break
            response = "\n".join(lines[start:end])
        
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response}")
            # Create a fallback single-node plan
            return self._create_fallback_plan(question, generation_time)
        
        # Convert to PlanGraph
        try:
            nodes = []
            for node in data.get("nodes", []):
                # Parse produces if present
                produces = []
                for p in node.get("produces", []):
                    try:
                        produces.append(ProducedVariable(
                            var=p.get("var", "entity"),
                            type=EntityType(p.get("type", "other")),
                            description=p.get("description", ""),
                        ))
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Failed to parse produces: {e}")
                
                plan_node = PlanNode(
                    id=node["id"],
                    query=node["query"],
                    op=OperatorType(node.get("op", "lookup")),
                    depends_on=node.get("depends_on", []),
                    confidence=node.get("confidence", 0.5),
                    budget_cost=node.get("budget_cost", 1),
                    produces=produces,
                    bindings=node.get("bindings", {}),
                    required_inputs=node.get("required_inputs", []),
                )
                nodes.append(plan_node)
            
            plan = PlanGraph(
                question=data.get("question", question),
                nodes=nodes,
                planner_model=self._get_model_name(),
                generation_time_ms=generation_time * 1000,
            )
            return plan
        except Exception as e:
            logger.error(f"Failed to create PlanGraph: {e}")
            return self._create_fallback_plan(question, generation_time)
    
    def _create_fallback_plan(self, question: str, generation_time: float) -> PlanGraph:
        """Create a simple fallback plan when parsing fails."""
        return PlanGraph(
            question=question,
            nodes=[
                PlanNode(
                    id="n1",
                    query=question,
                    op=OperatorType.LOOKUP,
                    depends_on=[],
                    confidence=0.5,
                    budget_cost=1,
                )
            ],
            planner_model=self._get_model_name(),
            generation_time_ms=generation_time * 1000,
        )
    
    async def generate_plan(self, question: str) -> PlanGraph:
        """Generate a retrieval plan for the given question."""
        start_time = time.time()
        
        prompt = f"Question: {question}"
        
        if self.provider == "ollama":
            response = await self._call_ollama(prompt)
        elif self.provider == "groq":
            response = await self._call_groq(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        generation_time = time.time() - start_time
        return self._parse_response(response, question, generation_time)
    
    def generate_plan_sync(self, question: str) -> PlanGraph:
        """Synchronous version of generate_plan."""
        start_time = time.time()
        
        prompt = f"Question: {question}"
        
        if self.provider == "ollama":
            response = self._call_ollama_sync(prompt)
        elif self.provider == "groq":
            response = self._call_groq_sync(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        generation_time = time.time() - start_time
        return self._parse_response(response, question, generation_time)


class ManualPlanner(BasePlanner):
    """
    Planner that generates prompts for manual execution.
    
    Use this when you want to run prompts manually on ChatGPT
    or another interface to avoid API costs.
    """
    
    def get_prompt(self, question: str) -> str:
        """Get the full prompt for manual execution."""
        return f"{PLANNER_SYSTEM_PROMPT}\n\nQuestion: {question}"
    
    def parse_manual_response(self, response: str, question: str) -> PlanGraph:
        """Parse a manually obtained response."""
        return LLMPlanner()._parse_response(response, question, 0.0)
    
    async def generate_plan(self, question: str) -> PlanGraph:
        """Not implemented for manual planner."""
        raise NotImplementedError(
            "ManualPlanner requires manual execution. "
            "Use get_prompt() to get the prompt and parse_manual_response() to parse the result."
        )
    
    def generate_plan_sync(self, question: str) -> PlanGraph:
        """Not implemented for manual planner."""
        raise NotImplementedError(
            "ManualPlanner requires manual execution. "
            "Use get_prompt() to get the prompt and parse_manual_response() to parse the result."
        )


def get_planner(
    provider: str | None = None,
    config: Config | None = None,
) -> BasePlanner:
    """Factory function to get a planner instance."""
    config = config or get_config()
    provider = provider or config.llm.provider
    
    if provider == "manual":
        return ManualPlanner(config)
    else:
        return LLMPlanner(config, provider)
