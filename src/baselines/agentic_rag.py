from __future__ import annotations
"""
Agentic RAG Baseline (ReAct-style).

Implements iterative retrieval with reasoning traces.
This simulates agentic behavior with sequential tool calls.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.config import Config, get_config
from src.retriever import VectorStore, SearchResult
from src.synthesizer import Synthesizer

logger = logging.getLogger(__name__)


REACT_SYSTEM_PROMPT = """You are a question-answering agent. You can search for information to answer questions.

You work in a loop of Thought, Action, Observation:
1. Thought: Analyze what information you need
2. Action: Search for specific information. Use format: SEARCH[query]
3. Observation: You will receive search results
4. Repeat until you have enough information, then provide the final answer

When you have enough information, respond with:
ANSWER[your final answer here]

Rules:
- Maximum {max_steps} search actions allowed
- Be specific in your search queries
- Analyze search results carefully before deciding next action"""


REACT_USER_TEMPLATE = """Question: {question}

{history}

Now continue with the next step. Remember to use SEARCH[query] or ANSWER[your answer]."""


@dataclass
class AgentStep:
    """A single step in the agent's reasoning."""
    
    step_num: int
    thought: str
    action_type: str  # "search" or "answer"
    action_input: str
    observation: str | None = None
    latency_ms: float = 0.0


@dataclass
class AgenticRAGResult:
    """Result from Agentic RAG."""
    
    question: str
    answer: str
    steps: list[AgentStep]
    all_retrieved_chunks: list[SearchResult]
    
    # Metrics
    total_latency_ms: float
    num_llm_calls: int
    num_search_calls: int
    
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "num_steps": len(self.steps),
            "num_search_calls": self.num_search_calls,
            "num_chunks": len(self.all_retrieved_chunks),
            "total_latency_ms": self.total_latency_ms,
            "num_llm_calls": self.num_llm_calls,
        }
    
    def get_reasoning_trace(self) -> str:
        """Get the full reasoning trace."""
        lines = []
        for step in self.steps:
            lines.append(f"Step {step.step_num}:")
            lines.append(f"  Thought: {step.thought}")
            lines.append(f"  Action: {step.action_type}[{step.action_input}]")
            if step.observation:
                lines.append(f"  Observation: {step.observation[:200]}...")
        return "\n".join(lines)


class AgenticRAG:
    """
    Agentic RAG baseline (ReAct-style).
    
    Pipeline:
    1. Start reasoning loop
    2. Agent decides to search or answer
    3. If search: retrieve and provide observation
    4. Repeat until answer or max steps
    """
    
    def __init__(
        self,
        config: Config | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.config = config or get_config()
        self.vector_store = vector_store or VectorStore(self.config)
        self.synthesizer = Synthesizer(self.config)
    
    def _call_llm(self, system: str, user: str) -> tuple[str, float]:
        """Call LLM and return response with latency."""
        start_time = time.time()
        
        if self.config.llm.provider == "ollama":
            response = self._call_ollama(system, user)
        elif self.config.llm.provider == "groq":
            response = self._call_groq(system, user)
        else:
            raise ValueError(f"Unsupported provider: {self.config.llm.provider}")
        
        latency = (time.time() - start_time) * 1000
        return response, latency
    
    def _call_ollama(self, system: str, user: str) -> str:
        """Call Ollama API."""
        url = f"{self.config.llm.ollama.host}/api/generate"
        
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                url,
                json={
                    "model": self.config.llm.ollama.synthesizer_model,
                    "prompt": user,
                    "system": system,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 1024,
                    },
                },
            )
            response.raise_for_status()
            return response.json()["response"]
    
    def _call_groq(self, system: str, user: str) -> str:
        """Call Groq API."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.config.llm.groq.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.llm.groq.synthesizer_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    
    def _parse_action(self, response: str) -> tuple[str, str, str]:
        """Parse agent response into thought, action_type, and action_input."""
        lines = response.strip().split("\n")
        
        thought = ""
        action_type = ""
        action_input = ""
        
        for line in lines:
            line = line.strip()
            
            # Extract thought
            if line.lower().startswith("thought:"):
                thought = line[8:].strip()
            
            # Extract SEARCH action
            if "SEARCH[" in line:
                start = line.index("SEARCH[") + 7
                end = line.index("]", start)
                action_type = "search"
                action_input = line[start:end]
            
            # Extract ANSWER action
            if "ANSWER[" in line:
                start = line.index("ANSWER[") + 7
                # Find matching bracket
                bracket_count = 1
                end = start
                while bracket_count > 0 and end < len(line):
                    if line[end] == "[":
                        bracket_count += 1
                    elif line[end] == "]":
                        bracket_count -= 1
                    end += 1
                action_type = "answer"
                action_input = line[start:end-1]
        
        return thought, action_type, action_input
    
    def _execute_search(
        self,
        query: str,
        top_k: int = 3,
    ) -> tuple[list[SearchResult], str]:
        """Execute search and format observation."""
        results = self.vector_store.search(query, top_k=top_k)
        
        if not results:
            observation = "No results found."
        else:
            observation_lines = []
            for i, result in enumerate(results):
                content = result.document.content[:300]
                observation_lines.append(f"[{i+1}] {content}")
            observation = "\n".join(observation_lines)
        
        return results, observation
    
    def run(
        self,
        question: str,
        max_steps: int = 5,
    ) -> AgenticRAGResult:
        """
        Run Agentic RAG pipeline.
        
        Args:
            question: The question to answer
            max_steps: Maximum number of search steps
            
        Returns:
            AgenticRAGResult
        """
        start_time = time.time()
        
        system_prompt = REACT_SYSTEM_PROMPT.format(max_steps=max_steps)
        
        steps: list[AgentStep] = []
        all_chunks: list[SearchResult] = []
        num_llm_calls = 0
        num_search_calls = 0
        history = ""
        answer = ""
        
        logger.info(f"Agentic RAG: Starting for question: {question[:100]}...")
        
        for step_num in range(1, max_steps + 2):  # +1 for final answer step
            # Build user prompt with history
            user_prompt = REACT_USER_TEMPLATE.format(
                question=question,
                history=history,
            )
            
            # Call LLM
            response, latency = self._call_llm(system_prompt, user_prompt)
            num_llm_calls += 1
            
            # Parse response
            thought, action_type, action_input = self._parse_action(response)
            
            if not action_type:
                # Try to find answer in response directly
                if any(word in response.lower() for word in ["the answer is", "therefore", "in conclusion"]):
                    action_type = "answer"
                    action_input = response
                else:
                    # Default to search with question
                    action_type = "search"
                    action_input = question
            
            step = AgentStep(
                step_num=step_num,
                thought=thought or "Analyzing...",
                action_type=action_type,
                action_input=action_input,
                latency_ms=latency,
            )
            
            if action_type == "answer":
                answer = action_input
                steps.append(step)
                logger.info(f"Agentic RAG: Found answer at step {step_num}")
                break
            
            elif action_type == "search":
                # Execute search
                results, observation = self._execute_search(action_input)
                step.observation = observation
                all_chunks.extend(results)
                num_search_calls += 1
                
                # Update history
                history += f"\nStep {step_num}:\n"
                history += f"Thought: {thought}\n"
                history += f"Action: SEARCH[{action_input}]\n"
                history += f"Observation: {observation}\n"
                
                steps.append(step)
                logger.info(f"Agentic RAG: Step {step_num} - searched '{action_input[:50]}...'")
        
        # If no answer found, synthesize from collected evidence
        if not answer:
            logger.info("Agentic RAG: Max steps reached, synthesizing from collected evidence...")
            context = "\n\n".join([
                chunk.document.content
                for chunk in all_chunks[:10]
            ])
            synthesis = self.synthesizer.synthesize_simple(question, context)
            answer = synthesis.answer
            num_llm_calls += 1
        
        total_latency = (time.time() - start_time) * 1000
        
        return AgenticRAGResult(
            question=question,
            answer=answer,
            steps=steps,
            all_retrieved_chunks=all_chunks,
            total_latency_ms=total_latency,
            num_llm_calls=num_llm_calls,
            num_search_calls=num_search_calls,
        )
