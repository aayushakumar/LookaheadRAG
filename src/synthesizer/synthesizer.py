from __future__ import annotations
"""
Synthesizer Module.

Generates final answers from assembled context using LLM.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import Config, get_config
from src.synthesizer.context import AssembledContext
from src.synthesizer.prompts import SynthesisPromptBuilder

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A citation extracted from the answer."""
    
    node_id: str
    chunk_index: int
    raw: str  # Original citation string like "[n1.2]"


@dataclass
class SynthesisResult:
    """Result from synthesis."""
    
    answer: str
    citations: list[Citation]
    raw_response: str
    latency_ms: float
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    @property
    def has_citations(self) -> bool:
        return len(self.citations) > 0
    
    @property
    def answer_without_citations(self) -> str:
        """Get answer with citation markers removed."""
        return re.sub(r'\[n\d+\.\d+\]', '', self.answer).strip()


class Synthesizer:
    """
    Generates final answers from assembled context.
    
    Supports multiple LLM providers: Ollama (local), Groq (free tier).
    """
    
    def __init__(
        self,
        config: Config | None = None,
        provider: str | None = None,
    ):
        self.config = config or get_config()
        self.provider = provider or self.config.llm.provider
        self.prompt_builder = SynthesisPromptBuilder()
    
    def _get_model_name(self) -> str:
        """Get the synthesizer model name based on provider."""
        if self.provider == "ollama":
            return self.config.llm.ollama.synthesizer_model
        elif self.provider == "groq":
            return self.config.llm.groq.synthesizer_model
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
    def _call_ollama(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Call Ollama API."""
        url = f"{self.config.llm.ollama.host}/api/generate"
        
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                url,
                json={
                    "model": self._get_model_name(),
                    "prompt": user_prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self._get_temperature(),
                        "num_predict": self.config.llm.ollama.max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            return {
                "content": data["response"],
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            }
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=30))
    def _call_groq(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Call Groq API with rate limit handling."""
        import time as t
        
        # Add a small delay to avoid hitting rate limits (30 req/min = 2 sec between)
        t.sleep(2.0)
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.config.llm.groq.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._get_model_name(),
                    "messages": messages,
                    "temperature": self._get_temperature(),
                    "max_tokens": self.config.llm.groq.max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
            return {
                "content": data["choices"][0]["message"]["content"],
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
            }
    
    def _extract_citations(self, answer: str) -> list[Citation]:
        """Extract citations from the answer."""
        # Match patterns like [n1.2] or [n2.1]
        pattern = r'\[(n\d+)\.(\d+)\]'
        citations = []
        
        for match in re.finditer(pattern, answer):
            citations.append(Citation(
                node_id=match.group(1),
                chunk_index=int(match.group(2)),
                raw=match.group(0),
            ))
        
        return citations
    
    def synthesize(self, context: AssembledContext) -> SynthesisResult:
        """
        Generate an answer from assembled context.
        
        Args:
            context: Assembled context with evidence
            
        Returns:
            SynthesisResult with answer and metadata
        """
        start_time = time.time()
        
        system_prompt = self.prompt_builder.build_system_prompt()
        user_prompt = self.prompt_builder.build_user_prompt(context)
        messages = self.prompt_builder.build_messages(context)
        
        try:
            if self.provider == "ollama":
                result = self._call_ollama(system_prompt, user_prompt)
            elif self.provider == "groq":
                result = self._call_groq(messages)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            latency = (time.time() - start_time) * 1000
            answer = result["content"].strip()
            citations = self._extract_citations(answer)
            
            return SynthesisResult(
                answer=answer,
                citations=citations,
                raw_response=answer,
                latency_ms=latency,
                model=self._get_model_name(),
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
            )
        
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            latency = (time.time() - start_time) * 1000
            return SynthesisResult(
                answer=f"Error: {str(e)}",
                citations=[],
                raw_response="",
                latency_ms=latency,
                model=self._get_model_name(),
            )
    
    def synthesize_simple(
        self,
        question: str,
        context: str,
    ) -> SynthesisResult:
        """
        Simple synthesis without structured context (for baselines).
        
        Args:
            question: The question to answer
            context: Raw context string
            
        Returns:
            SynthesisResult
        """
        from src.synthesizer.prompts import SimpleRAGPromptBuilder
        
        start_time = time.time()
        builder = SimpleRAGPromptBuilder()
        
        messages = builder.build_messages(question, context)
        
        try:
            if self.provider == "ollama":
                result = self._call_ollama(
                    messages[0]["content"],
                    messages[1]["content"],
                )
            elif self.provider == "groq":
                result = self._call_groq(messages)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            latency = (time.time() - start_time) * 1000
            answer = result["content"].strip()
            
            return SynthesisResult(
                answer=answer,
                citations=[],
                raw_response=answer,
                latency_ms=latency,
                model=self._get_model_name(),
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
            )
        
        except Exception as e:
            logger.error(f"Simple synthesis failed: {e}")
            latency = (time.time() - start_time) * 1000
            return SynthesisResult(
                answer=f"Error: {str(e)}",
                citations=[],
                raw_response="",
                latency_ms=latency,
                model=self._get_model_name(),
            )
