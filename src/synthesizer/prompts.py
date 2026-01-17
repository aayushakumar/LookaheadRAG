from __future__ import annotations
"""
Synthesis Prompt Templates.

Provides structured prompts for the final synthesis step.
"""

from src.synthesizer.context import AssembledContext


SYNTHESIS_SYSTEM_PROMPT = """You are a precise question-answering assistant. Your task is to answer questions based ONLY on the provided evidence.

## Rules
1. Answer ONLY using information from the provided evidence
2. Cite your sources using [node_id.chunk_number] format
3. If the evidence is insufficient, say "I cannot answer this based on the provided evidence"
4. Be concise and direct
5. For multi-part questions, address each part

## Output Format
Provide your answer followed by citations. For example:
"The author was Arthur Conan Doyle [n1.1], who was born in 1859 [n2.1]."
"""


SYNTHESIS_USER_TEMPLATE = """## Question
{question}

## Evidence
{evidence}

Please answer the question based on the evidence above. Cite your sources."""


SYNTHESIS_USER_TEMPLATE_WITH_PLAN = """## Question
{question}

## Plan Summary
{plan_summary}

## Evidence
{evidence}

Please answer the question based on the evidence above. Cite your sources."""


class SynthesisPromptBuilder:
    """Builds prompts for the synthesis step."""
    
    def __init__(self, include_plan: bool = True):
        self.include_plan = include_plan
    
    def build_system_prompt(self) -> str:
        """Get the system prompt for synthesis."""
        return SYNTHESIS_SYSTEM_PROMPT
    
    def build_user_prompt(self, context: AssembledContext) -> str:
        """Build the user prompt from assembled context."""
        evidence = context.to_structured_string()
        
        if self.include_plan:
            return SYNTHESIS_USER_TEMPLATE_WITH_PLAN.format(
                question=context.question,
                plan_summary=context.plan_summary,
                evidence=evidence,
            )
        else:
            return SYNTHESIS_USER_TEMPLATE.format(
                question=context.question,
                evidence=evidence,
            )
    
    def build_messages(
        self,
        context: AssembledContext,
    ) -> list[dict[str, str]]:
        """Build chat messages for the synthesis call."""
        return [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(context)},
        ]
    
    def build_single_prompt(self, context: AssembledContext) -> str:
        """Build a single prompt (for models without system prompt)."""
        return (
            f"{self.build_system_prompt()}\n\n"
            f"{self.build_user_prompt(context)}"
        )


# Simple RAG prompt (for baseline)
SIMPLE_RAG_SYSTEM_PROMPT = """You are a helpful assistant. Answer the question based on the provided context.
If you cannot find the answer in the context, say so."""

SIMPLE_RAG_USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""


class SimpleRAGPromptBuilder:
    """Simple prompt builder for RAG baseline."""
    
    def build_prompt(self, question: str, context: str) -> str:
        """Build a simple RAG prompt."""
        return SIMPLE_RAG_USER_TEMPLATE.format(
            question=question,
            context=context,
        )
    
    def build_messages(
        self,
        question: str,
        context: str,
    ) -> list[dict[str, str]]:
        """Build chat messages for simple RAG."""
        return [
            {"role": "system", "content": SIMPLE_RAG_SYSTEM_PROMPT},
            {"role": "user", "content": self.build_prompt(question, context)},
        ]
