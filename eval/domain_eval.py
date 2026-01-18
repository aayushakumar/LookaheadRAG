"""
Domain Evaluation Harness.

Supports evaluation on multiple domains beyond HotpotQA:
- QASPER: Scientific papers with multi-hop questions
- MultiDoc2Dial: Conversational multi-document QA
- PubHealth: Health claim verification (faithfulness focus)
- Custom: Any dataset with (question, answer, context) format

Reference: Generalizes LookaheadRAG to domain-specific corpora.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.config import Config, get_config

logger = logging.getLogger(__name__)


@dataclass
class DomainExample:
    """Single example from a domain dataset."""
    
    id: str
    question: str
    answer: str
    
    # Optional fields
    context: list[str] = field(default_factory=list)  # Gold passages
    supporting_facts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Domain info
    domain: str = "unknown"
    difficulty: str = "unknown"  # easy, medium, hard


@dataclass
class DomainEvaluationResult:
    """Evaluation result for a domain."""
    
    domain: str
    num_examples: int
    
    # Core metrics
    exact_match: float
    f1: float
    
    # Latency
    avg_latency_ms: float
    p95_latency_ms: float
    
    # Domain-specific
    supporting_fact_f1: float = 0.0
    faithfulness_score: float = 0.0  # For claim verification
    
    # Breakdown by difficulty
    metrics_by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "num_examples": self.num_examples,
            "exact_match": self.exact_match,
            "f1": self.f1,
            "avg_latency_ms": self.avg_latency_ms,
            "supporting_fact_f1": self.supporting_fact_f1,
            "faithfulness_score": self.faithfulness_score,
        }


class DomainDataset(ABC):
    """Abstract base class for domain datasets."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        pass
    
    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain identifier."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Number of examples."""
        pass
    
    @abstractmethod
    def __iter__(self) -> Iterator[DomainExample]:
        """Iterate over examples."""
        pass
    
    @abstractmethod
    def load(self, split: str = "test") -> None:
        """Load dataset split."""
        pass


class QASPERDataset(DomainDataset):
    """
    QASPER: Scientific paper QA dataset.
    
    Multi-hop questions over academic papers requiring
    information synthesis across sections.
    """
    
    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or Path("data/qasper")
        self.examples: list[DomainExample] = []
    
    @property
    def name(self) -> str:
        return "QASPER"
    
    @property
    def domain(self) -> str:
        return "scientific"
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __iter__(self) -> Iterator[DomainExample]:
        return iter(self.examples)
    
    def load(self, split: str = "test") -> None:
        """Load QASPER dataset."""
        path = self.data_dir / f"{split}.json"
        if not path.exists():
            logger.warning(f"QASPER data not found at {path}")
            return
        
        with open(path) as f:
            data = json.load(f)
        
        for paper_id, paper in data.items():
            title = paper.get("title", "")
            full_text = paper.get("full_text", {})
            
            # Flatten paper sections into context
            context = []
            for section_name, paragraphs in full_text.items():
                for para in paragraphs:
                    context.append(f"{section_name}: {para}")
            
            # Extract QA pairs
            for qa in paper.get("qas", []):
                question = qa.get("question", "")
                
                # Get answer (may have multiple annotators)
                answers = qa.get("answers", [])
                if answers:
                    answer = answers[0].get("answer", {})
                    if isinstance(answer, dict):
                        answer_text = answer.get("free_form_answer", "")
                    else:
                        answer_text = str(answer)
                else:
                    continue
                
                self.examples.append(DomainExample(
                    id=f"{paper_id}_{qa.get('question_id', len(self.examples))}",
                    question=question,
                    answer=answer_text,
                    context=context[:20],  # Limit context size
                    domain=self.domain,
                    metadata={"paper_title": title},
                ))
        
        logger.info(f"Loaded {len(self.examples)} QASPER examples")


class PubHealthDataset(DomainDataset):
    """
    PubHealth: Health claim verification dataset.
    
    Focus on faithfulness - claims must be supported by evidence.
    """
    
    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or Path("data/pubhealth")
        self.examples: list[DomainExample] = []
    
    @property
    def name(self) -> str:
        return "PubHealth"
    
    @property
    def domain(self) -> str:
        return "health"
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __iter__(self) -> Iterator[DomainExample]:
        return iter(self.examples)
    
    def load(self, split: str = "test") -> None:
        """Load PubHealth dataset."""
        path = self.data_dir / f"{split}.tsv"
        if not path.exists():
            logger.warning(f"PubHealth data not found at {path}")
            return
        
        import csv
        
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                claim = row.get("claim", "")
                label = row.get("label", "")  # true, false, mixture, unproven
                explanation = row.get("explanation", "")
                
                # Frame as verification question
                question = f"Is the following claim true? {claim}"
                
                self.examples.append(DomainExample(
                    id=row.get("claim_id", str(len(self.examples))),
                    question=question,
                    answer=label,
                    context=[explanation] if explanation else [],
                    domain=self.domain,
                    metadata={"original_claim": claim},
                ))
        
        logger.info(f"Loaded {len(self.examples)} PubHealth examples")


class CustomDataset(DomainDataset):
    """
    Custom dataset from JSON file.
    
    Expected format:
    [
        {"id": "1", "question": "...", "answer": "..."},
        ...
    ]
    """
    
    def __init__(self, path: Path, domain_name: str = "custom"):
        self.path = path
        self._domain = domain_name
        self.examples: list[DomainExample] = []
    
    @property
    def name(self) -> str:
        return self.path.stem
    
    @property
    def domain(self) -> str:
        return self._domain
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __iter__(self) -> Iterator[DomainExample]:
        return iter(self.examples)
    
    def load(self, split: str = "test") -> None:
        """Load custom dataset."""
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        
        with open(self.path) as f:
            data = json.load(f)
        
        for item in data:
            self.examples.append(DomainExample(
                id=str(item.get("id", len(self.examples))),
                question=item["question"],
                answer=item["answer"],
                context=item.get("context", []),
                supporting_facts=item.get("supporting_facts", []),
                domain=self.domain,
            ))
        
        logger.info(f"Loaded {len(self.examples)} examples from {self.path}")


class DomainEvaluator:
    """
    Evaluates LookaheadRAG across multiple domains.
    
    Supports:
    - Domain-specific metrics (faithfulness for health, etc.)
    - Cross-domain comparison
    - Difficulty stratification
    """
    
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
    
    async def evaluate_single(
        self,
        method,  # LookaheadRAG or baseline
        example: DomainExample,
    ) -> dict[str, Any]:
        """Evaluate on single example."""
        import time
        from eval.metrics import exact_match, f1_score
        
        start = time.time()
        result = await method.run(example.question)
        latency = (time.time() - start) * 1000
        
        prediction = result.answer if hasattr(result, "answer") else str(result)
        
        return {
            "id": example.id,
            "prediction": prediction,
            "ground_truth": example.answer,
            "exact_match": exact_match(prediction, example.answer),
            "f1": f1_score(prediction, example.answer),
            "latency_ms": latency,
            "difficulty": example.difficulty,
        }
    
    async def evaluate_domain(
        self,
        method,
        dataset: DomainDataset,
        max_examples: int | None = None,
    ) -> DomainEvaluationResult:
        """Evaluate on entire domain dataset."""
        results = []
        
        for i, example in enumerate(dataset):
            if max_examples and i >= max_examples:
                break
            
            if i % 10 == 0:
                logger.info(f"Evaluating {dataset.name}: {i}/{len(dataset)}")
            
            result = await self.evaluate_single(method, example)
            results.append(result)
        
        if not results:
            return DomainEvaluationResult(
                domain=dataset.domain,
                num_examples=0,
                exact_match=0.0,
                f1=0.0,
                avg_latency_ms=0.0,
                p95_latency_ms=0.0,
            )
        
        # Aggregate
        avg_em = sum(r["exact_match"] for r in results) / len(results)
        avg_f1 = sum(r["f1"] for r in results) / len(results)
        
        latencies = sorted(r["latency_ms"] for r in results)
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = latencies[int(len(latencies) * 0.95)]
        
        return DomainEvaluationResult(
            domain=dataset.domain,
            num_examples=len(results),
            exact_match=avg_em,
            f1=avg_f1,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
        )
    
    def compare_domains(
        self,
        results: dict[str, DomainEvaluationResult],
    ) -> str:
        """Generate comparison table across domains."""
        lines = [
            "=== Cross-Domain Evaluation ===",
            f"{'Domain':<15} {'EM':<8} {'F1':<8} {'Latency':<10}",
            "-" * 41,
        ]
        
        for domain, result in sorted(results.items()):
            lines.append(
                f"{domain:<15} {result.exact_match:<8.3f} "
                f"{result.f1:<8.3f} {result.avg_latency_ms:<10.0f}ms"
            )
        
        return "\n".join(lines)
