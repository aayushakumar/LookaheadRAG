"""
Distilled Planner Module.

Implements knowledge distillation from large LLM planner to smaller models.
Target: Flan-T5-small/base with LoRA for efficient fine-tuning.

Workflow:
1. Generate training data from teacher model (include binding info)
2. Fine-tune student with LoRA
3. Evaluate against teacher on planning quality metrics

Reference: Knowledge distillation for RAG planners.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.config import Config, get_config
from src.planner.schema import PlanGraph, PlanNode, OperatorType

logger = logging.getLogger(__name__)


@dataclass
class DistillationExample:
    """Single training example for distillation."""
    
    question: str
    teacher_output: str  # Raw JSON from teacher
    plan_graph: PlanGraph | None = None
    
    # Quality metrics (for filtering)
    num_nodes: int = 0
    has_bindings: bool = False
    teacher_confidence: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "teacher_output": self.teacher_output,
            "num_nodes": self.num_nodes,
            "has_bindings": self.has_bindings,
            "teacher_confidence": self.teacher_confidence,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DistillationExample":
        return cls(
            question=data["question"],
            teacher_output=data["teacher_output"],
            num_nodes=data.get("num_nodes", 0),
            has_bindings=data.get("has_bindings", False),
            teacher_confidence=data.get("teacher_confidence", 0.0),
        )


@dataclass
class DistillationDataset:
    """Dataset for planner distillation."""
    
    examples: list[DistillationExample] = field(default_factory=list)
    teacher_model: str = ""
    created_at: str = ""
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def save(self, path: Path) -> None:
        """Save dataset to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "teacher_model": self.teacher_model,
            "created_at": self.created_at,
            "num_examples": len(self.examples),
            "examples": [ex.to_dict() for ex in self.examples],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.examples)} examples to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "DistillationDataset":
        """Load dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        examples = [DistillationExample.from_dict(ex) for ex in data["examples"]]
        return cls(
            examples=examples,
            teacher_model=data.get("teacher_model", ""),
            created_at=data.get("created_at", ""),
        )
    
    def filter_quality(
        self,
        min_nodes: int = 1,
        min_confidence: float = 0.5,
        require_bindings: bool = False,
    ) -> "DistillationDataset":
        """Filter to high-quality examples."""
        filtered = [
            ex for ex in self.examples
            if ex.num_nodes >= min_nodes
            and ex.teacher_confidence >= min_confidence
            and (not require_bindings or ex.has_bindings)
        ]
        return DistillationDataset(
            examples=filtered,
            teacher_model=self.teacher_model,
            created_at=self.created_at,
        )
    
    def to_training_format(self) -> list[dict[str, str]]:
        """Convert to input/output pairs for training."""
        return [
            {
                "input": f"Generate a retrieval plan for: {ex.question}",
                "output": ex.teacher_output,
            }
            for ex in self.examples
        ]


class DistillationDataGenerator:
    """
    Generates training data from teacher model.
    
    Uses the full LLM planner to generate high-quality plans
    that include binding information.
    """
    
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        from src.planner import LLMPlanner
        self.teacher = LLMPlanner(self.config)
    
    async def generate_example(self, question: str) -> DistillationExample | None:
        """Generate single distillation example."""
        try:
            plan = await self.teacher.generate_plan(question)
            
            # Get raw output (serialize plan)
            teacher_output = json.dumps({
                "question": plan.question,
                "nodes": [node.to_dict() for node in plan.nodes],
            }, indent=2)
            
            # Extract quality metrics
            has_bindings = any(
                node.produces or node.bindings
                for node in plan.nodes
            )
            avg_confidence = plan.average_confidence()
            
            return DistillationExample(
                question=question,
                teacher_output=teacher_output,
                plan_graph=plan,
                num_nodes=len(plan.nodes),
                has_bindings=has_bindings,
                teacher_confidence=avg_confidence,
            )
        except Exception as e:
            logger.warning(f"Failed to generate example for: {question}: {e}")
            return None
    
    async def generate_dataset(
        self,
        questions: list[str],
        output_path: Path | None = None,
    ) -> DistillationDataset:
        """Generate dataset from list of questions."""
        import asyncio
        from datetime import datetime
        
        examples = []
        for i, question in enumerate(questions):
            if i % 10 == 0:
                logger.info(f"Generating example {i+1}/{len(questions)}")
            
            example = await self.generate_example(question)
            if example:
                examples.append(example)
            
            # Rate limiting
            await asyncio.sleep(0.1)
        
        dataset = DistillationDataset(
            examples=examples,
            teacher_model=self.teacher._get_model_name(),
            created_at=datetime.now().isoformat(),
        )
        
        if output_path:
            dataset.save(output_path)
        
        return dataset


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    
    r: int = 8                    # LoRA rank
    alpha: int = 16               # LoRA alpha
    dropout: float = 0.1          # LoRA dropout
    target_modules: list[str] = field(
        default_factory=lambda: ["q", "v"]  # Query and value projections
    )
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Model
    base_model: str = "google/flan-t5-small"
    max_input_length: int = 512
    max_output_length: int = 1024


class DistilledPlanner:
    """
    Planner using distilled small model.
    
    Options:
    1. Flan-T5-small (80M params) - fastest
    2. Flan-T5-base (250M params) - better quality
    
    Uses LoRA for efficient fine-tuning.
    """
    
    def __init__(
        self,
        model_path: Path | None = None,
        lora_config: LoRAConfig | None = None,
        use_cpu: bool = False,
    ):
        self.model_path = model_path
        self.lora_config = lora_config or LoRAConfig()
        self.use_cpu = use_cpu
        
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
    
    def load(self) -> None:
        """Load the distilled model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )
        
        model_name = str(self.model_path) if self.model_path else self.lora_config.base_model
        
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        if not self.use_cpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._model = self._model.to("mps")
            except Exception:
                pass
        
        self._is_loaded = True
        logger.info(f"Loaded model from {model_name}")
    
    def generate_plan(self, question: str) -> PlanGraph:
        """Generate plan using distilled model."""
        if not self._is_loaded:
            self.load()
        
        start_time = time.time()
        
        # Format input
        input_text = f"Generate a retrieval plan for: {question}"
        
        # Tokenize
        inputs = self._tokenizer(
            input_text,
            max_length=self.lora_config.max_input_length,
            truncation=True,
            return_tensors="pt",
        )
        
        if not self.use_cpu and self._model.device.type != "cpu":
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Generate
        outputs = self._model.generate(
            **inputs,
            max_length=self.lora_config.max_output_length,
            num_beams=4,
            early_stopping=True,
        )
        
        # Decode
        output_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        
        # Parse output
        try:
            data = json.loads(output_text)
            plan = self._parse_plan(data, question, generation_time)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse distilled output, using fallback")
            plan = self._fallback_plan(question, generation_time)
        
        return plan
    
    def _parse_plan(
        self,
        data: dict,
        question: str,
        generation_time: float,
    ) -> PlanGraph:
        """Parse JSON output into PlanGraph."""
        nodes = []
        for node_data in data.get("nodes", []):
            node = PlanNode(
                id=node_data.get("id", f"n{len(nodes)}"),
                query=node_data.get("query", question),
                op=OperatorType(node_data.get("op", "lookup")),
                depends_on=node_data.get("depends_on", []),
                confidence=node_data.get("confidence", 0.7),
                budget_cost=node_data.get("budget_cost", 1),
            )
            nodes.append(node)
        
        return PlanGraph(
            question=question,
            nodes=nodes,
            planner_model="distilled-flan-t5",
            generation_time_ms=generation_time * 1000,
        )
    
    def _fallback_plan(self, question: str, generation_time: float) -> PlanGraph:
        """Create simple fallback plan."""
        return PlanGraph(
            question=question,
            nodes=[
                PlanNode(
                    id="n0",
                    query=question,
                    op=OperatorType.LOOKUP,
                    confidence=0.5,
                )
            ],
            planner_model="distilled-flan-t5-fallback",
            generation_time_ms=generation_time * 1000,
        )


class LoRATrainer:
    """
    Trainer for LoRA fine-tuning of distilled planner.
    
    Uses PEFT (Parameter-Efficient Fine-Tuning) library.
    """
    
    def __init__(self, config: LoRAConfig | None = None):
        self.config = config or LoRAConfig()
    
    def train(
        self,
        dataset: DistillationDataset,
        output_dir: Path,
        eval_split: float = 0.1,
    ) -> dict[str, float]:
        """
        Fine-tune model with LoRA.
        
        Returns training metrics.
        """
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSeq2SeqLM,
                Seq2SeqTrainer,
                Seq2SeqTrainingArguments,
                DataCollatorForSeq2Seq,
            )
            from peft import LoraConfig, get_peft_model, TaskType
            from datasets import Dataset
        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {e}\n"
                "Run: pip install transformers peft datasets"
            )
        
        # Load base model
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.base_model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.alpha,
            lora_dropout=self.config.dropout,
            target_modules=self.config.target_modules,
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        
        logger.info(f"Trainable params: {model.print_trainable_parameters()}")
        
        # Prepare dataset
        training_data = dataset.to_training_format()
        
        def preprocess(examples):
            inputs = tokenizer(
                examples["input"],
                max_length=self.config.max_input_length,
                truncation=True,
                padding="max_length",
            )
            outputs = tokenizer(
                examples["output"],
                max_length=self.config.max_output_length,
                truncation=True,
                padding="max_length",
            )
            inputs["labels"] = outputs["input_ids"]
            return inputs
        
        hf_dataset = Dataset.from_list(training_data)
        hf_dataset = hf_dataset.map(preprocess, batched=True)
        
        # Split for evaluation
        split = hf_dataset.train_test_split(test_size=eval_split)
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            predict_with_generate=True,
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        )
        
        # Train
        result = trainer.train()
        
        # Save
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        return {
            "train_loss": result.training_loss,
            "train_samples": len(split["train"]),
            "eval_samples": len(split["test"]),
        }


# === Evaluation Metrics ===

@dataclass
class PlanQualityMetrics:
    """Metrics comparing distilled vs teacher plans."""
    
    num_examples: int = 0
    
    # Structure similarity
    avg_node_count_diff: float = 0.0    # |teacher_nodes - student_nodes|
    exact_structure_match: float = 0.0   # Fraction with same DAG structure
    
    # Query similarity
    avg_query_overlap: float = 0.0       # Jaccard on query terms
    
    # Binding preservation
    binding_preservation_rate: float = 0.0  # Fraction preserving bindings
    
    # Latency comparison
    teacher_avg_latency_ms: float = 0.0
    student_avg_latency_ms: float = 0.0
    speedup: float = 1.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "num_examples": self.num_examples,
            "avg_node_count_diff": self.avg_node_count_diff,
            "exact_structure_match": self.exact_structure_match,
            "avg_query_overlap": self.avg_query_overlap,
            "binding_preservation_rate": self.binding_preservation_rate,
            "speedup": self.speedup,
        }
