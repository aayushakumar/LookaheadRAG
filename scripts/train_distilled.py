#!/usr/bin/env python
"""
Distillation Training Script.

Generates training data from teacher LLM and trains distilled planner.

Usage:
    # Generate data
    python scripts/train_distilled.py generate --questions data/questions.txt --output data/distillation.json
    
    # Train
    python scripts/train_distilled.py train --dataset data/distillation.json --output models/distilled_planner
    
    # Evaluate
    python scripts/train_distilled.py evaluate --model models/distilled_planner --test data/test_questions.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_data(args):
    """Generate distillation data from teacher model."""
    from src.planner.distilled import DistillationDataGenerator, DistillationDataset
    
    # Load questions
    if args.questions.endswith(".json"):
        with open(args.questions) as f:
            data = json.load(f)
        questions = [ex["question"] for ex in data]
    else:
        with open(args.questions) as f:
            questions = [line.strip() for line in f if line.strip()]
    
    if args.limit:
        questions = questions[:args.limit]
    
    logger.info(f"Generating data for {len(questions)} questions")
    
    generator = DistillationDataGenerator()
    dataset = asyncio.run(generator.generate_dataset(
        questions,
        output_path=Path(args.output) if args.output else None,
    ))
    
    logger.info(f"Generated {len(dataset)} examples")
    logger.info(f"  With bindings: {sum(1 for e in dataset.examples if e.has_bindings)}")
    logger.info(f"  Avg nodes: {sum(e.num_nodes for e in dataset.examples) / max(len(dataset), 1):.1f}")


def train_model(args):
    """Train distilled planner with LoRA."""
    from src.planner.distilled import (
        DistillationDataset,
        LoRAConfig,
        LoRATrainer,
    )
    
    # Load dataset
    dataset = DistillationDataset.load(Path(args.dataset))
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Filter quality
    dataset = dataset.filter_quality(
        min_nodes=args.min_nodes,
        min_confidence=args.min_confidence,
    )
    logger.info(f"After filtering: {len(dataset)} examples")
    
    if len(dataset) < 10:
        raise ValueError("Need at least 10 examples for training")
    
    # Configure LoRA
    config = LoRAConfig(
        base_model=args.base_model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    # Train
    trainer = LoRATrainer(config)
    metrics = trainer.train(
        dataset,
        output_dir=Path(args.output),
        eval_split=args.eval_split,
    )
    
    logger.info(f"Training complete:")
    logger.info(f"  Train loss: {metrics['train_loss']:.4f}")
    logger.info(f"  Model saved to: {args.output}")


def evaluate_model(args):
    """Evaluate distilled planner vs teacher."""
    from src.planner.distilled import DistilledPlanner
    from src.planner import LLMPlanner
    import time
    
    # Load models
    distilled = DistilledPlanner(model_path=Path(args.model))
    distilled.load()
    
    teacher = LLMPlanner()
    
    # Load test questions
    with open(args.test) as f:
        questions = [line.strip() for line in f if line.strip()]
    
    if args.limit:
        questions = questions[:args.limit]
    
    logger.info(f"Evaluating on {len(questions)} questions")
    
    results = []
    for question in questions:
        # Teacher
        t_start = time.time()
        t_plan = asyncio.run(teacher.generate_plan(question))
        t_time = (time.time() - t_start) * 1000
        
        # Student
        s_start = time.time()
        s_plan = distilled.generate_plan(question)
        s_time = (time.time() - s_start) * 1000
        
        results.append({
            "question": question,
            "teacher_nodes": len(t_plan.nodes),
            "student_nodes": len(s_plan.nodes),
            "teacher_time_ms": t_time,
            "student_time_ms": s_time,
        })
    
    # Aggregate
    avg_t_time = sum(r["teacher_time_ms"] for r in results) / len(results)
    avg_s_time = sum(r["student_time_ms"] for r in results) / len(results)
    speedup = avg_t_time / max(avg_s_time, 1)
    
    node_diff = sum(abs(r["teacher_nodes"] - r["student_nodes"]) for r in results) / len(results)
    
    print("\n=== Evaluation Results ===")
    print(f"Examples: {len(results)}")
    print(f"Avg teacher time: {avg_t_time:.1f}ms")
    print(f"Avg student time: {avg_s_time:.1f}ms")
    print(f"Speedup: {speedup:.1f}x")
    print(f"Avg node count diff: {node_diff:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Distillation training script")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate training data")
    gen_parser.add_argument("--questions", required=True, help="Path to questions file")
    gen_parser.add_argument("--output", required=True, help="Output path for dataset")
    gen_parser.add_argument("--limit", type=int, help="Limit number of questions")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train distilled model")
    train_parser.add_argument("--dataset", required=True, help="Path to distillation dataset")
    train_parser.add_argument("--output", required=True, help="Output directory for model")
    train_parser.add_argument("--base-model", default="google/flan-t5-small")
    train_parser.add_argument("--lora-r", type=int, default=8)
    train_parser.add_argument("--lora-alpha", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--min-nodes", type=int, default=1)
    train_parser.add_argument("--min-confidence", type=float, default=0.5)
    train_parser.add_argument("--eval-split", type=float, default=0.1)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate distilled model")
    eval_parser.add_argument("--model", required=True, help="Path to distilled model")
    eval_parser.add_argument("--test", required=True, help="Path to test questions")
    eval_parser.add_argument("--limit", type=int, help="Limit number of questions")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        generate_data(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "evaluate":
        evaluate_model(args)


if __name__ == "__main__":
    main()
