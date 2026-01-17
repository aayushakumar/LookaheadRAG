from __future__ import annotations
"""
Dataset Loaders.

Provides loaders for HotpotQA and other multi-hop QA datasets.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetExample:
    """A single example from a dataset."""
    
    id: str
    question: str
    answer: str
    supporting_facts: list[dict] = field(default_factory=list)
    context: list[dict] = field(default_factory=list)  # Optional gold context
    level: str = ""  # Difficulty level (easy, medium, hard)
    type: str = ""   # Question type (comparison, bridge, etc.)
    
    def get_supporting_titles(self) -> list[str]:
        """Get titles of supporting documents."""
        return list({sf.get("title", sf[0]) for sf in self.supporting_facts})


class HotpotQADataset:
    """
    HotpotQA dataset loader.
    
    HotpotQA is a multi-hop question answering dataset with
    supporting fact annotations.
    """
    
    def __init__(
        self,
        split: str = "validation",
        subset_size: int | None = None,
    ):
        self.split = split
        self.subset_size = subset_size
        self._data = None
    
    def load(self) -> None:
        """Load the dataset from HuggingFace."""
        logger.info(f"Loading HotpotQA {self.split} split...")
        
        # Load from HuggingFace datasets
        dataset = load_dataset("hotpot_qa", "fullwiki", split=self.split)
        
        if self.subset_size:
            dataset = dataset.select(range(min(self.subset_size, len(dataset))))
        
        self._data = dataset
        logger.info(f"Loaded {len(self._data)} examples")
    
    def __len__(self) -> int:
        if self._data is None:
            self.load()
        return len(self._data)
    
    def __iter__(self) -> Iterator[DatasetExample]:
        if self._data is None:
            self.load()
        
        for item in self._data:
            yield self._parse_example(item)
    
    def __getitem__(self, idx: int) -> DatasetExample:
        if self._data is None:
            self.load()
        return self._parse_example(self._data[idx])
    
    def _parse_example(self, item: dict) -> DatasetExample:
        """Parse a dataset item into DatasetExample."""
        # Parse supporting facts
        supporting_facts = []
        if "supporting_facts" in item:
            sf = item["supporting_facts"]
            if isinstance(sf, dict):
                titles = sf.get("title", [])
                sent_ids = sf.get("sent_id", [])
                for title, sent_id in zip(titles, sent_ids):
                    supporting_facts.append({
                        "title": title,
                        "sent_id": sent_id,
                    })
        
        # Parse context
        context = []
        if "context" in item:
            ctx = item["context"]
            if isinstance(ctx, dict):
                titles = ctx.get("title", [])
                sentences = ctx.get("sentences", [])
                for title, sents in zip(titles, sentences):
                    context.append({
                        "title": title,
                        "sentences": sents,
                    })
        
        return DatasetExample(
            id=item.get("id", str(hash(item["question"]))),
            question=item["question"],
            answer=item["answer"],
            supporting_facts=supporting_facts,
            context=context,
            level=item.get("level", ""),
            type=item.get("type", ""),
        )
    
    def get_by_type(self, question_type: str) -> list[DatasetExample]:
        """Get examples of a specific type (e.g., 'comparison', 'bridge')."""
        return [ex for ex in self if ex.type == question_type]
    
    def get_by_level(self, level: str) -> list[DatasetExample]:
        """Get examples of a specific difficulty level."""
        return [ex for ex in self if ex.level == level]


class TwoWikiMultiHopDataset:
    """
    2WikiMultiHopQA dataset loader.
    
    Another multi-hop QA dataset with more diverse reasoning paths.
    """
    
    def __init__(
        self,
        split: str = "validation",
        subset_size: int | None = None,
    ):
        self.split = split
        self.subset_size = subset_size
        self._data = None
    
    def load(self) -> None:
        """Load the dataset."""
        logger.info(f"Loading 2WikiMultiHopQA {self.split} split...")
        
        try:
            dataset = load_dataset("multi_x_science_sum", split=self.split)  # Placeholder
            if self.subset_size:
                dataset = dataset.select(range(min(self.subset_size, len(dataset))))
            self._data = dataset
        except Exception as e:
            logger.warning(f"Could not load 2WikiMultiHopQA: {e}")
            self._data = []
    
    def __len__(self) -> int:
        if self._data is None:
            self.load()
        return len(self._data)
    
    def __iter__(self) -> Iterator[DatasetExample]:
        if self._data is None:
            self.load()
        for item in self._data:
            yield self._parse_example(item)
    
    def _parse_example(self, item: dict) -> DatasetExample:
        """Parse a dataset item."""
        return DatasetExample(
            id=item.get("id", ""),
            question=item.get("question", ""),
            answer=item.get("answer", ""),
        )


def get_dataset(
    name: str,
    split: str = "validation",
    subset_size: int | None = None,
) -> HotpotQADataset | TwoWikiMultiHopDataset:
    """Factory function to get a dataset by name."""
    datasets = {
        "hotpotqa": HotpotQADataset,
        "2wikimultihopqa": TwoWikiMultiHopDataset,
    }
    
    if name.lower() not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets.keys())}")
    
    return datasets[name.lower()](split=split, subset_size=subset_size)
