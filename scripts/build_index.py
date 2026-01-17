from __future__ import annotations
#!/usr/bin/env python3
"""
Build Vector Index.

This script builds a vector index from Wikipedia articles for HotpotQA.
"""

import argparse
import json
import logging
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from tqdm import tqdm

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_index(
    data_path: Path,
    output_dir: Path,
    collection_name: str = "hotpotqa_wiki",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> None:
    """
    Build vector index from HotpotQA context documents.
    
    Args:
        data_path: Path to HotpotQA jsonl file
        output_dir: Directory to save the index
        collection_name: Name for the collection
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    """
    from src.config import Config
    from src.retriever.vector_store import VectorStore, Document
    
    console.print(f"[bold blue]Building vector index from {data_path}...[/]")
    
    # Create config with custom persist dir
    config = Config()
    config.vector_store.persist_dir = str(output_dir)
    config.vector_store.collection_name = collection_name
    
    # Initialize vector store
    vector_store = VectorStore(config, collection_name=collection_name)
    
    # Collect all documents from context
    console.print("[yellow]Extracting documents from context...[/]")
    
    documents: dict[str, Document] = {}  # Use dict to deduplicate by title
    
    with open(data_path) as f:
        for line in tqdm(f, desc="Processing examples"):
            item = json.loads(line)
            
            # Extract context documents
            context = item.get("context", {})
            titles = context.get("title", [])
            sentences_list = context.get("sentences", [])
            
            for title, sentences in zip(titles, sentences_list):
                if title not in documents:
                    # Combine sentences into document
                    content = " ".join(sentences)
                    
                    # Create document with title-based ID to ensure uniqueness
                    import hashlib
                    doc_id = hashlib.md5(title.encode()).hexdigest()[:12]
                    
                    from src.retriever.vector_store import Document
                    doc = Document(
                        id=doc_id,
                        content=content,
                        metadata={
                            "title": title,
                            "source": "hotpotqa_wikipedia",
                        },
                    )
                    documents[title] = doc
    
    console.print(f"[green]Found {len(documents)} unique documents[/]")
    
    # Add documents to vector store
    console.print("[yellow]Building vector index (this may take a while)...[/]")
    
    doc_list = list(documents.values())
    batch_size = 100
    
    with Progress() as progress:
        task = progress.add_task("Indexing...", total=len(doc_list))
        
        for i in range(0, len(doc_list), batch_size):
            batch = doc_list[i:i + batch_size]
            vector_store.add_documents(batch)
            progress.update(task, advance=len(batch))
    
    console.print(f"[bold green]✓ Indexed {vector_store.count()} documents[/]")
    console.print(f"[bold green]✓ Index saved to {output_dir}/{collection_name}[/]")


def main():
    parser = argparse.ArgumentParser(description="Build vector index")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("./data/hotpotqa_validation.jsonl"),
        help="Path to HotpotQA data file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/chroma_db"),
        help="Output directory for index",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="hotpotqa_wiki",
        help="Name for the collection",
    )
    
    args = parser.parse_args()
    
    if not args.data_path.exists():
        console.print(f"[bold red]Data file not found: {args.data_path}[/]")
        console.print("[yellow]Run 'python scripts/download_data.py' first[/]")
        return
    
    build_index(
        data_path=args.data_path,
        output_dir=args.output_dir,
        collection_name=args.collection_name,
    )


if __name__ == "__main__":
    main()
