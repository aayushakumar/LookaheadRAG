"""
LookaheadRAG Web API Server.

FastAPI backend for the web UI.
"""

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from src.config import get_config
from src.engine.lookahead import LookaheadRAG
from src.baselines.standard_rag import StandardRAG
from src.retriever import VectorStore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
engine = None
standard_rag = None
vector_store = None


class QueryRequest(BaseModel):
    question: str
    method: str = "lookahead"  # lookahead or standard


class QueryResponse(BaseModel):
    answer: str
    method: str
    latency_ms: float
    plan_nodes: list = []
    retrieved_chunks: int = 0
    llm_calls: int = 0
    citations: list = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup."""
    global engine, standard_rag, vector_store
    
    logger.info("Initializing LookaheadRAG engine...")
    try:
        config = get_config()
        vector_store = VectorStore(config)
        engine = LookaheadRAG(config, vector_store)
        standard_rag = StandardRAG(config, vector_store)
        logger.info("Engine initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
    
    yield
    
    logger.info("Shutting down...")


app = FastAPI(
    title="LookaheadRAG API",
    description="Near-Agentic Accuracy with RAG-Like Latency",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Serve the main UI."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>LookaheadRAG API</h1><p>UI not found. Run from the web/ directory.</p>")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "engine_ready": engine is not None,
        "vector_store_ready": vector_store is not None,
    }


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Run a query through the RAG pipeline."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        if request.method == "lookahead":
            result = await engine.run(request.question)
            
            # Extract plan nodes for visualization
            plan_nodes = []
            for node in result.plan.nodes:
                plan_nodes.append({
                    "id": node.id,
                    "query": node.query,
                    "op": node.op.value,
                    "confidence": node.confidence,
                    "depends_on": node.depends_on,
                })
            
            return QueryResponse(
                answer=result.answer,
                method="lookahead",
                latency_ms=result.latency.total_ms,
                plan_nodes=plan_nodes,
                retrieved_chunks=result.context.total_chunks,
                llm_calls=result.num_llm_calls,
                citations=[c.raw for c in result.synthesis_result.citations],
            )
        
        elif request.method == "standard":
            result = await standard_rag.run(request.question)
            
            return QueryResponse(
                answer=result.answer,
                method="standard",
                latency_ms=result.latency_ms,
                plan_nodes=[],
                retrieved_chunks=result.num_chunks,
                llm_calls=1,
                citations=[],
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/examples")
async def examples():
    """Get example questions."""
    return {
        "examples": [
            "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
            "Who was known by his stage name Aladin and worked as a consultant?",
            "The director of the romantic comedy 'Big Stone Gap' is based in what New York city?",
            "Were Scott Derrickson and Ed Wood of the same nationality?",
            "Who is older, Annie Morton or Terry Richardson?",
        ]
    }


# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
