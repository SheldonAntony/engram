"""Engram API server — cloud deployment mode.

Provides a REST API for memory storage and retrieval. Companies can deploy
this as a service with Qdrant backend for shared team memory.

Endpoints:
    POST /ingest     — add conversation sessions
    POST /search     — search memories
    GET  /health     — health check
    GET  /stats      — store statistics

Runs with any ASGI server: uvicorn engram.server:app
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger("engram.server")

# Lazy imports for optional dependencies
_app = None


def _create_app():
    """Create the FastAPI app with lazy imports."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    from .backends.base import Document
    from .retrieval.embedder import Embedder

    app = FastAPI(
        title="Engram",
        description="High-recall conversational memory retrieval API",
        version="0.1.0",
    )

    # --- Configuration from env ---
    BACKEND_TYPE = os.getenv("ENGRAM_BACKEND", "faiss")
    STORE_PATH = os.getenv("ENGRAM_STORE_PATH", "./engram_store")
    EMBED_MODEL = os.getenv("ENGRAM_EMBED_MODEL", "bge-large")
    QDRANT_URL = os.getenv("ENGRAM_QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("ENGRAM_QDRANT_API_KEY")

    # --- Initialize backend ---
    if BACKEND_TYPE == "qdrant":
        from .backends.qdrant_backend import QdrantBackend

        backend = QdrantBackend(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        logger.info("Using Qdrant backend at %s", QDRANT_URL)
    else:
        from .backends.faiss_backend import FaissBackend

        backend = FaissBackend(path=STORE_PATH)
        logger.info("Using FAISS backend at %s", STORE_PATH)

    embedder = Embedder(EMBED_MODEL)

    # --- Models ---
    class IngestRequest(BaseModel):
        sessions: List[dict] = []
        include_assistant: bool = True

    class SearchRequest(BaseModel):
        query: str
        top_k: int = 5
        min_score: float = 0.45
        question_date: Optional[str] = None

    class SearchResult(BaseModel):
        id: str
        text: str
        score: float
        metadata: Optional[dict] = None

    # --- Routes ---
    @app.get("/health")
    def health():
        return {"status": "ok", "backend": BACKEND_TYPE}

    @app.get("/stats")
    def stats():
        return {
            "documents": backend.count(),
            "backend": BACKEND_TYPE,
            "embed_model": EMBED_MODEL,
        }

    @app.post("/ingest")
    def ingest(req: IngestRequest):
        from .ingestion.parser import session_to_documents

        all_docs = []
        for i, session_data in enumerate(req.sessions):
            turns = session_data.get("turns", [])
            session_id = session_data.get("id", f"session_{i}")
            timestamp = session_data.get("timestamp", "")
            parsed = session_to_documents(
                session=turns,
                session_id=session_id,
                timestamp=timestamp,
                include_assistant=req.include_assistant,
            )
            all_docs.extend(parsed)

        if not all_docs:
            return {"ingested": 0}

        texts = [d["text"] for d in all_docs]
        embeddings = embedder.encode_documents(texts)

        documents = []
        for i, doc_info in enumerate(all_docs):
            documents.append(
                Document(
                    id=doc_info["id"],
                    text=doc_info["text"],
                    embedding=embeddings[i].tolist(),
                    metadata=doc_info["metadata"],
                )
            )

        backend.add(documents)
        return {"ingested": len(documents), "total": backend.count()}

    @app.post("/search")
    def search(req: SearchRequest):
        count = backend.count()
        if count == 0:
            raise HTTPException(status_code=404, detail="No documents in store")

        # For the API, we do direct vector search + BM25 reranking
        query_vec = embedder.encode_query(req.query)
        candidates = backend.query(
            embedding=query_vec.tolist(),
            top_k=req.top_k * 3,
            min_score=req.min_score,
        )

        if not candidates:
            return {"query": req.query, "results": []}

        # BM25 rerank
        from .retrieval.sparse import BM25

        bm25 = BM25()
        doc_texts = [c.text for c in candidates]
        bm25_scores = bm25.score_query_against_docs(req.query, doc_texts)

        # Simple fusion
        for i, candidate in enumerate(candidates):
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
            candidate.score = 0.6 * candidate.score + 0.4 * (bm25_scores[i] / max_bm25)

        candidates.sort(key=lambda d: d.score, reverse=True)
        top = candidates[: req.top_k]

        return {
            "query": req.query,
            "results": [
                SearchResult(
                    id=d.id,
                    text=d.text[:500],
                    score=round(d.score, 4),
                    metadata=d.metadata,
                ).model_dump()
                for d in top
            ],
        }

    return app


def get_app():
    """Get or create the FastAPI app singleton."""
    global _app
    if _app is None:
        _app = _create_app()
    return _app


# For uvicorn: `uvicorn engram.server:app`
app = None
try:
    app = get_app()
except ImportError:
    pass  # FastAPI not installed — CLI-only mode
