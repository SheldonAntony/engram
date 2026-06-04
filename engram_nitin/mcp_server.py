"""Engram MCP server — expose memory retrieval as MCP tools.

Runs over stdio. Register with Claude Desktop / Cursor / Windsurf via
`claude_desktop_config.json` (or equivalent):

    {
      "mcpServers": {
        "engram": {
          "command": "engram-mcp",
          "env": {
            "ENGRAM_STORE_PATH": "/path/to/engram_store",
            "ENGRAM_EMBED_MODEL": "bge-large"
          }
        }
      }
    }

Tools:
    search_memory(query, top_k=5, min_score=0.45) -> list[dict]
    add_memory(text, metadata=None)               -> str
    memory_stats()                                 -> dict
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:
    raise ImportError("MCP not installed. Run: pip install 'engram-search[mcp]'") from e


STORE_PATH = os.getenv("ENGRAM_STORE_PATH", "./engram_store")
EMBED_MODEL = os.getenv("ENGRAM_EMBED_MODEL", "bge-large")
MIN_SCORE_DEFAULT = float(os.getenv("ENGRAM_MIN_SCORE", "0.45"))

mcp = FastMCP("engram")

_backend = None
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from .retrieval.embedder import Embedder

        _embedder = Embedder(EMBED_MODEL)
        _embedder._load()
    return _embedder


def _get_backend():
    global _backend
    if _backend is None:
        from .backends.faiss_backend import FaissBackend

        emb = _get_embedder()
        _backend = FaissBackend(path=Path(STORE_PATH).resolve(), dimension=emb.dimension)
    return _backend


@mcp.tool()
def search_memory(
    query: str,
    top_k: int = 5,
    min_score: float = MIN_SCORE_DEFAULT,
) -> list[dict]:
    """Retrieve memories relevant to a query.

    Use this when the user references something from past conversations,
    or when context from prior sessions would help answer the current turn.
    """
    backend = _get_backend()
    if backend.count() == 0:
        return []

    embedder = _get_embedder()
    query_vec = embedder.encode_query(query)
    candidates = backend.query(
        embedding=query_vec.tolist(),
        top_k=top_k * 3,
        min_score=min_score,
    )
    if not candidates:
        return []

    from .retrieval.sparse import BM25

    bm25 = BM25()
    bm25_scores = bm25.score_query_against_docs(query, [c.text for c in candidates])
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    for i, c in enumerate(candidates):
        c.score = 0.6 * c.score + 0.4 * (bm25_scores[i] / max_bm25)

    candidates.sort(key=lambda d: d.score, reverse=True)
    top = candidates[:top_k]

    return [
        {
            "id": d.id,
            "text": d.text[:800],
            "score": round(d.score, 4),
            "metadata": d.metadata or {},
        }
        for d in top
    ]


@mcp.tool()
def add_memory(text: str, metadata: Optional[dict] = None) -> str:
    """Store a new memory fact or conversation turn.

    Call this when the user shares something worth remembering for future
    sessions — preferences, decisions, personal facts, project context.
    Returns the generated document ID.
    """
    from .backends.base import Document

    if not text or not text.strip():
        raise ValueError("text cannot be empty")

    embedder = _get_embedder()
    backend = _get_backend()

    doc_id = hashlib.sha1(f"{time.time_ns()}:{text}".encode()).hexdigest()[:16]
    embedding = embedder.encode_documents([text])[0].tolist()

    backend.add(
        [
            Document(
                id=doc_id,
                text=text,
                embedding=embedding,
                metadata=metadata or {"type": "note"},
            )
        ]
    )
    return doc_id


@mcp.tool()
def memory_stats() -> dict:
    """Return statistics about the memory store."""
    backend = _get_backend()
    return {
        "documents": backend.count(),
        "store_path": str(Path(STORE_PATH).resolve()),
        "embed_model": EMBED_MODEL,
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
