"""Configuration for Engram — local and cloud modes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class EngramConfig:
    """Runtime configuration.

    Local mode (default):
        backend=faiss, store_path=./engram_store
        Everything runs on your machine. No API keys. No network.

    Cloud mode:
        backend=qdrant, qdrant_url=..., qdrant_api_key=...
        Documents stored in Qdrant (managed or self-hosted).
        Embeddings still computed locally (no data sent to embedding APIs).
    """

    # Backend
    backend: str = "faiss"  # "faiss" or "qdrant"
    store_path: str = "./engram_store"

    # Cloud backend (Qdrant)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "engram"

    # Embedding model
    embed_model: str = "bge-large"

    # Retrieval
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    dense_top_k: int = 50

    # Ingestion
    include_assistant_turns: bool = True
    generate_preference_docs: bool = True

    @classmethod
    def from_env(cls) -> "EngramConfig":
        """Load configuration from environment variables."""
        return cls(
            backend=os.getenv("ENGRAM_BACKEND", "faiss"),
            store_path=os.getenv("ENGRAM_STORE_PATH", "./engram_store"),
            qdrant_url=os.getenv("ENGRAM_QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("ENGRAM_QDRANT_API_KEY"),
            qdrant_collection=os.getenv("ENGRAM_QDRANT_COLLECTION", "engram"),
            embed_model=os.getenv("ENGRAM_EMBED_MODEL", "bge-large"),
            use_reranker=os.getenv("ENGRAM_USE_RERANKER", "").lower() in ("1", "true"),
        )

    def get_backend(self):
        """Create and return the configured backend."""
        from .retrieval.embedder import Embedder

        embedder = Embedder(self.embed_model)
        dim = embedder.dimension or 1024

        if self.backend == "qdrant":
            from .backends.qdrant_backend import QdrantBackend

            return QdrantBackend(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                collection_name=self.qdrant_collection,
                dimension=dim,
            )
        else:
            from .backends.faiss_backend import FaissBackend

            return FaissBackend(path=self.store_path, dimension=dim)
