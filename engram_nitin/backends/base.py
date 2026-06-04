"""Abstract backend interface for vector storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Document:
    """A stored document with its metadata."""

    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Optional[dict] = None
    score: float = 0.0


class VectorBackend(ABC):
    """Interface for vector storage backends.

    Implementations must support:
    - Adding documents with embeddings
    - Querying by vector similarity (top-k)
    - Metadata filtering
    - Deletion
    """

    @abstractmethod
    def add(self, docs: list[Document]) -> None:
        """Add documents with precomputed embeddings."""

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        metadata_filter: Optional[dict] = None,
        min_score: float = 0.0,
    ) -> list[Document]:
        """Return top-k nearest documents by cosine similarity.

        Args:
            min_score: Minimum similarity score threshold (0.0 to 1.0).
                Documents below this score are excluded. Default 0.0 (no filtering).
                Recommended: 0.3 for loose matching, 0.5 for strict relevance.
        """

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""

    @abstractmethod
    def count(self) -> int:
        """Return total document count."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all documents."""
