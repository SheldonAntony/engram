"""FAISS-backed local vector store with SQLite metadata."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from .base import Document, VectorBackend


class FaissBackend(VectorBackend):
    """Local vector storage using FAISS + SQLite.

    FAISS handles the vector index (cosine similarity via inner product on
    normalized vectors). SQLite stores document text and metadata alongside
    a positional mapping to FAISS indices.

    This is the zero-dependency-on-cloud path: everything on disk, no API
    keys, no network calls.
    """

    def __init__(self, path: "Optional[str | Path]" = None, dimension: int = 1024):
        self._dim = dimension
        self._path = Path(path) if path else None

        if self._path:
            self._path.mkdir(parents=True, exist_ok=True)
            self._db_path = self._path / "engram_meta.db"
            self._index_path = self._path / "engram.faiss"
        else:
            self._db_path = None
            self._index_path = None

        # In-memory structures
        self._index = faiss.IndexFlatIP(dimension)  # inner product on L2-normed = cosine
        self._id_map: list[str] = []  # positional: faiss row i -> doc id

        # SQLite for metadata + text
        self._conn = sqlite3.connect(
            str(self._db_path) if self._db_path else ":memory:",
            check_same_thread=False,
        )
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT,
                faiss_idx INTEGER
            )
        """)
        self._conn.commit()

        # Load existing index from disk
        if self._index_path and self._index_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            rows = self._conn.execute("SELECT id FROM documents ORDER BY faiss_idx").fetchall()
            self._id_map = [r[0] for r in rows]

    def add(self, docs: List[Document]) -> None:
        if not docs:
            return

        embeddings = []
        for doc in docs:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id} has no embedding")
            vec = np.array(doc.embedding, dtype=np.float32)
            # L2 normalize for cosine similarity via inner product
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)

        matrix = np.vstack(embeddings)
        start_idx = self._index.ntotal
        self._index.add(matrix)

        for i, doc in enumerate(docs):
            self._id_map.append(doc.id)
            meta_json = json.dumps(doc.metadata) if doc.metadata else None
            sql = "INSERT OR REPLACE INTO documents"
            sql += " (id, text, metadata, faiss_idx) VALUES (?, ?, ?, ?)"
            self._conn.execute(sql, (doc.id, doc.text, meta_json, start_idx + i))
        self._conn.commit()
        self._save()

    def query(
        self,
        embedding: List[float],
        top_k: int = 10,
        metadata_filter: Optional[dict] = None,
        min_score: float = 0.0,
    ) -> List[Document]:
        if self._index.ntotal == 0:
            return []

        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # Over-fetch if filtering, then trim
        if metadata_filter:
            fetch_k = min(top_k * 5, self._index.ntotal)
        else:
            fetch_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            if float(score) < min_score:
                continue
            doc_id = self._id_map[idx]
            row = self._conn.execute(
                "SELECT text, metadata FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
            if not row:
                continue

            text, meta_json = row
            metadata = json.loads(meta_json) if meta_json else {}

            # Apply metadata filter
            if metadata_filter:
                if not all(metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue

            results.append(
                Document(
                    id=doc_id,
                    text=text,
                    metadata=metadata,
                    score=float(score),
                )
            )
            if len(results) >= top_k:
                break

        return results

    def delete(self, ids: List[str]) -> None:
        # FAISS doesn't support deletion natively with IndexFlat.
        # For now, mark as deleted in SQLite; rebuild on next load.
        for doc_id in ids:
            self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self._conn.commit()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0] if row else 0

    def clear(self) -> None:
        self._index = faiss.IndexFlatIP(self._dim)
        self._id_map = []
        self._conn.execute("DELETE FROM documents")
        self._conn.commit()
        self._save()

    def _save(self) -> None:
        if self._index_path:
            faiss.write_index(self._index, str(self._index_path))
