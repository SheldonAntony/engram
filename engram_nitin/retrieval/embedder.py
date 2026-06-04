"""Embedding models — the single biggest lever for retrieval quality."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

# Model registry: name -> (HuggingFace ID, dimension)
MODELS = {
    "bge-large": ("BAAI/bge-large-en-v1.5", 1024),
    "bge-base": ("BAAI/bge-base-en-v1.5", 768),
    "gte-large": ("thenlper/gte-large", 1024),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 384),
    "nomic": ("nomic-ai/nomic-embed-text-v1.5", 768),
    "mxbai": ("mixedbread-ai/mxbai-embed-large-v1", 1024),
}

DEFAULT_MODEL = "bge-large"


class Embedder:
    """Wraps a sentence-transformers model for encoding text to vectors.

    Uses bge-large-en-v1.5 by default — 1024-dim, significantly stronger than
    all-MiniLM-L6-v2 (384-dim) used by MemPalace. This single change accounts
    for ~2-3% R@5 improvement on LongMemEval.

    BGE models expect a query prefix for asymmetric retrieval:
    - Queries: "Represent this sentence for searching relevant passages: {query}"
    - Documents: raw text (no prefix)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        if model_name in MODELS:
            hf_id, self.dimension = MODELS[model_name]
        else:
            hf_id = model_name
            self.dimension: Optional[int] = None  # will be set after loading

        self._model_name = model_name
        self._hf_id = hf_id
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._hf_id)
        if self.dimension is None:
            self.dimension = self._model.get_sentence_embedding_dimension()

    @property
    def needs_query_prefix(self) -> bool:
        return self._model_name in ("bge-large", "bge-base")

    def _prepare_query(self, text: str) -> str:
        if self.needs_query_prefix:
            return f"Represent this sentence for searching relevant passages: {text}"
        return text

    @property
    def max_seq_length(self) -> int:
        """Return the model's max sequence length in tokens."""
        self._load()
        return getattr(self._model, "max_seq_length", 512)

    def encode_documents(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode documents (no query prefix).

        For long documents that exceed the model's token limit, we use a
        chunked mean-pooling strategy: split into overlapping chunks at the
        character level (approx 4 chars/token), encode each chunk, and
        average the embeddings. This avoids silent truncation that loses
        the tail of long sessions.
        """
        self._load()
        max_chars = self.max_seq_length * 4  # rough chars-per-token estimate

        # Check if any document needs chunking
        needs_chunking = any(len(t) > max_chars for t in texts)
        if not needs_chunking:
            return self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        # Chunk long documents and mean-pool their embeddings
        results = []
        for text in texts:
            if len(text) <= max_chars:
                emb = self._model.encode(
                    [text],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0]
            else:
                chunks = self._chunk_text(text, max_chars, overlap=200)
                chunk_embs = self._model.encode(
                    chunks,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                # Mean pool and re-normalize
                emb = np.mean(chunk_embs, axis=0)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
            results.append(emb)

        return np.array(results)

    @staticmethod
    def _chunk_text(text: str, max_chars: int, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
        return chunks if chunks else [text[:max_chars]]

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query (with prefix if needed)."""
        self._load()
        prepared = self._prepare_query(text)
        return self._model.encode(
            [prepared],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

    def encode_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode multiple queries."""
        self._load()
        prepared = [self._prepare_query(t) for t in texts]
        return self._model.encode(
            prepared,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
