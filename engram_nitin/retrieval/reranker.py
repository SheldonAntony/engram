"""Cross-encoder reranker — high-accuracy reranking without LLM API calls.

This is the key advantage over MemPalace: they use Claude Haiku ($0.001/query)
for reranking. We use a local cross-encoder model (free, faster, no API key).

Cross-encoders score (query, document) pairs jointly — much more accurate than
bi-encoder cosine similarity, but too slow for full-corpus search. Perfect for
reranking a top-k candidate set.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


class CrossEncoderReranker:
    """Reranks candidates using a cross-encoder model.

    Default: BAAI/bge-reranker-v2-m3 (568M params, strong multilingual reranker).
    Fallback: cross-encoder/ms-marco-MiniLM-L-6-v2 (smaller, English-only).
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self._model_name)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Rerank documents by cross-encoder relevance score.

        Returns list of (original_index, score) sorted by score descending.
        """
        if not documents:
            return []

        self._load()

        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            indexed_scores = indexed_scores[:top_k]

        return [(idx, float(score)) for idx, score in indexed_scores]
