#!/usr/bin/env python3
"""Shared utilities for embeddings and similarity.

Used by memory.py and tasks.py to avoid duplicating model initialization.

Embedding backend selection (env vars, read once at import time):
  ENGRAM_EMBED_BACKEND  : "fastembed" (default) | "sentence-transformers"
  ENGRAM_EMBED_MODEL    : model name or local path
                          fastembed default  -> BAAI/bge-small-en-v1.5
                          st default         -> BAAI/bge-small-en-v1.5

When ENGRAM_EMBED_BACKEND=sentence-transformers the SentenceTransformer library
is used, enabling fine-tuned local models.  The fastembed path is unchanged
and remains the production default so existing DBs are never silently broken.
"""

import os

_EMBED_BACKEND = os.environ.get("ENGRAM_EMBED_BACKEND", "fastembed").lower()
_EMBED_MODEL   = os.environ.get("ENGRAM_EMBED_MODEL", "")   # "" = use backend default

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        if _EMBED_BACKEND == "sentence-transformers":
            from sentence_transformers import SentenceTransformer  # lazy
            model_name = _EMBED_MODEL or "BAAI/bge-small-en-v1.5"
            _embedding_model = SentenceTransformer(model_name)
        else:
            from fastembed import TextEmbedding  # lazy: only loaded when embeddings are needed
            _embedding_model = TextEmbedding(_EMBED_MODEL) if _EMBED_MODEL else TextEmbedding()
    return _embedding_model


def _normalize(vec: list[float]) -> list[float]:
    norm = sum(x ** 2 for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def embed_text(text: str) -> list[float]:
    """Generate a normalized embedding vector for the given text."""
    model = get_embedding_model()
    if _EMBED_BACKEND == "sentence-transformers":
        vec = model.encode(text, normalize_embeddings=False).tolist()
    else:
        vec = list(model.embed([text]))[0]
        vec = [float(x) for x in vec]
    return _normalize(vec)


def embed_texts_batch(texts: list) -> list:
    """Generate normalized embedding vectors for a list of texts in one batch.

    Returns a list of float lists, one per input text, in input order.
    """
    if not texts:
        return []
    model = get_embedding_model()
    result = []
    if _EMBED_BACKEND == "sentence-transformers":
        vecs = model.encode(texts, normalize_embeddings=False, batch_size=64,
                            show_progress_bar=False)
        for vec in vecs:
            result.append(_normalize([float(x) for x in vec]))
    else:
        for vec in model.embed(texts):
            vec = [float(x) for x in vec]
            result.append(_normalize(vec))
    return result


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two pre-normalized vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x ** 2 for x in a) ** 0.5
    nb = sum(x ** 2 for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


_cross_encoder = None
_cross_encoder_tried = False
_cross_encoder_model_name: str = ""

_CE_MODEL_NAME = os.environ.get(
    "PREFLIGHT_CE_MODEL", "BAAI/bge-reranker-v2-m3"
)


def get_cross_encoder():
    """Lazy-load a cross-encoder for Phase 4 reranking.

    Model is controlled by PREFLIGHT_CE_MODEL env var (default:
    mixedbread-ai/mxbai-rerank-xsmall-v1). Returns None silently if
    sentence-transformers is not installed.
    """
    global _cross_encoder, _cross_encoder_tried, _cross_encoder_model_name
    if not _cross_encoder_tried:
        _cross_encoder_tried = True
        _cross_encoder_model_name = _CE_MODEL_NAME
        try:
            from sentence_transformers import CrossEncoder  # lazy import
            _cross_encoder = CrossEncoder(_CE_MODEL_NAME)
        except Exception:
            pass
    return _cross_encoder
