#!/usr/bin/env python3
"""Shared utilities for embeddings and similarity.

Used by memory.py and tasks.py to avoid duplicating model initialization.
"""

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from fastembed import TextEmbedding  # lazy: only loaded when embeddings are needed
        _embedding_model = TextEmbedding()
    return _embedding_model


def embed_text(text: str) -> list[float]:
    """Generate a normalized embedding vector for the given text."""
    model = get_embedding_model()
    vec = list(model.embed([text]))[0]
    vec = [float(x) for x in vec]
    norm = sum(x ** 2 for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two pre-normalized vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x ** 2 for x in a) ** 0.5
    nb = sum(x ** 2 for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
