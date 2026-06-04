#!/usr/bin/env python3
"""Shared utilities for embeddings and similarity.

Used by memory.py and tasks.py to avoid duplicating model initialization.

Embedding backend selection (env vars, read once at import time):
  ENGRAM_EMBED_BACKEND  : "fastembed" (default) | "sentence-transformers"
  ENGRAM_EMBED_MODEL    : model name or local path
                          fastembed default  -> BAAI/bge-small-en-v1.5
                          st default         -> BAAI/bge-large-en-v1.5  (v20+)
  ENGRAM_BGE_LARGE      : "1" -> use bge-large (1024d) even with fastembed path
                          "0" -> use bge-small (384d) [faster, less precise]
                          Default: "0" (preserves existing production DBs)

When ENGRAM_EMBED_BACKEND=sentence-transformers the SentenceTransformer library
is used, enabling fine-tuned local models.  The fastembed path is unchanged
and remains the production default so existing DBs are never silently broken.

v20+ chunked encoding: long texts (over 4*max_seq_length chars) are split into
overlapping chunks, encoded separately, and mean-pooled. Avoids silent truncation
of long sessions (was losing the tail of 30-turn conversations).
"""

import os

_EMBED_BACKEND = os.environ.get("ENGRAM_EMBED_BACKEND", "fastembed").lower()

# v20 — bge-large opt-in.  Default stays bge-small for backward compat with
# existing production DBs.  Set ENGRAM_BGE_LARGE=1 to upgrade.
_USE_BGE_LARGE = os.environ.get("ENGRAM_BGE_LARGE", "0") == "1"
_BGE_LARGE_NAME = "BAAI/bge-large-en-v1.5"
_BGE_SMALL_NAME = "BAAI/bge-small-en-v1.5"
_DEFAULT_FASTEMBED_MODEL = _BGE_LARGE_NAME if _USE_BGE_LARGE else _BGE_SMALL_NAME
_DEFAULT_ST_MODEL       = _BGE_LARGE_NAME if _USE_BGE_LARGE else _BGE_SMALL_NAME

_EMBED_MODEL   = os.environ.get("ENGRAM_EMBED_MODEL", "")   # "" = use backend default

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        if _EMBED_BACKEND == "sentence-transformers":
            from sentence_transformers import SentenceTransformer  # lazy
            model_name = _EMBED_MODEL or _DEFAULT_ST_MODEL
            _embedding_model = SentenceTransformer(model_name)
        else:
            from fastembed import TextEmbedding  # lazy: only loaded when embeddings are needed
            _embedding_model = TextEmbedding(_EMBED_MODEL) if _EMBED_MODEL else TextEmbedding(_DEFAULT_FASTEMBED_MODEL)
    return _embedding_model


def get_embedding_dim() -> int:
    """Return the current embedding model's output dimension."""
    if _USE_BGE_LARGE:
        return 1024
    return 384


def _normalize(vec) -> list[float]:
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    norm = sum(x ** 2 for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def _needs_bge_query_prefix(model_name: str) -> bool:
    """BGE models need a query prefix for asymmetric retrieval."""
    return model_name.startswith("BAAI/bge-")


def _prepare_query(text: str, model_name: str) -> str:
    if _needs_bge_query_prefix(model_name):
        return f"Represent this sentence for searching relevant passages: {text}"
    return text


def _chunk_text(text: str, max_chars: int, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks for long-doc encoding."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks if chunks else [text[:max_chars]]


def _resolve_model_name() -> str:
    if _EMBED_MODEL:
        return _EMBED_MODEL
    if _EMBED_BACKEND == "sentence-transformers":
        return _DEFAULT_ST_MODEL
    return _DEFAULT_FASTEMBED_MODEL


def embed_text(text: str) -> list[float]:
    """Generate a normalized embedding vector for the given text.
    
    For long texts, the sentence-transformers path applies chunked mean-pooling
    to avoid silent truncation. The fastembed path delegates to its native embed().
    """
    model = get_embedding_model()
    if _EMBED_BACKEND == "sentence-transformers":
        model_name = _resolve_model_name()
        max_chars = 2048  # default bge max_seq_length=512, 4 chars/token
        if len(text) <= max_chars:
            prepared = _prepare_query(text, model_name) if not text else text
            vec = model.encode(prepared, normalize_embeddings=False)
        else:
            chunks = _chunk_text(text, max_chars, overlap=200)
            vecs = model.encode(chunks, normalize_embeddings=False, batch_size=8,
                                show_progress_bar=False)
            import numpy as _np
            mean = _np.mean(_np.array([v.tolist() for v in vecs]), axis=0)
            vec = mean
        return _normalize(vec)
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
        model_name = _resolve_model_name()
        max_chars = 2048
        needs_chunking = any(len(t) > max_chars for t in texts)
        if not needs_chunking:
            vecs = model.encode(texts, normalize_embeddings=False, batch_size=64,
                                show_progress_bar=False)
            for vec in vecs:
                result.append(_normalize(vec))
        else:
            import numpy as _np
            for text in texts:
                if len(text) <= max_chars:
                    vec = model.encode([text], normalize_embeddings=False,
                                       show_progress_bar=False)[0]
                else:
                    chunks = _chunk_text(text, max_chars, overlap=200)
                    chunk_vecs = model.encode(chunks, normalize_embeddings=False,
                                              batch_size=8, show_progress_bar=False)
                    vec = _np.mean(_np.array([v.tolist() for v in chunk_vecs]), axis=0)
                result.append(_normalize(vec))
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

# v20 — default CE upgraded to bge-reranker-v2-m3 (Nitin-Gupta1109/engram default).
# Set PREFLIGHT_CE_MODEL=... to override.  Production never silently changed:
# existing env override is respected, and PREFLIGHT_USE_CE_LEGACY=1 restores
# the prior mxbai-rerank-xsmall default.
_USE_CE_LEGACY = os.environ.get("PREFLIGHT_USE_CE_LEGACY", "0") == "1"
_DEFAULT_CE = (
    "mixedbread-ai/mxbai-rerank-xsmall-v1" if _USE_CE_LEGACY
    else "BAAI/bge-reranker-v2-m3"
)
_CE_MODEL_NAME = os.environ.get("PREFLIGHT_CE_MODEL", _DEFAULT_CE)
# Better CE models (set via PREFLIGHT_CE_MODEL):
#   BAAI/bge-reranker-v2-m3                (568M, multilingual, default v20+)
#   BAAI/bge-reranker-v2-base              (~290M, good for technical queries)
#   mixedbread-ai/mxbai-rerank-base-v1     (~200M, +3 BEIR over xsmall)
#   mixedbread-ai/mxbai-rerank-xsmall-v1   (~50M, legacy default, fastest)


def get_cross_encoder():
    """Lazy-load a cross-encoder for Phase 4 reranking.
    
    Model is controlled by PREFLIGHT_CE_MODEL env var (default v20+:
    BAAI/bge-reranker-v2-m3). Returns None silently if sentence-transformers
    is not installed.
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


def get_cross_encoder_name() -> str:
    return _CE_MODEL_NAME
