"""Bridge to vendored Nitin-Gupta1109/engram (v0.1.7).

Provides safe imports with lazy fallbacks so memory.py can call Nitin's
extraction and ranking helpers without breaking when an optional dep
(sentence-transformers, faiss, mcp) is missing.

Nitin's package lives at /home/sheldon_antony/.config/opencode/engram_nitin/.
We import from there directly — not installed via pip — so changes here
propagate without reinstall.

Environment toggles (all default off, preserving production behavior):
  ENGRAM_USE_NITIN_PREFS    "1"  -> extract_preferences() at store time
  ENGRAM_USE_NITIN_TOPICS   "1"  -> extract_topics() at store time
  ENGRAM_USE_NITIN_BOOSTS   "1"  -> add 3 RRF signals (name, phrase, temporal)
  ENGRAM_USE_NITIN_CHUNK    "1"  -> chunked turn ingestion (6 turns, overlap 1)
  ENGRAM_USE_NITIN_SPEAKER  "1"  -> speaker-name injection
  ENGRAM_USE_NITIN_FAISS    "1"  -> FAISS cache (replaces numpy LRU for big projects)
  ENGRAM_USE_NITIN_MCP      "1"  -> expose native MCP server (replaces TS plugin)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_NITIN_PATH = Path(__file__).parent / "engram_nitin"
if _NITIN_PATH.exists() and str(_NITIN_PATH.parent) not in sys.path:
    sys.path.insert(0, str(_NITIN_PATH.parent))

_USE_PREFS   = os.environ.get("ENGRAM_USE_NITIN_PREFS",   "0") == "1"
_USE_TOPICS  = os.environ.get("ENGRAM_USE_NITIN_TOPICS",  "0") == "1"
_USE_BOOSTS  = os.environ.get("ENGRAM_USE_NITIN_BOOSTS",  "0") == "1"
_USE_CHUNK   = os.environ.get("ENGRAM_USE_NITIN_CHUNK",   "0") == "1"
_USE_SPEAKER = os.environ.get("ENGRAM_USE_NITIN_SPEAKER", "0") == "1"
_USE_FAISS   = os.environ.get("ENGRAM_USE_NITIN_FAISS",   "0") == "1"
_USE_MCP     = os.environ.get("ENGRAM_USE_NITIN_MCP",     "0") == "1"

_nitin_parser: Any = None
_nitin_pipeline: Any = None
_nitin_faiss: Any = None
_nitin_sparse: Any = None
_import_warnings: list[str] = []


def _try_import_nitin():
    """Lazy-load Nitin's modules once.  Falls back to None on any failure."""
    global _nitin_parser, _nitin_pipeline, _nitin_faiss, _nitin_sparse
    if _nitin_parser is not None:
        return
    try:
        from engram_nitin.ingestion import parser as _p
        from engram_nitin.retrieval import pipeline as _pl
        from engram_nitin.retrieval import sparse as _sp
        _nitin_parser = _p
        _nitin_pipeline = _pl
        _nitin_sparse = _sp
    except Exception as e:
        _import_warnings.append(f"engram_nitin import failed: {e}")
        return
    try:
        from engram_nitin.backends import faiss_backend as _fb
        _nitin_faiss = _fb
    except Exception as e:
        _import_warnings.append(f"faiss backend import failed: {e}")


def is_available() -> bool:
    """True if Nitin's engram_nitin package is importable."""
    _try_import_nitin()
    return _nitin_parser is not None


def get_warnings() -> list[str]:
    _try_import_nitin()
    return list(_import_warnings)


# ─── Preference extraction (Nitin's 27 patterns) ──────────────────────────────

def extract_preferences(turns: list) -> list[str]:
    """Extract preference expressions using Nitin's comprehensive patterns.

    Returns deduplicated list of 5-80 char preference strings.
    Empty list if Nitin unavailable or feature disabled.
    """
    if not _USE_PREFS:
        return []
    _try_import_nitin()
    if _nitin_parser is None:
        return []
    try:
        return _nitin_parser.extract_preferences(turns)
    except Exception:
        return []


def extract_topics(turns: list) -> list[str]:
    """Extract key topic nouns from user turns for vocabulary bridging.

    Returns deduplicated list of topic strings.
    Empty list if Nitin unavailable or feature disabled.
    """
    if not _USE_TOPICS:
        return []
    _try_import_nitin()
    if _nitin_parser is None:
        return []
    try:
        return _nitin_parser.extract_topics(turns)
    except Exception:
        return []


def extract_person_names(text: str) -> list[str]:
    """Extract likely person names from text via regex.

    Returns deduped list of name strings.  Always available (no deps).
    """
    _try_import_nitin()
    if _nitin_pipeline is None:
        return []
    try:
        return _nitin_pipeline.extract_person_names(text)
    except Exception:
        return []


def extract_quoted_phrases(text: str) -> list[str]:
    """Extract quoted phrases from text (3-60 chars inside quotes)."""
    _try_import_nitin()
    if _nitin_pipeline is None:
        return []
    try:
        return _nitin_pipeline.extract_quoted_phrases(text)
    except Exception:
        return []


def parse_temporal_offset(question: str):
    """Extract temporal offset like 'two weeks ago' from question.

    Returns (days_back, tolerance_days) tuple or None.
    """
    _try_import_nitin()
    if _nitin_pipeline is None:
        return None
    try:
        return _nitin_pipeline.parse_temporal_offset(question)
    except Exception:
        return None


def parse_date(date_str: str):
    """Parse common date formats.  Returns datetime or None."""
    _try_import_nitin()
    if _nitin_pipeline is None:
        return None
    try:
        return _nitin_pipeline.parse_date(date_str)
    except Exception:
        return None


# ─── Chunked turn ingestion ──────────────────────────────────────────────────

def session_to_chunks(turns: list, max_turns: int = 6, overlap: int = 1) -> list:
    """Split turn list into overlapping chunks for long-session ingestion.

    Returns list of turn sublists.  Nitin's default: 6 turns, overlap 1.
    Empty list if feature disabled or Nitin unavailable.
    """
    if not _USE_CHUNK:
        return []
    _try_import_nitin()
    if _nitin_parser is None:
        return []
    try:
        return _nitin_parser._chunk_turns(turns, max_turns=max_turns, overlap=overlap)
    except Exception:
        return []


def render_turn_with_speaker(turn: dict, speaker_names: dict | None) -> str:
    """Render a turn prepending the speaker name if available.

    Bridges first-person facts ("I got a PS5") with entity-attribute queries
    ("What console does Nate own?") by injecting "Nate: I got a PS5".
    """
    if not _USE_SPEAKER:
        return turn.get("content", "")
    _try_import_nitin()
    if _nitin_parser is None:
        return turn.get("content", "")
    try:
        return _nitin_parser._render_turn(turn, speaker_names)
    except Exception:
        return turn.get("content", "")


# ─── FAISS cache for large projects ───────────────────────────────────────────

def get_faiss_backend_class():
    """Return Nitin's FaissBackend class, or None if disabled/unavailable."""
    if not _USE_FAISS:
        return None
    _try_import_nitin()
    if _nitin_faiss is None:
        return None
    return getattr(_nitin_faiss, "FaissBackend", None)


# ─── Sparse BM25 (Nitin's pure-Python impl) ───────────────────────────────────

def get_bm25_class():
    """Return Nitin's BM25 class (alternative to SQLite FTS5)."""
    _try_import_nitin()
    return _nitin_sparse


# ─── Diagnostic: feature flag summary ─────────────────────────────────────────

def status() -> dict:
    """Return current feature-flag state for diagnostics."""
    return {
        "nitin_available":      is_available(),
        "nitin_warnings":       get_warnings(),
        "nitin_use_prefs":      _USE_PREFS,
        "nitin_use_topics":     _USE_TOPICS,
        "nitin_use_boosts":     _USE_BOOSTS,
        "nitin_use_chunk":      _USE_CHUNK,
        "nitin_use_speaker":    _USE_SPEAKER,
        "nitin_use_faiss":      _USE_FAISS,
        "nitin_use_mcp":        _USE_MCP,
    }
