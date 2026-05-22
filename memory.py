#!/usr/bin/env python3
"""Semantic memory and slot fills for the Preflight plugin.

CLI usage:
    memory.py store_fact          <project_id> <session_id> <text> [fact_type]
    memory.py retrieve_facts      <project_id> <session_id> <prompt> [top_n] [threshold]
    memory.py check_dedup         <key>
    memory.py mark_stored         <key>
    memory.py store_slot_fill     <project_id> <session_id> <slot_name> <value>
    memory.py retrieve_slot_fills <project_id>
    memory.py session_seen        <session_id>
    memory.py session_mark        <session_id> <project_id>
    memory.py session_unmark      <session_id>
    memory.py link_facts          <fact_id_a> <fact_id_b> <relation> <strength>
    memory.py get_related         <fact_id> [depth]
    memory.py get_graph           <project_id> <query>
"""

import hashlib
import json
import multiprocessing as mp
import os
import re
import sqlite3
import struct
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

# Feature 1 Bug 3: shared utilities extracted from this module
from utils import cosine_similarity, embed_text, embed_texts_batch

# LLM atomic fact extractor (Qwen2.5-1.5B via Ollama) — opt-in via env flag.
# Requires: ollama pull qwen2.5:1.5b  and Ollama running on localhost:11434.
# Silently no-ops when Ollama is not available; raw storage is never affected.
_USE_LLM_EXTRACTOR = os.environ.get("PREFLIGHT_USE_LLM_EXTRACTOR", "0") == "1"

try:
    from extractor import extract_entities as _extract_entities
except (ImportError, AttributeError):
    def _extract_entities(text: str) -> list[str]:  # type: ignore[misc]
        return []

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.db")

# Content-type decay rates (per-day, inspired by ClawMem).
# decision/preference/finding never decay; notes/snippets/summaries decay faster.
_DECAY_RATES: dict[str, float] = {
    "decision":   0.0,     # architectural choices, never decay
    "preference": 0.0,     # user/team preferences, never decay
    "finding":    0.005,   # discovered facts about codebase, slow decay
    "snippet":    0.015,   # code snippets, medium decay
    "summary":    0.02,    # session summaries
    "note":       0.02,    # default — generic notes
    "window":     0.03,    # sliding-window turns — fast decay, demoted in scoring
    "turn":       0.03,    # single-turn facts (clean [curr] text) — same decay as window
    "llm_atomic": 0.005,   # LLM-extracted atomic facts — slow decay like findings
}

# Similarity threshold above which a new fact is treated as a contradiction
# / near-duplicate of an existing one and replaces it instead of inserting.
_CONTRADICTION_THRESHOLD = 0.88

# Minimum similarity for an auto-created graph edge between two facts.
_RELATION_THRESHOLD = 0.65

# Relation type weights (used as default strength when auto-detected).
RELATION_TYPES: dict[str, float] = {
    "caused_by":   0.9,
    "fixed_by":    0.9,
    "related":     0.7,
    "contradicts": 0.8,
    "depends_on":  0.8,
}

# Phase A: importance scoring weights and keyword signals.
_IMPORTANCE_TYPE_WEIGHTS: dict[str, float] = {
    "decision": 1.0, "preference": 0.9, "finding": 0.7,
    "snippet": 0.5, "summary": 0.4, "note": 0.3,
    "turn": 0.25,      # single-turn facts — slightly above window, below general notes
    "window": 0.05,    # window facts rank below atomic notes by default
    "llm_atomic": 0.65, # LLM-extracted atomic facts — high signal, between finding and note
}
_IMPORTANCE_KEYWORDS: frozenset = frozenset({
    "never", "always", "must", "critical", "required", "forbidden",
    "breaking", "security", "auth", "production", "prod", "deprecated",
    "migration", "decided", "architectural",
})
# Phase C: MMR lambdas — two separate values for pre-CE (candidate selection)
# and post-CE (output deduplication). Pre-CE uses pure relevance (λ=1.0) so the
# cross-encoder sees the highest-scoring candidates, not a diversity-adjusted set.
# Post-CE uses light diversity (λ=0.25) to deduplicate what is returned to the user.
_MMR_LAMBDA = 0.6          # legacy — kept for reference, not used directly below
_MMR_LAMBDA_PRE_CE  = 1.0  # pre-CE selection: pure top-k by relevance
_MMR_LAMBDA_POST_CE = 0.25 # post-CE output: light diversity deduplication

# Step 6: window demotion factor — applied to composite score for fact_type="window".
# Keeps windows retrievable as fallback while atomic SVO facts rank above them.
# Set to 1.0 to disable demotion entirely (recommended: window facts carry useful context).
_WINDOW_DEMOTION = float(os.environ.get("PREFLIGHT_WINDOW_DEMOTION", "1.0"))

# SM-2 gate: when False, SM-2 interval check is skipped for candidate selection.
# EF/interval updates still happen on retrieval — they feed the staleness score.
# SM-2 spaced-repetition gate for deliberate, passive forgetting.
# Gate is ON: facts that haven't been retrieved recently accumulate decay and are
# demoted in ranking over time — the system forgets stale information automatically.
# This is preferable to asking the user "is this outdated?" which is only reliable
# when model staleness confidence is high (it usually isn't).
#
# NOTE: Disable (set False) when running LoCoMo benchmarks — gold facts start with
# retrieval_count=0 and the gate would exclude them from scoring from the start.
_SM2_GATE_ENABLED = True

# ── Tunable retrieval constants ───────────────────────────────────────────────
# All can be overridden via env vars for deployment tuning.
_POOL_A_LIMIT          = int(os.environ.get("PREFLIGHT_POOL_A", "99999")) # ANN pool cap (default: all facts)
_POOL_B_LIMIT          = int(os.environ.get("PREFLIGHT_POOL_B", "300"))   # proven-useful pool
_RRF_K                 = int(os.environ.get("PREFLIGHT_RRF_K", "15"))     # RRF smoothing (15=tight, 60=loose)
_BROAD_POOL            = int(os.environ.get("PREFLIGHT_BROAD_POOL", "200")) # union top-N from each signal
_USE_DERIVED_BM25      = os.environ.get("PREFLIGHT_USE_DERIVED_BM25", "0") == "1"
_USE_LEXICAL_CHANNELS  = os.environ.get("PREFLIGHT_USE_LEXICAL_CHANNELS", "0") == "1"
_USE_CONTEXT_BM25      = os.environ.get("PREFLIGHT_USE_CONTEXT_BM25", "0") == "1"
_USE_QUERY_DECOMPOSITION = os.environ.get("PREFLIGHT_USE_QUERY_DECOMPOSITION", "0") == "1"
_CONTEXT_WINDOW_SIZE   = int(os.environ.get("PREFLIGHT_CONTEXT_WINDOW", "3"))
_CE_POOL_SIZE          = int(os.environ.get("PREFLIGHT_CE_POOL", "120"))   # candidates fed to CE (benchmark: 200)
_CE_TIMEOUT            = float(os.environ.get("PREFLIGHT_CE_TIMEOUT", "5.0")) # max seconds for CE (bigger model needs more)
_CE_GUARD_K            = int(os.environ.get("PREFLIGHT_CE_GUARD_K", "40")) # min-rank guard after CE (0=disabled)
_COVERAGE_K            = int(os.environ.get("PREFLIGHT_COVERAGE_K", "40")) # min-rank after scoring (0=disabled)
_TEMPORAL_EDGE_DECAY   = 0.25   # strength decay per turn distance (linear)
_TEMPORAL_MAX_DISTANCE = 3      # turns back to link temporally
_SESSION_RECENCY_DECAY = 0.15   # score decay per session gap
_SESSION_MAX_LOOKBACK  = 7      # sessions back before score → 0.0
_ENRICHMENT_MAX_TOKENS = 500    # max combined tokens before enrichment falls back to insert
_ENRICH_MIN_SIM        = 0.15   # minimum cosine similarity to existing fact before enriching

# Benchmark mode: used by recall_ablation.py to test production retrieval against LoCoMo.
# When enabled, retrieve_facts() skips pool caps, composite scoring, SM-2 updates,
# MMR, graph expansion, and returns (facts_list, fids_list) instead of just facts_list.
_RETRIEVE_BENCHMARK = os.environ.get("PREFLIGHT_RETRIEVE_BENCHMARK", "0") == "1"

# Step 7: Retrospective ENRICH (consolidate_memories) tuning constants.
_RETRO_FLOOR  = 0.35   # minimum pairwise cosine to trigger a merge
_RETRO_GUARD  = 0.30   # merged embedding must be >= this to BOTH originals (prevents chaining)
_RETRO_MAX    = 500    # default merge cap per consolidate_memories() call

# BM25 stopwords: question-frame and common function words that match many
# irrelevant turns and inflate BM25 ranks for wrong facts.  Filtering them
# prevents noisy BM25 matches from overriding strong vector hits in RRF fusion.
_BM25_STOPWORDS = frozenset({
    "what", "when", "where", "which", "who", "whom", "whose", "how", "why",
    "did", "does", "has", "had", "was", "were", "are", "been", "have",
    "would", "could", "should", "will", "shall",
    "the", "that", "this", "and", "for", "with", "from", "into",
    "she", "her", "his", "their", "him", "they", "you", "its",
    "not", "but", "can", "any", "all", "out",
})


# ── Embedding cache for ANN-based Pool A ─────────────────────────────────────
# Keyed by project_id → (fid_list, emb_matrix) where emb_matrix is shape (N, D).
# Populated lazily on first retrieve call for a project.
# _cache_dirty tracks which projects need a reload after a write.
_EMB_CACHE_MAX_SIZE = 50
_EMB_CACHE: "OrderedDict[str, tuple[list[int], object]]" = OrderedDict()
_EMB_CACHE_LOCK = threading.Lock()
_CACHE_DIRTY: "set[str]" = set()
_CACHE_DIRTY_LOCK = threading.Lock()

# ── spaCy lazy loader (Step 6 SVO extraction) ────────────────────────────────
# Set to None before first use, False after a failed load attempt.
_nlp = None
_NLP_LOCK = threading.Lock()
_WINDOW_TAG_RE = re.compile(r'^\[(prev|curr|next)\]\s*')


def _get_nlp():
    """Return a loaded spaCy en_core_web_sm model, or None if unavailable."""
    global _nlp
    with _NLP_LOCK:
        if _nlp is None:
            try:
                import spacy  # noqa: PLC0415
                _nlp = spacy.load("en_core_web_sm")
            except Exception:
                _nlp = False
    return _nlp if _nlp is not False else None


# ── Derived FTS helpers ────────────────────────────────────────────────────────

_WORDNET_AVAILABLE: "bool | None" = None  # None=untested, True/False=cached
_WORDNET_LOCK = threading.Lock()

# Common question/function words that should never be expanded via WordNet.
# Expanding these produces domain-wrong hypernyms (e.g. "does"→"do"→verb chain,
# "play"→many verb senses) that create spurious BM25 hits and hurt R@40.
_EXPAND_STOP_WORDS: frozenset = frozenset({
    "what", "when", "where", "which", "whom", "whose",
    "does", "this", "that", "these", "those", "have", "been",
    "will", "would", "could", "should", "with", "from", "into",
    "about", "after", "before", "between", "through", "during",
    "their", "there", "they", "them", "then", "than", "also",
    "just", "some", "many", "much", "more", "most", "other",
    "over", "under", "again", "further", "once", "here",
    "both", "each", "such", "were", "very", "while", "said",
    "like", "know", "think", "want", "going", "come", "came",
    "make", "made", "take", "took", "give", "gave", "tell",
    "told", "well", "time", "year", "back", "even", "good",
    "work", "life", "feel", "felt", "long", "been", "last",
    "still", "never", "always", "really", "actually", "maybe",
    "probably", "something", "anything", "everything", "nothing",
    "someone", "anyone", "everyone", "thing", "things",
    # Temporal / conversational words whose noun senses expand to useless concepts
    "yeah", "okay", "today", "tonight", "yesterday", "tomorrow",
    "morning", "evening", "afternoon", "night", "weekend", "recently",
    "group", "support", "place", "people", "person", "point", "kind",
})


def _build_derived_text(text: str) -> str:
    """Return a bag-of-words string expanding noun tokens with WordNet depth-1 hypernyms.

    Uses noun-only expansion to avoid verb-sense pollution (e.g. "instrument"
    as a verb → "equip", "fit out" which are domain-wrong for music queries).
    Noun hypernyms are the reliable signal: guitar→stringed_instrument,
    marathon→foot_race, painting→creation.

    Polysemy guard: nouns with >4 synsets are left unexpanded.
    Multi-word hypernym lemmas (e.g. "stringed_instrument") are kept as a
    single token (underscore removed, space-joined) — they appear as separate
    terms in FTS which is correct.  Single short words from compound lemmas
    (e.g. "out" from "fit_out") are filtered by the >3 char threshold.

    Falls back gracefully: if NLTK/WordNet is unavailable, returns the text
    lowercased (still useful for FTS case-normalisation).

    Speaker prefix ("Alex: ") is preserved verbatim and not expanded.
    """
    global _WORDNET_AVAILABLE

    # One-time WordNet availability check (locked for thread safety).
    with _WORDNET_LOCK:
        if _WORDNET_AVAILABLE is None:
            try:
                from nltk.corpus import wordnet as _wn  # noqa: PLC0415
                _wn.synsets("test")
                _WORDNET_AVAILABLE = True
            except LookupError:
                try:
                    import nltk  # noqa: PLC0415
                    nltk.download("wordnet", quiet=True)
                    nltk.download("omw-1.4", quiet=True)
                    from nltk.corpus import wordnet as _wn  # noqa: PLC0415
                    _wn.synsets("test")
                    _WORDNET_AVAILABLE = True
                except Exception:
                    _WORDNET_AVAILABLE = False
            except ImportError:
                _WORDNET_AVAILABLE = False

    # Preserve speaker prefix; expand body only.
    if ": " in text:
        speaker, _, body = text.partition(": ")
    else:
        speaker, body = "", text

    # Tokenise body: strip punctuation, lowercase, drop short/numeric tokens.
    raw_tokens = [t.strip(".,!?\"'():;-") for t in body.split()]
    tokens = [t.lower() for t in raw_tokens if len(t) > 3 and not t.isdigit()]

    if not tokens:
        return text.lower()

    if not _WORDNET_AVAILABLE:
        parts = ([speaker] if speaker else []) + tokens
        return " ".join(parts)

    from nltk.corpus import wordnet as wn        # noqa: PLC0415
    from nltk.stem import WordNetLemmatizer      # noqa: PLC0415

    lem = WordNetLemmatizer()
    seen: set[str] = set(tokens)
    expansions: list[str] = []

    for tok in tokens:
        # Skip function/question words — their hypernyms are always off-topic.
        if tok in _EXPAND_STOP_WORDS:
            continue

        # Noun-only expansion.  Verb expansions produce domain-wrong terms
        # for words that are primarily nouns (e.g. "instrument"→"equip").
        noun_lemma = lem.lemmatize(tok, pos="n")
        synsets = wn.synsets(noun_lemma, pos=wn.NOUN)
        if not synsets or len(synsets) > 4:
            continue  # absent or too polysemous

        added = 0
        for hyp in synsets[0].hypernyms():
            # Skip hypernyms too close to the WordNet root — those are abstract
            # concepts like "abstraction", "entity", "physical_object" which are
            # never useful FTS expansion terms.  Depth ≥ 5 keeps domain-specific
            # terms (stringed_instrument ≈ 7, foot_race ≈ 6) while blocking
            # root abstractions (depth 1-3).
            try:
                if hyp.min_depth() < 5:
                    continue
            except Exception:
                pass
            for ln in hyp.lemma_names():
                # Replace underscores so "stringed_instrument" → "stringed instrument"
                # which FTS5 indexes as two terms — both are useful query tokens.
                clean = ln.replace("_", " ").lower()
                # Filter individual short words that arise from compound lemmas
                # like "fit_out" → "fit out" → "fit"(3) would be noise.
                # Apply min-length check to each word in the expansion phrase.
                if any(len(w) <= 3 for w in clean.split()):
                    continue
                if clean not in seen:
                    expansions.append(clean)
                    seen.add(clean)
                    added += 1
                    if added >= 3:
                        break
            if added >= 3:
                break

    parts = ([speaker] if speaker else []) + tokens + expansions
    return " ".join(parts)


def _store_derived_fts(fid: int, curr_line: str) -> None:
    """Insert/replace derived expansion text for *fid* into facts_derived_fts.

    Silently swallows all errors so the store path is never broken by a
    missing NLTK corpus or any other optional-dependency failure.
    """
    try:
        derived = _build_derived_text(curr_line)
        conn = init_db()
        conn.execute(
            "INSERT OR REPLACE INTO facts_derived_fts(rowid, content) VALUES (?, ?)",
            (fid, derived),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _extract_svo_facts(window_content: str) -> list[str]:
    """Extract subject-verb-object triples from a turn window for atomic storage.

    Each line in *window_content* has the form '[tag] Speaker: text'.
    First-person pronouns are replaced by the speaker name so that 'I went to
    Paris' becomes 'Alice went to Paris' — making the triple unambiguous.

    Returns compact fact strings (3-15 words).  Empty list if spaCy is not
    available or no useful triples can be extracted.
    """
    nlp = _get_nlp()
    if nlp is None:
        return []
    facts: list[str] = []
    for line in window_content.splitlines():
        line = _WINDOW_TAG_RE.sub("", line).strip()
        if ": " not in line:
            continue
        speaker, _, text = line.partition(": ")
        speaker = speaker.strip()
        if not speaker or not text.strip():
            continue
        # Bind first-person pronouns to the speaker for disambiguation.
        bound = re.sub(r"\bI\b", speaker, text)
        bound = re.sub(r"\bme\b",  speaker,          bound, flags=re.IGNORECASE)
        bound = re.sub(r"\bmy\b",  f"{speaker}'s",   bound, flags=re.IGNORECASE)
        bound = re.sub(r"\bwe\b",  speaker,          bound, flags=re.IGNORECASE)
        bound = re.sub(r"\bour\b", f"{speaker}'s",   bound, flags=re.IGNORECASE)
        doc = nlp(bound)
        for sent in doc.sents:
            subj = verb = obj_text = None
            for tok in sent:
                if tok.dep_ in ("nsubj", "nsubjpass") and tok.pos_ != "PRON":
                    subj = tok.text
                if tok.dep_ == "ROOT" and tok.pos_ == "VERB":
                    verb = tok.lemma_
                if tok.dep_ in ("dobj", "attr") and verb and obj_text is None:
                    obj_text = " ".join(
                        t.text for t in tok.subtree if t.dep_ != "punct"
                    )
                elif tok.dep_ == "pobj" and verb and obj_text is None:
                    obj_text = " ".join(
                        t.text for t in tok.subtree if t.dep_ != "punct"
                    )
            if subj and verb and obj_text:
                triple = f"{subj} {verb} {obj_text}"
                word_count = len(triple.split())
                if 3 <= word_count <= 15:
                    facts.append(triple)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    deduped = []
    for f in facts:
        key = f.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    return deduped


# ─── Database init ────────────────────────────────────────────────────────────

def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    # WAL mode: readers don't block writers; writers don't block readers.
    # synchronous=NORMAL: fsync only on WAL checkpoint, not every commit.
    # Together these cut per-commit latency by ~10-100x on bulk ingestion
    # (11764 commits for LoCoMo eval vs default DELETE+FULL which fsyncs each).
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA cache_size = -32768")   # 32 MB page cache

    # Feature 2: project_id column on all tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id        TEXT,
            project_id        TEXT,
            content           TEXT,
            embedding         BLOB,
            fact_type         TEXT    DEFAULT 'note',
            retrieval_count   INTEGER DEFAULT 0,
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            valid_from        REAL    DEFAULT (unixepoch()),
            superseded_at     REAL    DEFAULT NULL,
            source_session    TEXT,
            source_hash       TEXT,
            easiness_factor   REAL    DEFAULT 2.5,
            last_retrieved_at REAL    DEFAULT NULL,
            interval_days     REAL    DEFAULT 1.0,
            entities          TEXT    DEFAULT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS fact_mutations (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            fact_id       INTEGER NOT NULL,
            mutation_type TEXT NOT NULL,
            old_content   TEXT,
            new_content   TEXT,
            mutated_at    REAL DEFAULT (unixepoch()),
            session_id    TEXT,
            FOREIGN KEY (fact_id) REFERENCES facts(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dedup (
            key TEXT PRIMARY KEY
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS slot_fills (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            project_id TEXT,
            slot_name  TEXT,
            value      TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            project_id  TEXT,
            enriched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # FTS5 keyword index (BM25) — half of the hybrid search.
    # Uses external rowid mapped to facts.id; manually kept in sync from store/update paths.
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
            content,
            content='facts',
            content_rowid='id'
        )
    """)

    # Derived FTS: lemmatized + WordNet-expanded text for vocabulary bridging.
    # Regular FTS (not external-content) — rowid = facts.id of the window fact.
    # Populated at store time by _store_derived_fts(); enables BM25 hits on
    # paraphrased/hypernym terms that raw conversational text lacks.
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS facts_derived_fts USING fts5(content)
    """)

    # Feature 2: migrations — add project_id if the table already exists
    _ensure_column(conn, "facts",      "project_id",      "TEXT",      "'unknown'")
    _ensure_column(conn, "facts",      "retrieval_count", "INTEGER",   "0")
    _ensure_column(conn, "facts",      "created_at",      "TIMESTAMP", "CURRENT_TIMESTAMP")
    _ensure_column(conn, "facts",      "fact_type",       "TEXT",      "'note'")
    _ensure_column(conn, "facts",      "valid_from",        "REAL",      "(unixepoch())")
    _ensure_column(conn, "facts",      "superseded_at",     "REAL",      "NULL")
    _ensure_column(conn, "facts",      "source_session",    "TEXT",      "NULL")
    _ensure_column(conn, "facts",      "source_hash",       "TEXT",      "NULL")
    _ensure_column(conn, "facts",      "easiness_factor",   "REAL",      "2.5")
    _ensure_column(conn, "facts",      "last_retrieved_at", "REAL",      "NULL")
    _ensure_column(conn, "facts",      "interval_days",     "REAL",      "1.0")
    _ensure_column(conn, "facts",      "entities",          "TEXT",      "NULL")
    _ensure_column(conn, "facts",      "importance",        "REAL",      "0.5")
    _ensure_column(conn, "slot_fills", "project_id",        "TEXT",      "'unknown'")
    _ensure_column(conn, "slot_fills", "created_at",        "TIMESTAMP", "CURRENT_TIMESTAMP")
    _ensure_column(conn, "sessions",   "session_index",     "INTEGER",   "0")

    # One-time backfill: assign sequential session_index per project ordered by enriched_at.
    try:
        projects = conn.execute(
            "SELECT DISTINCT project_id FROM sessions WHERE session_index = 0"
        ).fetchall()
        for (pid,) in projects:
            sids = conn.execute(
                "SELECT session_id FROM sessions WHERE project_id = ? "
                "AND session_index = 0 ORDER BY enriched_at",
                (pid,),
            ).fetchall()
            for idx, (sid,) in enumerate(sids, start=1):
                conn.execute(
                    "UPDATE sessions SET session_index = ? WHERE session_id = ?",
                    (idx, sid),
                )
    except Exception:
        pass

    # Unique constraint so store_slot_fill can upsert instead of always inserting.
    # Must deduplicate first — old insert-always behaviour may have left multiple
    # rows per (project_id, slot_name); SQLite rejects the index if they exist.
    conn.execute(
        """DELETE FROM slot_fills
           WHERE id NOT IN (
               SELECT MAX(id) FROM slot_fills GROUP BY project_id, slot_name
           )"""
    )
    try:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_slot_fills_project_slot "
            "ON slot_fills (project_id, slot_name)"
        )
    except sqlite3.OperationalError:
        pass  # index already exists

    # ── Memory graph ──────────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fact_relations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            fact_id_a   INTEGER NOT NULL,
            fact_id_b   INTEGER NOT NULL,
            relation    TEXT DEFAULT 'related',
            strength    REAL DEFAULT 0.0,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (fact_id_a) REFERENCES facts(id),
            FOREIGN KEY (fact_id_b) REFERENCES facts(id),
            UNIQUE(fact_id_a, fact_id_b)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_relations_a ON fact_relations(fact_id_a)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_relations_b ON fact_relations(fact_id_b)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_live ON facts (superseded_at)"
    )

    # Backfill FTS5 index from any pre-existing facts (one-time, cheap if empty).
    fts_count = conn.execute("SELECT COUNT(*) FROM facts_fts").fetchone()[0]
    facts_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    if fts_count < facts_count:
        conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")

    # Phase 7: One-time migration — re-encode JSON embeddings as binary blobs.
    # JSON rows are str; binary rows are bytes. Safe to run on every init_db():
    # rows already migrated are bytes and are skipped by the isinstance check.
    try:
        for fid, emb_data in conn.execute(
            "SELECT id, embedding FROM facts WHERE embedding IS NOT NULL"
        ).fetchall():
            if isinstance(emb_data, str):
                vec = json.loads(emb_data)
                blob = struct.pack(f"{len(vec)}f", *vec)
                conn.execute("UPDATE facts SET embedding = ? WHERE id = ?", (blob, fid))
    except Exception:
        pass

    conn.commit()
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, col: str,
                   col_type: str, default: str) -> None:
    try:
        conn.execute(f"SELECT {col} FROM {table} LIMIT 1")
    except sqlite3.OperationalError:
        _is_expr = default.startswith("(") and default.endswith(")")
        try:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {col} {col_type} "
                f"DEFAULT {'0' if _is_expr else default}"
            )
        except sqlite3.OperationalError:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {col} {col_type} DEFAULT 0"
            )


def _encode_embedding(vec: list[float]) -> bytes:
    """Pack a float vector into a compact binary blob (struct.pack, 4 bytes/float)."""
    return struct.pack(f"{len(vec)}f", *vec)


def _decode_embedding(blob) -> "list[float] | None":
    """Unpack a binary blob or legacy JSON string into a float list."""
    if blob is None:
        return None
    if isinstance(blob, (bytes, bytearray)):
        n = len(blob) // 4
        return list(struct.unpack(f"{n}f", blob))
    try:
        return json.loads(blob)
    except Exception:
        return None


# ─── Deduplication ───────────────────────────────────────────────────────────

def check_dedup(key: str) -> str:
    conn = init_db()
    row = conn.execute("SELECT key FROM dedup WHERE key = ?", (key,)).fetchone()
    conn.close()
    return "EXISTS" if row else "NEW"


def mark_stored(key: str) -> None:
    conn = init_db()
    conn.execute("INSERT OR IGNORE INTO dedup (key) VALUES (?)", (key,))
    conn.commit()
    conn.close()


# ─── Memory graph ───────────────────────────────────────────────────────────

def _infer_relation(content_a: str, content_b: str, similarity: float) -> str:
    """Infer an edge label from keyword signals in the two fact texts."""
    both = (content_a + " " + content_b).lower()
    if any(k in both for k in ("switch", "migrat", "replac", "instead of")):
        return "contradicts"
    if any(k in both for k in ("fix", "solve", "resolv", "patch")):
        return "fixed_by"
    if any(k in both for k in ("caus", "because", "due to", "result")):
        return "caused_by"
    if any(k in both for k in ("depend", "require", "need", "use")):
        return "depends_on"
    return "related"


def link_facts(
    fact_id_a: int,
    fact_id_b: int,
    relation: str = "related",
    strength: float = 0.0,
) -> None:
    """Create a directed edge between two facts (INSERT OR IGNORE — idempotent)."""
    conn = init_db()
    try:
        conn.execute(
            """INSERT OR IGNORE INTO fact_relations
               (fact_id_a, fact_id_b, relation, strength)
               VALUES (?, ?, ?, ?)""",
            (fact_id_a, fact_id_b, relation, strength),
        )
        conn.commit()
    finally:
        conn.close()


def get_related_facts(fact_id: int, depth: int = 1) -> list[dict]:
    """BFS over the graph starting from fact_id, up to `depth` hops.

    depth=1 returns direct neighbours; depth=2 returns neighbours of neighbours.
    Capped at 2 to avoid context explosion.
    """
    depth = min(depth, 2)
    conn = init_db()
    visited: set[int] = {fact_id}
    results: list[dict] = []
    queue: list[int] = [fact_id]

    for _ in range(depth):
        next_queue: list[int] = []
        for fid in queue:
            rows = conn.execute(
                """SELECT f.id, f.content, f.fact_type, r.relation, r.strength
                   FROM fact_relations r
                   JOIN facts f ON (
                       CASE WHEN r.fact_id_a = ? THEN r.fact_id_b
                            ELSE r.fact_id_a END = f.id
                   )
                   WHERE r.fact_id_a = ? OR r.fact_id_b = ?
                   ORDER BY r.strength DESC""",
                (fid, fid, fid),
            ).fetchall()
            for row_id, content, fact_type, relation, strength in rows:
                if row_id not in visited:
                    visited.add(row_id)
                    next_queue.append(row_id)
                    results.append({
                        "id": row_id,
                        "content": content,
                        "fact_type": fact_type,
                        "relation": relation,
                        "strength": strength,
                    })
        queue = next_queue

    conn.close()
    return results


def get_graph(project_id: str, query: str, depth: int = 1) -> dict:
    """Find the closest fact for `query` and return it with its graph neighbourhood."""
    conn = init_db()
    cursor = conn.execute(
        """SELECT id, content, embedding FROM facts
           WHERE project_id = ? AND superseded_at IS NULL
           ORDER BY id DESC LIMIT 200""",
        (project_id,),
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {"root": None, "neighbours": []}

    query_emb = embed_text(query)
    best_id, best_content, best_sim = None, "", 0.0
    for fid, content, emb_data in rows:
        emb = _decode_embedding(emb_data)
        if emb is None:
            continue
        sim = cosine_similarity(query_emb, emb)
        if sim > best_sim:
            best_sim, best_id, best_content = sim, fid, content

    if best_id is None:
        return {"root": None, "neighbours": []}

    neighbours = get_related_facts(best_id, depth=depth)
    return {
        "root": {"id": best_id, "content": best_content, "similarity": round(best_sim, 4)},
        "neighbours": neighbours,
    }


# ─── Facts (semantic memory) ──────────────────────────────────────────────────

_compacted_this_process = False
_COMPACTED_LOCK = threading.Lock()


def _compact_old_mutations(conn: sqlite3.Connection) -> None:
    """Delete INSERT mutation log entries older than 90 days.

    SUPERSEDE / EXPLICIT_EXPIRE events are kept forever for audit history.
    """
    cutoff = int(time.time()) - 90 * 86400
    conn.execute(
        "DELETE FROM fact_mutations WHERE mutation_type = 'INSERT' AND mutated_at < ?",
        (cutoff,),
    )


def _get_cross_encoder():
    """Lazy-load the MS-MARCO cross-encoder for Phase 4 reranking.

    Returns None if sentence-transformers is not installed or the stub
    utils module (used in tests) does not expose get_cross_encoder.
    """
    try:
        from utils import get_cross_encoder  # noqa: PLC0415
        return get_cross_encoder()
    except (ImportError, AttributeError):
        return None


def _ce_predict_worker(pairs: list, result_queue) -> None:
    """Load cross-encoder in subprocess and predict with timeout killability.

    Must be a top-level function (not nested) to be picklable by multiprocessing.
    Uses fork semantics (Linux default) so the model is inherited copy-on-write
    and does not need to be reloaded from disk.
    """
    try:
        from utils import get_cross_encoder  # noqa: PLC0415
        ce = get_cross_encoder()
        if ce is None:
            return
        result = ce.predict(pairs)
        result_queue.put(result)
    except Exception:
        pass


def store_fact(project_id: str, session_id: str, text: str,
               fact_type: str = "note", enrich: bool = True,
               _embed_text: "str | None" = None,
               _precomputed_emb: "list[float] | None" = None,
               _source_hash: "str | None" = None) -> "int | None":
    """Store a fact with soft-expire on contradiction, binary embedding, entity extraction.

    Contradiction detection: cosine similarity >= _CONTRADICTION_THRESHOLD writes a
    SUPERSEDE mutation, sets superseded_at on the old row, and inserts a new row.
    Phase 3: extracted entities stored as JSON list for entity-overlap retrieval.
    Phase 7: embeddings stored as compact binary blobs (struct.pack, 4 bytes/float).
    Phase 7.5: old INSERT mutations (> 90 days) compacted once per process.

    _embed_text: when provided, the embedding is computed from this string instead of
    `text`.  The DB content column still stores `text`.  Use this in store_turn_window
    to embed only the [curr] turn so ANN search is precise, while storing the full
    3-turn window for context display.

    _precomputed_emb: when provided, skip the embed_text() call entirely and use this
    pre-computed vector.  Takes priority over _embed_text.  Use in store_turn_window
    to share a single embedding between the turn row and the window row (both embed
    the same curr_line text), halving the number of model inference calls.
    """
    global _compacted_this_process
    with _COMPACTED_LOCK:
        if not _compacted_this_process:
            _compact_old_mutations(conn)
            _compacted_this_process = True
    if _precomputed_emb is not None:
        emb = _precomputed_emb
    else:
        emb = embed_text(_embed_text if _embed_text is not None else text)
    conn = init_db()

    # Find the most-similar live fact in this project (last 200).
    # Window facts are intentionally overlapping (sharing 2 of 3 turns) so
    # their cosine similarity is always high — skip the scan to prevent them
    # from chain-superseding each other and destroying historical context.
    # Turn facts embed the same [curr] text as their window counterpart, so
    # consecutive turns from the same speaker would also hit the threshold —
    # skip the scan for turn facts too.
    best_id: int | None = None
    best_sim: float = 0.0
    if fact_type not in ("window", "turn", "llm_atomic"):
        cursor = conn.execute(
            """SELECT id, embedding FROM facts
               WHERE project_id = ? AND superseded_at IS NULL
               ORDER BY id DESC LIMIT 200""",
            (project_id,),
        )
        for row_id, emb_data in cursor.fetchall():
            existing_emb = _decode_embedding(emb_data)
            if existing_emb is None:
                continue
            sim = cosine_similarity(emb, existing_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = row_id

    source_hash = _source_hash if _source_hash is not None else hashlib.sha256(text.encode()).hexdigest()[:16]
    # Skip spaCy NER for window/turn rows — conversational turns contain mostly
    # speaker names (low discriminability) and the call costs ~0.5s each.
    # Window rows store blended 3-turn text; entity overlap signal is noisy.
    # Turn rows are identical speaker:text — entities are subset of window row.
    # fact_type='note'/'finding'/'decision' etc. still get full entity extraction.
    entities = [] if fact_type in ("window", "turn", "llm_atomic") else _extract_entities(text)
    ents_json = json.dumps(entities)
    emb_blob = _encode_embedding(emb)
    # Phase A: importance scoring
    type_weight = _IMPORTANCE_TYPE_WEIGHTS.get(fact_type, 0.3)
    words = text.split()
    entity_density = min(len(entities) / max(len(words), 1) * 5, 1.0)
    kw_boost = 0.2 if any(kw in text.lower() for kw in _IMPORTANCE_KEYWORDS) else 0.0
    importance = min(1.0, type_weight * 0.5 + entity_density * 0.3 + kw_boost * 0.2)
    init_ef = max(1.3, 3.0 - importance)  # high importance -> shorter review interval

    if best_id is not None and best_sim >= _CONTRADICTION_THRESHOLD:
        # Soft-expire: record SUPERSEDE mutation, mark old row superseded, insert new row.
        old_content_row = conn.execute(
            "SELECT content FROM facts WHERE id = ?", (best_id,)
        ).fetchone()
        old_content = old_content_row[0] if old_content_row else ""
        print(
            f"[engram] Replaced contradicting fact (sim={best_sim:.2f}): "
            f'"{old_content[:60]}" \u2192 "{text[:60]}"',
            file=sys.stderr, flush=True,
        )
        conn.execute(
            """INSERT INTO fact_mutations (fact_id, mutation_type, old_content, new_content, session_id)
               VALUES (?, 'SUPERSEDE', ?, ?, ?)""",
            (best_id, old_content, text, session_id),
        )
        conn.execute(
            "UPDATE facts SET superseded_at = unixepoch() WHERE id = ?",
            (best_id,),
        )
        cur = conn.execute(
            """INSERT INTO facts
               (project_id, session_id, content, embedding, fact_type,
                source_session, source_hash, entities, importance, easiness_factor)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (project_id, session_id, text, emb_blob, fact_type,
             session_id, source_hash, ents_json, round(importance, 4), round(init_ef, 4)),
        )
        saved_id = cur.lastrowid
        conn.execute(
            "INSERT INTO facts_fts(rowid, content) VALUES (?, ?)",
            (saved_id, text),
        )
    else:
        # ── Entity enrichment: merge into a recent same-session fact that shares entities ──
        # Avoids storing a disconnected row when the new text is a continuation
        # of an existing same-session fact about the same real-world entity.
        # Skipped when enrich=False (e.g. store_turn_window) or entities are empty.
        enrich_id: "int | None" = None
        if enrich and entities:
            recent_rows = conn.execute(
                """SELECT id, entities FROM facts
                   WHERE project_id = ? AND session_id = ?
                     AND superseded_at IS NULL
                   ORDER BY id DESC LIMIT 10""",
                (project_id, session_id),
            ).fetchall()
            for efid, e_ents_json in recent_rows:
                try:
                    existing_ents = set(json.loads(e_ents_json or "[]"))
                except Exception:
                    existing_ents = set()
                if not (existing_ents & set(entities)):
                    continue
                # Require >=2 shared entities OR ratio >0.3 to avoid false merges
                # on single generic entities (e.g. "Python", "API").
                _shared_e = existing_ents & set(entities)
                _ratio_e  = len(_shared_e) / max(len(existing_ents), len(entities), 1)
                if len(_shared_e) < 2 and _ratio_e <= 0.3:
                    continue
                # Semantic similarity check: only enrich when facts are related.
                existing_emb_row = conn.execute(
                    "SELECT embedding FROM facts WHERE id = ?", (efid,)
                ).fetchone()
                existing_emb = _decode_embedding(existing_emb_row[0]) if existing_emb_row else None
                if existing_emb is None or cosine_similarity(emb, existing_emb) < _ENRICH_MIN_SIM:
                    continue
                existing_row = conn.execute(
                    "SELECT content FROM facts WHERE id = ?", (efid,)
                ).fetchone()
                existing_content = existing_row[0] if existing_row else ""
                if len((existing_content + "\n" + text).split()) <= _ENRICHMENT_MAX_TOKENS:
                    enrich_id = efid
                    break

        if enrich_id is not None:
            old_row = conn.execute(
                "SELECT content, entities FROM facts WHERE id = ?", (enrich_id,)
            ).fetchone()
            old_content   = old_row[0] if old_row else ""
            old_ents_json = old_row[1] if old_row else "[]"
            enriched_content = old_content + "\n" + text
            try:
                merged_ents = list(set(json.loads(old_ents_json or "[]")) | set(entities))
            except Exception:
                merged_ents = entities
            enriched_emb  = embed_text(enriched_content)
            enriched_blob = _encode_embedding(enriched_emb)
            e_words   = enriched_content.split()
            e_density = min(len(merged_ents) / max(len(e_words), 1) * 5, 1.0)
            e_kw      = 0.2 if any(kw in enriched_content.lower() for kw in _IMPORTANCE_KEYWORDS) else 0.0
            e_importance = min(1.0, type_weight * 0.5 + e_density * 0.3 + e_kw * 0.2)
            e_ef         = max(1.3, 3.0 - e_importance)
            # FTS5 external-content: remove old entry before updating facts row.
            conn.execute(
                "INSERT INTO facts_fts(facts_fts, rowid, content) VALUES('delete', ?, ?)",
                (enrich_id, old_content),
            )
            conn.execute(
                """UPDATE facts
                   SET content = ?, embedding = ?, entities = ?,
                       importance = ?, easiness_factor = ?,
                       last_retrieved_at = ?
                   WHERE id = ?""",
                (enriched_content, enriched_blob, json.dumps(merged_ents),
                 round(e_importance, 4), round(e_ef, 4), time.time(), enrich_id),
            )
            conn.execute(
                "INSERT INTO facts_fts(rowid, content) VALUES (?, ?)",
                (enrich_id, enriched_content),
            )
            conn.execute(
                """INSERT INTO fact_mutations
                   (fact_id, mutation_type, old_content, new_content, session_id)
                   VALUES (?, 'ENRICH', ?, ?, ?)""",
                (enrich_id, old_content, enriched_content, session_id),
            )
            saved_id = enrich_id
        else:
            cur = conn.execute(
                """INSERT INTO facts
                   (project_id, session_id, content, embedding, fact_type,
                    source_session, source_hash, entities, importance, easiness_factor)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (project_id, session_id, text, emb_blob, fact_type,
                 session_id, source_hash, ents_json, round(importance, 4), round(init_ef, 4)),
            )
            saved_id = cur.lastrowid
            conn.execute(
                "INSERT INTO facts_fts(rowid, content) VALUES (?, ?)",
                (saved_id, text),
            )
            conn.execute(
                """INSERT INTO fact_mutations (fact_id, mutation_type, new_content, session_id)
                   VALUES (?, 'INSERT', ?, ?)""",
                (saved_id, text, session_id),
            )

    # ── Auto-link: graph edges to semantically related existing facts ─────
    # Skip for window/turn rows: they embed the same [curr] turn text, so
    # adjacent turns hit _RELATION_THRESHOLD for every neighbour (cosine ~0.8+)
    # producing O(n²) inserts into fact_relations. The graph is not consulted
    # by retrieve_facts for these row types anyway (pure cosine ANN).
    # Also skip for llm_atomic: large volume of short facts would produce
    # excessive O(n²) edges and adds no retrieval signal during bulk ingestion.
    if fact_type not in ("window", "turn", "llm_atomic"):
        link_cursor = conn.execute(
            """SELECT id, content, embedding FROM facts
               WHERE project_id = ? AND id != ? AND superseded_at IS NULL
               ORDER BY id DESC LIMIT 50""",
            (project_id, saved_id),
        )
        for neighbor_id, neighbor_content, neighbor_emb_data in link_cursor.fetchall():
            neighbor_emb = _decode_embedding(neighbor_emb_data)
            if neighbor_emb is None:
                continue
            sim = cosine_similarity(emb, neighbor_emb)
            if sim >= _RELATION_THRESHOLD:
                relation = _infer_relation(text, neighbor_content, sim)
                id_a, id_b = min(saved_id, neighbor_id), max(saved_id, neighbor_id)
                conn.execute(
                    """INSERT OR IGNORE INTO fact_relations
                       (fact_id_a, fact_id_b, relation, strength)
                       VALUES (?, ?, ?, ?)""",
                    (id_a, id_b, relation, round(sim, 4)),
                )

    # ── Temporal proximity linking ─────────────────────────────────────────
    # Link to the last _TEMPORAL_MAX_DISTANCE facts in the same session.
    # Bridges conversationally adjacent facts that may be semantically unrelated.
    # Also skip for window/turn rows — temporal links between adjacent turns are
    # already implicit in the sliding window structure; the graph adds no signal.
    # Also skip for llm_atomic — same reasoning; avoids O(n²) edges in bulk.
    if fact_type not in ("window", "turn", "llm_atomic"):
        temporal_neighbors = conn.execute(
            """SELECT id FROM facts
               WHERE project_id = ? AND session_id = ? AND id != ?
                 AND superseded_at IS NULL
               ORDER BY id DESC LIMIT ?""",
            (project_id, session_id, saved_id, _TEMPORAL_MAX_DISTANCE),
        ).fetchall()
        for distance, (neighbor_id,) in enumerate(temporal_neighbors, start=1):
            t_strength = round(max(0.0, 1.0 - (distance - 1) * _TEMPORAL_EDGE_DECAY), 4)
            t_id_a, t_id_b = min(saved_id, neighbor_id), max(saved_id, neighbor_id)
            conn.execute(
                """INSERT INTO fact_relations (fact_id_a, fact_id_b, relation, strength)
                   VALUES (?, ?, 'temporal', ?)
                   ON CONFLICT(fact_id_a, fact_id_b) DO UPDATE SET
                       strength = excluded.strength,
                       relation = 'temporal'
                   WHERE excluded.strength > strength""",
                (t_id_a, t_id_b, t_strength),
            )

    conn.commit()
    conn.close()
    # Invalidate the ANN embedding cache for this project so the next
    # retrieve_facts() call reloads fresh embeddings including this new fact.
    with _CACHE_DIRTY_LOCK:
        _CACHE_DIRTY.add(project_id)
    return saved_id


def _fts5_query(prompt: str) -> str:
    """Sanitize a user prompt into a safe FTS5 MATCH query.

    Strips FTS5 special chars and joins remaining tokens with OR so any
    keyword can match. Returns empty string when no usable tokens remain.
    """
    safe = "".join(c if c.isalnum() or c.isspace() else " " for c in prompt)
    tokens = [t for t in safe.split() if len(t) > 2 and t.lower() not in _BM25_STOPWORDS]
    if not tokens:
        return ""
    # Quote each token to avoid FTS5 keyword collisions (AND/OR/NOT/NEAR).
    return " OR ".join(f'"{t}"' for t in tokens)


def _decompose_query(query: str) -> list[str]:
    """ToR-Lite query decomposition via embedding similarity valleys.

    Embeds each word independently, computes sliding-window similarity,
    splits at valleys (sim < 0.7), snaps splits to nearest conjunction,
    and returns sub-queries.  Short queries (<=4 words) pass through unchanged.
    Returns [query] when only one sub-query would be produced.
    """
    words = query.strip().split()
    if len(words) <= 4:
        return [query]

    word_embs = [embed_text(w) for w in words]
    similarities = [
        cosine_similarity(word_embs[i], word_embs[i + 1])
        for i in range(len(words) - 1)
    ]

    splits = [i for i, sim in enumerate(similarities) if sim < 0.7]

    conjunctions = {"and", "but", "or", "then", "before", "after"}
    final_splits: list[int] = []
    for split in splits:
        snapped = False
        for offset in range(-2, 3):
            idx = split + offset
            if 0 <= idx < len(words) and words[idx].lower() in conjunctions:
                final_splits.append(idx)
                snapped = True
                break
        if not snapped:
            final_splits.append(split)

    sub_queries: list[str] = []
    prev = 0
    for split in sorted(set(final_splits)):
        sub = " ".join(words[prev:split + 1])
        if len(sub.split()) >= 3:
            sub_queries.append(sub)
        prev = split + 1
    sub_queries.append(" ".join(words[prev:]))

    return sub_queries if len(sub_queries) > 1 else [query]


def _session_recency_score(
    fact_session_id: "str | None",
    current_session_id: str,
    session_idx_map: dict,
) -> float:
    """Return a session proximity score in [0.0, 1.0].

    1.0 when the fact comes from the current session; decays by
    _SESSION_RECENCY_DECAY per session gap; 0.0 at _SESSION_MAX_LOOKBACK or beyond.
    Returns 0.5 (neutral) when either session is unknown so facts stored
    before session_index tracking was added are not penalised.
    """
    if fact_session_id == current_session_id:
        return 1.0
    if not fact_session_id:
        return 0.5
    fact_idx = session_idx_map.get(fact_session_id)
    curr_idx = session_idx_map.get(current_session_id)
    if fact_idx is None or curr_idx is None:
        return 0.5
    gap = abs(curr_idx - fact_idx)
    if gap >= _SESSION_MAX_LOOKBACK:
        return 0.0
    return max(0.0, 1.0 - gap * _SESSION_RECENCY_DECAY)


def store_turn_window(
    project_id: str,
    session_id: str,
    turns: list,
    current_index: int,
    fact_type: str = "window",
    extract_svo: bool = True,
    store_turn: bool = True,
    _precomputed_curr_emb: "list[float] | None" = None,
) -> "int | None":
    """Store a 3-turn sliding window centred on current_index as a single fact.

    Tags: [prev] for preceding turn, [curr] for current, [next] for following.
    Each turn dict must have 'speaker' and 'text' keys.

    Designed for batch ingestion of conversational data (e.g. LoCoMo eval).
    For real-time coding sessions, use store_fact() directly.

    Semantic duplication across overlapping windows is intentional — different
    neighbouring context yields different embeddings and different retrieval matches.

    Both the window row and the companion turn row are embedded from the [curr]
    turn text only, not the full 3-turn window.  ANN search must match the
    retrieval question against the specific turn that contains the answer;
    blending three speakers into one vector degrades recall.

    extract_svo: when False, skip spaCy SVO extraction.  Pass False during
    bulk benchmark ingestion — conversational text yields <1% SVO triples but
    the spaCy calls dominate ingestion time (~0.5s each × 5882 turns = 3200s).

    store_turn: when False, skip the companion clean-turn row.  During benchmark
    ingestion with mode B the window row already provides ANN recall; the turn
    row adds a second embed call that is redundant.  Defaults to True so that
    production and test behaviour are unchanged.

    _precomputed_curr_emb: optional pre-computed embedding for curr_line (from
    caller's batch embed pass).  Skips one embed_text() call per turn.

    Returns the fact_id of the stored or enriched fact.
    """
    window: list[str] = []
    curr_line: str = ""
    for i in range(max(0, current_index - 1), min(len(turns), current_index + 2)):
        turn = turns[i]
        tag = "[curr]" if i == current_index else ("[prev]" if i < current_index else "[next]")
        line = f"{tag} {turn['speaker']}: {turn['text']}"
        window.append(line)
        if i == current_index:
            curr_line = f"{turn['speaker']}: {turn['text']}"
    content = "\n".join(window)
    # Both the turn row and the window row embed curr_line only — ANN search must
    # match the question against the specific turn that contains the answer.
    # Embedding the full [prev][curr][next] blends three speakers' text into one
    # vector and degrades recall (validated: R@40 unchanged, F1 drops).
    # Accept a pre-computed embedding from a batch pass (benchmark ingest speed).
    window_emb = (
        _precomputed_curr_emb
        if _precomputed_curr_emb is not None
        else (embed_text(curr_line) if curr_line else embed_text(content))
    )
    turn_emb = window_emb
    # Store the clean single-turn fact FIRST (fact_type="turn") so it gets a lower
    # row id than the window row.  Tests that fetch ORDER BY id DESC LIMIT 1 will
    # get the window row (richer context); retrieval can return either row.
    # store_turn=False skips this during benchmark ingestion (window row is sufficient).
    if curr_line and store_turn:
        store_fact(project_id, session_id, curr_line, "turn", enrich=False,
                   _precomputed_emb=turn_emb)
    # Window row stored second — last inserted, richer 3-turn context for display.
    window_fid = store_fact(project_id, session_id, content, fact_type, enrich=False,
                            _precomputed_emb=window_emb)
    # Populate derived FTS: lemmatized + WordNet-expanded text for the [curr] turn.
    # Bridges vocab gap between conversational text and information-seeking queries.
    if window_fid and curr_line:
        _store_derived_fts(window_fid, curr_line)
    # LLM atomic fact extraction (opt-in via PREFLIGHT_USE_LLM_EXTRACTOR=1).
    # Requires Ollama running locally with qwen2.5:1.5b pulled.
    # Each extracted fact is stored with the turn's content hash as source_hash
    # so build_dia_id_map can link llm_atomic fids back to their source dia_id.
    # Silently skips (never raises) if Ollama is unavailable.
    # Run in a daemon thread so live MCP calls are not blocked waiting for Ollama.
    # For batch ingestion (recall_ablation --reingest) this is also safe: the
    # outer loop stays alive until all turns are processed, giving threads time
    # to complete; eval queries happen only after the loop exits.
    if _USE_LLM_EXTRACTOR and curr_line:
        import threading as _threading  # noqa: PLC0415

        def _run_llm_extract(_cl=curr_line, _pid=project_id, _sid=session_id):
            try:
                from llm_extractor import extract_atomic_facts as _llm_extract  # noqa: PLC0415
                _turn_hash = hashlib.sha256(_cl.encode()).hexdigest()[:16]
                for _fact_text in _llm_extract(_cl):
                    _fact_emb = embed_text(_fact_text)
                    store_fact(
                        _pid, _sid, _fact_text, "llm_atomic",
                        enrich=False, _precomputed_emb=_fact_emb,
                        _source_hash=_turn_hash,
                    )
            except Exception:
                pass  # never let extractor errors affect raw storage

        _threading.Thread(target=_run_llm_extract, daemon=True).start()
    # Extract and store atomic SVO facts for higher-precision retrieval.
    # Skip during bulk ingestion (extract_svo=False) — spaCy is ~0.5s/window.
    if extract_svo:
        svo_facts = _extract_svo_facts(content)
        for svo in svo_facts:
            store_fact(project_id, session_id, svo, "note", enrich=True)
    return window_fid


def retrieve_facts(
    project_id: str,
    session_id: str,
    prompt: str,
    top_n: int = 3,
    threshold: float = 0.25,
    include_budget_info: bool = False,
    max_tokens: int = 2000,
    _gold_fid: "int | None" = None,
) -> "list[str] | dict":
    """Hybrid BM25 + vector + entity-overlap retrieval via three-way RRF.

    Score = 0.40*rrf + 0.25*recency + 0.15*freq + 0.20*staleness

    Phase 2 SM-2 gate: only facts whose spaced-repetition interval has elapsed
    are ranked. Soft-relax: gate dropped when fewer than 3 facts would pass.
    Phase 3: entity-overlap adds a third RRF signal.
    Phase 4: cross-encoder reranks top-20 when sentence-transformers is loaded
    (500ms latency cap; falls back to RRF-only on timeout or missing dep).
    Phase 5: greedy token budget; returns list[str] or dict per include_budget_info.
    """
    now_ts = time.time()
    conn = init_db()

    # ── 1. Pull candidate pool: Pool A (ANN cosine) + Pool B (proven useful) ─
    # Pool A: top-_POOL_A_LIMIT by cosine similarity to query — query-anchored, not
    #         time-anchored. Falls back to recency order if numpy unavailable.
    # Pool B: proven-useful facts (retrieval_count > 0), capped at _POOL_B_LIMIT.
    # Separate caps prevent Pool B from drowning Pool A's relevance results.
    _COLS = (
        "id, content, embedding, retrieval_count, created_at, fact_type, "
        "easiness_factor, last_retrieved_at, interval_days, entities, "
        "COALESCE(importance, 0.5), session_id"
    )
    _WHERE = (
        "project_id = ? AND superseded_at IS NULL"
    )

    # ── Pool A: ANN via in-process embedding cache ────────────────────────
    # Cache stores (fid_list, emb_matrix) per project. Rebuilt when dirty.
    pool_a: list = []
    if _NUMPY_AVAILABLE:
        with _EMB_CACHE_LOCK:
            if project_id in _CACHE_DIRTY or project_id not in _EMB_CACHE:
                # Load all live fact IDs + embeddings for this project.
                cache_rows = conn.execute(
                    "SELECT id, embedding FROM facts WHERE project_id = ? "
                    "AND superseded_at IS NULL",
                    (project_id,),
                ).fetchall()
                cache_fids: list[int] = []
                cache_vecs: list = []
                for cfid, cemb_blob in cache_rows:
                    vec = _decode_embedding(cemb_blob)
                    if vec is not None:
                        cache_fids.append(cfid)
                        cache_vecs.append(vec)
                if cache_vecs:
                    mat = np.array(cache_vecs, dtype=np.float32)
                    # Normalise rows so dot product == cosine similarity.
                    norms = np.linalg.norm(mat, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    mat = mat / norms
                    _EMB_CACHE[project_id] = (cache_fids, mat)
                    _EMB_CACHE.move_to_end(project_id)
                    # LRU eviction: drop least recently used project cache.
                    while len(_EMB_CACHE) > _EMB_CACHE_MAX_SIZE:
                        _EMB_CACHE.popitem(last=False)
                else:
                    _EMB_CACHE[project_id] = ([], None)
                with _CACHE_DIRTY_LOCK:
                    _CACHE_DIRTY.discard(project_id)

        with _EMB_CACHE_LOCK:
            cached_fids, cached_mat = _EMB_CACHE.get(project_id, ([], None))
        if project_id in _EMB_CACHE:
            with _EMB_CACHE_LOCK:
                _EMB_CACHE.move_to_end(project_id)
        if cached_mat is not None and len(cached_fids) > 0:
            # Embed the bare prompt for ANN pool selection.
            # (augmented_prompt is built later after pool; vector ranking step uses it.)
            prompt_emb_raw = embed_text(prompt)
            qvec = np.array(prompt_emb_raw, dtype=np.float32)
            qnorm = np.linalg.norm(qvec)
            if qnorm > 0:
                qvec = qvec / qnorm
            sims = cached_mat @ qvec          # shape (N,), cosine similarity
            top_k = len(cached_fids)
            top_indices = np.argpartition(sims, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
            top_fids = [cached_fids[i] for i in top_indices]
            if top_fids:
                placeholders = ",".join("?" for _ in top_fids)
                pool_a = conn.execute(
                    f"SELECT {_COLS} FROM facts WHERE id IN ({placeholders}) "
                    f"AND superseded_at IS NULL",
                    top_fids,
                ).fetchall()
                # Re-sort to match cosine similarity order.
                fid_order = {fid: rank for rank, fid in enumerate(top_fids)}
                pool_a = sorted(pool_a, key=lambda r: fid_order.get(r[0], 9999))

    if not pool_a:
        # Fallback: recency order (original behaviour), used when numpy unavailable
        # or cache is empty (no facts stored yet).
        pool_a = conn.execute(
            f"SELECT {_COLS} FROM facts WHERE {_WHERE}",
            (project_id,),
        ).fetchall()

    pool_a_ids = {r[0] for r in pool_a}
    if not _RETRIEVE_BENCHMARK:
        pool_b_raw = conn.execute(
            f"SELECT {_COLS} FROM facts WHERE {_WHERE} AND retrieval_count > 0 "
            f"ORDER BY retrieval_count DESC, last_retrieved_at DESC",
            (project_id,),
        ).fetchall()
        pool_b = [r for r in pool_b_raw if r[0] not in pool_a_ids][:_POOL_B_LIMIT]
        all_candidate_rows = pool_a + pool_b
    else:
        all_candidate_rows = pool_a

    # Diagnostic stage tracking (only active when _gold_fid is provided).
    _stages: dict = {}
    if _gold_fid is not None:
        _pool_fids = [r[0] for r in all_candidate_rows]
        _stages["pool_pos"] = _pool_fids.index(_gold_fid) if _gold_fid in _pool_fids else -1
        _stages["pool_size"] = len(_pool_fids)

    rows: list = []
    def _safe_ef(v, default=2.5):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, (bytes, bytearray)) and len(v) == 4:
            import struct as _struct
            return _struct.unpack("f", v)[0]
        return default
    for fid, content, emb_data, rc, ca, ft, ef, lra, ivd, ents, imp, fsid in all_candidate_rows:
        emb = _decode_embedding(emb_data)
        if emb is None:
            continue
        rows.append((
            fid, content, emb,
            rc or 0, ca, ft or "note",
            _safe_ef(ef) if ef is not None else 2.5,
            lra,
            ivd if ivd is not None else 1.0,
            ents,
            imp if imp is not None else 0.5,
            fsid,
        ))

    if not rows:
        conn.close()
        if include_budget_info:
            return {"facts": [], "budget_hit": False, "retrieved_count": 0, "total_candidates": 0}
        return []

    max_rc_row = conn.execute(
        "SELECT COALESCE(MAX(retrieval_count), 1) FROM facts WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    max_rc = max_rc_row[0] if max_rc_row and max_rc_row[0] else 1

    # Phase B: augment vector query with slot fills (BM25 uses bare prompt).
    if _RETRIEVE_BENCHMARK:
        augmented_prompt = prompt
    else:
        try:
            slot_rows = conn.execute(
                "SELECT slot_name, value FROM slot_fills WHERE project_id = ? LIMIT 5",
                (project_id,),
            ).fetchall()
            augmented_prompt = (
                " ".join(f"{k}={str(v)[:50]}" for k, v in slot_rows) + ": " + prompt
                if slot_rows else prompt
            )
        except Exception:
            augmented_prompt = prompt

    # ── 2-X. Query decomposition ─────────────────────────────────────────
    sub_queries = _decompose_query(prompt) if _USE_QUERY_DECOMPOSITION else [prompt]

    emb_by_fid: dict[int, list] = {fid: emb for fid, _c, emb, *_ in rows}
    content_by_fid: dict[int, str] = {fid: content for fid, content, *_ in rows}
    candidate_ids = tuple(fid for fid, *_ in rows)
    n_facts = len(rows)

    # Multi-query ranking: accumulate max RRF per fact across all sub-queries.
    # For single-query mode (sub_queries == [prompt]), this behaves identically
    # to the original pipeline.
    raw_rrf: dict[int, float] = {}
    _broad_parts: list[int] = []
    bm25_rank: dict[int, int] = {}
    phrase_fids: set[int] = set()
    derived_rank: dict[int, int] = {}
    entity_rank: dict[int, int] = {}
    context_rank: dict[int, int] = {}
    vec_rank: dict[int, int] = {}

    for sq in sub_queries:
        augmented_sq = augmented_prompt.replace(prompt, sq, 1) if len(sub_queries) > 1 else augmented_prompt

        # Vector ranking
        sq_emb = embed_text(augmented_sq)
        sq_vec_scored = sorted(
            ((cosine_similarity(sq_emb, emb), fid) for fid, emb in emb_by_fid.items()),
            reverse=True,
        )
        sq_vec_rank: dict[int, int] = {fid: rank for rank, (_, fid) in enumerate(sq_vec_scored)}

        # BM25
        sq_bm25_rank: dict[int, int] = {}
        sq_phrase_fids: set[int] = set()
        sq_fts = _fts5_query(sq)
        if sq_fts:
            placeholders = ",".join("?" for _ in candidate_ids)
            try:
                bm_cursor = conn.execute(
                    f"""SELECT rowid FROM facts_fts
                        WHERE facts_fts MATCH ?
                          AND rowid IN ({placeholders})
                        ORDER BY bm25(facts_fts)""",
                    (sq_fts, *candidate_ids),
                )
                for rank, (fid,) in enumerate(bm_cursor.fetchall()):
                    sq_bm25_rank[fid] = rank
            except sqlite3.OperationalError:
                pass

            sq_words = sq.strip().split()
            if len(sq_words) >= 2:
                ph_q = f'"{sq.strip()}"'
                try:
                    ph_cursor = conn.execute(
                        f"""SELECT rowid FROM facts_fts
                            WHERE facts_fts MATCH ?
                              AND rowid IN ({placeholders})""",
                        (ph_q, *candidate_ids),
                    )
                    sq_phrase_fids = {row[0] for row in ph_cursor.fetchall()}
                except sqlite3.OperationalError:
                    pass

        # Derived BM25
        sq_derived_rank: dict[int, int] = {}
        if _USE_DERIVED_BM25 and sq_fts:
            try:
                derived_q = _build_derived_text(sq)
                safe_d = "".join(c if c.isalnum() or c.isspace() else " " for c in derived_q)
                dtokens = [t for t in safe_d.split() if len(t) > 2]
                if dtokens:
                    dfts_q = " OR ".join(f'"{t}"' for t in dtokens)
                    dr_rows = conn.execute(
                        "SELECT rowid FROM facts_derived_fts"
                        " WHERE facts_derived_fts MATCH ? ORDER BY bm25(facts_derived_fts)",
                        (dfts_q,),
                    ).fetchall()
                    dr_r = 0
                    fids_set_d = set(candidate_ids)
                    for (dfid,) in dr_rows:
                        if dfid in fids_set_d:
                            sq_derived_rank[dfid] = dr_r
                            dr_r += 1
            except Exception:
                pass

        # Entity overlap
        sq_entity_rank: dict[int, int] = {}
        if not _RETRIEVE_BENCHMARK:
            try:
                prompt_ents = set(e.lower() for e in _extract_entities(sq))
                if prompt_ents:
                    ent_scores = []
                    for fidi, _c, _e, _rc, _ca, _ft, _ef, _lra, _ivd, ents_json, _imp, _fsid in rows:
                        try:
                            fact_ents = set(e.lower() for e in json.loads(ents_json or "[]"))
                        except Exception:
                            fact_ents = set()
                        shared = prompt_ents & fact_ents
                        n_shared = len(shared)
                        ratio = n_shared / max(len(prompt_ents), len(fact_ents), 1)
                        if n_shared >= 2 or ratio > 0.3:
                            overlap = ratio
                        else:
                            overlap = 0.0
                        ent_scores.append((overlap, fidi))
                    ent_scores.sort(reverse=True)
                    sq_entity_rank = {fidi: rank for rank, (_, fidi) in enumerate(ent_scores)}
            except Exception:
                pass

        # Context BM25
        sq_context_rank: dict[int, int] = {}
        if _USE_CONTEXT_BM25:
            try:
                _q_ctx_tokens = [
                    t for t in re.sub(r'[^a-z\s]', ' ', sq.lower()).split()
                    if len(t) > 2 and t not in _BM25_STOPWORDS
                ]
                if _q_ctx_tokens:
                    _ctx_content: dict[int, str] = {r[0]: r[1] for r in all_candidate_rows}
                    _pool_min = min(r[0] for r in all_candidate_rows)
                    _pool_max = max(r[0] for r in all_candidate_rows)
                    _missing_fids = set()
                    for _fid, *_ in rows:
                        for _off in range(-_CONTEXT_WINDOW_SIZE, _CONTEXT_WINDOW_SIZE + 1):
                            _nfid = _fid + _off
                            if _pool_min <= _nfid <= _pool_max and _nfid not in _ctx_content:
                                _missing_fids.add(_nfid)
                    if _missing_fids:
                        _ph = ",".join("?" for _ in _missing_fids)
                        try:
                            for _mfid, _mcontent in conn.execute(
                                f"SELECT id, content FROM facts WHERE id IN ({_ph}) "
                                "AND superseded_at IS NULL",
                                list(_missing_fids),
                            ).fetchall():
                                _ctx_content[_mfid] = _mcontent
                        except Exception:
                            pass
                    _ctx_scores: dict[int, int] = {}
                    for _fid, _content, *_ in rows:
                        _window_parts = [_content]
                        for _off in range(-_CONTEXT_WINDOW_SIZE, _CONTEXT_WINDOW_SIZE + 1):
                            if _off == 0:
                                continue
                            _nc = _ctx_content.get(_fid + _off)
                            if _nc:
                                _window_parts.append(_nc)
                        _window_text = " ".join(_window_parts).lower()
                        _ctx_score = sum(1 for _t in _q_ctx_tokens if _t in _window_text)
                        if _ctx_score:
                            _ctx_scores[_fid] = _ctx_score
                    if _ctx_scores:
                        _ctx_ranked = sorted(_ctx_scores, key=_ctx_scores.__getitem__, reverse=True)
                        sq_context_rank = {fid: i for i, fid in enumerate(_ctx_ranked)}
            except Exception:
                pass

        # RRF for this sub-query — accumulate max per fact across sub-queries
        for fid, *_ in rows:
            s = 1.0 / (_RRF_K + sq_vec_rank.get(fid, n_facts))
            if fid in sq_bm25_rank:
                bm25_component = 1.0 / (_RRF_K + sq_bm25_rank[fid])
                if fid in sq_phrase_fids:
                    bm25_component *= 1.5
                s += bm25_component
            if _USE_DERIVED_BM25 and fid in sq_derived_rank:
                s += 1.0 / (_RRF_K + sq_derived_rank[fid])
            if _USE_CONTEXT_BM25 and fid in sq_context_rank:
                s += 1.0 / (_RRF_K + sq_context_rank[fid])
            if fid in sq_entity_rank:
                s += 1.0 / (_RRF_K + sq_entity_rank[fid])
            if fid not in raw_rrf or s > raw_rrf[fid]:
                raw_rrf[fid] = s
            # Track per-signal best ranks for broad pool (use first sub-query's ranks for simplicity)
            if len(sub_queries) == 1:
                vec_rank[fid] = sq_vec_rank.get(fid, n_facts)
                if fid in sq_bm25_rank:
                    bm25_rank[fid] = sq_bm25_rank[fid]
                if fid in sq_phrase_fids:
                    phrase_fids.add(fid)
                if fid in sq_derived_rank:
                    derived_rank[fid] = sq_derived_rank[fid]
                if fid in sq_entity_rank:
                    entity_rank[fid] = sq_entity_rank[fid]
                if fid in sq_context_rank:
                    context_rank[fid] = sq_context_rank[fid]

    # For decomposed queries, rebuild per-signal ranks from merged RRF for broad pool
    if len(sub_queries) > 1:
        vec_rank = {fid: rank for rank, fid in enumerate(sorted(raw_rrf, key=raw_rrf.__getitem__, reverse=True))}

    max_rrf = max(raw_rrf.values()) if raw_rrf else 1.0

    # ── 7. Multi-signal broad pool ────────────────────────────────────────
    if _BROAD_POOL > 0:
        _cos_order = sorted((fid for fid, *_ in rows), key=lambda f: vec_rank.get(f, n_facts))
        _bm25_order = sorted((fid for fid, *_ in rows), key=lambda f: bm25_rank.get(f, n_facts))
        _broad_parts = list(_cos_order[:_BROAD_POOL]) + _bm25_order[:_BROAD_POOL]
        if _USE_DERIVED_BM25:
            _broad_parts += sorted(
                (fid for fid, *_ in rows), key=lambda f: derived_rank.get(f, n_facts)
            )[:_BROAD_POOL]
        if _USE_CONTEXT_BM25:
            _broad_parts += sorted(
                (fid for fid, *_ in rows), key=lambda f: context_rank.get(f, n_facts)
            )[:_BROAD_POOL]
        if _USE_LEXICAL_CHANNELS:
            import re as _re_lx
            _STOPNAME_LX = frozenset({
                'The', 'What', 'Who', 'When', 'Where', 'How', 'Does', 'Did',
                'Was', 'Are', 'Can', 'Will', 'Is', 'Do', 'Has', 'Have', 'Had',
                'Would', 'Could', 'Should', 'Which', 'Why', 'That', 'This',
                'His', 'Her', 'Its', 'Our', 'Their', 'Then', 'Than', 'From',
            })
            _name_toks = [w for w in _re_lx.findall(r'\b[A-Z][a-z]{2,}\b', prompt)
                          if w not in _STOPNAME_LX]
            if _name_toks:
                _name_sc: dict[int, int] = {}
                for _fid, _c, *_ in rows:
                    _s = sum(_c.count(_n) for _n in _name_toks)
                    if _s:
                        _name_sc[_fid] = _s
                _broad_parts += sorted(_name_sc, key=_name_sc.__getitem__, reverse=True)[:_BROAD_POOL]
            _MONTH_RE = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
            _date_toks = list(dict.fromkeys(
                _re_lx.findall(rf'\b{_MONTH_RE}\s+\d{{4}}\b|\b\d{{4}}\b', prompt)
            ))
            if _date_toks:
                _date_sc: dict[int, int] = {}
                for _fid, _c, *_ in rows:
                    _s = sum(_c.lower().count(_d.lower()) for _d in _date_toks)
                    if _s:
                        _date_sc[_fid] = _s
                _broad_parts += sorted(_date_sc, key=_date_sc.__getitem__, reverse=True)[:_BROAD_POOL]
            _q_words_lx = [w for w in _re_lx.sub(r'[^a-z\s]', ' ', prompt.lower()).split()
                           if len(w) > 2 and w not in _BM25_STOPWORDS]
            if len(_q_words_lx) >= 2:
                _bigrams_lx = [f"{_q_words_lx[i]} {_q_words_lx[i+1]}"
                               for i in range(len(_q_words_lx) - 1)]
                _bgram_hits: list[int] = []
                try:
                    _phrase_q = " OR ".join(f'"{bg}"' for bg in _bigrams_lx)
                    _br_rows = conn.execute(
                        "SELECT rowid FROM facts_fts WHERE facts_fts MATCH ?",
                        (_phrase_q,),
                    ).fetchall()
                    _fids_set_b = set(candidate_ids)
                    for (_bfid,) in _br_rows:
                        if _bfid in _fids_set_b:
                            _bgram_hits.append(_bfid)
                except Exception:
                    _bgram_hits = []
                if not _bgram_hits:
                    for _fid, _c, *_ in rows:
                        _cl = _c.lower()
                        if any(_bg in _cl for _bg in _bigrams_lx):
                            _bgram_hits.append(_fid)
                _broad_parts += _bgram_hits[:_BROAD_POOL]
        _broad_cands = list(dict.fromkeys(_broad_parts))
        _broad_set = set(_broad_cands)
    else:
        _broad_cands = [fid for fid, *_ in rows]
        _broad_set = set(_broad_cands)

    _broad_sorted = sorted(_broad_cands, key=lambda f: raw_rrf.get(f, 0.0), reverse=True)
    _tail_sorted = sorted(
        (fid for fid, *_ in rows if fid not in _broad_set),
        key=lambda f: raw_rrf.get(f, 0.0), reverse=True
    )
    _rerank_order = _broad_sorted + _tail_sorted
    _pre_score_order = list(_rerank_order) if _COVERAGE_K > 0 else None

    # ── 8. Combined score per fact ─────────────────────────────────────────
    if _RETRIEVE_BENCHMARK:
        # Benchmark mode: use pure RRF order as scores (no composite scoring).
        scored_all = [
            (raw_rrf.get(fid, 0.0), fid, content_by_fid[fid], 2.5, None, 0)
            for fid in _rerank_order
        ]
    else:
        # Preload session indices for session_recency scoring (one DB read, not per-fact).
        session_idx_map: dict[str, int] = {}
        try:
            for _sid, _sidx in conn.execute(
                "SELECT session_id, session_index FROM sessions WHERE project_id = ?",
                (project_id,),
            ).fetchall():
                session_idx_map[_sid] = _sidx
        except Exception:
            pass

        now = datetime.now(timezone.utc)
        row_by_fid: dict[int, tuple] = {r[0]: r for r in rows}
        scored_all: list = []
        for fid in _rerank_order:
            r = row_by_fid.get(fid)
            if r is None:
                continue
            _fid, content, _emb, rc, ca, ft, ef, lra, ivd, _ents, _imp, fsid = r
            rrf = (raw_rrf[fid] / max_rrf) if max_rrf > 0 else 0.0
            try:
                ca_dt = _parse_dt(ca)
            except Exception:
                ca_dt = now
            days = max(0, (now - ca_dt).days)
            decay = _DECAY_RATES.get(ft, _DECAY_RATES["note"])
            recency = 1.0 / (1.0 + decay * days) if decay > 0 else 1.0
            freq = (rc / max_rc) if max_rc > 0 else 0.0
            staleness = min((now_ts - lra) / (30 * 86400), 1.0) if lra is not None else 1.0
            session_rec = _session_recency_score(fsid, session_id, session_idx_map)
            # Weights sum to 1.0.
            score = 0.35 * rrf + 0.20 * recency + 0.20 * staleness + 0.15 * session_rec + 0.10 * freq
            # Window demotion (adjustable via PREFLIGHT_WINDOW_DEMOTION, default 1.0 = no demotion).
            if ft == "window":
                score *= _WINDOW_DEMOTION
            # Speaker boost: if a speaker name appears in the question AND this is a turn
            # fact whose content starts with that speaker, boost by 1.3x.
            if ft == "turn" and ": " in content:
                turn_speaker = content.split(":", 1)[0].strip()
                if turn_speaker and turn_speaker.lower() in prompt.lower():
                    score *= 1.3
            scored_all.append((score, fid, content, ef, lra, rc))

        # Coverage guard: min-rank ensemble of score rank and pre-score (RRF) rank.
        # Ensures R@K cannot regress below the baseline ordering.
        if _COVERAGE_K > 0 and _pre_score_order is not None:
            _score_rank = {r[1]: i for i, r in enumerate(scored_all)}
            _baseline_rank = {fid: i for i, fid in enumerate(_pre_score_order)}
            _n_ids = len(scored_all)
            scored_all.sort(key=lambda r: min(
                _score_rank.get(r[1], _n_ids),
                _baseline_rank.get(r[1], _n_ids),
            ))

    # Stage 1: post-scoring position.
    if _gold_fid is not None:
        _sa_fids = [r[1] for r in scored_all]
        _stages["scored_pos"] = _sa_fids.index(_gold_fid) if _gold_fid in _sa_fids else -1
        _stages["scored_size"] = len(_sa_fids)

    # ── 7. SM-2 gate (disabled by default — _SM2_GATE_ENABLED=False) ─────
    # Gate is skipped because unseen facts (retrieval_count=0) would be blocked
    # before scoring. EF/interval fields still update on retrieval for staleness.
    def _safe_ivd(v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, (bytes, bytearray)) and len(v) == 4:
            import struct as _si
            return float(_si.unpack("f", v)[0])
        return 1.0

    _lra_by_fid = {r[0]: r[7] for r in rows}
    _ivd_by_fid = {r[0]: _safe_ivd(r[8]) for r in rows}
    _ef_by_fid  = {r[0]: float(r[6]) for r in rows}
    _sta_by_fid = {r[0]: min((now_ts - r[7]) / (30 * 86400), 1.0)
                   if r[7] is not None else 1.0 for r in rows}
    _rc_by_fid  = {r[0]: r[3] for r in rows}

    if _SM2_GATE_ENABLED:
        def _is_due(fid: int) -> bool:
            lra = _lra_by_fid.get(fid)
            ivd = _ivd_by_fid.get(fid, 1.0)
            if _rc_by_fid.get(fid, 0) == 0 or lra is None:
                return True
            if _ef_by_fid.get(fid, 2.5) <= 1.5 and _sta_by_fid.get(fid, 1.0) > 0.5:
                return True
            return (now_ts - lra) >= (ivd * 86400)
        due_scored = [row for row in scored_all if _is_due(row[1])]
        scored = due_scored if len(due_scored) >= 3 else scored_all
    else:
        scored = scored_all

    total_candidates = len(scored)

    # Stage 2: post-gate position.
    if _gold_fid is not None:
        _g_fids = [r[1] for r in scored]
        _stages["gated_pos"] = _g_fids.index(_gold_fid) if _gold_fid in _g_fids else -1
        _stages["gated_size"] = len(_g_fids)

    # ── 9. Cross-encoder reranking (Phase 4) — bigger pool, clean [curr] text ─
    # CE extracts only the [curr] line from window facts — the full [prev]/[curr]/[next]
    # format confuses cross-encoder models trained on clean (query, passage) pairs.
    # Pool size controlled by _CE_POOL_SIZE (default 120; benchmark found 200 optimal).
    quality_by_fid: dict[int, float] = {}
    _pre_ce_order = list(scored) if _CE_GUARD_K > 0 else None
    try:
        cross_enc = _get_cross_encoder()
        if cross_enc is not None and len(scored) > 5 and _CE_POOL_SIZE > 0:
            ce_pool = scored[:_CE_POOL_SIZE]
            if not ce_pool:
                raise StopIteration  # skip CE — empty pool
            # BM25 anchor: guarantee top-5 BM25 facts enter the CE pool even if
            # composite scoring pushed them past the pool boundary.
            if bm25_rank:
                pool_fids = {r[1] for r in ce_pool}
                bm25_anchors = [
                    row for row in scored[_CE_POOL_SIZE:]
                    if row[1] in bm25_rank and bm25_rank[row[1]] < 5
                    and row[1] not in pool_fids
                ]
                if bm25_anchors:
                    ce_pool = ce_pool + bm25_anchors
            ce_pool_fids = {r[1] for r in ce_pool}

            # Extract [curr] line for CE input (full window format confuses CE)
            from functools import lru_cache as _lru_cache

            @_lru_cache(maxsize=256)
            def _curr_text(raw: str) -> str:
                for ln in raw.split("\n"):
                    if ln.startswith("[curr] "):
                        return ln[len("[curr] "):]
                return raw

            pairs = [(prompt, _curr_text(c)) for _, _, c, *_ in ce_pool]

            # Run CE in a subprocess so it can be killed if it exceeds the timeout.
            # Direct cross_enc.predict() hangs cannot be interrupted — the model
            # call is in C (ONNX Runtime) and Python-level timeout doesn't help.
            # Skip in benchmark mode and when pool is empty to avoid 5s subprocess
            # fork overhead on every query.
            _ce_queue: mp.Queue = mp.Queue()
            _ce_proc = mp.Process(target=_ce_predict_worker, args=(pairs, _ce_queue))
            _ce_proc.start()
            _ce_proc.join(timeout=_CE_TIMEOUT)
            ce_raw = None
            if _ce_proc.is_alive():
                _ce_proc.terminate()
                _ce_proc.join()
            else:
                try:
                    ce_raw = _ce_queue.get_nowait()
                except Exception:
                    pass

            if ce_raw is not None:
                ce_min = min(ce_raw)
                ce_max = max(ce_raw)
                ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
                for i, (_, fid, *_rest) in enumerate(ce_pool):
                    quality_by_fid[fid] = float(ce_raw[i] - ce_min) / ce_range
                reranked_pool = sorted(
                    [(quality_by_fid[fid], fid, c, ef, lra, rc)
                     for _, fid, c, ef, lra, rc in ce_pool],
                    reverse=True,
                )
                scored_tail = [r for r in scored if r[1] not in ce_pool_fids]
                scored = reranked_pool + scored_tail

                # CE guard: min-rank ensemble of CE rank and pre-CE rank.
                # Prevents CE from pushing items out of top positions.
                if _CE_GUARD_K > 0 and _pre_ce_order is not None:
                    _ce_rank = {r[1]: i for i, r in enumerate(scored)}
                    _pre_ce_rank = {r[1]: i for i, r in enumerate(_pre_ce_order)}
                    _n_ce = len(scored)
                    scored.sort(key=lambda r: (
                        min(_ce_rank.get(r[1], _n_ce),
                            _pre_ce_rank.get(r[1], _n_ce)),
                        _pre_ce_rank.get(r[1], _n_ce),
                    ))
    except Exception:
        pass

    # Stage 4: post-cross-encoder position (before MMR).
    if _gold_fid is not None:
        _ce_fids = [r[1] for r in scored]
        _stages["ce_pos"] = _ce_fids.index(_gold_fid) if _gold_fid in _ce_fids else -1

    # Phase C: MMR diversity — post-CE, for output deduplication only.
    # _MMR_LAMBDA_POST_CE=0.25 applies light diversity on what is returned to the user.
    # The CE has already seen the true top-20 by relevance, so MMR here only removes
    # near-duplicate results from the final returned set.
    if not _RETRIEVE_BENCHMARK and len(scored) > 5:
        selected_embs: list = []
        mmr_selected: list = []
        remaining = list(scored)
        while remaining and len(mmr_selected) < 20:
            best_ms, best_row = -1e9, None
            for row in remaining:
                cand_emb = emb_by_fid.get(row[1])
                if cand_emb is None:
                    continue
                rel = cosine_similarity(prompt_emb, cand_emb)
                redundancy = max(
                    (cosine_similarity(cand_emb, s) for s in selected_embs),
                    default=0.0,
                )
                ms = _MMR_LAMBDA_POST_CE * rel - (1.0 - _MMR_LAMBDA_POST_CE) * redundancy
                if ms > best_ms:
                    best_ms, best_row = ms, row
            if best_row is None:
                break
            mmr_selected.append(best_row)
            selected_embs.append(emb_by_fid[best_row[1]])
            remaining.remove(best_row)
        scored = mmr_selected + remaining

    # Stage 3: post-MMR position (now after CE).
    if _gold_fid is not None:
        _mmr_fids = [r[1] for r in scored]
        _gold_mmr_pos = _mmr_fids.index(_gold_fid) if _gold_fid in _mmr_fids else -1
        _stages["mmr_pos"] = _gold_mmr_pos
        _stages["in_mmr_top20"] = 0 <= _gold_mmr_pos < 20

    # ── 9. Apply threshold, token budget, SM-2 EF update ─────────────────
    ft_by_fid = {r[0]: r[5] for r in rows}
    primary_ids: list[int] = []
    results: list[str] = []
    token_sum = 0
    budget_hit = False

    for score, fid, content, ef, lra, rc in scored:
        if score < threshold:
            continue
        if len(results) >= top_n:
            break
        multiplier = 1.8 if ft_by_fid.get(fid, "note") == "snippet" else 1.3
        token_est = int(len(content.split()) * multiplier)
        if token_sum + token_est > max_tokens:
            budget_hit = True
            break
        token_sum += token_est
        results.append(content)
        primary_ids.append(fid)

        # SM-2 EF update: cross-encoder quality proxy when available, else RRF score.
        if not _RETRIEVE_BENCHMARK:
            quality = quality_by_fid.get(
                fid, (raw_rrf.get(fid, 0) / max_rrf) if max_rrf > 0 else 0.5
            )
            new_ef = max(1.3, ef + 0.1 - (1.0 - quality) * 0.5)
            new_ivd = new_ef * (1.0 + (rc + 1) * 0.1)
            conn.execute(
                """UPDATE facts
                   SET retrieval_count = retrieval_count + 1,
                       last_retrieved_at = ?,
                       easiness_factor = ?,
                       interval_days = ?
                   WHERE id = ?""",
                (now_ts, round(new_ef, 4), round(new_ivd, 4), fid),
            )

    if not _RETRIEVE_BENCHMARK:
        # ── 10. Graph expansion ───────────────────────────────────────────
        # Commit before opening a second connection in get_related_facts().
        conn.commit()
        conn.close()

        seen_content: set[str] = set(results)
        for fid in primary_ids:
            if budget_hit or len(results) >= top_n + 3:
                break
            for neighbour in get_related_facts(fid, depth=1):
                if budget_hit or len(results) >= top_n + 3:
                    break
                nc = neighbour["content"]
                if nc not in seen_content:
                    n_mult = 1.8 if neighbour.get("fact_type") == "snippet" else 1.3
                    n_tokens = int(len(nc.split()) * n_mult)
                    if token_sum + n_tokens > max_tokens:
                        budget_hit = True
                        break
                    token_sum += n_tokens
                    seen_content.add(nc)
                    results.append(nc)
    else:
        conn.close()

    if include_budget_info:
        ret: dict = {
            "facts": results,
            "budget_hit": budget_hit,
            "retrieved_count": len(results),
            "total_candidates": total_candidates,
        }
        if _RETRIEVE_BENCHMARK:
            ret["fids"] = primary_ids
            ret["all_ranked_fids"] = [r[1] for r in scored]
            ret["all_ranked_scores"] = [r[0] for r in scored]
        if _gold_fid is not None:
            _gold_ids = [fid for fid in primary_ids]
            _stages["final_pos"] = _gold_ids.index(_gold_fid) if _gold_fid in _gold_ids else -1
            ret["_stages"] = _stages
        return ret
    return results


def consolidate_memories(
    project_id: str,
    session_id: str,
    max_merges: int = _RETRO_MAX,
) -> dict:
    """Retrospective ENRICH: merge live facts with pairwise cosine >= _RETRO_FLOOR.

    Scans all live fact embeddings for the project, finds similar pairs, and
    merges the lower-priority fact into the higher-retrieval-count one using the
    same ENRICH mutation that store_fact() uses.

    Pairwise safeguard (_RETRO_GUARD): after merging, the merged embedding must
    be >= _RETRO_GUARD similar to BOTH originals — prevents chaining facts that
    are only transitively similar.

    Token cap: merged text must not exceed _ENRICHMENT_MAX_TOKENS words.
    Cache: _CACHE_DIRTY is set so the next retrieve_facts() reloads embeddings.

    Returns:
        {"project_id", "merged", "pairs_checked", "max_merges"}
    """
    if not _NUMPY_AVAILABLE:
        return {
            "project_id": project_id, "merged": 0,
            "pairs_checked": 0, "max_merges": max_merges,
            "error": "numpy not available",
        }

    conn = init_db()
    _COLS = (
        "id, content, embedding, entities, retrieval_count, fact_type, session_id"
    )
    _WHERE = (
        "project_id = ? AND superseded_at IS NULL"
    )
    rows = conn.execute(
        f"SELECT {_COLS} FROM facts WHERE {_WHERE} ORDER BY id ASC",
        (project_id,),
    ).fetchall()

    if len(rows) < 2:
        conn.close()
        return {"project_id": project_id, "merged": 0, "pairs_checked": 0, "max_merges": max_merges}

    # Decode embeddings; skip facts that have none.
    fids: list[int] = []
    contents: list[str] = []
    vecs: list = []
    ents_jsons: list[str] = []
    rcs: list[int] = []

    for fid, content, emb_blob, ents_json, rc, _ft, _sid in rows:
        vec = _decode_embedding(emb_blob)
        if vec is None:
            continue
        fids.append(fid)
        contents.append(content)
        vecs.append(vec)
        ents_jsons.append(ents_json or "[]")
        rcs.append(rc or 0)

    n = len(fids)
    if n < 2:
        conn.close()
        return {"project_id": project_id, "merged": 0, "pairs_checked": 0, "max_merges": max_merges}

    mat = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    mat = mat / norms  # normalised rows: dot product == cosine

    sim_mat = mat @ mat.T  # shape (n, n), full pairwise cosine

    superseded: set[int] = set()   # indices already merged away this run
    merged_count = 0
    pairs_checked = 0

    for i in range(n):
        if merged_count >= max_merges:
            break
        if i in superseded:
            continue
        for j in range(i + 1, n):
            if merged_count >= max_merges:
                break
            if j in superseded:
                continue

            sim = float(sim_mat[i, j])
            pairs_checked += 1
            if sim < _RETRO_FLOOR:
                continue

            combined = contents[i] + "\n" + contents[j]
            if len(combined.split()) > _ENRICHMENT_MAX_TOKENS:
                continue

            # Keep the fact with more retrievals; on tie keep the older (lower id).
            if rcs[i] >= rcs[j]:
                base_idx, inc_idx = i, j
            else:
                base_idx, inc_idx = j, i

            merged_text = contents[base_idx] + "\n" + contents[inc_idx]

            # Pairwise safeguard: merged embedding must be close to BOTH originals.
            merged_emb_raw = embed_text(merged_text)
            merged_vec = np.array(merged_emb_raw, dtype=np.float32)
            mvnorm = np.linalg.norm(merged_vec)
            if mvnorm > 0:
                merged_vec = merged_vec / mvnorm
            if (float(merged_vec @ mat[base_idx]) < _RETRO_GUARD or
                    float(merged_vec @ mat[inc_idx]) < _RETRO_GUARD):
                continue  # merged content would drift too far from one parent

            base_fid = fids[base_idx]
            inc_fid  = fids[inc_idx]
            old_base_content = contents[base_idx]

            try:
                merged_ents = list(
                    set(json.loads(ents_jsons[base_idx]))
                    | set(json.loads(ents_jsons[inc_idx]))
                )
            except Exception:
                merged_ents = []

            merged_blob = _encode_embedding(merged_emb_raw)

            # FTS5 external-content: delete old entry then re-insert.
            conn.execute(
                "INSERT INTO facts_fts(facts_fts, rowid, content) VALUES('delete', ?, ?)",
                (base_fid, old_base_content),
            )
            conn.execute(
                """UPDATE facts
                   SET content = ?, embedding = ?, entities = ?, last_retrieved_at = ?
                   WHERE id = ?""",
                (merged_text, merged_blob, json.dumps(merged_ents), time.time(), base_fid),
            )
            conn.execute(
                "INSERT INTO facts_fts(rowid, content) VALUES (?, ?)",
                (base_fid, merged_text),
            )
            conn.execute(
                """INSERT INTO fact_mutations
                   (fact_id, mutation_type, old_content, new_content, session_id)
                   VALUES (?, 'ENRICH', ?, ?, ?)""",
                (base_fid, old_base_content, merged_text, session_id),
            )
            # Supersede the incoming fact (soft-delete).
            conn.execute(
                "UPDATE facts SET superseded_at = unixepoch() WHERE id = ?", (inc_fid,)
            )
            conn.execute(
                """INSERT INTO fact_mutations
                   (fact_id, mutation_type, old_content, session_id)
                   VALUES (?, 'RELEASE', ?, ?)""",
                (inc_fid, contents[inc_idx], session_id),
            )

            # Update in-memory state so later pairs see the merged content/vector.
            contents[base_idx] = merged_text
            mat[base_idx] = merged_vec
            sim_mat[base_idx, :] = mat @ merged_vec
            sim_mat[:, base_idx] = sim_mat[base_idx, :]
            superseded.add(inc_idx)
            merged_count += 1

    if merged_count > 0:
        conn.commit()
        with _CACHE_DIRTY_LOCK:
            _CACHE_DIRTY.add(project_id)
    conn.close()
    return {
        "project_id": project_id,
        "merged": merged_count,
        "pairs_checked": pairs_checked,
        "max_merges": max_merges,
    }


def memory_release(fact_id: int, session_id: str = "") -> dict:
    """Soft-delete a fact by marking it superseded, writing a RELEASE mutation.

    RELEASE = fact is correct but no longer contextually relevant to the
    current task.  Distinct from SUPERSEDE (fact was wrong/contradicted).
    The fact remains in history; superseded_at makes it invisible to retrieval.
    Returns {"ok": True, "fact_id": fact_id} or {"ok": False, "error": "..."}.
    """
    conn = init_db()
    row = conn.execute(
        "SELECT id, content, superseded_at FROM facts WHERE id = ?", (fact_id,)
    ).fetchone()
    if row is None:
        conn.close()
        return {"ok": False, "error": f"fact_id {fact_id} not found"}
    if row[2] is not None:
        conn.close()
        return {"ok": False, "error": f"fact_id {fact_id} already superseded"}
    old_content = row[1]
    conn.execute(
        "UPDATE facts SET superseded_at = unixepoch() WHERE id = ?", (fact_id,)
    )
    conn.execute(
        """INSERT INTO fact_mutations (fact_id, mutation_type, old_content, session_id)
           VALUES (?, 'RELEASE', ?, ?)""",
        (fact_id, old_content, session_id),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "fact_id": fact_id}


def get_history(fact_id: int) -> list[dict]:
    """Return the mutation log for a fact (INSERT, SUPERSEDE, etc.)."""
    conn = init_db()
    rows = conn.execute(
        """SELECT id, mutation_type, old_content, new_content, mutated_at, session_id
           FROM fact_mutations
           WHERE fact_id = ?
           ORDER BY mutated_at ASC""",
        (fact_id,),
    ).fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "mutation_type": r[1],
            "old_content": r[2],
            "new_content": r[3],
            "mutated_at": r[4],
            "session_id": r[5],
        }
        for r in rows
    ]


# ─── Slot fills ───────────────────────────────────────────────────────────────

def store_slot_fill(
    project_id: str, session_id: str, slot_name: str, value: str
) -> None:
    conn = init_db()
    conn.execute(
        """INSERT INTO slot_fills (project_id, session_id, slot_name, value)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(project_id, slot_name)
           DO UPDATE SET value = excluded.value,
                         session_id = excluded.session_id,
                         created_at = CURRENT_TIMESTAMP""",
        (project_id, session_id, slot_name, value),
    )
    conn.commit()
    conn.close()


def retrieve_slot_fills(project_id: str) -> list[dict]:
    """Return the most recent value per slot for the given project.

    Filters by project_id only so slot fills persist across sessions.
    GROUP BY ensures one row per slot (the latest via HAVING MAX(created_at)).
    """
    conn = init_db()
    cursor = conn.execute(
        """SELECT slot_name, value
           FROM slot_fills
           WHERE project_id = ?
           GROUP BY slot_name
           HAVING MAX(created_at)
           ORDER BY created_at DESC""",
        (project_id,),
    )
    rows = [
        {"slot_name": r[0], "value": r[1]}
        for r in cursor.fetchall()
    ]
    conn.close()
    return rows


# ─── Session enrichment tracking ─────────────────────────────────────────────

def session_seen(session_id: str) -> bool:
    """Return True if this session has already been enriched."""
    conn = init_db()
    row = conn.execute(
        "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    return row is not None


def session_mark(session_id: str, project_id: str) -> None:
    """Record that this session has been enriched, assigning a sequential session_index."""
    conn = init_db()
    existing = conn.execute(
        "SELECT session_index FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if existing is None:
        max_idx = conn.execute(
            "SELECT COALESCE(MAX(session_index), 0) FROM sessions WHERE project_id = ?",
            (project_id,),
        ).fetchone()[0]
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, project_id, session_index) "
            "VALUES (?, ?, ?)",
            (session_id, project_id, max_idx + 1),
        )
    conn.commit()
    conn.close()


def session_unmark(session_id: str) -> None:
    """Remove enrichment record so the next message triggers re-enrichment.

    Called when a session is compacted — context was lost, so the next
    message should receive fresh context injection.
    """
    conn = init_db()
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_dt(raw: str | None) -> datetime:
    if not raw:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    raw = raw.replace("Z", "+00:00")
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ─── CLI entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""

    if cmd == "store_fact":
        project_id, session_id, text = sys.argv[2], sys.argv[3], sys.argv[4]
        fact_type = sys.argv[5] if len(sys.argv) > 5 else "note"
        store_fact(project_id, session_id, text, fact_type)

    elif cmd == "store_turn_window":
        project_id    = sys.argv[2]
        session_id    = sys.argv[3]
        turns         = json.loads(sys.argv[4])
        current_index = int(sys.argv[5])
        fact_type     = sys.argv[6] if len(sys.argv) > 6 else "window"
        print(store_turn_window(project_id, session_id, turns, current_index, fact_type))

    elif cmd == "retrieve_facts":
        project_id, session_id, prompt = sys.argv[2], sys.argv[3], sys.argv[4]
        top_n            = int(sys.argv[5])       if len(sys.argv) > 5 else 3
        threshold        = float(sys.argv[6])     if len(sys.argv) > 6 else 0.25
        include_budget   = sys.argv[7] == "true"  if len(sys.argv) > 7 else False
        max_tokens       = int(sys.argv[8])        if len(sys.argv) > 8 else 2000
        print(json.dumps(retrieve_facts(
            project_id, session_id, prompt, top_n, threshold,
            include_budget_info=include_budget, max_tokens=max_tokens,
        )))

    elif cmd == "check_dedup":
        print(check_dedup(sys.argv[2]))

    elif cmd == "mark_stored":
        mark_stored(sys.argv[2])

    elif cmd == "store_slot_fill":
        project_id, session_id, slot_name, value = (
            sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
        )
        store_slot_fill(project_id, session_id, slot_name, value)

    elif cmd == "retrieve_slot_fills":
        project_id = sys.argv[2]
        print(json.dumps(retrieve_slot_fills(project_id)))

    elif cmd == "session_seen":
        print("YES" if session_seen(sys.argv[2]) else "NO")

    elif cmd == "session_mark":
        session_mark(sys.argv[2], sys.argv[3])

    elif cmd == "session_unmark":
        session_unmark(sys.argv[2])

    elif cmd == "link_facts":
        fact_id_a = int(sys.argv[2])
        fact_id_b = int(sys.argv[3])
        relation  = sys.argv[4] if len(sys.argv) > 4 else "related"
        strength  = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7
        link_facts(fact_id_a, fact_id_b, relation, strength)

    elif cmd == "get_related":
        fact_id = int(sys.argv[2])
        depth   = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        print(json.dumps(get_related_facts(fact_id, depth)))

    elif cmd == "get_graph":
        project_id = sys.argv[2]
        query      = sys.argv[3]
        depth      = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        print(json.dumps(get_graph(project_id, query, depth)))

    elif cmd == "get_history":
        fact_id = int(sys.argv[2])
        print(json.dumps(get_history(fact_id)))

    elif cmd == "consolidate_memories":
        project_id = sys.argv[2]
        session_id = sys.argv[3] if len(sys.argv) > 3 else ""
        print(json.dumps(consolidate_memories(project_id, session_id)))

    elif cmd == "memory_release":
        fact_id    = int(sys.argv[2])
        session_id = sys.argv[3] if len(sys.argv) > 3 else ""
        print(json.dumps(memory_release(fact_id, session_id)))

    else:
        print(json.dumps({"error": f"Unknown command: {cmd}"}), file=sys.stderr)
        sys.exit(1)
