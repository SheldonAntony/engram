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
import os
import re
import sqlite3
import struct
import sys
import time
from datetime import datetime, timezone

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

# Feature 1 Bug 3: shared utilities extracted from this module
from utils import cosine_similarity, embed_text

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
_WINDOW_DEMOTION = 0.55

# SM-2 gate: when False, SM-2 interval check is skipped for candidate selection.
# EF/interval updates still happen on retrieval — they feed the staleness score.
# Gate is OFF because LoCoMo gold facts (and most unseen production facts) have
# retrieval_count=0 and would be gated out before scoring even starts.
_SM2_GATE_ENABLED = False

# ── Tunable retrieval constants ───────────────────────────────────────────────
_POOL_A_LIMIT          = 750    # most-recent facts (recency pool) — raised from 200
_POOL_B_LIMIT          = 300    # proven-useful facts (retrieval_count > 0)
_TEMPORAL_EDGE_DECAY   = 0.25   # strength decay per turn distance (linear)
_TEMPORAL_MAX_DISTANCE = 3      # turns back to link temporally
_SESSION_RECENCY_DECAY = 0.15   # score decay per session gap
_SESSION_MAX_LOOKBACK  = 7      # sessions back before score → 0.0
_ENRICHMENT_MAX_TOKENS = 500    # max combined tokens before enrichment falls back to insert
_ENRICH_MIN_SIM        = 0.15   # minimum cosine similarity to existing fact before enriching

# Step 7: Retrospective ENRICH (consolidate_memories) tuning constants.
_RETRO_FLOOR  = 0.35   # minimum pairwise cosine to trigger a merge
_RETRO_GUARD  = 0.30   # merged embedding must be >= this to BOTH originals (prevents chaining)
_RETRO_MAX    = 500    # default merge cap per consolidate_memories() call


# ── Embedding cache for ANN-based Pool A ─────────────────────────────────────
# Keyed by project_id → (fid_list, emb_matrix) where emb_matrix is shape (N, D).
# Populated lazily on first retrieve call for a project.
# _cache_dirty tracks which projects need a reload after a write.
_EMB_CACHE: "dict[str, tuple[list[int], object]]" = {}
_CACHE_DIRTY: "set[str]" = set()

# ── spaCy lazy loader (Step 6 SVO extraction) ────────────────────────────────
# Set to None before first use, False after a failed load attempt.
_nlp = None
_WINDOW_TAG_RE = re.compile(r'^\[(prev|curr|next)\]\s*')


def _get_nlp():
    """Return a loaded spaCy en_core_web_sm model, or None if unavailable."""
    global _nlp
    if _nlp is None:
        try:
            import spacy  # noqa: PLC0415
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp = False
    return _nlp if _nlp is not False else None


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
            valid_to          REAL    DEFAULT NULL,
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

    # Feature 2: migrations — add project_id if the table already exists
    _ensure_column(conn, "facts",      "project_id",      "TEXT",      "'unknown'")
    _ensure_column(conn, "facts",      "retrieval_count", "INTEGER",   "0")
    _ensure_column(conn, "facts",      "created_at",      "TIMESTAMP", "CURRENT_TIMESTAMP")
    _ensure_column(conn, "facts",      "fact_type",       "TEXT",      "'note'")
    _ensure_column(conn, "facts",      "valid_from",        "REAL",      "(unixepoch())")
    _ensure_column(conn, "facts",      "superseded_at",     "REAL",      "NULL")
    _ensure_column(conn, "facts",      "valid_to",          "REAL",      "NULL")
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
        "CREATE INDEX IF NOT EXISTS idx_facts_live ON facts (superseded_at, valid_to)"
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
        conn.execute(
            f"ALTER TABLE {table} ADD COLUMN {col} {col_type} DEFAULT {default}"
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
           WHERE project_id = ?
             AND superseded_at IS NULL
             AND (valid_to IS NULL OR valid_to > unixepoch())
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


def store_fact(project_id: str, session_id: str, text: str,
               fact_type: str = "note", enrich: bool = True,
               _embed_text: "str | None" = None) -> "int | None":
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
    """
    global _compacted_this_process
    emb = embed_text(_embed_text if _embed_text is not None else text)
    conn = init_db()

    # Phase 7.5: compact old INSERT mutations once per process lifetime.
    if not _compacted_this_process:
        _compact_old_mutations(conn)
        _compacted_this_process = True

    # Find the most-similar live fact in this project (last 200).
    # Window facts are intentionally overlapping (sharing 2 of 3 turns) so
    # their cosine similarity is always high — skip the scan to prevent them
    # from chain-superseding each other and destroying historical context.
    # Turn facts embed the same [curr] text as their window counterpart, so
    # consecutive turns from the same speaker would also hit the threshold —
    # skip the scan for turn facts too.
    best_id: int | None = None
    best_sim: float = 0.0
    if fact_type not in ("window", "turn"):
        cursor = conn.execute(
            """SELECT id, embedding FROM facts
               WHERE project_id = ?
                 AND superseded_at IS NULL
                 AND (valid_to IS NULL OR valid_to > unixepoch())
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

    source_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    entities = _extract_entities(text)
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
    link_cursor = conn.execute(
        """SELECT id, content, embedding FROM facts
           WHERE project_id = ? AND id != ?
             AND superseded_at IS NULL
             AND (valid_to IS NULL OR valid_to > unixepoch())
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
    _CACHE_DIRTY.add(project_id)
    return saved_id


def _fts5_query(prompt: str) -> str:
    """Sanitize a user prompt into a safe FTS5 MATCH query.

    Strips FTS5 special chars and joins remaining tokens with OR so any
    keyword can match. Returns empty string when no usable tokens remain.
    """
    safe = "".join(c if c.isalnum() or c.isspace() else " " for c in prompt)
    tokens = [t for t in safe.split() if len(t) > 2]
    if not tokens:
        return ""
    # Quote each token to avoid FTS5 keyword collisions (AND/OR/NOT/NEAR).
    return " OR ".join(f'"{t}"' for t in tokens)


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
) -> "int | None":
    """Store a 3-turn sliding window centred on current_index as a single fact.

    Tags: [prev] for preceding turn, [curr] for current, [next] for following.
    Each turn dict must have 'speaker' and 'text' keys.

    Designed for batch ingestion of conversational data (e.g. LoCoMo eval).
    For real-time coding sessions, use store_fact() directly.

    Semantic duplication across overlapping windows is intentional — different
    neighbouring context yields different embeddings and different retrieval matches.

    The embedding is computed from the [curr] turn text only (not the full
    3-turn window) so that ANN search matches the question against the specific
    turn that contains the answer, not a blended 3-turn vector.  The full window
    is still stored as content for context display.

    extract_svo: when False, skip spaCy SVO extraction.  Pass False during
    bulk benchmark ingestion — conversational text yields <1% SVO triples but
    the spaCy calls dominate ingestion time (~0.5s each × 5882 turns = 3200s).

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
    # Embed only the [curr] turn so ANN search is precise; store full window for context.
    embed_hint = curr_line if curr_line else content
    # Store the clean single-turn fact FIRST (fact_type="turn") so it gets a lower
    # row id than the window row.  Tests that fetch ORDER BY id DESC LIMIT 1 will
    # get the window row (richer context); retrieval can return either row.
    if curr_line:
        store_fact(project_id, session_id, curr_line, "turn", enrich=False,
                   _embed_text=curr_line)
    # Window row stored second — last inserted, richer 3-turn context for display.
    window_fid = store_fact(project_id, session_id, content, fact_type, enrich=False,
                            _embed_text=embed_hint)
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
        "project_id = ? AND superseded_at IS NULL "
        "AND (valid_to IS NULL OR valid_to > unixepoch())"
    )

    # ── Pool A: ANN via in-process embedding cache ────────────────────────
    # Cache stores (fid_list, emb_matrix) per project. Rebuilt when dirty.
    pool_a: list = []
    if _NUMPY_AVAILABLE:
        if project_id in _CACHE_DIRTY or project_id not in _EMB_CACHE:
            # Load all live fact IDs + embeddings for this project.
            cache_rows = conn.execute(
                "SELECT id, embedding FROM facts WHERE project_id = ? "
                "AND superseded_at IS NULL "
                "AND (valid_to IS NULL OR valid_to > unixepoch())",
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
            else:
                _EMB_CACHE[project_id] = ([], None)
            _CACHE_DIRTY.discard(project_id)

        cached_fids, cached_mat = _EMB_CACHE.get(project_id, ([], None))
        if cached_mat is not None and len(cached_fids) > 0:
            # Embed the bare prompt for ANN pool selection.
            # (augmented_prompt is built later after pool; vector ranking step uses it.)
            prompt_emb_raw = embed_text(prompt)
            qvec = np.array(prompt_emb_raw, dtype=np.float32)
            qnorm = np.linalg.norm(qvec)
            if qnorm > 0:
                qvec = qvec / qnorm
            sims = cached_mat @ qvec          # shape (N,), cosine similarity
            top_k = min(_POOL_A_LIMIT, len(cached_fids))
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
            f"SELECT {_COLS} FROM facts WHERE {_WHERE} ORDER BY id DESC LIMIT ?",
            (project_id, _POOL_A_LIMIT),
        ).fetchall()

    pool_a_ids = {r[0] for r in pool_a}
    pool_b_raw = conn.execute(
        f"SELECT {_COLS} FROM facts WHERE {_WHERE} AND retrieval_count > 0 "
        f"ORDER BY retrieval_count DESC, last_retrieved_at DESC LIMIT ?",
        (project_id, _POOL_A_LIMIT + _POOL_B_LIMIT),
    ).fetchall()
    pool_b = [r for r in pool_b_raw if r[0] not in pool_a_ids][:_POOL_B_LIMIT]
    all_candidate_rows = pool_a + pool_b

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

    # ── 2. Vector ranking ──────────────────────────────────────────────────
    prompt_emb = embed_text(augmented_prompt)
    emb_by_fid: dict[int, list] = {fid: emb for fid, _c, emb, *_ in rows}
    vec_scored = sorted(
        ((cosine_similarity(prompt_emb, emb), fid) for fid, emb in emb_by_fid.items()),
        reverse=True,
    )
    vec_rank: dict[int, int] = {fid: rank for rank, (_, fid) in enumerate(vec_scored)}

    # ── 3. BM25 via FTS5 ──────────────────────────────────────────────────
    bm25_rank: dict[int, int] = {}
    phrase_fids: set[int] = set()   # facts matching exact phrase — get 1.5× BM25 boost
    fts_query = _fts5_query(prompt)
    if fts_query:
        candidate_ids = tuple(fid for fid, *_ in rows)
        placeholders = ",".join("?" for _ in candidate_ids)
        try:
            bm_cursor = conn.execute(
                f"""SELECT rowid FROM facts_fts
                    WHERE facts_fts MATCH ?
                      AND rowid IN ({placeholders})
                    ORDER BY bm25(facts_fts)""",
                (fts_query, *candidate_ids),
            )
            for rank, (fid,) in enumerate(bm_cursor.fetchall()):
                bm25_rank[fid] = rank
        except sqlite3.OperationalError:
            pass

        # Phrase boost: exact phrase match via FTS5 quoted syntax.
        # Applies 1.5× weight to the BM25 component in RRF for phrase-matching facts.
        prompt_words = prompt.strip().split()
        if len(prompt_words) >= 2:
            phrase_query = f'"{prompt.strip()}"'
            try:
                ph_cursor = conn.execute(
                    f"""SELECT rowid FROM facts_fts
                        WHERE facts_fts MATCH ?
                          AND rowid IN ({placeholders})""",
                    (phrase_query, *candidate_ids),
                )
                phrase_fids = {row[0] for row in ph_cursor.fetchall()}
            except sqlite3.OperationalError:
                pass

    # ── 4. Entity-overlap ranking (Phase 3) ───────────────────────────────
    entity_rank: dict[int, int] = {}
    try:
        prompt_ents = set(e.lower() for e in _extract_entities(prompt))
        if prompt_ents:
            ent_scores = []
            for fid, _c, _e, _rc, _ca, _ft, _ef, _lra, _ivd, ents_json, _imp, _fsid in rows:
                try:
                    fact_ents = set(e.lower() for e in json.loads(ents_json or "[]"))
                except Exception:
                    fact_ents = set()
                shared = prompt_ents & fact_ents
                n_shared = len(shared)
                ratio = n_shared / max(len(prompt_ents), len(fact_ents), 1)
                # Require >=2 shared entities OR ratio >0.3 to prevent
                # single generic-entity false matches (e.g. "Python", "the project").
                if n_shared >= 2 or ratio > 0.3:
                    overlap = ratio
                else:
                    overlap = 0.0
                ent_scores.append((overlap, fid))
            ent_scores.sort(reverse=True)
            entity_rank = {fid: rank for rank, (_, fid) in enumerate(ent_scores)}
    except Exception:
        pass

    # ── 5. Three-way RRF fusion ────────────────────────────────────────────
    _RRF_K = 60
    n = len(rows)
    raw_rrf: dict[int, float] = {}
    for fid, *_ in rows:
        s = 1.0 / (_RRF_K + vec_rank.get(fid, n))
        if fid in bm25_rank:
            bm25_component = 1.0 / (_RRF_K + bm25_rank[fid])
            if fid in phrase_fids:
                bm25_component *= 1.5   # phrase match boost — exact phrase in content
            s += bm25_component
        if fid in entity_rank:
            s += 1.0 / (_RRF_K + entity_rank[fid])
        raw_rrf[fid] = s
    max_rrf = max(raw_rrf.values()) if raw_rrf else 1.0

    # ── 6. Combined score per fact ─────────────────────────────────────────
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
    scored_all: list = []
    for fid, content, _emb, rc, ca, ft, ef, lra, ivd, _ents, _imp, fsid in rows:
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
        # Weights sum to 1.0. importance dropped — encoded in easiness_factor at insert time.
        score = 0.35 * rrf + 0.20 * recency + 0.20 * staleness + 0.15 * session_rec + 0.10 * freq
        # Step 6: demote window facts so atomic SVO notes rank above them.
        # Turn facts (clean [curr] text) are NOT demoted — they are the precise answer unit.
        if ft == "window":
            score *= _WINDOW_DEMOTION
        # Speaker boost: if a speaker name appears in the question AND this is a turn
        # fact whose content starts with that speaker, boost by 1.3x.  Helps single-hop
        # questions like "What did Alice say about X?" where the answer is Alice's turn.
        if ft == "turn" and ": " in content:
            turn_speaker = content.split(":", 1)[0].strip()
            if turn_speaker and turn_speaker.lower() in prompt.lower():
                score *= 1.3
        # (score, fid, content, ef, lra, rc)
        scored_all.append((score, fid, content, ef, lra, rc))
    scored_all.sort(reverse=True, key=lambda x: x[0])

    # Stage 1: post-scoring position.
    if _gold_fid is not None:
        _sa_fids = [r[1] for r in scored_all]
        _stages["scored_pos"] = _sa_fids.index(_gold_fid) if _gold_fid in _sa_fids else -1
        _stages["scored_size"] = len(_sa_fids)

    # ── 7. SM-2 gate (disabled by default — _SM2_GATE_ENABLED=False) ─────
    # Gate is skipped because unseen facts (retrieval_count=0) would be blocked
    # before scoring. EF/interval fields still update on retrieval for staleness.
    _lra_by_fid = {r[0]: r[7] for r in rows}
    _ivd_by_fid = {r[0]: r[8] for r in rows}
    _ef_by_fid  = {r[0]: float(r[6]) for r in rows}
    _sta_by_fid = {r[0]: min((now_ts - r[7]) / (30 * 86400), 1.0)
                   if r[7] is not None else 1.0 for r in rows}

    if _SM2_GATE_ENABLED:
        def _is_due(fid: int) -> bool:
            lra = _lra_by_fid.get(fid)
            ivd = _ivd_by_fid.get(fid, 1.0)
            if lra is None:
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

    # ── 8. Cross-encoder reranking (Phase 4) — runs on pure top-20 by score ─
    # CE sees the top-20 by composite score (no MMR demotion yet), giving it the
    # highest-relevance candidates. MMR runs post-CE for output deduplication only.
    quality_by_fid: dict[int, float] = {}
    try:
        cross_enc = _get_cross_encoder()
        if cross_enc is not None and len(scored) > 5:
            top20 = scored[:40]
            # BM25 anchor: guarantee top-5 BM25 facts enter the CE pool even if
            # composite scoring pushed them past position 40.
            if bm25_rank:
                top20_fids = {r[1] for r in top20}
                bm25_anchors = [
                    row for row in scored[40:]
                    if row[1] in bm25_rank and bm25_rank[row[1]] < 5
                    and row[1] not in top20_fids
                ]
                if bm25_anchors:
                    top20 = top20 + bm25_anchors
            ce_pool_fids = {r[1] for r in top20}
            pairs = [(prompt, c) for _, _, c, *_ in top20]
            t0 = time.time()
            ce_raw = cross_enc.predict(pairs)
            if time.time() - t0 < 3.0:
                ce_min = min(ce_raw)
                ce_max = max(ce_raw)
                ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
                for i, (_, fid, *_rest) in enumerate(top20):
                    quality_by_fid[fid] = float(ce_raw[i] - ce_min) / ce_range
                reranked_top20 = sorted(
                    [(quality_by_fid[fid], fid, c, ef, lra, rc)
                     for _, fid, c, ef, lra, rc in top20],
                    reverse=True,
                )
                scored_tail = [r for r in scored if r[1] not in ce_pool_fids]
                scored = reranked_top20 + scored_tail
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
    if len(scored) > 5:
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

    # ── 10. Graph expansion ───────────────────────────────────────────────
    # Commit before opening a second connection in get_related_facts().
    conn.commit()
    conn.close()

    seen_content: set[str] = set(results)
    extra_cap = top_n + 3
    for fid in primary_ids:
        if len(results) >= extra_cap:
            break
        for neighbour in get_related_facts(fid, depth=1):
            if len(results) >= extra_cap:
                break
            nc = neighbour["content"]
            if nc not in seen_content:
                seen_content.add(nc)
                results.append(nc)

    if include_budget_info:
        ret = {
            "facts": results,
            "budget_hit": budget_hit,
            "retrieved_count": len(results),
            "total_candidates": total_candidates,
        }
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
        "project_id = ? AND superseded_at IS NULL "
        "AND (valid_to IS NULL OR valid_to > unixepoch())"
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
