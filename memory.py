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
import sqlite3
import struct
import sys
import time
from datetime import datetime, timezone

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
}
_IMPORTANCE_KEYWORDS: frozenset = frozenset({
    "never", "always", "must", "critical", "required", "forbidden",
    "breaking", "security", "auth", "production", "prod", "deprecated",
    "migration", "decided", "architectural",
})
# Phase C: MMR trade-off between relevance and diversity (0=diversity, 1=relevance).
_MMR_LAMBDA = 0.6

# ── Tunable retrieval constants ───────────────────────────────────────────────
_POOL_A_LIMIT          = 200    # most-recent facts (recency pool)
_POOL_B_LIMIT          = 300    # proven-useful facts (retrieval_count > 0)
_TEMPORAL_EDGE_DECAY   = 0.25   # strength decay per turn distance (linear)
_TEMPORAL_MAX_DISTANCE = 3      # turns back to link temporally
_SESSION_RECENCY_DECAY = 0.15   # score decay per session gap
_SESSION_MAX_LOOKBACK  = 7      # sessions back before score → 0.0
_ENRICHMENT_MAX_TOKENS = 500    # max combined tokens before enrichment falls back to insert
_ENRICH_MIN_SIM        = 0.15   # minimum cosine similarity to existing fact before enriching


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
               fact_type: str = "note", enrich: bool = True) -> "int | None":
    """Store a fact with soft-expire on contradiction, binary embedding, entity extraction.

    Contradiction detection: cosine similarity >= _CONTRADICTION_THRESHOLD writes a
    SUPERSEDE mutation, sets superseded_at on the old row, and inserts a new row.
    Phase 3: extracted entities stored as JSON list for entity-overlap retrieval.
    Phase 7: embeddings stored as compact binary blobs (struct.pack, 4 bytes/float).
    Phase 7.5: old INSERT mutations (> 90 days) compacted once per process.
    """
    global _compacted_this_process
    emb = embed_text(text)
    conn = init_db()

    # Phase 7.5: compact old INSERT mutations once per process lifetime.
    if not _compacted_this_process:
        _compact_old_mutations(conn)
        _compacted_this_process = True

    # Find the most-similar live fact in this project (last 200).
    cursor = conn.execute(
        """SELECT id, embedding FROM facts
           WHERE project_id = ?
             AND superseded_at IS NULL
             AND (valid_to IS NULL OR valid_to > unixepoch())
           ORDER BY id DESC LIMIT 200""",
        (project_id,),
    )
    best_id: int | None = None
    best_sim: float = 0.0
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
    fact_type: str = "note",
) -> "int | None":
    """Store a 3-turn sliding window centred on current_index as a single fact.

    Tags: [prev] for preceding turn, [curr] for current, [next] for following.
    Each turn dict must have 'speaker' and 'text' keys.

    Designed for batch ingestion of conversational data (e.g. LoCoMo eval).
    For real-time coding sessions, use store_fact() directly.

    Semantic duplication across overlapping windows is intentional — different
    neighbouring context yields different embeddings and different retrieval matches.

    Returns the fact_id of the stored or enriched fact.
    """
    window: list[str] = []
    for i in range(max(0, current_index - 1), min(len(turns), current_index + 2)):
        turn = turns[i]
        tag = "[curr]" if i == current_index else ("[prev]" if i < current_index else "[next]")
        window.append(f"{tag} {turn['speaker']}: {turn['text']}")
    content = "\n".join(window)
    return store_fact(project_id, session_id, content, fact_type, enrich=False)


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

    # ── 1. Pull candidate pool: Pool A (recency) + Pool B (proven useful) ─
    # Pool A: _POOL_A_LIMIT most-recent facts by insert time.
    # Pool B: proven-useful facts (retrieval_count > 0), capped at _POOL_B_LIMIT,
    #         ordered by recency-of-use then total use count.
    # Separate caps prevent Pool B's proven hits from drowning Pool A's recency.
    _COLS = (
        "id, content, embedding, retrieval_count, created_at, fact_type, "
        "easiness_factor, last_retrieved_at, interval_days, entities, "
        "COALESCE(importance, 0.5), session_id"
    )
    _WHERE = (
        "project_id = ? AND superseded_at IS NULL "
        "AND (valid_to IS NULL OR valid_to > unixepoch())"
    )
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
    for fid, content, emb_data, rc, ca, ft, ef, lra, ivd, ents, imp, fsid in all_candidate_rows:
        emb = _decode_embedding(emb_data)
        if emb is None:
            continue
        rows.append((
            fid, content, emb,
            rc or 0, ca, ft or "note",
            ef if ef is not None else 2.5,
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
            s += 1.0 / (_RRF_K + bm25_rank[fid])
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
        # (score, fid, content, ef, lra, rc)
        scored_all.append((score, fid, content, ef, lra, rc))
    scored_all.sort(reverse=True, key=lambda x: x[0])

    # Stage 1: post-scoring position.
    if _gold_fid is not None:
        _sa_fids = [r[1] for r in scored_all]
        _stages["scored_pos"] = _sa_fids.index(_gold_fid) if _gold_fid in _sa_fids else -1
        _stages["scored_size"] = len(_sa_fids)

    # ── 7. SM-2 gate — soft relax when < 3 facts pass (Phase 2) ──────────
    _lra_by_fid = {r[0]: r[7] for r in rows}
    _ivd_by_fid = {r[0]: r[8] for r in rows}

    _ef_by_fid  = {r[0]: float(r[6]) for r in rows}   # easiness_factor per fact
    _sta_by_fid = {r[0]: min((now_ts - r[7]) / (30 * 86400), 1.0)
                   if r[7] is not None else 1.0 for r in rows}

    def _is_due(fid: int) -> bool:
        lra = _lra_by_fid.get(fid)
        ivd = _ivd_by_fid.get(fid, 1.0)
        if lra is None:
            return True
        if _ef_by_fid.get(fid, 2.5) <= 1.5 and _sta_by_fid.get(fid, 1.0) > 0.5:
            return True   # high-importance (low EF), stale — bypass SM-2 gate
        return (now_ts - lra) >= (ivd * 86400)

    due_scored = [row for row in scored_all if _is_due(row[1])]
    scored = due_scored if len(due_scored) >= 3 else scored_all
    total_candidates = len(scored)

    # Stage 2: post-gate position.
    if _gold_fid is not None:
        _g_fids = [r[1] for r in scored]
        _stages["gated_pos"] = _g_fids.index(_gold_fid) if _gold_fid in _g_fids else -1
        _stages["gated_size"] = len(_g_fids)

    # Phase C: MMR diversity — greedy select up to 20 diverse candidates for cross-encoder.
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
                ms = _MMR_LAMBDA * rel - (1.0 - _MMR_LAMBDA) * redundancy
                if ms > best_ms:
                    best_ms, best_row = ms, row
            if best_row is None:
                break
            mmr_selected.append(best_row)
            selected_embs.append(emb_by_fid[best_row[1]])
            remaining.remove(best_row)
        scored = mmr_selected + remaining

    # Stage 3: post-MMR position (in top-20 or pushed to remaining).
    if _gold_fid is not None:
        _mmr_fids = [r[1] for r in scored]
        _gold_mmr_pos = _mmr_fids.index(_gold_fid) if _gold_fid in _mmr_fids else -1
        _stages["mmr_pos"] = _gold_mmr_pos
        _stages["in_mmr_top20"] = 0 <= _gold_mmr_pos < 20

    # ── 8. Cross-encoder reranking (Phase 4) ──────────────────────────────
    # Reranks top-20 when sentence-transformers is loaded; 500ms latency cap.
    quality_by_fid: dict[int, float] = {}
    try:
        cross_enc = _get_cross_encoder()
        if cross_enc is not None and len(scored) > 5:
            top20 = scored[:20]
            pairs = [(prompt, c) for _, _, c, *_ in top20]
            t0 = time.time()
            ce_raw = cross_enc.predict(pairs)
            if time.time() - t0 < 0.5:
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
                scored = reranked_top20 + scored[20:]
    except Exception:
        pass

    # Stage 4: post-cross-encoder position.
    if _gold_fid is not None:
        _ce_fids = [r[1] for r in scored]
        _stages["ce_pos"] = _ce_fids.index(_gold_fid) if _gold_fid in _ce_fids else -1

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


def consolidate_memories(project_id: str, session_id: str) -> dict:
    """Return the last 50 live facts for LLM-assisted consolidation (Phase 6).

    The caller (LLM) reviews the list for contradictions and redundancies,
    then calls store_memory to update stale entries.
    """
    conn = init_db()
    cursor = conn.execute(
        """SELECT id, content, fact_type, created_at, retrieval_count
           FROM facts
           WHERE project_id = ?
             AND superseded_at IS NULL
             AND (valid_to IS NULL OR valid_to > unixepoch())
           ORDER BY id DESC LIMIT 50""",
        (project_id,),
    )
    facts = [
        {
            "id": r[0],
            "content": r[1],
            "fact_type": r[2],
            "created_at": r[3],
            "retrieval_count": r[4],
        }
        for r in cursor.fetchall()
    ]
    conn.close()
    return {"project_id": project_id, "facts": facts, "count": len(facts)}


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
        fact_type     = sys.argv[6] if len(sys.argv) > 6 else "note"
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
