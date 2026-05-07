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
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

# Feature 1 Bug 3: shared utilities extracted from this module
from utils import cosine_similarity, embed_text

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


# ─── Database init ────────────────────────────────────────────────────────────

def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)

    # Feature 2: project_id column on all tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT,
            project_id      TEXT,
            content         TEXT,
            embedding       TEXT,
            fact_type       TEXT DEFAULT 'note',
            retrieval_count INTEGER DEFAULT 0,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    _ensure_column(conn, "slot_fills", "project_id",      "TEXT",      "'unknown'")
    _ensure_column(conn, "slot_fills", "created_at",      "TIMESTAMP", "CURRENT_TIMESTAMP")

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

    # Backfill FTS5 index from any pre-existing facts (one-time, cheap if empty).
    fts_count = conn.execute("SELECT COUNT(*) FROM facts_fts").fetchone()[0]
    facts_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    if fts_count < facts_count:
        conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")

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


# ─── Facts (semantic memory) ──────────────────────────────────────────────────

def store_fact(project_id: str, session_id: str, text: str,
               fact_type: str = "note") -> None:
    """Store a fact, replacing near-duplicates instead of inserting them.

    Contradiction detection: if any existing fact in the same project has
    cosine similarity >= _CONTRADICTION_THRESHOLD with the new text, we
    overwrite that row in-place. This handles both verbatim duplicates and
    paraphrased restatements (e.g. "we use Postgres" → "switched to MySQL"
    when the latter is similar enough to the former).
    """
    emb = embed_text(text)
    conn = init_db()

    # Find the most-similar existing fact in this project (last 200 only —
    # bounded for speed; older near-duplicates are acceptable to keep).
    cursor = conn.execute(
        """SELECT id, embedding FROM facts
           WHERE project_id = ?
           ORDER BY id DESC LIMIT 200""",
        (project_id,),
    )
    best_id: int | None = None
    best_sim: float = 0.0
    for row_id, emb_json in cursor.fetchall():
        if not emb_json:
            continue
        try:
            existing_emb = json.loads(emb_json)
        except (json.JSONDecodeError, TypeError):
            continue
        sim = cosine_similarity(emb, existing_emb)
        if sim > best_sim:
            best_sim = sim
            best_id = row_id

    if best_id is not None and best_sim >= _CONTRADICTION_THRESHOLD:
        # Replace the contradicted/duplicate fact in place.
        conn.execute(
            """UPDATE facts
               SET content = ?, embedding = ?, fact_type = ?, session_id = ?
               WHERE id = ?""",
            (text, json.dumps(emb), fact_type, session_id, best_id),
        )
        # Keep the FTS5 mirror in sync.
        conn.execute("DELETE FROM facts_fts WHERE rowid = ?", (best_id,))
        conn.execute(
            "INSERT INTO facts_fts(rowid, content) VALUES (?, ?)",
            (best_id, text),
        )
    else:
        cur = conn.execute(
            """INSERT INTO facts (project_id, session_id, content, embedding, fact_type)
               VALUES (?, ?, ?, ?, ?)""",
            (project_id, session_id, text, json.dumps(emb), fact_type),
        )
        new_id = cur.lastrowid
        conn.execute(
            "INSERT INTO facts_fts(rowid, content) VALUES (?, ?)",
            (new_id, text),
        )

    conn.commit()
    conn.close()


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


def retrieve_facts(
    project_id: str,
    session_id: str,
    prompt: str,
    top_n: int = 3,
    threshold: float = 0.25,
) -> list[str]:
    """Hybrid retrieval: BM25 + vector via Reciprocal Rank Fusion (RRF),
    then weighted with content-type-aware recency and frequency.

    Score = 0.5 * fused_rank + 0.3 * type_weighted_recency + 0.2 * frequency
    """
    conn = init_db()

    # ── 1. Pull candidate pool (last 200 facts in project) ────────────────
    cursor = conn.execute(
        """SELECT id, content, embedding, retrieval_count, created_at, fact_type
           FROM facts
           WHERE project_id = ?
           ORDER BY id DESC LIMIT 200""",
        (project_id,),
    )
    rows: list[tuple[int, str, list[float], int, str, str]] = []
    for fid, content, emb_json, rc, ca, ft in cursor.fetchall():
        if not emb_json:
            continue
        try:
            emb = json.loads(emb_json)
        except (json.JSONDecodeError, TypeError):
            continue
        rows.append((fid, content, emb, rc, ca, ft or "note"))

    if not rows:
        conn.close()
        return []

    max_rc_row = conn.execute(
        "SELECT COALESCE(MAX(retrieval_count), 1) FROM facts WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    max_rc = max_rc_row[0] if max_rc_row and max_rc_row[0] else 1

    # ── 2. Vector ranking (cosine similarity) ─────────────────────────────
    prompt_emb = embed_text(prompt)
    vec_scored = sorted(
        (
            (cosine_similarity(prompt_emb, emb), fid)
            for fid, _c, emb, _rc, _ca, _ft in rows
        ),
        reverse=True,
        key=lambda x: x[0],
    )
    vec_rank: dict[int, int] = {fid: rank for rank, (_s, fid) in enumerate(vec_scored)}

    # ── 3. BM25 ranking via FTS5 (returns lowest = best match) ────────────
    bm25_rank: dict[int, int] = {}
    fts_query = _fts5_query(prompt)
    if fts_query:
        candidate_ids = tuple(fid for fid, *_rest in rows)
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
            # Malformed query or FTS5 unavailable — silently skip BM25 leg.
            bm25_rank = {}

    # ── 4. RRF fusion + weighted scoring ──────────────────────────────────
    # Standard RRF scores are tiny (~0.03 max) — normalize within the batch
    # so the relevance leg is comparable to recency/freq.
    K = 60
    raw_rrf: dict[int, float] = {}
    for fid, *_rest in rows:
        s = 1.0 / (K + vec_rank.get(fid, len(rows)))
        if fid in bm25_rank:
            s += 1.0 / (K + bm25_rank[fid])
        raw_rrf[fid] = s
    max_rrf = max(raw_rrf.values()) if raw_rrf else 1.0

    now = datetime.now(timezone.utc)
    scored: list[tuple[float, int, str]] = []
    for fid, content, _emb, rc, created_at, fact_type in rows:
        rrf = (raw_rrf[fid] / max_rrf) if max_rrf > 0 else 0.0

        # Content-type-aware recency decay (decisions don't decay).
        try:
            ca_dt = _parse_dt(created_at)
        except Exception:
            ca_dt = now
        days = max(0, (now - ca_dt).days)
        decay = _DECAY_RATES.get(fact_type, _DECAY_RATES["note"])
        recency = 1.0 / (1.0 + decay * days) if decay > 0 else 1.0

        # Frequency normalized by project max.
        freq = (rc / max_rc) if max_rc > 0 else 0

        combined = 0.5 * rrf + 0.3 * recency + 0.2 * freq
        scored.append((combined, fid, content))

    scored.sort(reverse=True, key=lambda x: x[0])

    # ── 5. Apply threshold and bump retrieval_count ───────────────────────
    # Note: rrf scores are small (<0.04), so the historical 0.25 threshold
    # mostly rejects via recency/freq contributions. Kept for compatibility.
    results: list[str] = []
    for score, fid, content in scored[:top_n]:
        if score >= threshold:
            results.append(content)
            conn.execute(
                "UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE id = ?",
                (fid,),
            )
    conn.commit()
    conn.close()
    return results


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
    """Record that this session has been enriched."""
    conn = init_db()
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, project_id) VALUES (?, ?)",
        (session_id, project_id),
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

    elif cmd == "retrieve_facts":
        project_id, session_id, prompt = sys.argv[2], sys.argv[3], sys.argv[4]
        top_n     = int(sys.argv[5])   if len(sys.argv) > 5 else 3
        threshold = float(sys.argv[6]) if len(sys.argv) > 6 else 0.25
        print(json.dumps(retrieve_facts(project_id, session_id, prompt, top_n, threshold)))

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

    else:
        print(json.dumps({"error": f"Unknown command: {cmd}"}), file=sys.stderr)
        sys.exit(1)
