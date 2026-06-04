#!/usr/bin/env python3
"""Task snapshots for the Preflight plugin.

Stores and retrieves summaries of past coding tasks using semantic search,
weighted by similarity, recency, frequency, and completion status.

CLI usage:
    tasks.py create_snapshot  <project_id> <session_id> <task_type> <orig_prompt> <summary>
    tasks.py retrieve_similar <project_id> <session_id> <prompt> [top_n] [threshold]
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

# Feature 1 Bug 3: shared utilities extracted from this module
from utils import cosine_similarity, embed_text

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.db")


# ─── Database init ────────────────────────────────────────────────────────────

def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)

    # Feature 2 + Feature 3: project_id, retrieval_count, completed columns
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT,
            project_id      TEXT,
            task_type       TEXT,
            orig_prompt     TEXT,
            summary         TEXT,
            embedding       TEXT,
            retrieval_count INTEGER DEFAULT 0,
            completed       BOOLEAN DEFAULT FALSE,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Feature 2: migrations
    _ensure_column(conn, "tasks", "project_id",      "TEXT",      "'unknown'")
    _ensure_column(conn, "tasks", "retrieval_count", "INTEGER",   "0")
    _ensure_column(conn, "tasks", "completed",       "BOOLEAN",   "FALSE")
    _ensure_column(conn, "tasks", "created_at",      "TIMESTAMP", "CURRENT_TIMESTAMP")

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


# ─── Task snapshots ───────────────────────────────────────────────────────────

def create_task_snapshot(
    project_id: str,
    session_id: str,
    task_type: str,
    orig_prompt: str,
    summary: str,
) -> None:
    combined = f"{task_type}: {orig_prompt} {summary}".strip()
    emb = embed_text(combined)
    conn = init_db()
    conn.execute(
        """INSERT INTO tasks
               (project_id, session_id, task_type, orig_prompt, summary, embedding)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (project_id, session_id, task_type, orig_prompt, summary, json.dumps(emb)),
    )
    conn.commit()
    conn.close()


def retrieve_similar_tasks(
    project_id: str,
    session_id: str,
    prompt: str,
    top_n: int = 3,
    threshold: float = 0.25,
) -> list[dict]:
    """Feature 3 + Feature 7: weighted scoring with completion bonus and decay."""
    conn = init_db()
    cursor = conn.execute(
        """SELECT task_type, orig_prompt, summary, embedding,
                  retrieval_count, completed, created_at
           FROM tasks
           WHERE project_id = ?
           ORDER BY id DESC LIMIT 200""",
        (project_id,),
    )
    rows = cursor.fetchall()

    max_rc_row = conn.execute(
        "SELECT COALESCE(MAX(retrieval_count), 1) FROM tasks WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    conn.close()

    if not rows:
        return []

    prompt_emb = embed_text(prompt)
    now = datetime.now(timezone.utc)
    max_rc = max_rc_row[0] if max_rc_row and max_rc_row[0] else 1

    scored: list[tuple[float, str, str, str]] = []
    for task_type, orig_prompt, summary, emb_str, rc, completed, created_at in rows:
        if not emb_str:
            continue
        emb = json.loads(emb_str)

        # 1. Semantic similarity (weight 0.5)
        sem = cosine_similarity(prompt_emb, emb)

        # 2. Recency score (weight 0.3)
        try:
            ca_dt = _parse_dt(created_at)
        except Exception:
            ca_dt = now
        days = max(0, (now - ca_dt).days)
        recency = 1 / (1 + days)

        # 3. Frequency score (weight 0.2)
        freq = (rc / max_rc) if max_rc > 0 else 0

        # Feature 3: completion bonus
        weighted = (0.5 * sem + 0.3 * recency + 0.2 * freq) * (1.2 if completed else 1.0)

        # Feature 7: snapshot recency decay (tasks decay faster than facts)
        decay = 1 / (1 + 0.02 * days)
        final_score = weighted * decay

        scored.append((final_score, task_type, orig_prompt, summary))

    scored.sort(reverse=True, key=lambda x: x[0])

    # Feature 4: configurable threshold; also increment retrieval_count
    results: list[dict] = []
    conn = init_db()
    for score, task_type, orig_prompt, summary in scored[:top_n]:
        if score >= threshold:
            results.append({
                "type":    task_type,
                "prompt":  orig_prompt,
                "summary": summary,
                "score":   round(score, 4),
            })
            conn.execute(
                """UPDATE tasks
                   SET retrieval_count = retrieval_count + 1
                   WHERE orig_prompt = ? AND project_id = ?""",
                (orig_prompt, project_id),
            )
    conn.commit()
    conn.close()
    return results


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

    if cmd == "create_snapshot":
        project_id, session_id, task_type, orig_prompt, summary = (
            sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
        )
        create_task_snapshot(project_id, session_id, task_type, orig_prompt, summary)

    elif cmd == "retrieve_similar":
        project_id, session_id, prompt = sys.argv[2], sys.argv[3], sys.argv[4]
        top_n     = int(sys.argv[5])   if len(sys.argv) > 5 else 3
        threshold = float(sys.argv[6]) if len(sys.argv) > 6 else 0.25
        print(json.dumps(retrieve_similar_tasks(
            project_id, session_id, prompt, top_n, threshold
        )))

    else:
        print(json.dumps({"error": f"Unknown command: {cmd}"}), file=sys.stderr)
        sys.exit(1)
