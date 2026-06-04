#!/usr/bin/env python3
"""Context Broker — MCP memory/context resolution layer.

Resolves missing context for the user's LLM using stored conversation
history, project/topic state, slot fills, and long-term memory.

Primary entry points:
    resolve_missing_context(...)  — fill what the LLM lacks
    submit_clarification_answer(...) — save user's answer, update slots
    save_turn(...)                — persist raw user/assistant turn
    get_active_task(...)          — load current task whiteboard

Search order (per design):
    1. Active task/session state
    2. Current project/topic memory
    3. Same conversation outside LLM's context window
    4. Older messages in same conversation
    5. Recent active projects list
    6. Other conversations in suspected project
    7. Long-term user/project memory
    8. Global semantic fact search
    9. Return needs_user with options
"""

import hashlib
import json
import os
import sqlite3
import time
from typing import Any

from utils import embed_text, cosine_similarity

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.db")

# ── Confidence thresholds ─────────────────────────────────────────────────────
_AUTOFILL_THRESHOLD    = 0.72   # autofill without asking
_PARTIAL_THRESHOLD     = 0.45   # include as suggestion, flag uncertain
_TOPIC_AUTOFILL        = 0.80   # auto-select project/topic without asking

# ── Task frame registry ───────────────────────────────────────────────────────
# Each frame defines which slots the LLM needs to answer/act correctly.
# The slot_names list is ordered by priority (earlier = ask first if missing).
TASK_FRAMES: dict[str, dict] = {
    "clarify_referent": {
        "description": "Resolve what a vague reference ('it', 'that', 'same') refers to.",
        "slot_names": ["current_referent", "active_topic"],
        "triggers": ["it", "that", "same", "this", "previous", "earlier",
                     "continue", "that run", "better one", "the one"],
    },
    "continue_existing_task": {
        "description": "Resume a task from a previous session.",
        "slot_names": ["active_topic", "active_task", "current_phase",
                       "open_threads", "recent_decisions", "next_action"],
        "triggers": ["continue", "resume", "where we left", "pick up",
                     "last time", "from before", "keep going"],
    },
    "debug_error": {
        "description": "Debug a code error or unexpected behavior.",
        "slot_names": ["project", "active_files", "error_message",
                       "command", "expected_behavior", "actual_behavior",
                       "attempts_tried"],
        "triggers": ["error", "crash", "fail", "bug", "exception",
                     "broken", "doesn't work", "not working", "traceback"],
    },
    "run_benchmark": {
        "description": "Run an evaluation or benchmark.",
        "slot_names": ["dataset", "db_path", "rrf_k", "bm25_weight",
                       "use_llm_extractor", "tag", "success_metric"],
        "triggers": ["run", "eval", "benchmark", "ablation", "recall",
                     "locomo", "evaluate"],
    },
    "compare_experiments": {
        "description": "Compare two experiment results.",
        "slot_names": ["baseline_run", "candidate_run", "metrics",
                       "acceptance_rule", "latest_result_status"],
        "triggers": ["compare", "vs", "better", "worse", "difference",
                     "baseline", "improvement", "regress"],
    },
    "choose_config": {
        "description": "Select an optimal configuration.",
        "slot_names": ["config_options", "success_metric", "current_best",
                       "tradeoffs"],
        "triggers": ["config", "setting", "parameter", "which one",
                     "best option", "which is better"],
    },
    "answer_memory_question": {
        "description": "Answer a factual question about past user/project context.",
        "slot_names": ["subject", "attribute", "time_scope"],
        "triggers": ["did i", "did we", "what did", "when did",
                     "do you remember", "what was", "have i ever",
                     "what is my", "what's my"],
    },
    "summarize_status": {
        "description": "Summarize current project or task status.",
        "slot_names": ["active_topic", "active_task", "open_threads",
                       "recent_decisions", "latest_result_status"],
        "triggers": ["status", "where are we", "what's left",
                     "progress", "summary", "overview"],
    },
}


# ── DB init for context broker tables ─────────────────────────────────────────

def _init_broker_tables(conn: sqlite3.Connection) -> None:
    """Create context broker tables if they don't exist."""

    # Active task/session whiteboard — one row per (project_id, session_id)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS active_task_state (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id      TEXT NOT NULL,
            session_id      TEXT NOT NULL,
            active_topic    TEXT,
            active_task     TEXT,
            current_phase   TEXT,
            open_threads    TEXT,          -- JSON list
            recent_decisions TEXT,         -- JSON list
            active_files    TEXT,          -- JSON list
            active_runs     TEXT,          -- JSON list
            unresolved_questions TEXT,     -- JSON list
            known_good_configs   TEXT,     -- JSON dict
            updated_at      REAL DEFAULT (unixepoch()),
            UNIQUE(project_id, session_id)
        )
    """)

    # Structured slot values with scope, confidence, provenance
    conn.execute("""
        CREATE TABLE IF NOT EXISTS context_slots (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id       TEXT NOT NULL,
            session_id       TEXT,
            slot_name        TEXT NOT NULL,
            value            TEXT NOT NULL,
            scope            TEXT DEFAULT 'session',   -- user|project|session|task
            confidence       REAL DEFAULT 0.5,
            confirmed_by_user INTEGER DEFAULT 0,       -- 1 = user confirmed
            source_fact_ids  TEXT,                     -- JSON list of fact ids
            source_message_ids TEXT,                   -- JSON list of message ids
            updated_at       REAL DEFAULT (unixepoch()),
            UNIQUE(project_id, session_id, slot_name)
        )
    """)

    # Project/topic index — track known topics per project
    conn.execute("""
        CREATE TABLE IF NOT EXISTS project_topics (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id     TEXT NOT NULL,
            topic_label    TEXT NOT NULL,
            last_active    REAL DEFAULT (unixepoch()),
            status_summary TEXT,
            linked_sessions TEXT,     -- JSON list of session_ids
            linked_files    TEXT,     -- JSON list of file paths
            linked_runs     TEXT,     -- JSON list of run tags
            UNIQUE(project_id, topic_label)
        )
    """)

    # Raw conversation transcript (separate from facts — always preserved)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_turns (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id   TEXT NOT NULL,
            session_id   TEXT NOT NULL,
            turn_index   INTEGER NOT NULL,
            role         TEXT NOT NULL,   -- 'user' | 'assistant'
            content      TEXT NOT NULL,
            embedding    BLOB,
            created_at   REAL DEFAULT (unixepoch())
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_turns_session
        ON conversation_turns(project_id, session_id, turn_index)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_turns_time
        ON conversation_turns(project_id, created_at DESC)
    """)

    conn.commit()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    _init_broker_tables(conn)
    return conn


# ── Save turn ─────────────────────────────────────────────────────────────────

def save_turn(
    project_id: str,
    session_id: str,
    role: str,
    content: str,
    embed: bool = True,
) -> int:
    """Persist a raw conversation turn. Returns the new turn id."""
    conn = _get_conn()
    idx = (conn.execute(
        "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM conversation_turns "
        "WHERE project_id = ? AND session_id = ?",
        (project_id, session_id),
    ).fetchone()[0])
    emb_blob: bytes | None = None
    if embed:
        try:
            import struct
            vec = embed_text(content)
            emb_blob = struct.pack(f"{len(vec)}f", *vec)
        except Exception:
            pass
    cur = conn.execute(
        """INSERT INTO conversation_turns
           (project_id, session_id, turn_index, role, content, embedding, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (project_id, session_id, idx, role, content, emb_blob, time.time()),
    )
    turn_id = cur.lastrowid
    conn.commit()
    conn.close()
    return turn_id


# ── Active task state ─────────────────────────────────────────────────────────

def get_active_task(project_id: str, session_id: str) -> dict:
    """Return the current task whiteboard for this project/session."""
    conn = _get_conn()
    row = conn.execute(
        """SELECT active_topic, active_task, current_phase, open_threads,
                  recent_decisions, active_files, active_runs,
                  unresolved_questions, known_good_configs
           FROM active_task_state
           WHERE project_id = ? AND session_id = ?""",
        (project_id, session_id),
    ).fetchone()
    conn.close()
    if not row:
        return {}
    keys = ["active_topic", "active_task", "current_phase", "open_threads",
            "recent_decisions", "active_files", "active_runs",
            "unresolved_questions", "known_good_configs"]
    result = {}
    for k, v in zip(keys, row):
        if v and k in ("open_threads", "recent_decisions", "active_files",
                        "active_runs", "unresolved_questions"):
            try:
                result[k] = json.loads(v)
            except Exception:
                result[k] = v
        elif v and k == "known_good_configs":
            try:
                result[k] = json.loads(v)
            except Exception:
                result[k] = v
        elif v:
            result[k] = v
    return result


def update_active_task(project_id: str, session_id: str, **fields) -> None:
    """Upsert the active task whiteboard. Pass any subset of fields."""
    conn = _get_conn()
    existing = get_active_task(project_id, session_id)
    # Merge new fields into existing
    for k, v in fields.items():
        if isinstance(v, (list, dict)):
            existing[k] = v
        elif v is not None:
            existing[k] = v

    def _j(k: str):
        val = existing.get(k)
        if val is None:
            return None
        if isinstance(val, (list, dict)):
            return json.dumps(val)
        return val

    conn.execute(
        """INSERT INTO active_task_state
           (project_id, session_id, active_topic, active_task, current_phase,
            open_threads, recent_decisions, active_files, active_runs,
            unresolved_questions, known_good_configs, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(project_id, session_id) DO UPDATE SET
               active_topic         = excluded.active_topic,
               active_task          = excluded.active_task,
               current_phase        = excluded.current_phase,
               open_threads         = excluded.open_threads,
               recent_decisions     = excluded.recent_decisions,
               active_files         = excluded.active_files,
               active_runs          = excluded.active_runs,
               unresolved_questions = excluded.unresolved_questions,
               known_good_configs   = excluded.known_good_configs,
               updated_at           = excluded.updated_at""",
        (
            project_id, session_id,
            existing.get("active_topic"),
            existing.get("active_task"),
            existing.get("current_phase"),
            _j("open_threads"),
            _j("recent_decisions"),
            _j("active_files"),
            _j("active_runs"),
            _j("unresolved_questions"),
            _j("known_good_configs"),
            time.time(),
        ),
    )
    conn.commit()
    conn.close()


# ── Context slots ─────────────────────────────────────────────────────────────

def get_slot(project_id: str, session_id: str, slot_name: str) -> dict | None:
    """Return a context slot value with metadata, or None if not found."""
    conn = _get_conn()
    row = conn.execute(
        """SELECT value, scope, confidence, confirmed_by_user, updated_at
           FROM context_slots
           WHERE project_id = ? AND session_id = ? AND slot_name = ?
           ORDER BY confidence DESC LIMIT 1""",
        (project_id, session_id, slot_name),
    ).fetchone()
    # Also check project-scoped and user-scoped slots
    if row is None:
        row = conn.execute(
            """SELECT value, scope, confidence, confirmed_by_user, updated_at
               FROM context_slots
               WHERE project_id = ? AND slot_name = ?
                 AND scope IN ('project', 'user')
               ORDER BY confidence DESC LIMIT 1""",
            (project_id, slot_name),
        ).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "value": row[0],
        "scope": row[1],
        "confidence": row[2],
        "confirmed_by_user": bool(row[3]),
        "updated_at": row[4],
    }


def set_slot(
    project_id: str,
    session_id: str,
    slot_name: str,
    value: str,
    scope: str = "session",
    confidence: float = 0.8,
    confirmed_by_user: bool = False,
    source_fact_ids: list[int] | None = None,
) -> None:
    """Upsert a context slot value."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO context_slots
           (project_id, session_id, slot_name, value, scope, confidence,
            confirmed_by_user, source_fact_ids, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(project_id, session_id, slot_name) DO UPDATE SET
               value             = excluded.value,
               scope             = excluded.scope,
               confidence        = excluded.confidence,
               confirmed_by_user = excluded.confirmed_by_user,
               source_fact_ids   = excluded.source_fact_ids,
               updated_at        = excluded.updated_at""",
        (
            project_id, session_id, slot_name, value, scope,
            confidence, int(confirmed_by_user),
            json.dumps(source_fact_ids) if source_fact_ids else None,
            time.time(),
        ),
    )
    conn.commit()
    conn.close()


# ── Project/topic index ───────────────────────────────────────────────────────

def upsert_topic(
    project_id: str,
    topic_label: str,
    status_summary: str | None = None,
    session_id: str | None = None,
    files: list[str] | None = None,
    run_tags: list[str] | None = None,
) -> None:
    conn = _get_conn()
    existing = conn.execute(
        "SELECT linked_sessions, linked_files, linked_runs FROM project_topics "
        "WHERE project_id = ? AND topic_label = ?",
        (project_id, topic_label),
    ).fetchone()
    if existing:
        sessions = json.loads(existing[0] or "[]")
        linked_files = json.loads(existing[1] or "[]")
        linked_runs = json.loads(existing[2] or "[]")
        if session_id and session_id not in sessions:
            sessions.append(session_id)
        if files:
            for f in files:
                if f not in linked_files:
                    linked_files.append(f)
        if run_tags:
            for r in run_tags:
                if r not in linked_runs:
                    linked_runs.append(r)
        conn.execute(
            """UPDATE project_topics SET
               last_active = ?, status_summary = ?,
               linked_sessions = ?, linked_files = ?, linked_runs = ?
               WHERE project_id = ? AND topic_label = ?""",
            (
                time.time(),
                status_summary or existing[0],
                json.dumps(sessions), json.dumps(linked_files), json.dumps(linked_runs),
                project_id, topic_label,
            ),
        )
    else:
        conn.execute(
            """INSERT INTO project_topics
               (project_id, topic_label, last_active, status_summary,
                linked_sessions, linked_files, linked_runs)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                project_id, topic_label, time.time(), status_summary,
                json.dumps([session_id] if session_id else []),
                json.dumps(files or []),
                json.dumps(run_tags or []),
            ),
        )
    conn.commit()
    conn.close()


def get_recent_topics(project_id: str, limit: int = 5) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        """SELECT topic_label, last_active, status_summary, linked_sessions,
                  linked_files, linked_runs
           FROM project_topics
           WHERE project_id = ?
           ORDER BY last_active DESC LIMIT ?""",
        (project_id, limit),
    ).fetchall()
    conn.close()
    topics = []
    for r in rows:
        topics.append({
            "topic": r[0],
            "last_active": r[1],
            "status_summary": r[2],
            "sessions": json.loads(r[3] or "[]"),
            "files": json.loads(r[4] or "[]"),
            "runs": json.loads(r[5] or "[]"),
        })
    return topics


def get_all_recent_topics(limit: int = 10) -> list[dict]:
    """Get recently active topics across all projects."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT project_id, topic_label, last_active, status_summary
           FROM project_topics
           ORDER BY last_active DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        {"project_id": r[0], "topic": r[1], "last_active": r[2], "status_summary": r[3]}
        for r in rows
    ]


# ── Conversation search ───────────────────────────────────────────────────────

def _decode_blob(blob) -> list[float] | None:
    if blob is None:
        return None
    if isinstance(blob, (bytes, bytearray)):
        import struct
        n = len(blob) // 4
        return list(struct.unpack(f"{n}f", blob))
    return None


def search_conversation_history(
    project_id: str,
    session_id: str,
    query: str,
    exclude_last_n: int = 20,
    limit: int = 5,
) -> list[dict]:
    """Search same-conversation turns that are outside the LLM's visible window.

    exclude_last_n: how many recent turns to skip (assumed in LLM's context).
    """
    conn = _get_conn()
    # Get all turns except the most recent `exclude_last_n`
    max_idx = conn.execute(
        "SELECT COALESCE(MAX(turn_index), -1) FROM conversation_turns "
        "WHERE project_id = ? AND session_id = ?",
        (project_id, session_id),
    ).fetchone()[0]
    cutoff = max(0, max_idx - exclude_last_n)

    rows = conn.execute(
        """SELECT id, turn_index, role, content, embedding
           FROM conversation_turns
           WHERE project_id = ? AND session_id = ? AND turn_index < ?
           ORDER BY turn_index DESC""",
        (project_id, session_id, cutoff),
    ).fetchall()
    conn.close()

    if not rows:
        return []

    try:
        q_emb = embed_text(query)
        scored = []
        for fid, idx, role, content, emb_blob in rows:
            emb = _decode_blob(emb_blob)
            sim = cosine_similarity(q_emb, emb) if emb else 0.0
            scored.append((sim, fid, idx, role, content))
        scored.sort(key=lambda x: -x[0])
        return [
            {"turn_index": r[2], "role": r[3], "content": r[4], "score": r[0]}
            for r in scored[:limit]
        ]
    except Exception:
        # Fallback: return most recent turns if embeddings fail
        return [
            {"turn_index": r[1], "role": r[2], "content": r[3], "score": 0.0}
            for r in rows[:limit]
        ]


def search_cross_session(
    project_id: str,
    exclude_session_id: str,
    query: str,
    limit: int = 5,
) -> list[dict]:
    """Search turns from other sessions in the same project."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, session_id, turn_index, role, content, embedding
           FROM conversation_turns
           WHERE project_id = ? AND session_id != ?
           ORDER BY created_at DESC LIMIT 500""",
        (project_id, exclude_session_id),
    ).fetchall()
    conn.close()

    if not rows:
        return []

    try:
        q_emb = embed_text(query)
        scored = []
        for fid, sid, idx, role, content, emb_blob in rows:
            emb = _decode_blob(emb_blob)
            sim = cosine_similarity(q_emb, emb) if emb else 0.0
            scored.append((sim, sid, idx, role, content))
        scored.sort(key=lambda x: -x[0])
        return [
            {"session_id": r[1], "turn_index": r[2], "role": r[3],
             "content": r[4], "score": r[0]}
            for r in scored[:limit]
        ]
    except Exception:
        return [
            {"session_id": r[1], "turn_index": r[2], "role": r[3],
             "content": r[4], "score": 0.0}
            for r in rows[:limit]
        ]


# ── Intent / frame detection ──────────────────────────────────────────────────

def detect_task_frame(user_message: str, missing: list[str]) -> str | None:
    """Detect the most likely task frame from the user message and missing slots."""
    import re as _re
    msg_lower = user_message.lower()
    missing_text = " ".join(missing).lower()
    combined = msg_lower + " " + missing_text

    scores: dict[str, int] = {}
    for frame_name, frame_def in TASK_FRAMES.items():
        score = sum(
            1 for t in frame_def["triggers"]
            if _re.search(r"\b" + _re.escape(t) + r"\b", combined)
        )
        if score > 0:
            scores[frame_name] = score

    if not scores:
        return None
    return max(scores, key=lambda k: scores[k])


def get_frame_slots(frame_name: str) -> list[str]:
    if frame_name in TASK_FRAMES:
        return TASK_FRAMES[frame_name]["slot_names"]
    return []


# ── Confidence scoring ────────────────────────────────────────────────────────

def _score_slot_confidence(
    slot_name: str,
    value: str,
    evidence: list[dict],
    confirmed_by_user: bool,
    is_recent: bool,
) -> float:
    score = 0.0
    if confirmed_by_user:
        return 1.0
    if evidence:
        score += 0.35 * min(1.0, len(evidence) / 3)  # multiple sources
        if is_recent:
            score += 0.15
        top_sim = max((e.get("score", 0.0) for e in evidence), default=0.0)
        score += 0.35 * top_sim
    score += 0.15  # prior for having a candidate at all
    return min(0.99, score)


# ── Topic resolver ────────────────────────────────────────────────────────────

def resolve_topic(
    project_id: str,
    session_id: str,
    user_message: str,
    project_hint: str | None = None,
    candidate_topics: list[str] | None = None,
) -> dict:
    """Resolve the most likely project/topic for this request.

    Returns:
        {
            "status": "resolved" | "needs_user",
            "topic": str | None,
            "confidence": float,
            "options": list[dict]  # only when needs_user
        }
    """
    # 1. Check active task state first
    task_state = get_active_task(project_id, session_id)
    if task_state.get("active_topic"):
        return {
            "status": "resolved",
            "topic": task_state["active_topic"],
            "confidence": 0.90,
            "source": "active_task_state",
        }

    # 2. Check project hint from LLM
    if project_hint:
        return {
            "status": "resolved",
            "topic": project_hint,
            "confidence": 0.75,
            "source": "llm_hint",
        }

    # 3. Search recent topics for this project
    topics = get_recent_topics(project_id, limit=5)

    # 4. If only one topic and it's recent (<1h), resolve it
    if len(topics) == 1:
        return {
            "status": "resolved",
            "topic": topics[0]["topic"],
            "confidence": 0.80,
            "source": "only_topic",
        }

    # 5. If multiple topics, try to match by semantic similarity to user message
    if topics:
        try:
            q_emb = embed_text(user_message)
            scored = []
            for t in topics:
                topic_text = f"{t['topic']} {t.get('status_summary', '')}"
                t_emb = embed_text(topic_text)
                sim = cosine_similarity(q_emb, t_emb)
                scored.append((sim, t))
            scored.sort(key=lambda x: -x[0])
            best_sim, best_topic = scored[0]
            if best_sim >= _TOPIC_AUTOFILL:
                return {
                    "status": "resolved",
                    "topic": best_topic["topic"],
                    "confidence": best_sim,
                    "source": "semantic_match",
                }
            # Multiple close candidates — ask user
            options = [
                {
                    "label": t["topic"],
                    "why": t.get("status_summary") or f"Last active {_format_age(t['last_active'])}",
                }
                for _, t in scored[:4]
            ]
            return {
                "status": "needs_user",
                "topic": None,
                "confidence": best_sim,
                "options": options + [{"label": "Something else", "why": ""}],
            }
        except Exception:
            pass

    # 6. Check all projects across DB for cold start ("continue where we left off")
    all_topics = get_all_recent_topics(limit=5)
    if all_topics:
        options = [
            {
                "label": t["topic"],
                "why": t.get("status_summary") or f"From project {t['project_id'][:6]}...",
            }
            for t in all_topics[:4]
        ]
        return {
            "status": "needs_user",
            "topic": None,
            "confidence": 0.0,
            "options": options + [{"label": "Something else", "why": ""}],
        }

    return {
        "status": "needs_user",
        "topic": None,
        "confidence": 0.0,
        "options": [{"label": "Something else", "why": "No previous topics found"}],
    }


def _format_age(ts: float) -> str:
    diff = time.time() - ts
    if diff < 3600:
        return f"{int(diff/60)}m ago"
    if diff < 86400:
        return f"{int(diff/3600)}h ago"
    return f"{int(diff/86400)}d ago"


# ── Slot autofill ─────────────────────────────────────────────────────────────

def _autofill_slot_from_memory(
    slot_name: str,
    project_id: str,
    session_id: str,
    user_message: str,
) -> dict | None:
    """Try to fill a slot from stored context/facts. Returns None if not found."""
    import memory as _mem

    # 1. Check context_slots table (previously confirmed/stored slots)
    existing = get_slot(project_id, session_id, slot_name)
    if existing and existing["confidence"] >= _AUTOFILL_THRESHOLD:
        return {
            "value": existing["value"],
            "confidence": existing["confidence"],
            "confirmed_by_user": existing["confirmed_by_user"],
            "source": "context_slot",
        }

    # 2. Check active task state for known fields that map to slots
    task_state = get_active_task(project_id, session_id)
    state_map = {
        "active_topic":      task_state.get("active_topic"),
        "active_task":       task_state.get("active_task"),
        "current_phase":     task_state.get("current_phase"),
        "next_action":       task_state.get("open_threads", [None])[0] if task_state.get("open_threads") else None,
        "recent_decisions":  json.dumps(task_state.get("recent_decisions", [])) if task_state.get("recent_decisions") else None,
        "active_files":      json.dumps(task_state.get("active_files", [])) if task_state.get("active_files") else None,
        "open_threads":      json.dumps(task_state.get("open_threads", [])) if task_state.get("open_threads") else None,
        "active_runs":       json.dumps(task_state.get("active_runs", [])) if task_state.get("active_runs") else None,
        "known_good_configs": json.dumps(task_state.get("known_good_configs", {})) if task_state.get("known_good_configs") else None,
    }
    if slot_name in state_map and state_map[slot_name] is not None:
        return {
            "value": state_map[slot_name],
            "confidence": 0.85,
            "confirmed_by_user": False,
            "source": "active_task_state",
        }

    # 3. Search conversation history (same session, outside context window)
    history = search_conversation_history(
        project_id, session_id, f"{slot_name} {user_message}", exclude_last_n=20, limit=5
    )
    if history and history[0]["score"] >= _PARTIAL_THRESHOLD:
        return {
            "value": history[0]["content"][:400],
            "confidence": _score_slot_confidence(
                slot_name, history[0]["content"], history, False, True
            ),
            "confirmed_by_user": False,
            "source": "conversation_history",
            "evidence_snippet": history[0]["content"][:200],
        }

    # 4. Search long-term memory (facts)
    try:
        result = _mem.retrieve_facts(
            project_id, session_id,
            f"{slot_name}: {user_message}",
            top_n=3, threshold=0.25, include_budget_info=True,
        )
        facts = result.get("facts", []) if isinstance(result, dict) else result
        if facts:
            return {
                "value": facts[0],
                "confidence": 0.60,
                "confirmed_by_user": False,
                "source": "long_term_memory",
                "evidence_snippet": facts[0][:200],
            }
    except Exception:
        pass

    return None


# ── Main: resolve_missing_context ─────────────────────────────────────────────

def resolve_missing_context(
    missing: list[str],
    project_id: str,
    session_id: str,
    user_message: str = "",
    project_hint: str | None = None,
    candidate_topics: list[str] | None = None,
    context_window_size: int = 20,
) -> dict:
    """Primary MCP tool: resolve what the user's LLM doesn't know.

    Args:
        missing: list of things the LLM lacks, e.g.
                 ["what does 'it' refer to?", "what was the accepted config?"]
        project_id: current project id
        session_id: current session/conversation id
        user_message: the current user message (for semantic search)
        project_hint: optional topic/project guess from the LLM
        candidate_topics: optional list of what LLM thinks user may mean
        context_window_size: how many recent turns to treat as already visible

    Returns:
        {
            "status": "resolved" | "partial" | "needs_user",
            "resolved_project": str,
            "filled": {slot_name: {value, confidence, source}},
            "evidence": [str],
            "task_state": dict,
            "needs_user_questions": [{"slot", "question", "options", "why"}],
        }
    """
    result: dict[str, Any] = {
        "status": "resolved",
        "resolved_project": project_id,
        "filled": {},
        "evidence": [],
        "task_state": {},
        "needs_user_questions": [],
    }

    # ── Step 1: Resolve topic / project scope ─────────────────────────────────
    topic_res = resolve_topic(
        project_id, session_id, user_message,
        project_hint=project_hint, candidate_topics=candidate_topics,
    )
    if topic_res["status"] == "needs_user":
        result["status"] = "needs_user"
        result["needs_user_questions"].append({
            "slot": "active_topic",
            "question": "Which topic or project are you referring to?",
            "options": topic_res.get("options", []),
            "why": "I couldn't determine the current topic from context.",
        })
    else:
        topic = topic_res["topic"]
        result["resolved_project"] = topic or project_id
        # Update active task state with resolved topic
        if topic:
            update_active_task(project_id, session_id, active_topic=topic)

    # ── Step 2: Load task state ────────────────────────────────────────────────
    task_state = get_active_task(project_id, session_id)
    result["task_state"] = task_state

    # ── Step 3: Detect task frame from missing items ───────────────────────────
    frame = detect_task_frame(user_message, missing)
    required_slots = get_frame_slots(frame) if frame else []

    # ── Step 4: Autofill — try to resolve each missing item from memory ────────
    # Build a unified list of slot names from `missing` strings + frame slots
    all_slots_to_fill: list[str] = []
    # Map missing natural-language items to known slot names
    for item in missing:
        item_lower = item.lower()
        matched = False
        for frame_name, frame_def in TASK_FRAMES.items():
            for slot in frame_def["slot_names"]:
                if slot in item_lower or item_lower in slot:
                    if slot not in all_slots_to_fill:
                        all_slots_to_fill.append(slot)
                    matched = True
                    break
        if not matched:
            # Use the missing string itself as a free-text slot key
            slug = item_lower.replace(" ", "_").replace("?", "")[:40]
            if slug not in all_slots_to_fill:
                all_slots_to_fill.append(slug)

    # Add frame-required slots not already in list
    for s in required_slots:
        if s not in all_slots_to_fill:
            all_slots_to_fill.append(s)

    unresolved_slots: list[str] = []
    for slot_name in all_slots_to_fill:
        filled = _autofill_slot_from_memory(
            slot_name, project_id, session_id, user_message
        )
        if filled and filled["confidence"] >= _AUTOFILL_THRESHOLD:
            result["filled"][slot_name] = filled
            if filled.get("evidence_snippet"):
                result["evidence"].append(filled["evidence_snippet"])
        elif filled and filled["confidence"] >= _PARTIAL_THRESHOLD:
            # Include with uncertainty flag
            result["filled"][slot_name] = {**filled, "uncertain": True}
        else:
            unresolved_slots.append(slot_name)

    # ── Step 5: Same-conversation history search for evidence ──────────────────
    if user_message:
        history_hits = search_conversation_history(
            project_id, session_id, user_message,
            exclude_last_n=context_window_size, limit=5,
        )
        for h in history_hits:
            snippet = f"[Turn {h['turn_index']}, {h['role']}]: {h['content'][:300]}"
            if snippet not in result["evidence"]:
                result["evidence"].append(snippet)

    # ── Step 6: Cross-session search for remaining unresolved items ───────────
    if unresolved_slots:
        for slot in unresolved_slots[:3]:  # limit cross-session lookups
            xhits = search_cross_session(
                project_id, session_id,
                f"{slot} {user_message}", limit=3,
            )
            for h in xhits:
                if h["score"] >= _PARTIAL_THRESHOLD:
                    result["evidence"].append(
                        f"[From previous session, {h['role']}]: {h['content'][:200]}"
                    )

    # ── Step 7: Long-term memory retrieval ────────────────────────────────────
    if user_message:
        try:
            import memory as _mem
            mem_result = _mem.retrieve_facts(
                project_id, session_id, user_message,
                top_n=5, threshold=0.25, include_budget_info=True,
            )
            facts = mem_result.get("facts", []) if isinstance(mem_result, dict) else mem_result
            for f in facts:
                if f not in result["evidence"]:
                    result["evidence"].append(f)
        except Exception:
            pass

    # ── Step 8: Generate clarification questions for unresolved high-impact slots
    high_impact = {"current_referent", "active_topic", "active_task", "project",
                   "error_message", "baseline_run", "candidate_run"}
    for slot in unresolved_slots:
        if slot in high_impact or slot in [s for s in required_slots]:
            question, options = _build_clarification(
                slot, project_id, session_id, user_message
            )
            result["needs_user_questions"].append({
                "slot": slot,
                "question": question,
                "options": options,
                "why": f"Required for: {frame or 'current request'}",
            })

    # ── Determine overall status ───────────────────────────────────────────────
    if result["needs_user_questions"]:
        result["status"] = "needs_user"
    elif unresolved_slots:
        result["status"] = "partial"
    else:
        result["status"] = "resolved"

    return result


# ── Clarification builder ─────────────────────────────────────────────────────

def _build_clarification(
    slot_name: str,
    project_id: str,
    session_id: str,
    user_message: str,
) -> tuple[str, list[dict]]:
    """Build a clarification question + options for a missing slot."""
    templates: dict[str, str] = {
        "current_referent":  "When you say 'it' / 'that', which one do you mean?",
        "active_topic":      "Which topic or project are you referring to?",
        "active_task":       "Which task should I resume?",
        "project":           "Which project does this relate to?",
        "baseline_run":      "Which run should be used as the baseline?",
        "candidate_run":     "Which run should be compared against the baseline?",
        "error_message":     "Could you paste the error message or traceback?",
        "active_files":      "Which file(s) are you working on?",
        "db_path":           "Which database file should I use?",
        "success_metric":    "What is the target metric for this?",
    }
    question = templates.get(slot_name, f"What is the {slot_name.replace('_', ' ')}?")

    # Build options from recent topics / task state
    options: list[dict] = []
    topics = get_recent_topics(project_id, limit=5)
    if slot_name in ("current_referent", "active_topic", "active_task"):
        task_state = get_active_task(project_id, session_id)
        if task_state.get("open_threads"):
            for t in task_state["open_threads"][:3]:
                options.append({"label": str(t), "why": "Open thread"})
        if task_state.get("active_runs"):
            for r in task_state["active_runs"][:3]:
                options.append({"label": str(r), "why": "Active run"})
        for t in topics[:3]:
            lbl = t["topic"]
            if not any(o["label"] == lbl for o in options):
                options.append({
                    "label": lbl,
                    "why": t.get("status_summary") or f"Last active {_format_age(t['last_active'])}",
                })
    if not options:
        options.append({"label": "Something else", "why": "Not listed"})

    return question, options


# ── Submit clarification answer ───────────────────────────────────────────────

def submit_clarification_answer(
    slot_name: str,
    value: str,
    project_id: str,
    session_id: str,
    update_task_state: bool = True,
) -> dict:
    """Save a user's clarification answer. Update slots and active task state.

    Returns the updated context for the LLM to continue.
    """
    # Save as confirmed slot
    set_slot(
        project_id, session_id, slot_name, value,
        scope="session", confidence=1.0, confirmed_by_user=True,
    )

    # Update active task state for topic/task related answers
    task_updates = {}
    if slot_name == "active_topic":
        task_updates["active_topic"] = value
        upsert_topic(project_id, value, session_id=session_id)
    elif slot_name == "active_task":
        task_updates["active_task"] = value
    elif slot_name == "current_referent":
        task_updates.setdefault("open_threads", [])
    elif slot_name in ("active_files",):
        try:
            task_updates["active_files"] = json.loads(value) if value.startswith("[") else [value]
        except Exception:
            task_updates["active_files"] = [value]

    if task_updates:
        update_active_task(project_id, session_id, **task_updates)

    # Also save as a raw turn
    save_turn(project_id, session_id, "user_clarification", f"{slot_name}: {value}", embed=True)

    return {
        "saved": True,
        "slot": slot_name,
        "value": value,
        "task_state": get_active_task(project_id, session_id),
    }


# ── User LLM tool-use instructions ───────────────────────────────────────────

USER_LLM_INSTRUCTIONS = """
You have access to an external memory MCP server (preflight).
You are NOT expected to remember past user/project/session context yourself.

== WHEN TO CALL resolve_missing_context ==

Call it BEFORE answering when:
1. The user refers to previous work, sessions, experiments, files, decisions, configs, or preferences.
2. The user uses vague references: "it", "that", "same", "previous", "earlier", "continue",
   "the better one", "that run", "what we decided", "last time", "where we left off".
3. The answer requires knowing past state not visible in the current context window.
4. The request is multi-part, complex, debugging-related, or task-continuation-related.
5. The user asks about their preferences, setup, project config, or past choices.
6. You are below 90% confident about what past context the user is referring to.

== HOW TO CALL ==

Send ONLY what you are missing. Do NOT summarize the whole conversation.

Example:
{
  "missing": ["what does 'that run' refer to?", "what was the accepted retrieval config?"],
  "user_message": "compare that run with baseline and tell me if we should keep it",
  "project_hint": "locomo retrieval benchmark"
}

== AFTER RECEIVING THE RESPONSE ==

- If status is "resolved" or "partial": use filled slots + evidence to answer.
- If status is "needs_user": ask the user EXACTLY the first question with its options.
  Do not ask multiple clarification questions at once.
- If status is "needs_user" for "active_topic": present the options and let the user choose.
  Then call submit_clarification_answer with their choice.

== WHAT NOT TO DO ==

- Do not invent past facts, experiments, or decisions.
- Do not answer from vague memory when the MCP can resolve it.
- Do not send the full conversation history in the missing_context call.
- Do not ask the user open-ended "what do you mean?" — always use option lists when available.
""".strip()
