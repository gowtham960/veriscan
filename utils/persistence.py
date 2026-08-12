"""
Persistence layer — audit log storage + human-review queue.

Uses SQLite: free, zero external dependency, and genuinely testable (Supabase
would require network access this environment doesn't have, and adds a paid/
hosted dependency for what's fundamentally the same relational data). Swapping
to Supabase/Postgres later is a connection-string change, not a schema rewrite
— the SQL here is plain, portable SQL.

Two tables:
  - audit_log: immutable, append-only record of every node execution across
    every run (matches the architecture doc's governance requirement)
  - review_queue: runs flagged requires_human_review=True, with a status a
    human reviewer can update (pending -> approved/rejected) — this is the
    actual "human-in-the-loop gate" made concrete, not just a boolean that
    nothing acts on
"""
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager

DB_PATH = "veriscan.db"


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                node TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                input_hash TEXT,
                output_summary TEXT,
                model_used TEXT,
                tokens_used INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS review_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                raw_input TEXT,
                final_verdict TEXT,
                confidence_score REAL,
                faithfulness_score REAL,
                reason TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                reviewed_at TEXT,
                reviewer_notes TEXT
            )
        """)


def persist_run(run_id: str, audit_entries: list) -> None:
    """Writes every audit entry from one graph run into the immutable audit_log table."""
    with _connect() as conn:
        for entry in audit_entries:
            conn.execute(
                "INSERT INTO audit_log (run_id, node, timestamp, input_hash, output_summary, model_used, tokens_used) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    entry.get("node", ""),
                    entry.get("timestamp", ""),
                    entry.get("input_hash", ""),
                    entry.get("output_summary", ""),
                    entry.get("model_used", ""),
                    entry.get("tokens_used", 0),
                ),
            )


def enqueue_for_review(run_id: str, raw_input: str, final_verdict: str,
                        confidence_score: float, faithfulness_score: float, reason: str) -> None:
    """Inserts a run into the human-review queue. This is the concrete action
    behind requires_human_review=True — without this, that flag was previously
    just a value in the return dict that nothing ever consumed."""
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO review_queue "
            "(run_id, created_at, raw_input, final_verdict, confidence_score, faithfulness_score, reason, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')",
            (run_id, datetime.now(timezone.utc).isoformat(), raw_input, final_verdict,
             confidence_score, faithfulness_score, reason),
        )


def get_pending_reviews() -> list:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM review_queue WHERE status = 'pending' ORDER BY created_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def resolve_review(run_id: str, status: str, notes: str = "") -> bool:
    """status should be 'approved' or 'rejected'."""
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE review_queue SET status = ?, reviewed_at = ?, reviewer_notes = ? WHERE run_id = ?",
            (status, datetime.now(timezone.utc).isoformat(), notes, run_id),
        )
        return cursor.rowcount > 0


def get_audit_trail(run_id: str) -> list:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log WHERE run_id = ? ORDER BY id ASC", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def new_run_id() -> str:
    return str(uuid.uuid4())
