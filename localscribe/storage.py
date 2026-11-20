"""SQLite session library with FTS5 full-text search across all transcripts."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import DEFAULT_DB_PATH
from .transcript import TranscriptEntry

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    started_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'system',
    model TEXT NOT NULL DEFAULT 'base',
    language TEXT NOT NULL DEFAULT 'auto',
    entry_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    text TEXT NOT NULL,
    start_seconds REAL NOT NULL DEFAULT 0.0,
    end_seconds REAL NOT NULL DEFAULT 0.0,
    starred INTEGER NOT NULL DEFAULT 0,
    speaker TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_entries_session ON entries(session_id, position);

CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
    text,
    session_id UNINDEXED,
    entry_id UNINDEXED
);
"""


@dataclass(frozen=True)
class SessionSummary:
    id: int
    title: str
    started_at: str
    source: str
    model: str
    language: str
    entry_count: int


def _auto_title(text: str, started_at: str) -> str:
    words = text.strip().split()
    if not words:
        return f"Session {started_at}"
    snippet = " ".join(words[:7])
    return snippet[:46] + ("…" if len(snippet) > 46 or len(words) > 7 else "")


def _fts_query(raw: str) -> str:
    tokens = [
        "".join(ch for ch in word if ch.isalnum()) for word in raw.strip().split()
    ]
    tokens = [t for t in tokens if t]
    return " AND ".join(f'"{t}"*' for t in tokens)


class SessionLibrary:
    """Thread-safe SQLite store for transcript sessions and FTS5 search."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path if db_path is not None else DEFAULT_DB_PATH
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _connection(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            with conn:
                yield conn

    def _init_db(self) -> None:
        with self._lock, self._connection() as conn:
            conn.executescript(SCHEMA)

    def create_session(
        self,
        title: str | None = None,
        source: str = "system",
        model: str = "base",
        language: str = "auto",
        started_at: datetime | None = None,
    ) -> int:
        stamp = (started_at or datetime.now()).strftime("%Y-%m-%d %H:%M")
        resolved_title = (title or "").strip() or f"Session {stamp}"
        with self._lock, self._connection() as conn:
            cur = conn.execute(
                "INSERT INTO sessions (title, started_at, source, model, language) "
                "VALUES (?, ?, ?, ?, ?)",
                (resolved_title, stamp, source, model, language),
            )
            return int(cur.lastrowid)

    def add_entry(self, session_id: int, entry: TranscriptEntry) -> int:
        cleaned = entry.text.strip()
        if not cleaned:
            return 0
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT entry_count, title, started_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return 0
            count, title, started_at = int(row[0]), str(row[1]), str(row[2])
            cur = conn.execute(
                "INSERT INTO entries "
                "(session_id, position, timestamp, text, start_seconds, end_seconds, "
                "starred, speaker) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    count,
                    entry.timestamp,
                    cleaned,
                    float(entry.start_seconds),
                    float(entry.end_seconds),
                    1 if entry.starred else 0,
                    entry.speaker,
                ),
            )
            entry_id = int(cur.lastrowid)
            conn.execute(
                "INSERT INTO entries_fts (rowid, text, session_id, entry_id) "
                "VALUES (?, ?, ?, ?)",
                (entry_id, cleaned, session_id, entry_id),
            )
            new_title = (
                _auto_title(cleaned, started_at)
                if count == 0 and title.startswith("Session ")
                else title
            )
            conn.execute(
                "UPDATE sessions SET entry_count = ?, title = ? WHERE id = ?",
                (count + 1, new_title, session_id),
            )
            return entry_id

    def set_starred(self, session_id: int, position: int, starred: bool) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE entries SET starred = ? WHERE session_id = ? AND position = ?",
                (1 if starred else 0, session_id, position),
            )

    def rename_session(self, session_id: int, title: str) -> None:
        cleaned = title.strip()
        if not cleaned:
            return
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?", (cleaned, session_id)
            )

    def delete_session(self, session_id: int) -> None:
        with self._lock, self._connection() as conn:
            conn.execute("DELETE FROM entries_fts WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def load_entries(self, session_id: int) -> list[TranscriptEntry]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT timestamp, text, start_seconds, end_seconds, starred, speaker "
                "FROM entries WHERE session_id = ? ORDER BY position ASC",
                (session_id,),
            ).fetchall()
        return [
            TranscriptEntry(
                timestamp=str(r[0]),
                text=str(r[1]),
                start_seconds=float(r[2]),
                end_seconds=float(r[3]),
                starred=bool(r[4]),
                speaker=str(r[5]),
            )
            for r in rows
        ]

    def list_sessions(self, query: str = "", limit: int = 100) -> list[SessionSummary]:
        fts = _fts_query(query)
        with self._lock, self._connection() as conn:
            if not fts:
                rows = conn.execute(
                    "SELECT id, title, started_at, source, model, language, entry_count "
                    "FROM sessions WHERE entry_count > 0 ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                like = f"%{query.strip()}%"
                rows = conn.execute(
                    "SELECT s.id, s.title, s.started_at, s.source, s.model, "
                    "s.language, s.entry_count "
                    "FROM sessions s "
                    "WHERE s.entry_count > 0 AND ("
                    "s.title LIKE ? OR s.id IN ("
                    "SELECT CAST(session_id AS INTEGER) FROM entries_fts "
                    "WHERE entries_fts MATCH ?"
                    ")) "
                    "ORDER BY s.id DESC LIMIT ?",
                    (like, fts, limit),
                ).fetchall()
        return [
            SessionSummary(
                id=int(r[0]),
                title=str(r[1]),
                started_at=str(r[2]),
                source=str(r[3]),
                model=str(r[4]),
                language=str(r[5]),
                entry_count=int(r[6]),
            )
            for r in rows
        ]
