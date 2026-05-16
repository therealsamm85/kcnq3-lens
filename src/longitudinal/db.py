"""SQLite-backed local storage for longitudinal recordings + diary.

This is the load-bearing layer for v0.12+. Replaces the per-file JSON
storage in `recordings/*.json` and `diary.jsonl` with a single SQLite
database at `~/.kcnq3-lens/kcnq3-lens.db` (or `$KCNQ3_LENS_DATA/...`).

Why SQLite over the previous JSON-on-disk:
- O(1) lookup by date instead of full directory scan
- Transactional writes (no half-written JSON on crash)
- Single-file backup / sync / inspection
- Real queries for the longitudinal trends UI
- Foundation for the federated registry (v0.12.1+): the submission
  builder reads from the same DB rather than re-scanning JSON blobs.

Schema versioning
-----------------
- v1 (initial): recordings, diary, meta, schema_version
- Bump `_SCHEMA_VERSION` and add migration steps in `_apply_migrations`
  when changing schema. NEVER drop existing data without an explicit
  user-confirmed migration path.

Legacy JSON migration
---------------------
On first open, if legacy `recordings/*.json` or `diary.jsonl` exist,
they are imported into SQLite. A `meta` row records that this ran so it
never repeats. Legacy files are LEFT IN PLACE (not deleted) for safety —
user can manually clean up after verifying.

Thread safety
-------------
Each call opens a fresh short-lived connection. SQLite WAL mode is
enabled so concurrent readers + a writer don't block. Streamlit's
per-rerun model makes connection pooling unnecessary.

Privacy
-------
- DB file is on the user's local disk only. No network calls.
- No PHI normalization or de-id happens here — that's the registry
  submission layer's job (v0.12.1). This module trusts its callers
  to store whatever they want.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


_SCHEMA_VERSION = 2
_INIT_LOCK = threading.Lock()
_INIT_DONE: set[str] = set()  # paths that have been initialized this process


def default_db_path() -> Path:
    """Resolve the SQLite file path, honoring $KCNQ3_LENS_DATA."""
    base = Path(os.environ.get(
        "KCNQ3_LENS_DATA", str(Path.home() / ".kcnq3-lens")
    ))
    return base / "kcnq3-lens.db"


def resolve_db_path(path: Path | str | None) -> Path:
    """Resolve an optional override to an absolute DB path.

    Rules:
    - None  → default_db_path()
    - directory → directory/kcnq3-lens.db
    - file path → used as-is (any extension; we don't enforce .db)
    """
    if path is None:
        return default_db_path()
    p = Path(path)
    if p.exists() and p.is_dir():
        return p / "kcnq3-lens.db"
    # If the parent looks like a directory but file doesn't exist yet,
    # treat it as the intended file path.
    return p


# ─── Connection + schema init ──────────────────────────────────────────────

@contextmanager
def connect(path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """Open a short-lived connection. Initializes schema on first use."""
    db_path = resolve_db_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=10.0)
    conn.row_factory = sqlite3.Row
    # WAL gives us concurrent reads while a write is in flight.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")

    key = str(db_path.resolve())
    if key not in _INIT_DONE:
        with _INIT_LOCK:
            if key not in _INIT_DONE:
                _init_schema(conn)
                _maybe_migrate_legacy_json(conn, db_path.parent)
                _INIT_DONE.add(key)

    try:
        yield conn
    finally:
        conn.close()


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_date TEXT NOT NULL,
            label TEXT NOT NULL DEFAULT '',
            source_filename TEXT NOT NULL DEFAULT '',
            saved_at TEXT NOT NULL,
            findings_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_recordings_date
            ON recordings(recording_date);
        CREATE TABLE IF NOT EXISTS diary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            word_count INTEGER,
            concentration_minutes REAL,
            sleep_onset_difficulty TEXT,
            sleep_quality_1to5 INTEGER,
            medication_change TEXT,
            new_milestone TEXT,
            seizure_event TEXT,
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_diary_date ON diary(date);
        CREATE TABLE IF NOT EXISTS submissions_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            submission_id TEXT NOT NULL UNIQUE,
            opened_at TEXT NOT NULL,
            issue_url TEXT NOT NULL DEFAULT '',
            submission_json TEXT NOT NULL,
            UNIQUE(submission_id)
        );
        CREATE INDEX IF NOT EXISTS idx_submissions_log_opened
            ON submissions_log(opened_at);
    """)
    _apply_migrations(conn)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply any schema-version migrations beyond v1. Idempotent.

    Migration history:
      v1 → initial schema (recordings, diary, meta, schema_version)
      v2 → added submissions_log table (v0.12.3)
    """
    cur = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version")
    current = cur.fetchone()[0]

    if current < 2:
        # v1 → v2: add submissions_log for upgraded databases.
        # New databases already have it via _init_schema's CREATE TABLE IF NOT
        # EXISTS, so this is a no-op for them and safe to run unconditionally.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS submissions_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                submission_id TEXT NOT NULL UNIQUE,
                opened_at TEXT NOT NULL,
                issue_url TEXT NOT NULL DEFAULT '',
                submission_json TEXT NOT NULL,
                UNIQUE(submission_id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_submissions_log_opened "
            "ON submissions_log(opened_at)"
        )
        _meta_set(conn, "v2_migration_applied", _now_iso())

    if current < _SCHEMA_VERSION:
        conn.execute(
            "INSERT OR REPLACE INTO schema_version (version, applied_at) "
            "VALUES (?, ?)",
            (_SCHEMA_VERSION, _now_iso()),
        )


def _meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM meta WHERE key = ?", (key,)
    ).fetchone()
    return row["value"] if row else None


def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


# ─── Legacy JSON → SQLite import (run once per DB) ─────────────────────────

def _maybe_migrate_legacy_json(
    conn: sqlite3.Connection, base_dir: Path
) -> dict[str, int]:
    """Import legacy recordings/*.json and diary.jsonl on first run.

    Returns counts: {"recordings": N, "diary": M, "skipped": K}.
    Idempotent — the meta row 'legacy_json_migrated' guards re-runs.
    """
    if _meta_get(conn, "legacy_json_migrated") == "1":
        return {"recordings": 0, "diary": 0, "skipped": 0}

    counts = {"recordings": 0, "diary": 0, "skipped": 0}

    # Recordings: ~/.kcnq3-lens/recordings/*.json
    rec_dir = base_dir / "recordings"
    if rec_dir.exists() and rec_dir.is_dir():
        for fpath in sorted(rec_dir.glob("*.json")):
            try:
                d = json.loads(fpath.read_text())
                conn.execute(
                    "INSERT INTO recordings("
                    "recording_date, label, source_filename, saved_at, "
                    "findings_json, metadata_json) VALUES(?,?,?,?,?,?)",
                    (
                        d.get("recording_date", ""),
                        d.get("label", ""),
                        d.get("source_filename", ""),
                        d.get("saved_at") or _now_iso(),
                        json.dumps(d.get("findings", {}), default=str),
                        json.dumps(d.get("metadata", {}), default=str),
                    ),
                )
                counts["recordings"] += 1
            except (json.JSONDecodeError, OSError, sqlite3.Error):
                counts["skipped"] += 1

    # Diary: ~/.kcnq3-lens/diary.jsonl
    diary_path = base_dir / "diary.jsonl"
    if diary_path.exists() and diary_path.is_file():
        try:
            with open(diary_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        conn.execute(
                            "INSERT INTO diary("
                            "date, word_count, concentration_minutes, "
                            "sleep_onset_difficulty, sleep_quality_1to5, "
                            "medication_change, new_milestone, "
                            "seizure_event, notes, created_at) "
                            "VALUES(?,?,?,?,?,?,?,?,?,?)",
                            (
                                d.get("date", ""),
                                d.get("word_count"),
                                d.get("concentration_minutes"),
                                d.get("sleep_onset_difficulty"),
                                d.get("sleep_quality_1to5"),
                                d.get("medication_change"),
                                d.get("new_milestone"),
                                d.get("seizure_event"),
                                d.get("notes", "") or "",
                                d.get("created_at") or _now_iso(),
                            ),
                        )
                        counts["diary"] += 1
                    except (json.JSONDecodeError, sqlite3.Error):
                        counts["skipped"] += 1
        except OSError:
            pass

    _meta_set(conn, "legacy_json_migrated", "1")
    _meta_set(conn, "legacy_json_migrated_at", _now_iso())
    _meta_set(conn, "legacy_json_migrated_counts", json.dumps(counts))
    return counts


# ─── Recordings CRUD ───────────────────────────────────────────────────────

def insert_recording(
    *,
    recording_date: str,
    label: str,
    source_filename: str,
    findings: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    saved_at: str | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Insert a recording row. Returns the new row id."""
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO recordings("
            "recording_date, label, source_filename, saved_at, "
            "findings_json, metadata_json) VALUES(?,?,?,?,?,?)",
            (
                recording_date,
                label or "",
                source_filename or "",
                saved_at or _now_iso(),
                json.dumps(findings, default=str),
                json.dumps(metadata or {}, default=str),
            ),
        )
        return int(cur.lastrowid)


def list_recordings(
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return all recordings sorted by date asc, then saved_at asc.

    Each item is a dict with the same keys the legacy StoredEntry used:
    recording_date, label, findings, metadata, saved_at, source_filename,
    plus 'id' (the SQLite row id).
    """
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, recording_date, label, source_filename, saved_at, "
            "findings_json, metadata_json FROM recordings "
            "ORDER BY recording_date ASC, saved_at ASC"
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({
            "id": r["id"],
            "recording_date": r["recording_date"],
            "label": r["label"],
            "source_filename": r["source_filename"],
            "saved_at": r["saved_at"],
            "findings": _safe_json_loads(r["findings_json"], default={}),
            "metadata": _safe_json_loads(r["metadata_json"], default={}),
        })
    return out


def delete_recording(
    *,
    row_id: int | None = None,
    legacy_filename: str | None = None,
    db_path: Path | str | None = None,
) -> bool:
    """Delete a recording. Returns True if a row was deleted.

    Accepts either `row_id` (new path) or `legacy_filename` (for
    backward compat with code that still has old filenames around —
    we parse `{date}_{label-slug}.json` back to (date, label) and
    delete the matching row).
    """
    with connect(db_path) as conn:
        if row_id is not None:
            cur = conn.execute(
                "DELETE FROM recordings WHERE id = ?", (row_id,)
            )
            return cur.rowcount > 0
        if legacy_filename:
            stem = Path(legacy_filename).stem  # strip .json
            # Format was "{YYYY-MM-DD}_{slug}" or "{YYYY-MM-DD}".
            parts = stem.split("_", 1)
            if len(parts) == 2:
                date_part, label_part = parts
                cur = conn.execute(
                    "DELETE FROM recordings WHERE recording_date = ? "
                    "AND label LIKE ? || '%'",
                    (date_part, label_part.replace("_", "%")),
                )
            else:
                cur = conn.execute(
                    "DELETE FROM recordings WHERE recording_date = ?",
                    (parts[0],),
                )
            return cur.rowcount > 0
    return False


# ─── Diary CRUD ────────────────────────────────────────────────────────────

def insert_diary(
    *,
    date: str,
    word_count: int | None = None,
    concentration_minutes: float | None = None,
    sleep_onset_difficulty: str | None = None,
    sleep_quality_1to5: int | None = None,
    medication_change: str | None = None,
    new_milestone: str | None = None,
    seizure_event: str | None = None,
    notes: str = "",
    created_at: str | None = None,
    db_path: Path | str | None = None,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO diary("
            "date, word_count, concentration_minutes, "
            "sleep_onset_difficulty, sleep_quality_1to5, "
            "medication_change, new_milestone, seizure_event, "
            "notes, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                date, word_count, concentration_minutes,
                sleep_onset_difficulty, sleep_quality_1to5,
                medication_change, new_milestone, seizure_event,
                notes or "", created_at or _now_iso(),
            ),
        )
        return int(cur.lastrowid)


def record_submission(
    *,
    submission_id: str,
    submission: dict[str, Any],
    issue_url: str = "",
    opened_at: str | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Record that the family opened a GitHub issue for a submission.

    `opened_at` is the moment the issue URL was constructed, which is
    when the family clicked through. Returns the row id.
    """
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO submissions_log("
            "submission_id, opened_at, issue_url, submission_json) "
            "VALUES(?,?,?,?)",
            (
                submission_id,
                opened_at or _now_iso(),
                issue_url or "",
                json.dumps(submission, default=str),
            ),
        )
        return int(cur.lastrowid)


def list_submissions_log(
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return all submission-log rows, newest first."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, submission_id, opened_at, issue_url, "
            "submission_json FROM submissions_log "
            "ORDER BY opened_at DESC"
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({
            "id": r["id"],
            "submission_id": r["submission_id"],
            "opened_at": r["opened_at"],
            "issue_url": r["issue_url"],
            "submission": _safe_json_loads(r["submission_json"], default={}),
        })
    return out


def find_submission_in_log(
    submission_id: str, db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Look up one logged submission by its uuid."""
    with connect(db_path) as conn:
        r = conn.execute(
            "SELECT id, submission_id, opened_at, issue_url, "
            "submission_json FROM submissions_log "
            "WHERE submission_id = ?",
            (submission_id,),
        ).fetchone()
    if not r:
        return None
    return {
        "id": r["id"],
        "submission_id": r["submission_id"],
        "opened_at": r["opened_at"],
        "issue_url": r["issue_url"],
        "submission": _safe_json_loads(r["submission_json"], default={}),
    }


def list_diary(
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, date, word_count, concentration_minutes, "
            "sleep_onset_difficulty, sleep_quality_1to5, "
            "medication_change, new_milestone, seizure_event, "
            "notes, created_at FROM diary "
            "ORDER BY date ASC, created_at ASC"
        ).fetchall()
    return [dict(r) for r in rows]


# ─── Helpers ───────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_json_loads(s: str, default: Any) -> Any:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return default


def reset_init_cache_for_tests() -> None:
    """Drop the per-process init cache. Tests use this between fixtures
    that swap KCNQ3_LENS_DATA mid-run."""
    _INIT_DONE.clear()


# ─── Maintenance / introspection ───────────────────────────────────────────

def stats(db_path: Path | str | None = None) -> dict[str, Any]:
    """Return counts + schema version + migration status. Useful in UI."""
    with connect(db_path) as conn:
        rec_n = conn.execute(
            "SELECT COUNT(*) AS n FROM recordings"
        ).fetchone()["n"]
        diary_n = conn.execute(
            "SELECT COUNT(*) AS n FROM diary"
        ).fetchone()["n"]
        ver = conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM schema_version"
        ).fetchone()["v"]
        migrated = _meta_get(conn, "legacy_json_migrated") == "1"
        migrated_at = _meta_get(conn, "legacy_json_migrated_at") or ""
        migrated_counts = _safe_json_loads(
            _meta_get(conn, "legacy_json_migrated_counts") or "{}",
            default={},
        )
    return {
        "schema_version": ver,
        "n_recordings": rec_n,
        "n_diary": diary_n,
        "legacy_json_migrated": migrated,
        "legacy_json_migrated_at": migrated_at,
        "legacy_json_migrated_counts": migrated_counts,
        "db_path": str(resolve_db_path(db_path)),
    }
