"""Longitudinal storage — recordings of a child's brain across time.

v0.12+: backed by SQLite (see `db.py`). The public API
(`save_entry`, `load_all_entries`, `delete_entry`, `StoredEntry`) is
preserved bit-for-bit so existing callers — `app.py`, the longitudinal
tracker, and the test suite — do not need to change.

The first call into this module from a given DB will silently import
any legacy JSON recordings from `recordings/*.json` into SQLite, so
upgrading from v0.11 → v0.12 is automatic and lossless. Legacy files
are kept on disk (not deleted) until the user verifies the migration.

Privacy: storage is local-only. No network calls. Same as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from . import db as _db


@dataclass
class StoredEntry:
    recording_date: str       # 'YYYY-MM-DD'
    label: str                # e.g. 'pre-Sultiam', 'post-treatment month 3'
    findings: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    saved_at: str = ""        # ISO timestamp filled by save()
    source_filename: str = "" # the EEG filename this was computed from

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StoredEntry":
        return cls(
            recording_date=d.get("recording_date", ""),
            label=d.get("label", ""),
            findings=d.get("findings", {}),
            metadata=d.get("metadata", {}),
            saved_at=d.get("saved_at", ""),
            source_filename=d.get("source_filename", ""),
        )


def default_storage_dir() -> Path:
    """Legacy alias — returns the directory containing the SQLite DB."""
    return _db.default_db_path().parent


def save_entry(
    entry: StoredEntry,
    storage_dir: Path | None = None,
) -> Path:
    """Persist a recording. Returns the DB file path (callers historically
    treated this as a 'path that exists' check, which still holds)."""
    db_path = _resolve(storage_dir)
    _db.insert_recording(
        recording_date=entry.recording_date,
        label=entry.label,
        source_filename=entry.source_filename,
        findings=entry.findings,
        metadata=entry.metadata,
        saved_at=entry.saved_at or None,
        db_path=db_path,
    )
    # Touch saved_at on the dataclass for caller visibility, mirroring
    # the legacy contract.
    if not entry.saved_at:
        entry.saved_at = _db._now_iso()
    return db_path


def load_all_entries(storage_dir: Path | None = None) -> list[StoredEntry]:
    """Load every stored recording, sorted by date then save-time."""
    db_path = _resolve(storage_dir)
    rows = _db.list_recordings(db_path=db_path)
    return [StoredEntry.from_dict(r) for r in rows]


def delete_entry(filename: str, storage_dir: Path | None = None) -> bool:
    """Delete a recording. `filename` may be a legacy
    `{date}_{label}.json` filename (back-compat) or the literal string
    representation of an integer row id."""
    db_path = _resolve(storage_dir)
    # Allow row-id deletion: e.g. "12" or "id:12"
    raw = filename
    if raw.startswith("id:"):
        raw = raw[3:]
    if raw.isdigit():
        return _db.delete_recording(row_id=int(raw), db_path=db_path)
    return _db.delete_recording(legacy_filename=filename, db_path=db_path)


def _resolve(storage_dir: Path | None) -> Path:
    """Map legacy 'storage_dir' (a directory) to a DB file path.

    The legacy contract was: a directory containing JSON files. We map
    that to: that directory's `kcnq3-lens.db` SQLite file. If callers
    pass None, defaults apply via db.default_db_path().
    """
    if storage_dir is None:
        return _db.default_db_path()
    p = Path(storage_dir)
    # If they passed a .db file directly, use it as-is.
    if p.suffix == ".db":
        return p
    # Otherwise treat as a directory.
    return p / "kcnq3-lens.db"
