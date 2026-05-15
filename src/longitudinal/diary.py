"""Development / symptom diary — milestones tracked alongside EEG findings.

v0.12+: backed by SQLite (see `db.py`). Same public API as before so
`app.py` doesn't change.

The diary's killer-feature pairing for any rare-epilepsy family: not
just "what does the EEG say" but "did the child gain new words this
month?" Plotted on the same timeline as quantitative EEG output,
parents can see how language development tracks medication + EEG.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from . import db as _db


@dataclass
class DiaryEntry:
    """One observation on one date."""

    date: str
    word_count: int | None = None
    concentration_minutes: float | None = None
    sleep_onset_difficulty: str | None = None
    sleep_quality_1to5: int | None = None
    medication_change: str | None = None
    new_milestone: str | None = None
    seizure_event: str | None = None
    notes: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DiaryEntry":
        return cls(
            date=d.get("date", ""),
            word_count=d.get("word_count"),
            concentration_minutes=d.get("concentration_minutes"),
            sleep_onset_difficulty=d.get("sleep_onset_difficulty"),
            sleep_quality_1to5=d.get("sleep_quality_1to5"),
            medication_change=d.get("medication_change"),
            new_milestone=d.get("new_milestone"),
            seizure_event=d.get("seizure_event"),
            notes=d.get("notes", "") or "",
            created_at=d.get("created_at", ""),
        )


def default_diary_path() -> Path:
    """Legacy alias — returns the SQLite DB path."""
    return _db.default_db_path()


def append_entry(entry: DiaryEntry, path: Path | None = None) -> Path:
    """Append a diary entry. Returns the DB file path."""
    db_path = _resolve(path)
    _db.insert_diary(
        date=entry.date,
        word_count=entry.word_count,
        concentration_minutes=entry.concentration_minutes,
        sleep_onset_difficulty=entry.sleep_onset_difficulty,
        sleep_quality_1to5=entry.sleep_quality_1to5,
        medication_change=entry.medication_change,
        new_milestone=entry.new_milestone,
        seizure_event=entry.seizure_event,
        notes=entry.notes,
        created_at=entry.created_at or None,
        db_path=db_path,
    )
    if not entry.created_at:
        entry.created_at = _db._now_iso()
    return db_path


def load_all_entries(path: Path | None = None) -> list[DiaryEntry]:
    """Load all diary entries, sorted by date then created_at."""
    db_path = _resolve(path)
    rows = _db.list_diary(db_path=db_path)
    return [DiaryEntry.from_dict(r) for r in rows]


def to_table(entries: list[DiaryEntry]) -> list[dict]:
    """Convert to a list-of-dicts suitable for pandas display."""
    out = []
    for e in entries:
        out.append({
            "date": e.date,
            "words": e.word_count if e.word_count is not None else "",
            "concentration_min": (
                e.concentration_minutes
                if e.concentration_minutes is not None else ""
            ),
            "sleep_quality": (
                e.sleep_quality_1to5
                if e.sleep_quality_1to5 is not None else ""
            ),
            "milestone": e.new_milestone or "",
            "med_change": e.medication_change or "",
            "seizure": e.seizure_event or "",
            "notes": e.notes,
        })
    return out


def _resolve(path: Path | None) -> Path:
    """Map legacy path argument to a SQLite DB path.

    Back-compat quirks:
    - None  → default DB
    - .jsonl path → use the .jsonl file's PARENT directory's DB
                    (so tests that pass tempfile .jsonl paths get isolation)
    - .db path  → use as-is
    - anything else: treat as DB file path
    """
    if path is None:
        return _db.default_db_path()
    p = Path(path)
    if p.suffix == ".jsonl":
        # Legacy tests passed a tempfile .jsonl; map to a sibling .db
        # so each test gets its own isolated database.
        return p.with_suffix(".db")
    if p.suffix == ".db":
        return p
    if p.exists() and p.is_dir():
        return p / "kcnq3-lens.db"
    return p
