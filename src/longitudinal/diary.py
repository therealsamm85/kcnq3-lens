"""Development / symptom diary — milestone tracking alongside EEG findings.

The killer-feature pairing for any rare-epilepsy family: not just "what
does the EEG say" but "did the child gain new words this month?" The
diary lives next to the recordings and gets plotted on the same timeline,
so a parent can see how language development tracks medication and EEG
changes.

Each entry is one observation on one date. Multiple entries per day are
allowed (a parent might log a word count Monday morning and a seizure
event Monday afternoon).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class DiaryEntry:
    """One observation on one date."""

    date: str                       # 'YYYY-MM-DD'
    word_count: int | None = None   # active vocabulary count
    concentration_minutes: float | None = None  # longest focused episode
    sleep_onset_difficulty: str | None = None   # 'normal' / 'mild' / 'severe' / None
    sleep_quality_1to5: int | None = None
    medication_change: str | None = None         # free text — what changed today
    new_milestone: str | None = None             # e.g. "first 2-word combination"
    seizure_event: str | None = None             # description if any
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
            notes=d.get("notes", ""),
            created_at=d.get("created_at", ""),
        )


def default_diary_path() -> Path:
    base = Path(os.environ.get("KCNQ3_LENS_DATA",
                                 str(Path.home() / ".kcnq3-lens")))
    return base / "diary.jsonl"


def append_entry(entry: DiaryEntry, path: Path | None = None) -> Path:
    """Append a diary entry to a JSONL file. One entry per line."""
    if path is None:
        path = default_diary_path()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry.created_at = datetime.now().isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_dict(), default=str) + "\n")
    return path


def load_all_entries(path: Path | None = None) -> list[DiaryEntry]:
    """Load all diary entries, sorted by date."""
    if path is None:
        path = default_diary_path()
    path = Path(path)
    if not path.exists():
        return []
    out: list[DiaryEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(DiaryEntry.from_dict(json.loads(line)))
            except json.JSONDecodeError:
                continue
    out.sort(key=lambda e: (e.date, e.created_at))
    return out


def to_table(entries: list[DiaryEntry]) -> list[dict]:
    """Convert to a list-of-dicts suitable for pandas display."""
    out = []
    for e in entries:
        out.append({
            "date": e.date,
            "words": e.word_count if e.word_count is not None else "",
            "concentration_min": (
                e.concentration_minutes if e.concentration_minutes is not None else ""
            ),
            "sleep_quality": (
                e.sleep_quality_1to5 if e.sleep_quality_1to5 is not None else ""
            ),
            "milestone": e.new_milestone or "",
            "med_change": e.medication_change or "",
            "seizure": e.seizure_event or "",
            "notes": e.notes,
        })
    return out
