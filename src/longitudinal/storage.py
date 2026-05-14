"""Longitudinal storage — save findings + metadata across multiple recordings.

The goal: a family with multiple EEGs (pre-treatment, post-Sultiam, post-
Amitriptylin, follow-up at 6 months, etc.) should be able to track the
evolution of their child's brain over time. This module persists each
analysis run as a JSON file on disk, indexed by recording date.

Default storage location: `~/.kcnq3-lens/recordings/` — local-only,
privacy-preserving. User can override.

Each saved entry has:
- `recording_date`: ISO date string (YYYY-MM-DD)
- `label`: free-text label ("pre-Sultiam", "post-Amitriptylin month 3", etc.)
- `findings`: full findings dict from `run_all_analyses`
- `metadata`: RecordingMetadata fields (current meds, indication, etc.)
- `saved_at`: ISO timestamp when the entry was written

The diary module appends to a SEPARATE file because it's edited more
frequently than recordings are run.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


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
    """Default per-user storage location."""
    base = Path(os.environ.get("KCNQ3_LENS_DATA",
                                 str(Path.home() / ".kcnq3-lens")))
    return base / "recordings"


def save_entry(
    entry: StoredEntry,
    storage_dir: Path | None = None,
) -> Path:
    """Persist an entry to disk. Filename: '{recording_date}_{label-slug}.json'."""
    if storage_dir is None:
        storage_dir = default_storage_dir()
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    entry.saved_at = datetime.now().isoformat(timespec="seconds")
    slug = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in entry.label
    )[:60]
    fname = f"{entry.recording_date}_{slug}.json" if slug else f"{entry.recording_date}.json"
    path = storage_dir / fname

    # If a file with this name already exists, append a suffix
    n = 1
    while path.exists():
        path = storage_dir / f"{entry.recording_date}_{slug}_{n}.json"
        n += 1

    path.write_text(json.dumps(entry.to_dict(), indent=2, default=str))
    return path


def load_all_entries(storage_dir: Path | None = None) -> list[StoredEntry]:
    """Load every stored entry, sorted by recording_date ascending."""
    if storage_dir is None:
        storage_dir = default_storage_dir()
    storage_dir = Path(storage_dir)
    if not storage_dir.exists():
        return []
    entries: list[StoredEntry] = []
    for fpath in sorted(storage_dir.glob("*.json")):
        try:
            data = json.loads(fpath.read_text())
            entries.append(StoredEntry.from_dict(data))
        except (json.JSONDecodeError, OSError):
            continue
    entries.sort(key=lambda e: (e.recording_date, e.saved_at))
    return entries


def delete_entry(filename: str, storage_dir: Path | None = None) -> bool:
    """Delete a stored entry by filename. Returns True if deleted."""
    if storage_dir is None:
        storage_dir = default_storage_dir()
    path = Path(storage_dir) / filename
    if path.exists() and path.is_file():
        path.unlink()
        return True
    return False
