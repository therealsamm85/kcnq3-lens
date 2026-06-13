"""Export detected events for human review (BIDS events.tsv + round-trip).

Why this exists
---------------
Every automated detector in this project produces candidates, not diagnoses.
The "is this actually epileptiform?" question can only be answered by a human
reading the trace. This module writes detected events to a BIDS-compatible
events.tsv (onset / duration / trial_type / channel / ...), which loads
directly into EDFbrowser and most review tools alongside the raw EEG, so a
clinician (e.g. at the clinic) can confirm or reject each one.

It round-trips: load_events() reads the file back, so a reviewer can mark
events (edit the trial_type / add a 'reviewed' column) and the result can be
re-ingested. The format is plain TSV — no dependency, Excel-openable.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

# BIDS events.tsv requires onset + duration; the rest are optional extensions.
_CORE_COLUMNS = ["onset", "duration"]
_EXT_COLUMNS = ["trial_type", "channel", "amplitude_uv", "detector", "reviewed"]


@dataclass
class EventExportResult:
    path: str
    n_events: int
    columns: list[str]


def export_events(
    events: list[dict],
    out_path: str | Path,
    *,
    default_duration: float = 0.0,
    detector: str = "",
    channel: str | None = None,
) -> EventExportResult:
    """Write events to a BIDS-compatible events.tsv.

    Parameters
    ----------
    events : list of dicts. Each must have a time in seconds under one of
        "onset", "time_s", or "onset_s". Optional per-event keys: "duration",
        "trial_type"/"type", "channel", "amplitude_uv"/"amplitude".
    out_path : output .tsv path.
    default_duration : duration to use when an event has none (0 = instantaneous).
    detector : name recorded in the "detector" column for provenance.
    channel : a fallback channel if an event has none.

    Returns
    -------
    EventExportResult with the path, count, and column order written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    columns = _CORE_COLUMNS + _EXT_COLUMNS
    rows = []
    for ev in events:
        onset = ev.get("onset", ev.get("time_s", ev.get("onset_s")))
        if onset is None:
            continue
        rows.append({
            "onset": round(float(onset), 3),
            "duration": round(float(ev.get("duration", default_duration)), 3),
            "trial_type": ev.get("trial_type", ev.get("type", "candidate")),
            "channel": ev.get("channel", channel or "n/a"),
            "amplitude_uv": (round(float(ev["amplitude_uv"]), 1)
                             if "amplitude_uv" in ev else
                             (round(float(ev["amplitude"]), 1)
                              if "amplitude" in ev else "n/a")),
            "detector": detector or "n/a",
            "reviewed": "",   # blank for the human to fill (e.g. yes/no/epileptiform)
        })
    rows.sort(key=lambda r: r["onset"])

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    return EventExportResult(
        path=str(out_path), n_events=len(rows), columns=columns,
    )


def load_events(path: str | Path) -> list[dict]:
    """Read an events.tsv back (round-trip / re-ingest a reviewer's edits)."""
    path = Path(path)
    out: list[dict] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rec = dict(row)
            # Coerce numerics where present.
            for k in ("onset", "duration", "amplitude_uv"):
                if rec.get(k) not in (None, "", "n/a"):
                    try:
                        rec[k] = float(rec[k])
                    except (TypeError, ValueError):
                        pass
            out.append(rec)
    return out
