"""Extract trend time-series from multiple stored entries.

Given a list of StoredEntry objects sorted by date, produce a tidy table:
one row per recording date, one column per metric. Suitable for plotting
or exporting.

Tracked metrics (each present only if the underlying analysis ran):
- spike rate per minute (morphology)
- spindle density per minute
- posterior dominant rhythm Hz
- SWI N3 percentage
- activation factor (NREM / wake)
- bursts ≥10s count
- sleep efficiency %
- REM latency minutes
- first-cycle N3 minutes
"""

from __future__ import annotations

from typing import Any

from .storage import StoredEntry


METRICS = [
    # (display_label, [path], formatter_function or None)
    ("spike_rate_per_min", ["morphology", "events_per_minute"]),
    ("spindle_density_per_min", ["spindles", "density_per_minute"]),
    ("pdr_hz", ["background", "posterior_dominant_rhythm_hz"]),
    ("delta_alpha_ratio", ["background", "delta_alpha_ratio"]),
    ("swi_n3_pct", ["swi", "swi_n3_only_pct"]),
    ("swi_nrem_combined_pct", ["swi", "swi_nrem_combined_pct"]),
    ("activation_factor", ["state_split", "activation_factor"]),
    ("nrem_rate_per_min", ["state_split", "nrem_rate_per_min"]),
    ("wake_rate_per_min", ["state_split", "wake_rate_per_min"]),
    ("bursts_10s_count", ["bursts", "n_bursts_10s_or_longer"]),
    ("sleep_efficiency_pct", ["sleep_stages", "sleep_efficiency_pct"]),
    ("rem_latency_minutes", ["sleep_architecture", "rem_latency_minutes"]),
    ("first_cycle_n3_minutes", ["sleep_architecture", "first_cycle_n3_minutes"]),
    ("fragmentation_index", ["sleep_architecture", "fragmentation_index_per_hour"]),
]


def _safe_path(d: dict, path: list[str]) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def build_trends_table(entries: list[StoredEntry]) -> list[dict]:
    """Return one dict per entry with all available metric values."""
    rows: list[dict] = []
    for e in entries:
        row: dict[str, Any] = {
            "recording_date": e.recording_date,
            "label": e.label,
        }
        for metric, path in METRICS:
            row[metric] = _safe_path(e.findings, path)
        rows.append(row)
    return rows


def get_metric_series(
    entries: list[StoredEntry], metric: str
) -> tuple[list[str], list[float]]:
    """Return (dates, values) where values is not None, suitable for plotting."""
    path = next((p for label, p in METRICS if label == metric), None)
    if path is None:
        return [], []
    dates: list[str] = []
    values: list[float] = []
    for e in entries:
        v = _safe_path(e.findings, path)
        if v is None:
            continue
        try:
            values.append(float(v))
            dates.append(e.recording_date)
        except (TypeError, ValueError):
            continue
    return dates, values
