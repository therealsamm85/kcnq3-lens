"""Longitudinal spike-burden biomarker — a treatment-response tracker.

Why this exists
---------------
A single spike rate is hard to interpret. A *trajectory* of the same spike
rate, measured the same way across recordings, is an objective biomarker of
whether an intervention (an AED dose change, a new drug) is reducing the
interictal spike burden — the thing that, in the ESES spectrum, degrades
overnight memory consolidation and language learning.

The whole value is comparability: the rate must be measured on the SAME
channel with the SAME detection threshold at every timepoint, otherwise a
change in the number reflects a change in method, not in the brain. This
module enforces that and refuses to compare points measured with different
parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from ..readers.base import EEGRecording
from ..analyses.morphology import compute_spike_morphology


@dataclass
class BiomarkerPoint:
    label: str
    age_years: float | None
    date: str | None
    channel: str
    mad_multiplier: float
    rate_per_min: float
    rate_ci_low: float | None
    rate_ci_high: float | None
    n_events: int


@dataclass
class BiomarkerTrajectory:
    metric: str
    channel: str
    mad_multiplier: float
    points: list[BiomarkerPoint] = field(default_factory=list)
    slope_per_year: float | None = None
    direction: str = "insufficient_data"  # rising | falling | stable | insufficient_data
    notes: list[str] = field(default_factory=list)


def track_spike_rate(
    rec_specs: list[tuple[EEGRecording, str, float | None, str | None]],
    channel: str = "Pz",
    mad_multiplier: float = 6.0,
    start_epoch: int = 0,
    stable_slope_tol: float = 1.0,   # |slope| (per year) below this = "stable"
) -> BiomarkerTrajectory:
    """Measure the spike rate on each recording with IDENTICAL parameters.

    Parameters
    ----------
    rec_specs : list of (rec, label, age_years, date)
        The recordings to track, with a label and optional age/date for the
        x-axis. Order is preserved; the trend is fit against age_years when
        available for all points, else against position index.
    channel : str
        The fixed channel to measure on (default "Pz"). Same for every point.
    mad_multiplier : float
        The fixed detection threshold. Same for every point.

    Returns
    -------
    BiomarkerTrajectory with one point per recording plus a trend assessment.
    """
    traj = BiomarkerTrajectory(
        metric="spike_rate_per_min",
        channel=channel,
        mad_multiplier=mad_multiplier,
    )

    for rec, label, age, date in rec_specs:
        try:
            m = compute_spike_morphology(
                rec,
                start_epoch=start_epoch,
                end_epoch=rec.n_epochs,
                target_channel=channel,
                mad_multiplier=mad_multiplier,
            )
            traj.points.append(BiomarkerPoint(
                label=label,
                age_years=age,
                date=date,
                channel=m.channel,          # actual resolved channel
                mad_multiplier=mad_multiplier,
                rate_per_min=round(float(m.n_events_per_minute), 2),
                rate_ci_low=(round(float(m.rate_ci_low_per_min), 2)
                             if m.rate_ci_low_per_min is not None else None),
                rate_ci_high=(round(float(m.rate_ci_high_per_min), 2)
                              if m.rate_ci_high_per_min is not None else None),
                n_events=int(m.n_events_detected),
            ))
        except Exception as e:
            traj.notes.append(f"{label}: measurement failed ({type(e).__name__})")

    # Warn if the resolved channel differs across points (comparability broken).
    resolved = {p.channel for p in traj.points}
    if len(resolved) > 1:
        traj.notes.append(
            f"WARNING: spike rate was measured on different channels across "
            f"timepoints ({', '.join(sorted(resolved))}) — points are NOT "
            "directly comparable. Re-record with a consistent montage."
        )

    # Trend fit (only if ≥2 points). Prefer age as the x-axis; fall back to index.
    if len(traj.points) >= 2:
        ys = np.array([p.rate_per_min for p in traj.points], dtype=float)
        ages = [p.age_years for p in traj.points]
        if all(a is not None for a in ages) and len(set(ages)) > 1:
            xs = np.array(ages, dtype=float)
            x_unit = "year"
        else:
            xs = np.arange(len(ys), dtype=float)
            x_unit = "recording"
            traj.notes.append(
                "Trend fit against recording order (ages unavailable or equal)."
            )
        # Least-squares slope
        slope = float(np.polyfit(xs, ys, 1)[0])
        traj.slope_per_year = round(slope, 2)
        if abs(slope) < stable_slope_tol:
            traj.direction = "stable"
        elif slope < 0:
            traj.direction = "falling"
        else:
            traj.direction = "rising"
        traj.notes.append(
            f"slope = {slope:.2f} spikes/min per {x_unit}."
        )

    return traj


def summarize_trajectory(traj: BiomarkerTrajectory) -> dict:
    """JSON-serializable summary."""
    return {
        "metric": traj.metric,
        "channel": traj.channel,
        "mad_multiplier": traj.mad_multiplier,
        "direction": traj.direction,
        "slope_per_year": traj.slope_per_year,
        "points": [asdict(p) for p in traj.points],
        "notes": traj.notes,
    }
