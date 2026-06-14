"""C1 — Spike-triggered averaging → peak voltage topography.  [BUILD on numpy]

Averaging the detected IEDs at their peaks yields a clean spike field whose
topography answers the ESES focal-vs-secondary-bilateral-synchrony question that
a per-channel count cannot. Reuses the spike event times the morphology detector
already exports (``_morphology_events``).

BUILD: window extraction + averaging + peak-topography is straightforward numpy.
Equivalent-dipole / template-MRI source localisation is intentionally NOT done —
it needs a head model + electrode coregistration a routine clinical montage
cannot reliably supply; the focal/regional/bilateral field-spread call is the
appropriate, honest proxy at the scalp.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..readers.base import EEGRecording


@dataclass
class SpikeAverageResult:
    n_spikes_averaged: int
    window_ms: tuple[float, float]
    peak_channel: str | None
    peak_latency_ms: float | None
    peak_topography: dict[str, float] = field(default_factory=dict)
    field_spread: str = "n/a"          # "focal" | "regional" | "bilateral" | "n/a"
    notes: list[str] = field(default_factory=list)


def _read_samples(rec: EEGRecording, start: int, n: int) -> np.ndarray | None:
    """Read n samples starting at absolute sample `start` (all file channels)."""
    if start < 0 or n <= 0:
        return None
    sf = float(rec.sfreq)
    spe = int(round(30.0 * sf))
    if spe <= 0:
        return None
    ep0, ep1 = start // spe, (start + n - 1) // spe
    chunks = []
    for ep in range(ep0, ep1 + 1):
        d = rec.read_epoch(ep, 30.0)
        if d is None:
            return None
        chunks.append(np.asarray(d, dtype=float))
    full = np.concatenate(chunks, axis=1)
    local = start - ep0 * spe
    seg = full[:, local: local + n]
    return seg if seg.shape[1] == n else None


def _hemisphere(name: str) -> str:
    """Left/right/mid from a 10-20 channel label (odd=left, even=right, z=mid)."""
    s = name.strip().lower()
    if s.endswith("z"):
        return "mid"
    for ch in reversed(s):
        if ch.isdigit():
            return "left" if int(ch) % 2 == 1 else "right"
    return "mid"


def compute_spike_average(
    rec: EEGRecording,
    spike_events: list[dict] | None = None,
    window_ms: tuple[float, float] = (-100.0, 100.0),
    max_spikes: int = 500,
    focal_fraction: float = 0.5,
    bilateral_fraction: float = 0.3,
) -> SpikeAverageResult:
    """Average detected spikes at their peaks → peak voltage topography.

    spike_events : list of {"time_s": float} (from ``_morphology_events``).
    window_ms : averaging window around each spike peak.
    """
    sf = float(rec.sfreq)
    eeg_idx = rec.eeg_channel_indices or list(range(rec.n_channels_in_file))
    names = [str(rec.channel_names[i]) for i in eeg_idx]
    pre = int(round(-window_ms[0] / 1000.0 * sf))
    post = int(round(window_ms[1] / 1000.0 * sf))
    n_win = pre + post
    if n_win <= 0:
        return SpikeAverageResult(0, window_ms, None, None,
                                  notes=["invalid averaging window"])

    times = [ev["time_s"] for ev in (spike_events or [])
             if isinstance(ev, dict) and ev.get("time_s") is not None]
    if not times:
        return SpikeAverageResult(0, window_ms, None, None,
                                  notes=["no spike events supplied — nothing to average"])
    if len(times) > max_spikes:
        # evenly subsample to bound cost while keeping coverage
        step = len(times) / max_spikes
        times = [times[int(k * step)] for k in range(max_spikes)]

    acc = np.zeros((len(eeg_idx), n_win))
    used = 0
    for t in times:
        center = int(round(t * sf))
        seg = _read_samples(rec, center - pre, n_win)
        if seg is None:
            continue
        acc += seg[eeg_idx]
        used += 1
    if used == 0:
        return SpikeAverageResult(0, window_ms, None, None,
                                  notes=["no spike windows were in range"])
    avg = acc / used

    # Peak latency = where the averaged scalp field is strongest (global power).
    field_power = np.sum(np.abs(avg), axis=0)
    peak_idx = int(np.argmax(field_power))
    topo_vals = avg[:, peak_idx]
    peak_ch_i = int(np.argmax(np.abs(topo_vals)))
    peak_channel = names[peak_ch_i]
    peak_latency_ms = (peak_idx - pre) / sf * 1000.0

    topo = {names[i]: round(float(topo_vals[i]), 2) for i in range(len(names))}

    # Field-spread classification from the peak topography.
    abs_vals = np.abs(topo_vals)
    total = float(abs_vals.sum())
    field_spread = "n/a"
    if total > 0:
        top_frac = float(abs_vals.max() / total)
        by_hemi = {"left": 0.0, "right": 0.0, "mid": 0.0}
        for nm, v in zip(names, abs_vals):
            by_hemi[_hemisphere(nm)] += float(v)
        peak_hemi_val = float(abs_vals.max())
        left_sig = by_hemi["left"] >= bilateral_fraction * total
        right_sig = by_hemi["right"] >= bilateral_fraction * total
        if left_sig and right_sig:
            field_spread = "bilateral"
        elif top_frac >= focal_fraction:
            field_spread = "focal"
        else:
            field_spread = "regional"

    notes: list[str] = []
    if used < 10:
        notes.append(f"only {used} spikes averaged — topography is unstable (<10).")
    notes.append("scalp field-spread proxy; equivalent-dipole/source localisation "
                 "not attempted (needs a head model + electrode coregistration).")

    return SpikeAverageResult(
        n_spikes_averaged=used, window_ms=window_ms,
        peak_channel=peak_channel, peak_latency_ms=round(peak_latency_ms, 1),
        peak_topography=topo, field_spread=field_spread, notes=notes,
    )


def summarize_spike_average(result: SpikeAverageResult) -> dict:
    return {
        "n_spikes_averaged": result.n_spikes_averaged,
        "window_ms": list(result.window_ms),
        "peak_channel": result.peak_channel,
        "peak_latency_ms": result.peak_latency_ms,
        "field_spread": result.field_spread,
        "peak_topography": result.peak_topography,
        "notes": result.notes,
    }
