"""Broadband sharpness-gated spike detector — an alternative to morphology.py.

Why this exists
---------------
The existing morphology.py detector finds peaks in a 10-30 Hz band. That band
overlaps mu/beta rhythmic activity, so a rhythmic run can be counted as
"spikes" (the IPI analysis on this project found ~22% of the age-4.9 "spikes"
sat in the 8-12 Hz rhythmic band). A true interictal epileptiform discharge is
a BROADBAND, SHARP transient — it stands out by its sharpness (steep slope /
high curvature at the peak), not by its band-limited amplitude.

This detector adds a second, independent estimate: detect candidate peaks on a
broadband trace, then keep only those that are genuinely sharp (peak curvature
well above the local background). Comparing its rate against morphology.py's
tells you how much of the morphology count is rhythmic rather than spike-like.

It is ADDITIVE — morphology.py is unchanged, so existing clinical numbers stay
comparable. Neither is a substitute for an expert read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

from ..readers.base import EEGRecording


@dataclass
class SharpSpikeResult:
    channel: str
    n_candidates: int
    n_sharp_spikes: int
    sharp_rate_per_min: float
    pct_candidates_sharp: float
    median_sharpness: float
    sharpness_threshold: float
    notes: list[str] = field(default_factory=list)


def _resolve_channel(rec: EEGRecording, target: str) -> tuple[int | None, str]:
    cands = [target, "Pz", "Cz", "C3", "C4", "Fz"]
    if hasattr(rec, "resolve_live_channel"):
        idx, name, _ = rec.resolve_live_channel(cands)
        return idx, name
    for nm in cands:
        i = rec.channel_index(nm)
        if i is not None:
            return i, nm
    return None, target


def detect_sharp_spikes(
    rec: EEGRecording,
    start_epoch: int = 0,
    end_epoch: int | None = None,
    epoch_seconds: float = 30.0,
    target_channel: str = "Pz",
    amplitude_mad_k: float = 6.0,
    sharpness_mad_k: float = 4.0,
    max_duration_ms: float = 200.0,
) -> SharpSpikeResult:
    """Detect broadband sharp transients (spikes) with a curvature gate.

    A candidate is a broadband peak above amplitude_mad_k·MAD (per epoch). It is
    kept as a "sharp spike" only if its curvature (second-derivative magnitude
    at the peak) exceeds sharpness_mad_k·MAD of the epoch's curvature — i.e. it
    is sharp relative to the local background, which rhythmic mu/beta is not.
    """
    if end_epoch is None:
        end_epoch = rec.n_epochs

    ch_idx, ch_name = _resolve_channel(rec, target_channel)
    if ch_idx is None:
        raise ValueError("No suitable channel for sharp-spike detection.")

    sf = rec.sfreq
    hi = min(70.0, sf / 2 - 1.0)
    sos = butter(4, [1.0, hi], btype="band", fs=sf, output="sos")
    min_dist = max(1, int(0.05 * sf))           # 50 ms refractory
    max_dur_samp = int(max_duration_ms * sf / 1000.0)

    n_candidates = 0
    n_sharp = 0
    sharp_values: list[float] = []

    for ep, d in rec.iter_epochs(
        epoch_seconds=epoch_seconds, start=start_epoch, end=end_epoch
    ):
        x = sosfiltfilt(sos, d[ch_idx])
        if x.size < 3:
            continue
        xc = x - np.median(x)
        amp_mad = np.median(np.abs(xc)) * 1.4826
        if amp_mad <= 0 or not np.isfinite(amp_mad):
            continue
        amp_thr = amplitude_mad_k * amp_mad

        # Second derivative (curvature) — sharp transients have large |d2|.
        d2 = np.abs(np.gradient(np.gradient(x)))
        d2_med = np.median(d2)
        d2_mad = np.median(np.abs(d2 - d2_med)) * 1.4826
        if d2_mad <= 0 or not np.isfinite(d2_mad):
            continue
        sharp_thr = d2_med + sharpness_mad_k * d2_mad

        peaks, _ = find_peaks(np.abs(xc), height=amp_thr, distance=min_dist)
        for p in peaks:
            n_candidates += 1
            # Local curvature at the peak (max over a small window).
            lo = max(0, p - 2)
            hi_i = min(len(d2), p + 3)
            local_sharp = float(d2[lo:hi_i].max())
            # Duration gate: width at half-amplitude must be short (spike-like).
            half = abs(xc[p]) * 0.5
            l = p
            while l > 0 and abs(xc[l]) > half:
                l -= 1
            r = p
            while r < len(xc) - 1 and abs(xc[r]) > half:
                r += 1
            dur_ok = (r - l) <= max_dur_samp
            if local_sharp >= sharp_thr and dur_ok:
                n_sharp += 1
                sharp_values.append(local_sharp / max(sharp_thr, 1e-9))

    total_min = ((end_epoch - start_epoch) * epoch_seconds) / 60.0
    rate = n_sharp / total_min if total_min > 0 else 0.0
    pct_sharp = (100.0 * n_sharp / n_candidates) if n_candidates else 0.0

    return SharpSpikeResult(
        channel=ch_name,
        n_candidates=n_candidates,
        n_sharp_spikes=n_sharp,
        sharp_rate_per_min=round(rate, 2),
        pct_candidates_sharp=round(pct_sharp, 1),
        median_sharpness=round(float(np.median(sharp_values)), 2)
        if sharp_values else 0.0,
        sharpness_threshold=round(float(sharpness_mad_k), 1),
    )


def summarize_sharp_spikes(result: SharpSpikeResult) -> dict:
    return {
        "channel": result.channel,
        "n_candidates": result.n_candidates,
        "n_sharp_spikes": result.n_sharp_spikes,
        "sharp_rate_per_min": result.sharp_rate_per_min,
        "pct_candidates_sharp": result.pct_candidates_sharp,
        "median_sharpness": result.median_sharpness,
    }
