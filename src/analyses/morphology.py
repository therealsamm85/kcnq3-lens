"""Spike morphology classification.

Classifies detected sharp transients by their broadband width:
- < 70 ms: classic simple spike (e.g. Rolandic)
- 70-200 ms: sharp wave
- ≥ 200 ms: spike followed by aftercoming slow wave (CSWS/atypical absence morphology)

The distribution matters clinically: predominantly simple spikes (>60%) suggest
benign focal epilepsy patterns; predominantly complex spike-wave (>50%) suggests
atypical absence / CSWS spectrum, which has different prognostic and treatment
implications.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

from ..readers.base import EEGRecording


@dataclass
class MorphologyResult:
    channel: str
    n_events_detected: int
    n_events_per_minute: float
    pct_simple_spikes: float       # < 70 ms
    pct_sharp_waves: float         # 70-200 ms
    pct_complex_spike_wave: float  # ≥ 200 ms
    polyspike_fraction: float      # peaks <250 ms apart
    classification: str            # "predominantly_simple", "mixed", "predominantly_complex"
    duration_percentiles_ms: tuple[float, float, float]  # p25, p50, p75


def compute_spike_morphology(
    rec: EEGRecording,
    start_epoch: int,
    end_epoch: int,
    epoch_seconds: float = 30.0,
    target_channel: str = "Pz",
    detection_bandpass: tuple[float, float] = (10.0, 30.0),
    morphology_bandpass: tuple[float, float] = (1.0, 35.0),
    mad_multiplier: float = 6.0,
) -> MorphologyResult:
    """Classify spike morphology on the highest-burden channel.

    Returns the distribution of broadband widths (FWHM) of detected events.
    """
    ch_idx = rec.channel_index(target_channel)
    if ch_idx is None:
        for fallback in ("Pz", "Cz", "C3", "C4", "Fz"):
            ch_idx = rec.channel_index(fallback)
            if ch_idx is not None:
                target_channel = fallback
                break
    if ch_idx is None:
        raise ValueError("No suitable channel for morphology analysis.")

    sos_det = butter(4, list(detection_bandpass), btype="band", fs=rec.sfreq, output="sos")
    sos_bb = butter(4, list(morphology_bandpass), btype="band", fs=rec.sfreq, output="sos")

    segments = []
    for _, d in rec.iter_epochs(epoch_seconds=epoch_seconds, start=start_epoch, end=end_epoch):
        segments.append(d[ch_idx])
    if not segments:
        raise ValueError("No data in window.")
    trace = np.concatenate(segments)

    det = sosfiltfilt(sos_det, trace)
    bb = sosfiltfilt(sos_bb, trace)
    mad = np.median(np.abs(det - np.median(det)))
    threshold = mad_multiplier * mad

    # Find peaks with minimum 80 ms separation (avoid double-counting one spike)
    peaks, _ = find_peaks(
        np.abs(det),
        height=threshold,
        distance=max(1, int(0.08 * rec.sfreq)),
    )

    # Subsample to speed up morphology measurement
    sample_every = max(1, len(peaks) // 3000)
    durations_ms: list[float] = []
    for p in peaks[::sample_every]:
        win_l = max(0, p - 100)
        win_r = min(len(bb), p + 100)
        seg = bb[win_l:win_r]
        rel_p = p - win_l
        if rel_p < 0 or rel_p >= len(seg):
            continue
        half = abs(seg[rel_p]) * 0.5
        l = rel_p
        while l > 0 and abs(seg[l]) > half:
            l -= 1
        r = rel_p
        while r < len(seg) - 1 and abs(seg[r]) > half:
            r += 1
        dur_ms = (r - l) * 1000 / rec.sfreq
        if 20 < dur_ms < 600:
            durations_ms.append(dur_ms)

    if not durations_ms:
        return MorphologyResult(
            channel=target_channel,
            n_events_detected=0,
            n_events_per_minute=0.0,
            pct_simple_spikes=0.0,
            pct_sharp_waves=0.0,
            pct_complex_spike_wave=0.0,
            polyspike_fraction=0.0,
            classification="no_events",
            duration_percentiles_ms=(0.0, 0.0, 0.0),
        )

    durs = np.array(durations_ms)
    pct_simple = 100 * float(np.mean(durs < 70))
    pct_sharp = 100 * float(np.mean((durs >= 70) & (durs < 200)))
    pct_complex = 100 * float(np.mean(durs >= 200))

    inter_peak_ms = np.diff(peaks) * 1000 / rec.sfreq if len(peaks) > 1 else np.array([])
    poly_frac = float(np.mean(inter_peak_ms < 250)) if len(inter_peak_ms) > 0 else 0.0

    total_min = len(trace) / rec.sfreq / 60
    events_per_min = len(peaks) / total_min if total_min > 0 else 0.0

    if pct_simple > 60:
        classification = "predominantly_simple"
    elif pct_complex > 50:
        classification = "predominantly_complex"
    else:
        classification = "mixed"

    p25, p50, p75 = np.percentile(durs, [25, 50, 75])

    return MorphologyResult(
        channel=target_channel,
        n_events_detected=len(peaks),
        n_events_per_minute=float(events_per_min),
        pct_simple_spikes=pct_simple,
        pct_sharp_waves=pct_sharp,
        pct_complex_spike_wave=pct_complex,
        polyspike_fraction=poly_frac,
        classification=classification,
        duration_percentiles_ms=(float(p25), float(p50), float(p75)),
    )


def summarize_morphology(result: MorphologyResult) -> dict:
    return {
        "channel": result.channel,
        "n_events": result.n_events_detected,
        "events_per_minute": round(result.n_events_per_minute, 1),
        "pct_simple_spikes": round(result.pct_simple_spikes, 1),
        "pct_sharp_waves": round(result.pct_sharp_waves, 1),
        "pct_complex_spike_wave": round(result.pct_complex_spike_wave, 1),
        "polyspike_fraction": round(100 * result.polyspike_fraction, 1),
        "classification": result.classification,
        "duration_p25_ms": round(result.duration_percentiles_ms[0]),
        "duration_p50_ms": round(result.duration_percentiles_ms[1]),
        "duration_p75_ms": round(result.duration_percentiles_ms[2]),
    }
