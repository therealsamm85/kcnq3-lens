"""Spike morphology classification.

Classifies detected sharp transients by their broadband width:
- < 70 ms: classic simple spike (e.g. Rolandic)
- 70-200 ms: sharp wave
- ≥ 200 ms: spike followed by aftercoming slow wave (CSWS/atypical absence morphology)

The distribution matters clinically: predominantly simple spikes (>60%) suggest
benign focal epilepsy patterns; predominantly complex spike-wave (>50%) suggests
atypical absence / CSWS spectrum, which has different prognostic and treatment
implications.

Detection gating (v0.3 — fixes audit-discovered over-counting)
--------------------------------------------------------------
v0.1/v0.2 computed MAD once over the entire concatenated sleep window. On
recordings with large CSWS bursts, this global MAD was inflated by the burst
amplitudes — but during quiet inter-burst intervals, the now-too-low threshold
still caught noise peaks in the 10-30 Hz band, producing event rates 6-17x
above published literature.

v0.3 fixes this by computing MAD **per 30-second epoch** and additionally
requiring each peak to exceed a fraction of the local (epoch) RMS. This makes
detection self-calibrating to each epoch's noise floor and rejects noise peaks
in quiet intervals while preserving real spikes inside CSWS bursts.

See data/_session/current/audit_findings.md for the validation report that
surfaced the original bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    rate_ci_low_per_min: float | None = None   # 95% CI lower bound for events/min
    rate_ci_high_per_min: float | None = None  # 95% CI upper bound for events/min
    # C4: spike event times for HFO co-occurrence coupling (v0.13.1)
    # Each dict has {"time_s": float} — analogous to _slow_waves_events convention.
    events: list[dict] = field(default_factory=list)


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

    Parameters
    ----------
    mad_multiplier : float
        Threshold = mad_multiplier × MAD where MAD = median(|x − median(x)|).
        Default 6.0 means threshold ≈ 4σ for Gaussian noise (MAD ≈ 0.6745σ,
        so 6 × MAD ≈ 4σ — NOT 6σ). To target N σ-units use ≈ N / 0.6745.
    """
    # v0.18.5: liveness-aware channel resolution (skips present-but-dead
    # channels, same guard as slow_waves). Candidate order preserves the prior
    # default of preferring the requested channel, then Pz/Cz/C3/C4/Fz.
    _cands = [target_channel] + [
        c for c in ("Pz", "Cz", "C3", "C4", "Fz") if c.upper() != target_channel.upper()
    ]
    if hasattr(rec, "resolve_live_channel"):
        ch_idx, resolved_name, _ = rec.resolve_live_channel(_cands)
    else:
        ch_idx, resolved_name = None, None
        for _nm in _cands:
            _i = rec.channel_index(_nm)
            if _i is not None:
                ch_idx, resolved_name = _i, _nm
                break
    if ch_idx is None:
        raise ValueError("No suitable channel for morphology analysis.")
    target_channel = resolved_name

    sos_det = butter(4, list(detection_bandpass), btype="band", fs=rec.sfreq, output="sos")
    sos_bb = butter(4, list(morphology_bandpass), btype="band", fs=rec.sfreq, output="sos")

    segments = []
    for _, d in rec.iter_epochs(epoch_seconds=epoch_seconds, start=start_epoch, end=end_epoch):
        segments.append(d[ch_idx])
    if not segments:
        raise ValueError("No data in window.")
    trace = np.concatenate(segments)

    # Filter once on the whole trace (filters need continuity at boundaries)
    det = sosfiltfilt(sos_det, trace)
    bb = sosfiltfilt(sos_bb, trace)

    # ── PER-EPOCH MAD detection (v0.3 fix) ──────────────────────────────────
    # Compute MAD inside each 30-second epoch separately so threshold tracks
    # the local noise floor. Inside quiet inter-burst intervals, this produces
    # a low local MAD and a correspondingly tight threshold — but we additionally
    # require each peak to exceed 3× the epoch's RMS, which rejects noise peaks
    # that just happen to be the local maximum.
    samples_per_epoch = int(epoch_seconds * rec.sfreq)
    min_dist = max(1, int(0.08 * rec.sfreq))

    all_peaks: list[int] = []
    per_epoch_counts: list[int] = []   # for bootstrap CI
    for ep_start in range(0, len(det), samples_per_epoch):
        ep_end = min(ep_start + samples_per_epoch, len(det))
        ep_det = det[ep_start:ep_end]
        if len(ep_det) < min_dist * 2:
            per_epoch_counts.append(0)
            continue
        ep_centered = ep_det - np.median(ep_det)
        local_mad = np.median(np.abs(ep_centered))
        local_rms = float(np.sqrt(np.mean(ep_det ** 2)))
        # Require: peak > mad_multiplier × MAD  AND  peak > 3 × RMS
        # mad_multiplier=6.0 means threshold = 6 × MAD where MAD = median(|x−med|).
        # For Gaussian data: MAD ≈ 0.6745σ, so 6×MAD ≈ 4σ (NOT 6σ).
        # If you want N σ-units, use mad_multiplier ≈ N / 0.6745.
        local_threshold = max(mad_multiplier * local_mad, 3.0 * local_rms)
        if local_threshold <= 0 or not np.isfinite(local_threshold):
            per_epoch_counts.append(0)
            continue
        local_peaks, _ = find_peaks(
            np.abs(ep_det),
            height=local_threshold,
            distance=min_dist,
        )
        # Offset peaks back to global index space
        all_peaks.extend((local_peaks + ep_start).tolist())
        per_epoch_counts.append(int(len(local_peaks)))

    peaks = np.array(all_peaks, dtype=int)

    # ── Morphology measurement on broadband ─────────────────────────────────
    # v0.18.17: the half-amplitude search window must be a fixed TIME span, not
    # a fixed sample count. A ±100-sample window is ±200 ms at 500 Hz but only
    # ±100 ms at 1000 Hz, making the simple/sharp/complex duration
    # classification sample-rate-dependent (a true 550 ms complex measured 398
    # ms at 500 Hz and 199 ms — misclassified "sharp" — at 1000 Hz). Use ±300 ms
    # so the window comfortably contains the longest events the classifier bins.
    half_win = int(0.3 * rec.sfreq)
    sample_every = max(1, len(peaks) // 3000) if len(peaks) > 0 else 1
    durations_ms: list[float] = []
    for p in peaks[::sample_every]:
        win_l = max(0, p - half_win)
        win_r = min(len(bb), p + half_win)
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

    # Bootstrap 95% CI on rate (per-epoch resampling)
    rate_ci_low = rate_ci_high = None
    if len(per_epoch_counts) >= 2:
        try:
            from ..utils.bootstrap import bootstrap_count_ci
            # Rate = mean count per epoch × (epochs per minute)
            epochs_per_min = 60.0 / epoch_seconds
            ci = bootstrap_count_ci(
                per_epoch_counts, aggregate="mean", n_bootstrap=500,
            )
            rate_ci_low = float(ci.ci_low * epochs_per_min)
            rate_ci_high = float(ci.ci_high * epochs_per_min)
        except Exception:
            pass

    # Build event list for HFO co-occurrence coupling (C4, v0.13.1).
    # Each entry exposes time_s so hfo_ripples.py can look up spike times
    # without needing to re-run detection.
    morph_events = [
        {"time_s": float(p) / rec.sfreq}
        for p in peaks
    ]

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
        rate_ci_low_per_min=rate_ci_low,
        rate_ci_high_per_min=rate_ci_high,
        events=morph_events,
    )


def summarize_morphology(result: MorphologyResult) -> dict:
    return {
        "channel": result.channel,
        "n_events": result.n_events_detected,
        "events_per_minute": round(result.n_events_per_minute, 1),
        "events_per_minute_ci_low": (
            round(result.rate_ci_low_per_min, 1)
            if result.rate_ci_low_per_min is not None else None
        ),
        "events_per_minute_ci_high": (
            round(result.rate_ci_high_per_min, 1)
            if result.rate_ci_high_per_min is not None else None
        ),
        "pct_simple_spikes": round(result.pct_simple_spikes, 1),
        "pct_sharp_waves": round(result.pct_sharp_waves, 1),
        "pct_complex_spike_wave": round(result.pct_complex_spike_wave, 1),
        "polyspike_fraction": round(100 * result.polyspike_fraction, 1),
        "classification": result.classification,
        "duration_p25_ms": round(result.duration_percentiles_ms[0]),
        "duration_p50_ms": round(result.duration_percentiles_ms[1]),
        "duration_p75_ms": round(result.duration_percentiles_ms[2]),
    }
