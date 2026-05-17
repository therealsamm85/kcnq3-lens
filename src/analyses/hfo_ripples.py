"""HFO Ripple Detection (80–250 Hz scalp ripples) — Staba-style energy detector.

High-frequency oscillations (HFOs) in the ripple band (80–250 Hz) have been
proposed as a biomarker of epileptogenic tissue. Staba et al. 2002 (PMID
12364503) originally validated energy-based detection in intracranial recordings
of hippocampus and entorhinal cortex. Scalp recordings of HFOs are feasible
(Kramer et al. 2019, PMID 30907404) but technically challenging due to
muscle/EMG contamination in the same frequency band.

Frequency-specificity check (Burnos et al. 2014, PMID 24722663): true HFOs
show substantially more power in the ripple band (80–250 Hz) than in the
high-gamma/fast-ripple band (250–500 Hz). Events where the power ratio falls
below 2 are rejected as likely broad-band transients (muscle, spike-ringing).

IMPORTANT — no validated pediatric normative ranges exist for scalp HFO
density as of 2026. This module therefore reports descriptive metrics only.
No "below / in / above" range classification is applied. Do not compare
output numbers against adult intracranial norms.

References
----------
Staba RJ et al. 2002      PMID 12364503  original energy-based HFO detector
Burnos S et al. 2014      PMID 24722663  frequency-specificity criterion
Kramer MA et al. 2019     PMID 30907404  scalp spike ripples in childhood epilepsy
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import firwin, filtfilt, iirnotch

from ..readers.base import EEGRecording
from .sleep_stages import SleepStageResult


# ─── Result dataclass ─────────────────────────────────────────────────────────

_DISCLAIMER = (
    "RESEARCH METRIC — No validated pediatric normative ranges exist for scalp "
    "HFO ripple density (80–250 Hz). Results are descriptive only. Scalp HFOs "
    "require confirmation with high-density EEG or intracranial recordings. "
    "Muscle/EMG artefacts can mimic HFO waveforms; frequency-specificity "
    "rejection is applied but not infallible."
)


@dataclass
class HFORippleResult:
    channel: str
    sfreq_used: float
    available: bool                     # False when sfreq < 600 Hz or rec < 30 s
    unavailable_reason: str             # "insufficient_sfreq" | ""
    n_ripples_total: int
    n_ripples_isolated: int             # without spike co-occurrence
    n_ripples_on_spike: int
    rate_per_minute_nrem: float         # NREM-only rate (or full-rec if no stages)
    median_duration_ms: float
    median_peak_freq_hz: float
    method: str                         # "energy_staba_style"
    artifact_warnings: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    # [{start_s, end_s, peak_s, duration_ms, peak_freq_hz, rms_z, co_occurs_with_spike}]
    notes: list[str] = field(default_factory=list)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _unavailable(
    reason: str,
    notes: list[str],
    sfreq: float,
    reason_detail: str = "",
) -> HFORippleResult:
    all_notes = list(notes)
    if reason_detail:
        all_notes.append(f"reason_detail:{reason_detail}")
    return HFORippleResult(
        channel="",
        sfreq_used=sfreq,
        available=False,
        unavailable_reason=reason,
        n_ripples_total=0,
        n_ripples_isolated=0,
        n_ripples_on_spike=0,
        rate_per_minute_nrem=0.0,
        median_duration_ms=0.0,
        median_peak_freq_hz=0.0,
        method="energy_staba_style",
        artifact_warnings=[],
        events=[],
        notes=all_notes,
    )


def _safe_float(x: float) -> float | None:
    """Return x if finite, else None (for JSON-safe summary fields)."""
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _notch(signal: np.ndarray, sfreq: float, freq_hz: float) -> np.ndarray:
    """Apply IIR notch filter at freq_hz."""
    b, a = iirnotch(freq_hz, Q=30.0, fs=sfreq)
    return filtfilt(b, a, signal)


def _bandpass_fir(signal: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    """FIR bandpass filter (linear-phase). Order ≈ 2 × sfreq."""
    numtaps = int(round(2.0 * sfreq))
    if numtaps % 2 == 0:
        numtaps += 1  # firwin requires odd order for bandpass
    nyq = sfreq / 2.0
    cutoffs = [lo / nyq, hi / nyq]
    # Clamp to valid range
    cutoffs = [max(0.001, min(c, 0.999)) for c in cutoffs]
    taps = firwin(numtaps, cutoffs, pass_zero=False)
    return filtfilt(taps, [1.0], signal)


def _rms_envelope(signal: np.ndarray, window_samp: int) -> np.ndarray:
    """Sliding-window RMS envelope."""
    window_samp = max(1, window_samp)
    kernel = np.ones(window_samp) / window_samp
    return np.sqrt(np.convolve(signal ** 2, kernel, mode="same"))


def _peak_freq(segment: np.ndarray, sfreq: float, lo: float = 80.0, hi: float = 250.0) -> float:
    """Dominant frequency of a short segment via FFT."""
    n = len(segment)
    if n < 4:
        return (lo + hi) / 2.0
    fft_mag = np.abs(np.fft.rfft(segment * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return (lo + hi) / 2.0
    peak_idx = int(np.argmax(fft_mag[mask]))
    return float(freqs[mask][peak_idx])


def _power_in_band(segment: np.ndarray, sfreq: float, lo: float, hi: float) -> float:
    """Total power in frequency band via un-windowed rFFT magnitude squared.

    Intentionally omits windowing (unlike _peak_freq which uses Hanning).
    Design decision: for the Burnos frequency-specificity ratio we need
    unbiased total band energy, not spectral resolution. Applying a Hanning
    window would reduce each band's power by ~50% and introduce non-uniform
    leakage suppression near the 250 Hz boundary, distorting the ratio when
    energy sits close to that edge. The inconsistency between _peak_freq
    (windowed for frequency precision) and _power_in_band (un-windowed for
    unbiased total power) is deliberate and documented here.
    """
    n = len(segment)
    if n < 4:
        return 0.0
    fft_mag = np.abs(np.fft.rfft(segment)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.sum(fft_mag[mask]))


# ─── Main entry point ─────────────────────────────────────────────────────────

def compute_hfo_ripples(
    rec: EEGRecording,
    sleep_stages: SleepStageResult | None = None,
    channel: str = "Cz",
    line_freq_hz: float = 50.0,
    morphology_events: list[dict] | None = None,
) -> HFORippleResult:
    """Detect HFO ripples (80–250 Hz) using a Staba-style energy threshold.

    Parameters
    ----------
    rec : EEGRecording
    sleep_stages : SleepStageResult, optional
        If provided, rate is computed over N2 + N3 epochs only.
    channel : str
        Preferred channel (default "Cz"). Falls back to C3 → C4 → first EEG
        channel. Case-insensitive.
    line_freq_hz : float
        Mains frequency for notch filter (default 50.0 Hz; use 60.0 for North
        America). Both fundamental and second harmonic are notched.
    morphology_events : list[dict], optional
        Spike events from morphology analysis. Each must have a ``time_s`` key
        (or ``neg_peak_s`` / ``peak_s`` as fallback). Ripples within 100 ms of
        a spike are flagged as co-occurring but NOT dropped.

    Returns
    -------
    HFORippleResult
        ``available=False`` when sfreq < 600 Hz or recording < 30 s.

    Raises
    ------
    ValueError
        If no EEG channel found after fallback.
    """
    notes: list[str] = []

    # ── Step 1: sfreq guard ───────────────────────────────────────────────────
    # Minimum 600 Hz required: at 500 Hz the Nyquist (250 Hz) equals the
    # ripple band upper edge, producing a degenerate FIR (cutoff clamped to
    # 0.999). 600 Hz gives a clean 250 Hz bandpass with 50 Hz headroom.
    if rec.sfreq < 600:
        notes.append(
            f"sfreq={rec.sfreq:.0f}_Hz_below_600_Hz_minimum_for_ripple_detection"
        )
        return _unavailable(
            "insufficient_sfreq",
            notes,
            rec.sfreq,
            reason_detail=f"need sfreq>=600, got {rec.sfreq:.0f}",
        )

    # ── Recording duration guard ──────────────────────────────────────────────
    if rec.duration_s < 30:
        notes.append(f"recording_too_short_{rec.duration_s:.1f}s")
        return _unavailable(
            "recording_too_short",
            notes,
            rec.sfreq,
            reason_detail=f"need >=30s, got {rec.duration_s:.1f}s",
        )

    # ── Step 2: Channel fallback chain (case-insensitive) ─────────────────────
    channel_upper = channel.upper()
    ch_idx = rec.channel_index(channel)
    resolved_channel = rec.channel_names[ch_idx] if ch_idx is not None else channel

    if ch_idx is None:
        for fallback in ("Cz", "C3", "C4"):
            if fallback.upper() == channel_upper:
                continue
            ch_idx = rec.channel_index(fallback)
            if ch_idx is not None:
                resolved_channel = rec.channel_names[ch_idx]
                break

    if ch_idx is None:
        if rec.eeg_channel_indices:
            ch_idx = rec.eeg_channel_indices[0]
            resolved_channel = rec.channel_names[ch_idx]
        else:
            raise ValueError(
                "No suitable EEG channel found for HFO ripple detection after "
                "fallback through Cz → C3 → C4."
            )

    # ── Build signal (full recording) ─────────────────────────────────────────
    segments = []
    for _, d in rec.iter_epochs(epoch_seconds=30.0):
        segments.append(d[ch_idx])
    if not segments:
        raise ValueError("No data readable from recording.")
    signal = np.concatenate(segments).astype(np.float64)

    # ── Unit-scale guard (Volts vs µV) ────────────────────────────────────────
    p99 = float(np.percentile(np.abs(signal[np.isfinite(signal)]) if np.any(np.isfinite(signal)) else np.array([0.0]), 99))
    if p99 < 1.0:
        signal = signal * 1e6
        notes.append("auto_scaled_volts_to_uv")

    # ── NaN/Inf guard ─────────────────────────────────────────────────────────
    nan_frac = float(np.isnan(signal).sum()) / max(signal.size, 1)
    if nan_frac > 0.05:
        notes.append("high_nan_fraction")
    if not np.all(np.isfinite(signal)):
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    sfreq = rec.sfreq
    total_s = len(signal) / sfreq

    # ── Step 3: Notch filter (line_freq and 2nd harmonic) ─────────────────────
    signal = _notch(signal, sfreq, line_freq_hz)
    second_harmonic = 2.0 * line_freq_hz
    if second_harmonic < sfreq / 2.0:
        signal = _notch(signal, sfreq, second_harmonic)

    # ── Step 4: Bandpass 80–250 Hz (FIR) ──────────────────────────────────────
    filtered = _bandpass_fir(signal, sfreq, 80.0, 250.0)

    # Also bandpass for Burnos specificity check: 250–min(500, nyq-1) Hz
    # Full check requires sfreq ≥ 1000 Hz (nyq ≥ 500) for the complete
    # 250–500 Hz comparison band per Burnos et al. 2014.
    nyq = sfreq / 2.0
    high_band_hi = min(500.0, nyq - 1.0)
    do_burnos = high_band_hi > 250.0
    # C2: flag explicitly when check is degraded or disabled due to low sfreq.
    artifact_warnings_pre: list[str] = []
    if sfreq < 1000.0:
        # Either Burnos is fully disabled (sfreq ≤ ~502) or runs with a
        # compressed comparison band (502 < sfreq < 1000). Either way,
        # frequency-specificity rejection is below the validated range.
        notes.append("burnos_check_disabled_low_sfreq")
        artifact_warnings_pre.append(
            "frequency_specificity_check_unavailable_below_1khz_sfreq"
        )
        if not do_burnos:
            # Nyquist too low for any comparison band — skip entirely.
            pass  # do_burnos already False, filtered_high will not be used
    if do_burnos:
        filtered_high = _bandpass_fir(signal, sfreq, 250.0, high_band_hi)

    # ── Step 5: RMS envelope (6 ms window) ────────────────────────────────────
    # 6 ms provides better temporal smoothing and reduces spurious single-sample
    # peaks compared to the original 3 ms window.
    window_samp = max(1, int(round(0.006 * sfreq)))
    rms = _rms_envelope(filtered, window_samp)

    # ── Step 6: Threshold (5 × SD of background RMS, bottom 50 percentile) ────
    # Bottom 50% (median) follows Staba/Burnos convention: use only the quieter
    # half of the signal as background to avoid threshold inflation from bursts.
    bg_rms = rms[rms <= np.percentile(rms, 50)]
    if len(bg_rms) == 0:
        bg_rms = rms
    threshold = 5.0 * float(np.std(bg_rms))
    # Minimum absolute floor: 1 µV. Floor prevents sub-physiological
    # detections from filter-edge artifacts. (Scalp ripple amplitude ranges
    # are described in Kramer et al. 2019, PMID 30907404, though the
    # specific 1 µV value here is a tool-convention conservative floor.)
    _MIN_THRESHOLD_UV = 1.0
    threshold = max(threshold, _MIN_THRESHOLD_UV)
    if threshold <= 0 or not math.isfinite(threshold):
        # Degenerate signal — return zero events cleanly
        notes.append("degenerate_signal_zero_threshold")
        return HFORippleResult(
            channel=resolved_channel,
            sfreq_used=sfreq,
            available=True,
            unavailable_reason="",
            n_ripples_total=0,
            n_ripples_isolated=0,
            n_ripples_on_spike=0,
            rate_per_minute_nrem=0.0,
            median_duration_ms=0.0,
            median_peak_freq_hz=0.0,
            method="energy_staba_style",
            artifact_warnings=[],
            events=[],
            notes=notes,
        )

    # Mean of bg_rms used for z-score normalization
    bg_mean = float(np.mean(bg_rms))
    bg_std = float(np.std(bg_rms))
    if bg_std <= 0:
        bg_std = 1.0

    # ── Step 7: Event detection ───────────────────────────────────────────────
    # Minimum duration: 6 cycles at center frequency 165 Hz ≈ 36 ms
    center_freq = 165.0  # Hz (midpoint of 80–250 Hz band)
    min_dur_s = 6.0 / center_freq       # ~0.036 s
    max_dur_s = 0.200                    # 200 ms max
    min_dur_samp = max(1, int(round(min_dur_s * sfreq)))
    max_dur_samp = int(round(max_dur_s * sfreq))

    above = rms > threshold
    raw_events: list[dict] = []

    i = 0
    n_samp = len(rms)
    while i < n_samp:
        if above[i]:
            # Find start of this region
            start_i = i
            # Extend while rms > threshold
            while i < n_samp and above[i]:
                i += 1
            end_i = i  # exclusive

            # Expand edges while rms > threshold/2 (hysteresis)
            expand_thresh = threshold / 2.0
            s = start_i
            while s > 0 and rms[s - 1] > expand_thresh:
                s -= 1
            e = end_i
            while e < n_samp - 1 and rms[e] > expand_thresh:
                e += 1

            dur_samp = e - s
            if dur_samp < min_dur_samp or dur_samp > max_dur_samp:
                continue

            # Peak sample within the core above-threshold region
            peak_offset = int(np.argmax(rms[s:e]))
            peak_i = s + peak_offset
            peak_s_time = peak_i / sfreq
            start_s_time = s / sfreq
            end_s_time = e / sfreq
            duration_ms = (e - s) / sfreq * 1000.0

            # RMS z-score at peak
            rms_z = (float(rms[peak_i]) - bg_mean) / bg_std

            # Extract the filtered segment for spectral analysis
            seg = filtered[s:e]

            # ── Step 8: Burnos frequency-specificity check ────────────────────
            if do_burnos:
                seg_high = filtered_high[s:e]
                power_ripple = _power_in_band(seg, sfreq, 80.0, 250.0)
                power_high = _power_in_band(seg_high, sfreq, 250.0, high_band_hi)
                if power_high > 0 and (power_ripple / power_high) < 2.0:
                    continue  # reject: likely muscle/broad-band artifact

            # ── Step 9: Peak frequency ────────────────────────────────────────
            pf = _peak_freq(seg, sfreq, lo=80.0, hi=250.0)

            raw_events.append({
                "start_s": start_s_time,
                "end_s": end_s_time,
                "peak_s": peak_s_time,
                "duration_ms": round(duration_ms, 2),
                "peak_freq_hz": round(pf, 1),
                "rms_z": round(rms_z, 2),
                "co_occurs_with_spike": False,
            })
        else:
            i += 1

    # ── Step 10: NREM restriction ─────────────────────────────────────────────
    n2n3_windows: list[tuple[float, float]] | None = None
    nrem_total_s: float = total_s

    if sleep_stages is not None:
        nrem_labels = {"N2", "N3"}
        epoch_s = sleep_stages.epoch_seconds
        nrem_indices = [
            idx for idx, lbl in enumerate(sleep_stages.epoch_labels)
            if lbl in nrem_labels
        ]
        if not nrem_indices:
            notes.append("no_nrem_sleep")
            # Still compute events but rate will be 0
            n2n3_windows = []
        else:
            windows: list[tuple[float, float]] = []
            for ep_idx in nrem_indices:
                t0 = ep_idx * epoch_s
                t1 = (ep_idx + 1) * epoch_s
                if t0 < total_s:
                    windows.append((t0, min(t1, total_s)))
            n2n3_windows = windows
            nrem_total_s = sum(e - s for s, e in windows)

    # Filter events to NREM windows for rate calculation
    if n2n3_windows is not None:
        if len(n2n3_windows) == 0:
            nrem_events = []
        else:
            nrem_events = [
                ev for ev in raw_events
                if any(ws <= ev["peak_s"] < we for ws, we in n2n3_windows)
            ]
    else:
        nrem_events = raw_events

    nrem_total_min = nrem_total_s / 60.0
    if "no_nrem_sleep" in notes:
        rate_per_min = 0.0
    else:
        rate_per_min = len(nrem_events) / nrem_total_min if nrem_total_min > 0 else 0.0

    # ── Step 11: Spike co-occurrence ──────────────────────────────────────────
    spike_times: list[float] = []
    if morphology_events:
        for mev in morphology_events:
            # C3 fix: use explicit key precedence to avoid falsy coercion bug
            # where time_s == 0.0 would incorrectly fall through to next key.
            if "time_s" in mev:
                t = mev["time_s"]
            elif "neg_peak_s" in mev:
                t = mev["neg_peak_s"]
            elif "peak_s" in mev:
                t = mev["peak_s"]
            else:
                continue  # malformed event, skip
            if t is None or not math.isfinite(float(t) if t is not None else float("nan")):
                continue
            try:
                spike_times.append(float(t))
            except (TypeError, ValueError):
                pass

    for ev in raw_events:
        if any(abs(ev["peak_s"] - st) < 0.1 for st in spike_times):
            ev["co_occurs_with_spike"] = True

    n_on_spike = sum(1 for ev in raw_events if ev["co_occurs_with_spike"])
    n_isolated = len(raw_events) - n_on_spike

    # ── Step 12: Artifact warnings ────────────────────────────────────────────
    # Pre-populate with any warnings accumulated before event detection
    # (e.g. frequency_specificity_check_unavailable from C2 above).
    artifact_warnings: list[str] = list(artifact_warnings_pre)
    median_rms = float(np.median(rms))
    sigma_rms = float(np.std(rms))
    high_power_frac = float(np.mean(rms > (median_rms + 5.0 * sigma_rms)))
    if high_power_frac > 0.30:
        artifact_warnings.append(
            f"excess_broadband_power_fraction={high_power_frac:.2f}_possible_muscle_EMG"
        )

    # ── Median stats ─────────────────────────────────────────────────────────
    if raw_events:
        median_dur = float(np.median([ev["duration_ms"] for ev in raw_events]))
        median_pf = float(np.median([ev["peak_freq_hz"] for ev in raw_events]))
    else:
        median_dur = 0.0
        median_pf = 0.0

    return HFORippleResult(
        channel=resolved_channel,
        sfreq_used=sfreq,
        available=True,
        unavailable_reason="",
        n_ripples_total=len(raw_events),
        n_ripples_isolated=n_isolated,
        n_ripples_on_spike=n_on_spike,
        rate_per_minute_nrem=round(rate_per_min, 3),
        median_duration_ms=round(median_dur, 2),
        median_peak_freq_hz=round(median_pf, 1),
        method="energy_staba_style",
        artifact_warnings=artifact_warnings,
        events=raw_events,
        notes=notes,
    )


# ─── Summary ──────────────────────────────────────────────────────────────────

def summarize_hfo_ripples(result: HFORippleResult) -> dict:
    """Return a JSON-serializable summary dict (no events, no NaN/Inf).

    The raw events list is intentionally excluded from this summary to keep
    the findings dict compact. Callers that need the events list should
    retrieve ``findings["_hfo_ripples_events"]`` which is set by runner.py.
    """
    def _sf(x: float) -> float | None:
        return _safe_float(x)

    if not result.available:
        return {
            "available": False,
            "unavailable_reason": result.unavailable_reason,
            "sfreq_used": result.sfreq_used,
            "notes": result.notes,
            "disclaimer": _DISCLAIMER,
        }

    return {
        "available": True,
        "channel": result.channel,
        "sfreq_used": _sf(result.sfreq_used),
        "n_ripples_total": result.n_ripples_total,
        "n_ripples_isolated": result.n_ripples_isolated,
        "n_ripples_on_spike": result.n_ripples_on_spike,
        "rate_per_minute_nrem": _sf(round(result.rate_per_minute_nrem, 3)),
        "median_duration_ms": _sf(round(result.median_duration_ms, 2)),
        "median_peak_freq_hz": _sf(round(result.median_peak_freq_hz, 1)),
        "method": result.method,
        "artifact_warnings": result.artifact_warnings,
        "notes": result.notes,
        "disclaimer": _DISCLAIMER,
        # events intentionally omitted — stored under findings["_hfo_ripples_events"]
    }
