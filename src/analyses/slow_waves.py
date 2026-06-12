"""Slow-wave (slow oscillation) detection via YASA.

Slow waves are large-amplitude, low-frequency (0.3–1.5 Hz) oscillations
that dominate NREM sleep and are essential for memory consolidation and
synaptic homeostasis (Tononi & Cirelli 2006). They consist of a negative
half-wave (down-state, near-silence) followed by a positive half-wave
(up-state, cortical firing).

Detection is based on the Massimini et al. 2004 criteria as implemented
in YASA (Vallat & Walker 2021): the algorithm detects candidate events in
the 0.3–1.5 Hz bandpass signal, then applies amplitude and duration
thresholds on the negative peak (down-state), positive peak (up-state),
and peak-to-peak deflection.

Pediatric amplitude tuning
---------------------------
Children under ~8 years produce substantially larger slow waves than
adults; default YASA thresholds (amp_neg 40–200 µV, amp_pos 10–150 µV)
would miss much of the upper-amplitude range and incorrectly reject
many genuine events. When age_years < 8 this module raises the upper
bounds: amp_neg=(60, 300) µV, amp_pos=(20, 200) µV.

**IMPORTANT — no validated pediatric normative values exist** for
slow-wave density in the clinical literature as of 2026. The Carrier et al.
2011 and Kurth et al. 2010 references describe age-related trends in
adults and adolescents respectively, but cover only age ≥12. This module
therefore reports descriptive metrics only — no "below/in/above" range
classification is applied. Do not interpret density numbers against adult
norms for children.

Detection methods
-----------------
1. **YASA** (default when installed) — uses the Massimini et al. 2004
   validated criteria. Recommended; produces far fewer false positives than
   the heuristic fallback.
2. **Heuristic fallback** — bandpass 0.3–1.5 Hz, detect negative half-waves
   exceeding a fixed amplitude threshold, duration-gated. Over-counts in
   the presence of slow artefacts or high-amplitude movement. Use only when
   YASA is unavailable.

Slope sign convention
---------------------
Both backends report ``slope_uv_per_s`` as the **rising slope** of the
slow wave:

    slope_uv_per_s = (zero_cross_uv - neg_peak_uv) / (zero_cross_s - neg_peak_s)

This quantity is **positive** for a well-formed slow wave (the signal
rises from the negative peak toward the zero crossing). YASA's own
``Slope`` column uses the same convention (Vallat & Walker 2021, Table 1).
If YASA is available its column is used directly; otherwise the heuristic
re-derives the slope from the detected neg-peak and zero-crossing samples
using the formula above.

+------------------+-----------------------------------------------------------+
| Field            | Convention                                                |
+==================+===========================================================+
| neg_peak_uv      | negative (down-state peak), typically −40 to −300 µV     |
+------------------+-----------------------------------------------------------+
| pos_peak_uv      | positive (up-state peak), typically +10 to +200 µV       |
+------------------+-----------------------------------------------------------+
| ptp_uv           | pos_peak_uv − neg_peak_uv, positive                      |
+------------------+-----------------------------------------------------------+
| slope_uv_per_s   | (zero_cross_uv − neg_peak_uv) / (zero_cross_s −          |
|                  | neg_peak_s), positive for valid slow waves               |
+------------------+-----------------------------------------------------------+

References
----------
Massimini M et al. 2004 PMID 15295020
Carrier J et al. 2011 PMID 21226772
Kurth S et al. 2010  PMID 20926647
Vallat R & Walker MP 2021 PMID 34648426
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..readers.base import EEGRecording
from .sleep_stages import SleepStageResult


# ─── Result dataclass ─────────────────────────────────────────────────────────


@dataclass
class SlowWaveResult:
    channel: str
    n_slow_waves: int
    density_per_minute: float
    mean_neg_peak_uv: float
    mean_pos_peak_uv: float
    mean_ptp_uv: float            # peak-to-peak
    mean_duration_s: float
    mean_slope_uv_per_s: float    # neg-peak → zero-cross slope
    method: str                   # "yasa" | "heuristic"
    events: list[dict] = field(default_factory=list)
    # [{start_s, neg_peak_s, zero_cross_s, end_s, neg_peak_uv, pos_peak_uv,
    #   ptp_uv, duration_s, slope_uv_per_s}]
    notes: list[str] = field(default_factory=list)
    # warnings/flags e.g. "pediatric_thresholds_applied", "yasa_no_detections"


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _yasa_available() -> bool:
    try:
        import yasa  # noqa: F401
        return True
    except ImportError:
        return False


def _zero_result(channel: str, method: str, notes: list[str]) -> SlowWaveResult:
    return SlowWaveResult(
        channel=channel,
        n_slow_waves=0,
        density_per_minute=0.0,
        mean_neg_peak_uv=0.0,
        mean_pos_peak_uv=0.0,
        mean_ptp_uv=0.0,
        mean_duration_s=0.0,
        mean_slope_uv_per_s=0.0,
        method=method,
        events=[],
        notes=notes,
    )


def _safe_float_for_summary(x: float) -> float | None:
    """Return x if finite, else None (for JSON-safe summary fields)."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _validate_event(ev: dict) -> bool:
    """Return True if all float fields in the event dict are finite."""
    for v in ev.values():
        if isinstance(v, float) and not math.isfinite(v):
            return False
    return True


# ─── Backend 1: YASA (recommended) ───────────────────────────────────────────


def _detect_with_yasa(
    signal_uv: np.ndarray,
    sfreq: float,
    age_years: float | None,
) -> tuple[int, float, float, float, float, float, list[dict]]:
    """Run YASA sw_detect.

    Returns (n, mean_neg_uv, mean_pos_uv, mean_ptp_uv, mean_dur_s,
             mean_slope, events).
    """
    import yasa

    # Pediatric amplitude tuning
    if age_years is not None and age_years < 8:
        amp_neg = (60, 300)
        amp_pos = (20, 200)
        amp_ptp = (75, 500)
    else:
        # YASA defaults
        amp_neg = (40, 200)
        amp_pos = (10, 150)
        amp_ptp = (75, 350)

    result = yasa.sw_detect(
        signal_uv.astype(np.float64),
        sf=sfreq,
        freq_sw=(0.3, 1.5),
        dur_neg=(0.3, 1.5),
        dur_pos=(0.1, 1.0),
        amp_neg=amp_neg,
        amp_pos=amp_pos,
        amp_ptp=amp_ptp,
        remove_outliers=False,
        verbose=False,
    )

    if result is None:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, []

    df = result.summary()
    if len(df) == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, []

    # YASA Slope column uses the same rising-slope convention we document.
    inf_dropped = 0
    events = []
    for _, row in df.iterrows():
        ev = {
            "start_s": float(row["Start"]),
            "neg_peak_s": float(row["NegPeak"]),
            "zero_cross_s": float(row["MidCrossing"]),
            "end_s": float(row["End"]),
            "neg_peak_uv": float(row["ValNegPeak"]),
            "pos_peak_uv": float(row["ValPosPeak"]),
            "ptp_uv": float(row["PTP"]),
            "duration_s": float(row["Duration"]),
            "slope_uv_per_s": float(row["Slope"]),
        }
        if _validate_event(ev):
            events.append(ev)
        else:
            inf_dropped += 1

    if not events:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, []

    neg_vals = [e["neg_peak_uv"] for e in events]
    pos_vals = [e["pos_peak_uv"] for e in events]
    ptp_vals = [e["ptp_uv"] for e in events]
    dur_vals = [e["duration_s"] for e in events]
    slope_vals = [e["slope_uv_per_s"] for e in events]

    return (
        len(events),
        float(np.mean(neg_vals)),
        float(np.mean(pos_vals)),
        float(np.mean(ptp_vals)),
        float(np.mean(dur_vals)),
        float(np.mean(slope_vals)),
        events,
    )


# ─── Backend 2: heuristic (fallback) ─────────────────────────────────────────


def _detect_with_heuristic(
    signal: np.ndarray,
    sfreq: float,
    age_years: float | None,
) -> tuple[int, float, float, float, float, float, list[dict]]:
    """Simple bandpass + negative half-wave threshold detection.

    Tends to over-count compared to YASA; use only when YASA unavailable.

    Slope is computed as:
        slope_uv_per_s = (zero_cross_uv - neg_peak_uv) / (zero_cross_s - neg_peak_s)
    This is positive for rising slow waves (see module docstring).
    """
    sos = butter(4, [0.3, 1.5], btype="band", fs=sfreq, output="sos")
    filtered = sosfiltfilt(sos, signal.astype(np.float64))

    # Amplitude threshold: 40 µV (adults) / 60 µV (children < 8)
    if age_years is not None and age_years < 8:
        threshold_neg = 60.0
    else:
        threshold_neg = 40.0

    below = filtered < -threshold_neg

    min_samp = int(0.3 * sfreq)
    max_samp = int(1.5 * sfreq)

    inf_dropped = 0
    events = []
    i = 0
    while i < len(below):
        if below[i]:
            j = i
            while j < len(below) and below[j]:
                j += 1
            dur_samp = j - i
            if min_samp <= dur_samp <= max_samp:
                seg = filtered[i:j]
                neg_peak_idx = int(np.argmin(seg))
                neg_peak_uv = float(seg[neg_peak_idx])
                # positive half-wave: next dur_samp samples after j
                pos_end = min(j + dur_samp, len(filtered))
                pos_seg = filtered[j:pos_end]
                pos_peak_uv = float(np.max(pos_seg)) if len(pos_seg) > 0 else 0.0
                ptp = pos_peak_uv - neg_peak_uv
                dur_s = dur_samp / sfreq
                # Slope: (zero_cross_uv - neg_peak_uv) / (zero_cross_s - neg_peak_s)
                # zero_cross_uv ≈ 0 (the bandpass value at j is near zero)
                zero_cross_s = j / sfreq
                neg_peak_s_abs = (i + neg_peak_idx) / sfreq
                dt = zero_cross_s - neg_peak_s_abs
                slope = (0.0 - neg_peak_uv) / max(dt, 1e-6)
                start_s = i / sfreq
                ev = {
                    "start_s": start_s,
                    "neg_peak_s": neg_peak_s_abs,
                    "zero_cross_s": zero_cross_s,
                    "end_s": pos_end / sfreq,
                    "neg_peak_uv": neg_peak_uv,
                    "pos_peak_uv": pos_peak_uv,
                    "ptp_uv": ptp,
                    "duration_s": dur_s,
                    "slope_uv_per_s": slope,
                }
                if _validate_event(ev):
                    events.append(ev)
                else:
                    inf_dropped += 1
            i = j
        else:
            i += 1

    if not events:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, []

    neg_vals = [e["neg_peak_uv"] for e in events]
    pos_vals = [e["pos_peak_uv"] for e in events]
    ptp_vals = [e["ptp_uv"] for e in events]
    dur_vals = [e["duration_s"] for e in events]
    slope_vals = [e["slope_uv_per_s"] for e in events]

    return (
        len(events),
        float(np.mean(neg_vals)),
        float(np.mean(pos_vals)),
        float(np.mean(ptp_vals)),
        float(np.mean(dur_vals)),
        float(np.mean(slope_vals)),
        events,
    )


# ─── Public entry point ───────────────────────────────────────────────────────


def compute_slow_waves(
    rec: EEGRecording,
    sleep_stages: SleepStageResult | None = None,
    channel: str = "Fz",
    age_years: float | None = None,
) -> SlowWaveResult:
    """Detect slow waves (0.3–1.5 Hz) on a frontal channel.

    Slow waves are reported descriptively. **No normative range comparison
    is applied** because validated pediatric norms for slow-wave density
    do not exist in the published literature as of 2026.

    Parameters
    ----------
    rec : EEGRecording
    sleep_stages : SleepStageResult, optional
        If provided, detection is restricted to N2 + N3 epochs only.
        Detection runs on the **full signal** (no concatenation artifact);
        events whose neg_peak_s falls outside any N2/N3 epoch window are
        discarded after detection.
    channel : str
        Preferred channel (default "Fz"). Falls back to Cz → C3 → first
        available EEG channel if the requested channel is absent.
        Channel name is matched case-insensitively.
    age_years : float, optional
        Child's age. When < 8, raises upper amplitude bounds to accommodate
        the larger slow waves seen in young children. NaN is treated as None
        (adult defaults).

    Returns
    -------
    SlowWaveResult

    Raises
    ------
    ValueError
        If the recording is shorter than 60 seconds, or if no EEG channel
        can be found at all.
    """
    # --- Validate recording length ------------------------------------------
    if rec.duration_s < 60:
        raise ValueError(
            f"Recording is only {rec.duration_s:.1f}s; slow-wave detection "
            "requires at least 60 seconds of data."
        )

    # --- Normalise age_years: NaN → None (treat as adult) -------------------
    notes: list[str] = []
    if age_years is not None:
        try:
            if math.isnan(float(age_years)):
                notes.append("age_years_was_nan")
                age_years = None
            elif float(age_years) < 0:
                age_years = None
        except (TypeError, ValueError):
            age_years = None

    # --- Channel fallback chain (case-insensitive) ---------------------------
    # channel_index() already compares case-insensitively; we normalise the
    # requested name to upper-case for the skip-check in the fallback loop.
    # resolved_channel is always set to the actual stored channel name (not the
    # requested name), so "fz" → "Fz" when the recording stores "Fz".
    channel_upper = channel.upper()

    # v0.18.4: channel-liveness guard. A present-but-dead channel (e.g. an
    # unplugged electrode that the long-form NK reader still maps to a name)
    # silently yields 0 slow waves if we run on it. Build the candidate chain
    # and pick the first channel that is actually live (non-flat). Without
    # this, FA06301E's "Fz" maps to a 0.4 µV flat trace and the detector
    # reports a false slow-wave deficit.
    def _channel_is_live(idx: int) -> bool:
        """Sample a few epochs; return False if the channel is essentially flat."""
        try:
            n = rec.n_epochs
            sample_eps = [n // 4, n // 2, (3 * n) // 4] if n >= 4 else list(range(n))
            stds = []
            for ep in sample_eps:
                d = rec.read_epoch(ep, 30.0)
                if d is None:
                    continue
                stds.append(float(d[idx].std()))
            if not stds:
                return True  # can't sample → don't block
            # < 1.5 µV trimmed std over deep-sleep-capable windows = dead/flat.
            # (Real EEG, even quiet wake, sits well above this.)
            return float(np.median(stds)) >= 1.5
        except Exception:
            return True  # never let the liveness probe crash detection

    # Ordered, de-duplicated candidate names: requested first, then fallbacks.
    _seen_up = set()
    candidate_names = []
    for nm in (channel, "Fz", "Cz", "C3", "C4", "Pz"):
        up = nm.upper()
        if up not in _seen_up:
            _seen_up.add(up)
            candidate_names.append(nm)

    ch_idx = None
    resolved_channel = channel
    first_present_idx = None
    first_present_name = channel
    for nm in candidate_names:
        idx = rec.channel_index(nm)
        if idx is None:
            continue
        if first_present_idx is None:
            first_present_idx = idx
            first_present_name = rec.channel_names[idx]
        if _channel_is_live(idx):
            ch_idx = idx
            resolved_channel = rec.channel_names[idx]
            break

    if ch_idx is None:
        # No named candidate was live — scan all EEG channels for a live one.
        for idx in rec.eeg_channel_indices:
            if _channel_is_live(idx):
                ch_idx = idx
                resolved_channel = rec.channel_names[idx]
                notes.append("fell_back_to_live_eeg_channel")
                break

    if ch_idx is None:
        # Everything looks flat — proceed on the first present candidate (or
        # first EEG channel) so behaviour is unchanged for genuinely odd files,
        # but flag it loudly so the 0-result isn't read as biology.
        if first_present_idx is not None:
            ch_idx = first_present_idx
            resolved_channel = first_present_name
        elif rec.eeg_channel_indices:
            ch_idx = rec.eeg_channel_indices[0]
            resolved_channel = rec.channel_names[ch_idx]
        else:
            raise ValueError(
                "none of preferred channels available for slow-wave detection."
            )
        notes.append("no_live_channel_found_results_may_be_artifact")

    # --- Determine method ----------------------------------------------------
    method = "yasa" if _yasa_available() else "heuristic"

    if age_years is not None and age_years < 8:
        notes.append("pediatric_thresholds_applied")

    # --- Build full signal (entire recording) --------------------------------
    # C1 fix: always run detector on the full signal to avoid step-discontinuity
    # ringing from concatenating non-adjacent N2/N3 segments.
    full_segments = []
    for _, d in rec.iter_epochs(epoch_seconds=30.0):
        full_segments.append(d[ch_idx])
    if not full_segments:
        raise ValueError("No data readable from recording.")
    signal = np.concatenate(full_segments).astype(np.float32)

    # --- H4: Unit-scale guard (Volts vs µV) ----------------------------------
    p99 = float(np.percentile(np.abs(signal), 99))
    if p99 < 1.0:
        signal = signal * 1e6
        notes.append("auto_scaled_volts_to_uv")
    elif p99 > 10000:
        notes.append(f"unexpected_signal_scale_p99_{p99:.0f}")

    # --- C3: NaN/Inf guard ---------------------------------------------------
    nan_frac = float(np.isnan(signal).sum()) / max(signal.size, 1)
    if nan_frac > 0.05:
        notes.append("high_nan_fraction")
    if nan_frac > 0:
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    total_s = len(signal) / rec.sfreq

    # --- Determine N2/N3 windows and total N2/N3 duration --------------------
    n2n3_windows: list[tuple[float, float]] | None = None
    n2n3_total_s: float = total_s  # default: full recording

    if sleep_stages is not None:
        nrem_labels = {"N2", "N3"}
        epoch_s = sleep_stages.epoch_seconds
        nrem_epoch_indices = [
            i for i, lbl in enumerate(sleep_stages.epoch_labels)
            if lbl in nrem_labels
        ]
        if not nrem_epoch_indices:
            return _zero_result(
                resolved_channel, method, notes + ["no_n2_n3_sleep"]
            )

        # H1: track dropped epochs
        requested = len(nrem_epoch_indices)
        dropped = 0
        windows: list[tuple[float, float]] = []
        for ep_idx in nrem_epoch_indices:
            start_s = ep_idx * epoch_s
            end_s = (ep_idx + 1) * epoch_s
            # Verify epoch is readable (check against signal length)
            if start_s < total_s:
                windows.append((start_s, min(end_s, total_s)))
            else:
                dropped += 1

        # Also try via read_epoch if available, but count failures
        read_dropped = 0
        if hasattr(rec, 'read_epoch'):
            for ep_idx in nrem_epoch_indices:
                d = rec.read_epoch(ep_idx, epoch_s)
                if d is None:
                    read_dropped += 1
            if requested > 0:
                drop_frac = read_dropped / requested
                if drop_frac > 0.10:
                    notes.append(
                        f"epoch_signal_mismatch_high_drop_{drop_frac:.2f}"
                    )

        if not windows:
            return _zero_result(
                resolved_channel, method, notes + ["no_n2_n3_sleep"]
            )

        n2n3_windows = windows
        n2n3_total_s = sum(end - start for start, end in windows)

    total_min = n2n3_total_s / 60.0

    # --- Run detector on full signal -----------------------------------------
    if method == "yasa":
        n, mean_neg, mean_pos, mean_ptp, mean_dur, mean_slope, events = (
            _detect_with_yasa(signal, rec.sfreq, age_years)
        )
        if n == 0:
            notes.append("yasa_no_detections")
    else:
        n, mean_neg, mean_pos, mean_ptp, mean_dur, mean_slope, events = (
            _detect_with_heuristic(signal, rec.sfreq, age_years)
        )

    # --- C1: Filter events to N2/N3 windows ---------------------------------
    if n2n3_windows is not None:
        events = [
            e for e in events
            if any(start <= e["neg_peak_s"] < end for start, end in n2n3_windows)
        ]
        n = len(events)
        if n > 0:
            mean_neg = float(np.mean([e["neg_peak_uv"] for e in events]))
            mean_pos = float(np.mean([e["pos_peak_uv"] for e in events]))
            mean_ptp = float(np.mean([e["ptp_uv"] for e in events]))
            mean_dur = float(np.mean([e["duration_s"] for e in events]))
            mean_slope = float(np.mean([e["slope_uv_per_s"] for e in events]))
        else:
            mean_neg = mean_pos = mean_ptp = mean_dur = mean_slope = 0.0

    density = n / total_min if total_min > 0 else 0.0

    return SlowWaveResult(
        channel=resolved_channel,
        n_slow_waves=n,
        density_per_minute=density,
        mean_neg_peak_uv=mean_neg,
        mean_pos_peak_uv=mean_pos,
        mean_ptp_uv=mean_ptp,
        mean_duration_s=mean_dur,
        mean_slope_uv_per_s=mean_slope,
        method=method,
        events=events,
        notes=notes,
    )


def summarize_slow_waves(result: SlowWaveResult) -> dict:
    """Return a JSON-serializable summary dict (no events, no NaN/Inf).

    The raw events list is intentionally excluded from this summary to keep
    the findings dict compact. Callers that need the events list should
    store the SlowWaveResult object directly, or use the private
    ``_slow_waves_events`` key that runner.py injects into findings.
    """
    def _sf(x: float) -> float | None:
        """Safe float: return None instead of NaN/Inf."""
        return _safe_float_for_summary(x)

    return {
        "channel": result.channel,
        "method": result.method,
        "n_slow_waves": result.n_slow_waves,
        "density_per_minute": _sf(round(result.density_per_minute, 2)),
        "mean_neg_peak_uv": _sf(round(result.mean_neg_peak_uv, 2)),
        "mean_pos_peak_uv": _sf(round(result.mean_pos_peak_uv, 2)),
        "mean_ptp_uv": _sf(round(result.mean_ptp_uv, 2)),
        "mean_duration_s": _sf(round(result.mean_duration_s, 3)),
        "mean_slope_uv_per_s": _sf(round(result.mean_slope_uv_per_s, 1)),
        "notes": result.notes,
        # events list omitted from summary to keep findings dict compact;
        # raw events are preserved under findings["_slow_waves_events"] by runner.
    }
