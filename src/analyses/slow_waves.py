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

References
----------
Massimini M et al. 2004 PMID 15282274
Carrier J et al. 2011 PMID 20813192
Kurth S et al. 2010  PMID 20534927
Vallat R & Walker MP 2021 PMID 34648426
"""

from __future__ import annotations

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

    events = []
    for _, row in df.iterrows():
        events.append({
            "start_s": float(row["Start"]),
            "neg_peak_s": float(row["NegPeak"]),
            "zero_cross_s": float(row["MidCrossing"]),
            "end_s": float(row["End"]),
            "neg_peak_uv": float(row["ValNegPeak"]),
            "pos_peak_uv": float(row["ValPosPeak"]),
            "ptp_uv": float(row["PTP"]),
            "duration_s": float(row["Duration"]),
            "slope_uv_per_s": float(row["Slope"]),
        })

    return (
        int(len(df)),
        float(df["ValNegPeak"].mean()),
        float(df["ValPosPeak"].mean()),
        float(df["PTP"].mean()),
        float(df["Duration"].mean()),
        float(df["Slope"].mean()),
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
                # slope: neg-peak → end of negative half-wave (zero crossing)
                slope = -neg_peak_uv / max(((dur_samp - neg_peak_idx) / sfreq), 1e-6)
                start_s = i / sfreq
                events.append({
                    "start_s": start_s,
                    "neg_peak_s": start_s + neg_peak_idx / sfreq,
                    "zero_cross_s": j / sfreq,
                    "end_s": pos_end / sfreq,
                    "neg_peak_uv": neg_peak_uv,
                    "pos_peak_uv": pos_peak_uv,
                    "ptp_uv": ptp,
                    "duration_s": dur_s,
                    "slope_uv_per_s": slope,
                })
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
    channel : str
        Preferred channel (default "Fz"). Falls back to Cz → C3 → first
        available EEG channel if the requested channel is absent.
    age_years : float, optional
        Child's age. When < 8, raises upper amplitude bounds to accommodate
        the larger slow waves seen in young children.

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

    # --- Channel fallback chain: Fz → Cz → C3 → first available -------------
    ch_idx = rec.channel_index(channel)
    resolved_channel = channel
    if ch_idx is None:
        for fallback in ("Fz", "Cz", "C3"):
            if fallback == channel:
                continue
            ch_idx = rec.channel_index(fallback)
            if ch_idx is not None:
                resolved_channel = fallback
                break
    if ch_idx is None:
        # Last resort: first EEG channel
        if rec.eeg_channel_indices:
            ch_idx = rec.eeg_channel_indices[0]
            resolved_channel = rec.channel_names[ch_idx]
        else:
            raise ValueError(
                "none of preferred channels available for slow-wave detection."
            )

    # --- Determine method ----------------------------------------------------
    method = "yasa" if _yasa_available() else "heuristic"
    notes: list[str] = []

    if age_years is not None and age_years < 8:
        notes.append("pediatric_thresholds_applied")

    # --- Build signal: full recording or N2/N3 only --------------------------
    if sleep_stages is not None:
        nrem_labels = {"N2", "N3"}
        nrem_epoch_indices = [
            i for i, lbl in enumerate(sleep_stages.epoch_labels)
            if lbl in nrem_labels
        ]
        if not nrem_epoch_indices:
            return _zero_result(
                resolved_channel, method, notes + ["no_n2_n3_sleep"]
            )

        segments = []
        for ep_idx in nrem_epoch_indices:
            d = rec.read_epoch(ep_idx, sleep_stages.epoch_seconds)
            if d is not None:
                segments.append(d[ch_idx])
        if not segments:
            return _zero_result(
                resolved_channel, method, notes + ["no_n2_n3_sleep"]
            )
        signal = np.concatenate(segments).astype(np.float32)
    else:
        # Use entire recording
        segments = []
        for _, d in rec.iter_epochs(epoch_seconds=30.0):
            segments.append(d[ch_idx])
        if not segments:
            raise ValueError("No data readable from recording.")
        signal = np.concatenate(segments).astype(np.float32)

    total_s = len(signal) / rec.sfreq
    total_min = total_s / 60.0

    # --- Run detector --------------------------------------------------------
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
    return {
        "channel": result.channel,
        "method": result.method,
        "n_slow_waves": result.n_slow_waves,
        "density_per_minute": round(result.density_per_minute, 2),
        "mean_neg_peak_uv": round(result.mean_neg_peak_uv, 2),
        "mean_pos_peak_uv": round(result.mean_pos_peak_uv, 2),
        "mean_ptp_uv": round(result.mean_ptp_uv, 2),
        "mean_duration_s": round(result.mean_duration_s, 3),
        "mean_slope_uv_per_s": round(result.mean_slope_uv_per_s, 1),
        "notes": result.notes,
        # events list omitted from summary to keep findings dict compact;
        # raw events are available on the SlowWaveResult object directly.
    }
