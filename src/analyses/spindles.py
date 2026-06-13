"""Sleep spindle detection at central electrodes.

Spindles are 11–16 Hz transient oscillations lasting 0.5–2 seconds, generated
by the thalamocortical loop during N2 sleep. They are critical for memory
consolidation. Reduced density correlates with worse cognitive outcomes in
pediatric epilepsy.

Normative ranges at central derivations (C3/C4/Cz) during NREM2.
Sources: McClain et al. 2016 (PMID 27110405; ages 2–5, n=8 longitudinal)
and Kwon et al. 2023 (PMID 36719044; ages 0–18, n=567):
- Age 2–5:   ~1 spindle/min (0.8–1.5)  — McClain 2016
- Age 6–10:  ~1.5–2.5 spindles/min     — Kwon 2023 (rising)
- Age 11–14: ~2.5–3.5 spindles/min     — Kwon 2023 (continued rise)
- Adults:    ~2–4 spindles/min         — plateau after ~14

NOTE: Older versions of this tool (≤ v0.11.0) cited a "3–5/min for age 5"
range attributed to Wamsley 2012. That was incorrect — the Wamsley paper
is on adult schizophrenia and contains no pediatric norms. The values
above are from the actual pediatric literature and are roughly 3× lower
at age 5 than previously claimed.

Detection methods
-----------------
This module provides two detection backends:

1. **YASA** (default when installed) — uses the validated three-criteria
   approach from Lacourse et al. 2019 (correlation with spindle template,
   relative power in spindle band, and RMS thresholding). This is the recommended
   choice — it has been validated against expert-scored polysomnograms and is
   far less prone to false positives than envelope-threshold methods.

2. **Heuristic fallback** — simple envelope-percentile detection. **Tends to
   over-count** because it cannot distinguish spindles from mu/alpha bursts or
   other 11–16 Hz transients. Use only when YASA is unavailable.

Integration history
-------------------
v0.1 used the heuristic only. v0.3 added YASA as default after a side-by-side
comparison on the reference the reference patient recording showed the heuristic produced
~150× more "detections" than YASA, the vast majority of which were almost
certainly false positives (mu rhythm, beta bursts, drowsy alpha, transient
artifact). See tests/compare_yasa.py and tests/yasa_sensitivity.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from ..readers.base import EEGRecording

_DISCLAIMER = (
    "Spindle density interpretation ('below'/'in'/'above') uses ±30% "
    "ranges around the values reported by McClain 2016 (n=8 longitudinal, "
    "ages 2-5) and Kwon 2023. These ranges are a TOOL CONVENTION for "
    "longitudinal tracking, not a published clinical cutoff. The McClain "
    "cohort is too small to support deterministic 'abnormal' calls. Use "
    "the interpretation label for intra-patient comparison only."
)


@dataclass
class SpindleResult:
    channel: str
    n_spindles: int
    duration_hours: float
    density_per_minute: float
    mean_duration_s: float
    median_peak_freq_hz: float
    age_normative_range: tuple[float, float] | None
    interpretation: str        # "below", "in", "above", "no_age_provided"
    method: str                # "yasa" or "heuristic"
    events: list[dict] = field(default_factory=list)
    # [{peak_time_s: float, start_s: float, end_s: float, duration_s: float}]


# Anchors from McClain 2016 (PMID 27110405) at ages 2-5 (≈1/min at C3/C4 NREM2)
# and Kwon 2023 (PMID 36719044) for the developmental rise to adolescent plateau.
# Ranges are ±~30% around the reported mean to account for inter-individual
# variability seen in those cohorts.
_AGE_NORMS = {
    2: (0.6, 1.3),
    3: (0.7, 1.4),
    4: (0.8, 1.5),
    5: (0.8, 1.5),
    6: (1.0, 2.0),
    7: (1.2, 2.5),
    8: (1.5, 2.8),
    9: (1.8, 3.0),
    10: (2.0, 3.2),
    12: (2.2, 3.5),
    15: (2.5, 3.8),
    18: (2.0, 4.0),
}


def _normative_range(age: float | None) -> tuple[float, float] | None:
    if age is None:
        return None
    ages = sorted(_AGE_NORMS.keys())
    closest = min(ages, key=lambda a: abs(a - age))
    return _AGE_NORMS[closest]


def _interp(density: float, norm: tuple[float, float] | None) -> str:
    if norm is None:
        return "no_age_provided"
    if density < norm[0]:
        return "below"
    if density > norm[1]:
        return "above"
    return "in"


def _yasa_available() -> bool:
    try:
        import yasa  # noqa: F401
        return True
    except ImportError:
        return False


# ─── Backend 1: YASA (recommended) ───────────────────────────────────────────


def _detect_with_yasa(
    signal_uv: np.ndarray, sfreq: float
) -> tuple[int, float, float, list[dict]]:
    """Run YASA spindles_detect.

    Returns (n_spindles, mean_duration_s, median_peak_hz, events).
    events: list of {peak_time_s, start_s, end_s, duration_s}
    """
    import yasa

    result = yasa.spindles_detect(
        signal_uv.astype(np.float64),
        sf=sfreq,
        freq_sp=(11, 16),
        freq_broad=(1, 30),
        duration=(0.5, 2.5),
        min_distance=500,
        thresh={"corr": 0.65, "rel_pow": 0.20, "rms": 1.5},
        multi_only=False,
        remove_outliers=False,
        verbose=False,
    )
    if result is None:
        return 0, 0.0, 0.0, []
    df = result.summary()
    if len(df) == 0:
        return 0, 0.0, 0.0, []

    # Build events list with peak_time_s.
    # YASA summary has "Start", "End", "Peak" columns (in seconds).
    events: list[dict] = []
    for _, row in df.iterrows():
        start_s = float(row["Start"])
        end_s = float(row["End"])
        # YASA "Peak" column is the peak sample time in seconds
        if "Peak" in df.columns:
            peak_s = float(row["Peak"])
        else:
            # Fallback: midpoint of start/end
            peak_s = (start_s + end_s) / 2.0
        dur_s = float(row["Duration"])
        if all(np.isfinite(v) for v in [start_s, end_s, peak_s, dur_s]):
            events.append({
                "start_s": start_s,
                "end_s": end_s,
                "peak_time_s": peak_s,
                "duration_s": dur_s,
            })

    return (
        int(len(df)),
        float(df["Duration"].mean()),
        float(df["Frequency"].median()),
        events,
    )


# ─── Backend 2: heuristic envelope (fallback) ────────────────────────────────


def _detect_with_heuristic(
    signal: np.ndarray, sfreq: float
) -> tuple[int, float, float, list[dict]]:
    """Envelope-percentile spindle detector.

    Returns (n_spindles, mean_duration_s, median_peak_hz, events).
    events: list of {peak_time_s, start_s, end_s, duration_s}
    """
    sos = butter(4, [11.0, 16.0], btype="band", fs=sfreq, output="sos")
    filtered = sosfiltfilt(sos, signal)
    envelope = np.abs(hilbert(filtered))
    win = max(1, int(0.2 * sfreq))
    smooth = np.convolve(envelope, np.ones(win) / win, mode="same")

    threshold = np.percentile(smooth, 90)
    above = smooth > threshold
    min_samples = int(0.5 * sfreq)
    max_samples = int(2.5 * sfreq)

    spindle_ranges = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if min_samples <= (j - i) <= max_samples:
                spindle_ranges.append((i, j))
            i = j
        else:
            i += 1

    durations_s = [(j - i) / sfreq for i, j in spindle_ranges]
    peak_freqs = []
    for i, j in spindle_ranges[:200]:
        seg = filtered[i:j]
        if len(seg) < 8:
            continue
        fft = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(len(seg), 1 / sfreq)
        band = (freqs >= 11) & (freqs <= 16)
        if band.any():
            peak_freqs.append(freqs[band][np.argmax(fft[band])])

    # Build events list: peak_time_s from envelope maximum within spindle
    events: list[dict] = []
    for i, j in spindle_ranges:
        seg_env = smooth[i:j]
        peak_offset = int(np.argmax(seg_env))
        peak_sample = i + peak_offset
        events.append({
            "start_s": float(i / sfreq),
            "end_s": float(j / sfreq),
            "peak_time_s": float(peak_sample / sfreq),
            "duration_s": float((j - i) / sfreq),
        })

    return (
        len(spindle_ranges),
        float(np.mean(durations_s)) if durations_s else 0.0,
        float(np.median(peak_freqs)) if peak_freqs else 0.0,
        events,
    )


# ─── Public entry point ──────────────────────────────────────────────────────


def compute_spindle_density(
    rec: EEGRecording,
    sleep_start_epoch: int,
    sleep_end_epoch: int,
    epoch_seconds: float = 30.0,
    channel: str = "Cz",
    age_years: float | None = None,
    method: str = "auto",
    target_amplitude_uv: float = 20.0,
) -> SpindleResult:
    """Detect sleep spindles in the 11–16 Hz band on a central channel.

    Parameters
    ----------
    rec : EEGRecording
    sleep_start_epoch, sleep_end_epoch : int
        Range of epochs covering the sleep period.
    epoch_seconds : float
    channel : str
        Channel name (default "Cz"). Cz, Pz, Fz are good choices.
    age_years : float, optional
        Child's age, used to look up age-appropriate normative range.
    method : str
        "auto"      → use YASA if installed, else heuristic
        "yasa"      → force YASA (raises if unavailable)
        "heuristic" → force the envelope-percentile fallback
    target_amplitude_uv : float
        For non-EDF formats (e.g. raw int16 ADC counts), scale the signal so
        its standard deviation matches this target before passing to YASA.
        YASA's amplitude thresholds expect µV-scaled input.
    """
    # Resolve channel — v0.18.5: liveness-aware (skip present-but-dead channels).
    # Falls back to the plain channel_index chain for minimal rec-like objects
    # (e.g. test doubles) that don't implement resolve_live_channel.
    _cands = [channel] + [
        c for c in ("Cz", "Pz", "Fz", "C3", "C4") if c.upper() != channel.upper()
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
        raise ValueError("No central EEG channel found for spindle detection.")
    channel = resolved_name

    # Build continuous Cz trace
    segments = []
    for _, d in rec.iter_epochs(
        epoch_seconds=epoch_seconds, start=sleep_start_epoch, end=sleep_end_epoch
    ):
        segments.append(d[ch_idx])
    if not segments:
        raise ValueError("No data in specified sleep window.")
    x = np.concatenate(segments).astype(np.float32)

    # Resolve method
    if method == "auto":
        method = "yasa" if _yasa_available() else "heuristic"
    if method == "yasa" and not _yasa_available():
        raise RuntimeError(
            "YASA not installed. Install with: pip install yasa, "
            "or pass method='heuristic'."
        )

    # Scale to µV-like range if using YASA (its amplitude thresholds need it)
    if method == "yasa":
        centered = x - x.mean()
        signal_for_yasa = centered * (target_amplitude_uv / max(centered.std(), 1e-9))
        n_spindles, mean_dur, peak_hz, events = _detect_with_yasa(
            signal_for_yasa, rec.sfreq
        )
    else:
        n_spindles, mean_dur, peak_hz, events = _detect_with_heuristic(
            x, rec.sfreq
        )

    total_hours = len(x) / rec.sfreq / 3600
    total_min = len(x) / rec.sfreq / 60
    density = n_spindles / total_min if total_min > 0 else 0.0
    norm = _normative_range(age_years)

    return SpindleResult(
        channel=channel,
        n_spindles=n_spindles,
        duration_hours=total_hours,
        density_per_minute=density,
        mean_duration_s=mean_dur,
        median_peak_freq_hz=peak_hz,
        age_normative_range=norm,
        interpretation=_interp(density, norm),
        method=method,
        events=events,
    )


def summarize_spindles(result: SpindleResult) -> dict:
    return {
        "channel": result.channel,
        "method": result.method,
        "n_spindles": result.n_spindles,
        "duration_hours": round(result.duration_hours, 2),
        "density_per_minute": round(result.density_per_minute, 2),
        "mean_duration_s": round(result.mean_duration_s, 2),
        "median_peak_freq_hz": round(result.median_peak_freq_hz, 2),
        "age_normative_range": result.age_normative_range,
        "interpretation": result.interpretation,
        "disclaimer": _DISCLAIMER,
    }
