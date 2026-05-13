"""Sleep spindle detection at central electrodes.

Spindles are 11–16 Hz transient oscillations lasting 0.5–2 seconds, generated
by the thalamocortical loop during N2 sleep. They are critical for memory
consolidation. Reduced density correlates with worse cognitive outcomes in
pediatric epilepsy.

Normative ranges at Cz during N2:
- Age 4–6:   3–5 spindles/min
- Age 7–9:   4–6 spindles/min
- Adults:    2–4 spindles/min (lower because N2 is a smaller fraction of sleep)

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

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from ..readers.base import EEGRecording


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


_AGE_NORMS = {
    3: (2.5, 4.5),
    4: (3.0, 5.0),
    5: (3.0, 5.0),
    6: (3.5, 5.5),
    7: (4.0, 6.0),
    8: (4.0, 6.0),
    9: (4.0, 6.0),
    10: (4.0, 6.0),
    12: (3.5, 5.5),
    15: (3.0, 5.0),
    18: (2.5, 4.5),
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
) -> tuple[int, float, float]:
    """Run YASA spindles_detect.

    Returns (n_spindles, mean_duration_s, median_peak_hz).
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
        return 0, 0.0, 0.0
    df = result.summary()
    if len(df) == 0:
        return 0, 0.0, 0.0
    return (
        int(len(df)),
        float(df["Duration"].mean()),
        float(df["Frequency"].median()),
    )


# ─── Backend 2: heuristic envelope (fallback) ────────────────────────────────


def _detect_with_heuristic(
    signal: np.ndarray, sfreq: float
) -> tuple[int, float, float]:
    sos = butter(4, [11.0, 16.0], btype="band", fs=sfreq, output="sos")
    filtered = sosfiltfilt(sos, signal)
    envelope = np.abs(hilbert(filtered))
    win = max(1, int(0.2 * sfreq))
    smooth = np.convolve(envelope, np.ones(win) / win, mode="same")

    threshold = np.percentile(smooth, 90)
    above = smooth > threshold
    min_samples = int(0.5 * sfreq)
    max_samples = int(2.5 * sfreq)

    spindles = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if min_samples <= (j - i) <= max_samples:
                spindles.append((i, j))
            i = j
        else:
            i += 1

    durations_s = [(j - i) / sfreq for i, j in spindles]
    peak_freqs = []
    for i, j in spindles[:200]:
        seg = filtered[i:j]
        if len(seg) < 8:
            continue
        fft = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(len(seg), 1 / sfreq)
        band = (freqs >= 11) & (freqs <= 16)
        if band.any():
            peak_freqs.append(freqs[band][np.argmax(fft[band])])

    return (
        len(spindles),
        float(np.mean(durations_s)) if durations_s else 0.0,
        float(np.median(peak_freqs)) if peak_freqs else 0.0,
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
    # Resolve channel
    ch_idx = rec.channel_index(channel)
    if ch_idx is None:
        for fallback in ("Cz", "Pz", "Fz", "C3", "C4"):
            ch_idx = rec.channel_index(fallback)
            if ch_idx is not None:
                channel = fallback
                break
    if ch_idx is None:
        raise ValueError("No central EEG channel found for spindle detection.")

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
        n_spindles, mean_dur, peak_hz = _detect_with_yasa(signal_for_yasa, rec.sfreq)
    else:
        n_spindles, mean_dur, peak_hz = _detect_with_heuristic(x, rec.sfreq)

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
    }
