"""Automatic sleep-onset and sleep-window detection.

For long overnight recordings, manually specifying when sleep starts and ends
is error-prone — and small window differences can shift SWI calculations
dramatically. This module estimates the main sleep period from spectral
features that are robust across pediatric recordings:

- Delta-band power rises during sleep
- Alpha-band power drops during sleep
- The delta/alpha ratio per epoch is a robust binary-ish signal

Strategy:
1. Compute delta_rms and alpha_rms per 30s epoch across the whole recording
2. Compute log(delta/alpha) per epoch
3. Smooth with a 5-minute moving average
4. Find the longest contiguous run where the smoothed ratio exceeds a
   threshold calibrated from the recording's own distribution

This is a heuristic — not a substitute for proper polysomnographic sleep
staging — but works well as a starting window for the other analyses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..readers.base import EEGRecording


@dataclass
class SleepWindowResult:
    sleep_start_epoch: int
    sleep_end_epoch: int
    sleep_start_hours: float    # hours from recording start
    sleep_end_hours: float
    sleep_duration_hours: float
    confidence: str             # "high" | "medium" | "low"
    wake_indices: list[int]     # epochs identified as wake (for background analysis)
    delta_alpha_ratio_log: list[float]  # one value per epoch (for plotting/debug)


def detect_sleep_window(
    rec: EEGRecording,
    epoch_seconds: float = 30.0,
    central_channels: tuple[str, ...] = ("Cz", "C3", "C4", "Fz", "Pz"),
    smoothing_minutes: float = 5.0,
    min_sleep_hours: float = 1.0,
) -> SleepWindowResult:
    """Auto-detect the main sleep window from spectral features.

    Parameters
    ----------
    rec : EEGRecording
    epoch_seconds : float
    central_channels : iterable of str
        Channels averaged for the spectral computation. Central is best for
        delta/alpha discrimination across sleep stages.
    smoothing_minutes : float
        Moving-average window for the delta/alpha ratio.
    min_sleep_hours : float
        Minimum duration to be considered a valid sleep window. Below this,
        returns low-confidence result.
    """
    # Resolve channels
    indices = []
    for name in central_channels:
        i = rec.channel_index(name)
        if i is not None:
            indices.append(i)
    if not indices:
        raise ValueError("No central channels available for sleep detection.")

    sos_delta = butter(4, [0.5, 4.0], btype="band", fs=rec.sfreq, output="sos")
    sos_alpha = butter(4, [8.0, 13.0], btype="band", fs=rec.sfreq, output="sos")

    n_epochs = rec.n_epochs
    delta_rms = np.zeros(n_epochs)
    alpha_rms = np.zeros(n_epochs)

    for ep, d in rec.iter_epochs(epoch_seconds=epoch_seconds, end=n_epochs):
        chans = d[indices].mean(axis=0)
        df = sosfiltfilt(sos_delta, chans)
        af = sosfiltfilt(sos_alpha, chans)
        delta_rms[ep] = float(np.sqrt(np.mean(df ** 2)))
        alpha_rms[ep] = float(np.sqrt(np.mean(af ** 2)))

    # log delta/alpha ratio (higher = more sleep-like)
    eps_floor = 1e-3
    ratio_log = np.log(np.maximum(delta_rms, eps_floor) /
                       np.maximum(alpha_rms, eps_floor))

    # Smooth with moving average
    win_epochs = max(1, int(smoothing_minutes * 60 / epoch_seconds))
    kernel = np.ones(win_epochs) / win_epochs
    ratio_smooth = np.convolve(ratio_log, kernel, mode="same")

    # Threshold: 50th percentile of the smoothed ratio. Sleep should be the
    # upper ~30-50% of the recording for a typical overnight, but spike
    # clusters during sleep can briefly suppress the ratio.
    threshold = np.percentile(ratio_smooth, 50)
    above = ratio_smooth > threshold

    # Bridge short gaps (≤ 15 minutes) — sleep is rarely interrupted by long
    # wake epochs on otherwise-continuous overnight studies, and spike clusters
    # commonly create brief dips below threshold mid-sleep.
    max_gap_epochs = int(15 * 60 / epoch_seconds)
    bridged = above.copy()
    i = 0
    while i < len(bridged):
        if not bridged[i]:
            j = i
            while j < len(bridged) and not bridged[j]:
                j += 1
            # If gap is short enough AND surrounded by "above" runs, bridge it
            gap = j - i
            if gap <= max_gap_epochs and i > 0 and j < len(bridged) and \
               bridged[i - 1] and (j < len(bridged) and above[j]):
                bridged[i:j] = True
            i = j
        else:
            i += 1

    # Find longest contiguous run on the bridged signal
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(bridged):
        if bridged[i]:
            j = i
            while j < len(bridged) and bridged[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1

    total_h = n_epochs * epoch_seconds / 3600

    def _fallback_window():
        """Sensible default for an overnight: skip first 6h, take next 8h."""
        if total_h >= 14:
            start = int(6 * 3600 / epoch_seconds)
            end = int(14 * 3600 / epoch_seconds)
        elif total_h >= 6:
            start = n_epochs // 4
            end = 3 * n_epochs // 4
        else:
            start = 0
            end = n_epochs
        return start, end

    if not runs:
        sleep_start, sleep_end = _fallback_window()
        confidence = "low"
    else:
        longest = max(runs, key=lambda r: r[1] - r[0])
        sleep_start, sleep_end = longest
        duration_h = (sleep_end - sleep_start) * epoch_seconds / 3600

        # Sanity check: for recordings ≥ 14h, the detected sleep window should
        # be at least 4 hours. If it isn't, the heuristic is unreliable — fall
        # back to a conventional overnight window.
        if total_h >= 14 and duration_h < 4.0:
            sleep_start, sleep_end = _fallback_window()
            confidence = "low"
        elif duration_h < min_sleep_hours:
            confidence = "low"
        elif duration_h < 3.0:
            confidence = "medium"
        else:
            confidence = "high"

    # Wake epochs = NOT sleep and bounded away from sleep onset/end by 30 min
    pad_eps = int(30 * 60 / epoch_seconds)
    wake_mask = np.ones(n_epochs, dtype=bool)
    wake_mask[max(0, sleep_start - pad_eps):min(n_epochs, sleep_end + pad_eps)] = False
    wake_indices = np.where(wake_mask)[0].tolist()

    sleep_start_h = sleep_start * epoch_seconds / 3600
    sleep_end_h = sleep_end * epoch_seconds / 3600

    return SleepWindowResult(
        sleep_start_epoch=int(sleep_start),
        sleep_end_epoch=int(sleep_end),
        sleep_start_hours=float(sleep_start_h),
        sleep_end_hours=float(sleep_end_h),
        sleep_duration_hours=float(sleep_end_h - sleep_start_h),
        confidence=confidence,
        wake_indices=wake_indices,
        delta_alpha_ratio_log=ratio_smooth.tolist(),
    )


def summarize_sleep_window(result: SleepWindowResult) -> dict:
    return {
        "sleep_start_epoch": result.sleep_start_epoch,
        "sleep_end_epoch": result.sleep_end_epoch,
        "sleep_start_hours": round(result.sleep_start_hours, 2),
        "sleep_end_hours": round(result.sleep_end_hours, 2),
        "sleep_duration_hours": round(result.sleep_duration_hours, 2),
        "confidence": result.confidence,
        "n_wake_epochs_available": len(result.wake_indices),
    }
