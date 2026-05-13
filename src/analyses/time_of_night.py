"""Time-of-night spike burden distribution.

For each time bin (default 30 minutes) across the recording, counts the
number of spike candidates on the highest-burden channel. This reveals:

- When during the night spike activity peaks (typical CSWS pattern peaks in
  the first NREM cycle, ~1-3 hours after sleep onset)
- Whether spikes cluster in deep sleep vs distribute uniformly
- Whether wake/sleep transitions show explosive activation (KCNQ3 R230H
  pattern)

Uses the same per-epoch local-MAD detection as morphology.py to avoid the
v0.1 global-MAD over-counting bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

from ..readers.base import EEGRecording


@dataclass
class TimeOfNightResult:
    target_channel: str
    bin_minutes: float
    bin_start_hours: list[float]
    bin_center_hours: list[float]
    counts_per_minute: list[float]
    peak_bin_hours: float
    peak_count_per_min: float
    total_events: int
    total_hours: float


def compute_time_of_night(
    rec: EEGRecording,
    start_epoch: int = 0,
    end_epoch: int | None = None,
    epoch_seconds: float = 30.0,
    bin_minutes: float = 30.0,
    target_channel: str = "Pz",
    detection_bandpass: tuple[float, float] = (5.0, 25.0),
    mad_multiplier: float = 6.0,
) -> TimeOfNightResult:
    """Compute spike burden binned across the recording timeline.

    Uses the same per-epoch local-MAD detection used by `morphology.py`,
    so counts are comparable to the morphology event rate.
    """
    if end_epoch is None:
        end_epoch = rec.n_epochs

    ch_idx = rec.channel_index(target_channel)
    if ch_idx is None:
        for fallback in ("Pz", "Cz", "C3", "C4"):
            ch_idx = rec.channel_index(fallback)
            if ch_idx is not None:
                target_channel = fallback
                break
    if ch_idx is None:
        raise ValueError("No suitable channel for time-of-night analysis.")

    sos = butter(4, list(detection_bandpass), btype="band", fs=rec.sfreq, output="sos")
    samples_per_epoch = int(epoch_seconds * rec.sfreq)
    min_dist = max(1, int(0.08 * rec.sfreq))
    epochs_per_bin = int(bin_minutes * 60 / epoch_seconds)

    bin_starts_hours: list[float] = []
    bin_centers_hours: list[float] = []
    bin_counts: list[int] = []

    current_bin_peaks = 0
    current_bin_epochs = 0
    current_bin_start_ep = start_epoch

    for ep_idx, d in rec.iter_epochs(epoch_seconds=epoch_seconds,
                                     start=start_epoch, end=end_epoch):
        signal = d[ch_idx]
        filtered = sosfiltfilt(sos, signal)
        centered = filtered - np.median(filtered)
        local_mad = np.median(np.abs(centered))
        local_rms = float(np.sqrt(np.mean(filtered ** 2)))
        local_threshold = max(mad_multiplier * local_mad, 3.0 * local_rms)
        if local_threshold > 0 and np.isfinite(local_threshold):
            peaks, _ = find_peaks(np.abs(filtered),
                                  height=local_threshold,
                                  distance=min_dist)
            current_bin_peaks += len(peaks)
        current_bin_epochs += 1

        # Flush bin when full
        if current_bin_epochs >= epochs_per_bin:
            start_h = current_bin_start_ep * epoch_seconds / 3600
            center_h = start_h + bin_minutes / 60 / 2
            bin_starts_hours.append(start_h)
            bin_centers_hours.append(center_h)
            bin_counts.append(current_bin_peaks)
            current_bin_peaks = 0
            current_bin_epochs = 0
            current_bin_start_ep = ep_idx + 1

    # Flush remainder
    if current_bin_epochs > 0:
        start_h = current_bin_start_ep * epoch_seconds / 3600
        center_h = start_h + (current_bin_epochs * epoch_seconds / 3600) / 2
        bin_starts_hours.append(start_h)
        bin_centers_hours.append(center_h)
        bin_counts.append(current_bin_peaks)

    counts_per_min = [c / bin_minutes for c in bin_counts]
    peak_idx = int(np.argmax(counts_per_min)) if counts_per_min else 0
    peak_bin_h = bin_centers_hours[peak_idx] if bin_centers_hours else 0.0
    peak_count = counts_per_min[peak_idx] if counts_per_min else 0.0

    return TimeOfNightResult(
        target_channel=target_channel,
        bin_minutes=bin_minutes,
        bin_start_hours=bin_starts_hours,
        bin_center_hours=bin_centers_hours,
        counts_per_minute=counts_per_min,
        peak_bin_hours=peak_bin_h,
        peak_count_per_min=peak_count,
        total_events=sum(bin_counts),
        total_hours=(end_epoch - start_epoch) * epoch_seconds / 3600,
    )


def summarize_time_of_night(result: TimeOfNightResult) -> dict:
    return {
        "target_channel": result.target_channel,
        "bin_minutes": result.bin_minutes,
        "total_hours": round(result.total_hours, 2),
        "total_events": result.total_events,
        "peak_bin_hours": round(result.peak_bin_hours, 2),
        "peak_count_per_min": round(result.peak_count_per_min, 1),
        "bin_centers": [round(h, 2) for h in result.bin_center_hours],
        "counts_per_min": [round(c, 1) for c in result.counts_per_minute],
    }
