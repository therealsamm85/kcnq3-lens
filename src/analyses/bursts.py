"""Sustained rhythmic burst detection.

Looks for periods of sustained high-amplitude rhythmic activity (≥3 seconds)
that may represent subclinical electrographic events. These are particularly
relevant in KCNQ3-spectrum patients where the standard "no behavioral seizure"
read can miss real cortical events.

Detection strategy:
1. Bandpass filter to 5-25 Hz (sensitive to spike-wave components)
2. Compute amplitude envelope with smoothing
3. Find regions exceeding threshold for ≥ minimum duration
4. Verify multi-channel involvement to rule out focal artifact
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..readers.base import EEGRecording


@dataclass
class Burst:
    start_s: float          # seconds from start of recording
    duration_s: float
    peak_channel: str
    peak_amplitude: float   # peak-to-peak amplitude on peak channel
    n_channels_involved: int  # channels with >500 µV peak-to-peak during burst
    dominant_freq_hz: float


@dataclass
class BurstResult:
    n_bursts: int
    total_burst_seconds: float
    median_duration_s: float
    max_duration_s: float
    n_bursts_5s_or_longer: int
    n_bursts_10s_or_longer: int
    longest_bursts: list[Burst]  # top 10 by duration
    primary_channel: str
    search_window_hours: float


def compute_sustained_bursts(
    rec: EEGRecording,
    start_epoch: int,
    end_epoch: int,
    epoch_seconds: float = 30.0,
    target_channel: str = "Pz",
    min_duration_s: float = 3.0,
    threshold_mad_multiplier: float = 4.0,
    bandpass: tuple[float, float] = (5.0, 25.0),
) -> BurstResult:
    """Find sustained rhythmic bursts ≥ min_duration_s on target_channel.

    Parameters
    ----------
    rec : EEGRecording
    start_epoch, end_epoch : int
    target_channel : str
        Channel to detect on. Default Pz; will fall back to other midline channels.
    min_duration_s : float
        Minimum sustained duration to count as a burst.
    threshold_mad_multiplier : float
        Threshold for envelope = N × median absolute deviation.
    """
    ch_idx = rec.channel_index(target_channel)
    if ch_idx is None:
        for fallback in ("Pz", "Cz", "C3", "C4"):
            ch_idx = rec.channel_index(fallback)
            if ch_idx is not None:
                target_channel = fallback
                break
    if ch_idx is None:
        raise ValueError("No suitable channel found for burst detection.")

    sos = butter(4, list(bandpass), btype="band", fs=rec.sfreq, output="sos")

    # Build continuous trace and remember epoch boundaries
    trace_segments = []
    multi_ch_segments: list[np.ndarray] = []
    eeg_idx = rec.eeg_channel_indices
    for _, d in rec.iter_epochs(epoch_seconds=epoch_seconds, start=start_epoch, end=end_epoch):
        trace_segments.append(d[ch_idx])
        multi_ch_segments.append(d[eeg_idx])
    if not trace_segments:
        raise ValueError("No data in window.")

    trace = np.concatenate(trace_segments)
    multi = np.concatenate(multi_ch_segments, axis=1)
    filtered = sosfiltfilt(sos, trace)

    mad = np.median(np.abs(filtered - np.median(filtered)))
    threshold = threshold_mad_multiplier * mad

    # Smoothed envelope
    envelope = np.abs(filtered)
    win = max(1, int(0.25 * rec.sfreq))
    env_smooth = np.convolve(envelope, np.ones(win) / win, mode="same")
    above = env_smooth > threshold

    min_samples = int(min_duration_s * rec.sfreq)
    sec_per_sample = 1 / rec.sfreq
    bursts: list[Burst] = []
    eeg_names = [rec.channel_names[c] for c in eeg_idx]

    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if (j - i) >= min_samples:
                dur_s = (j - i) * sec_per_sample
                start_s = i * sec_per_sample + start_epoch * epoch_seconds
                # Multi-channel involvement check
                ch_amps = np.ptp(multi[:, i:j], axis=1)
                n_involved = int(np.sum(ch_amps > 500))
                peak_ch_local = int(np.argmax(ch_amps))
                peak_ch_name = eeg_names[peak_ch_local]
                peak_amp = float(ch_amps[peak_ch_local])

                # Dominant frequency
                seg = filtered[i:j]
                fft = np.abs(np.fft.rfft(seg))
                freqs = np.fft.rfftfreq(len(seg), 1 / rec.sfreq)
                in_band = (freqs >= 1) & (freqs <= 15)
                dom_freq = float(freqs[in_band][np.argmax(fft[in_band])]) if in_band.any() else 0.0

                bursts.append(Burst(
                    start_s=start_s,
                    duration_s=dur_s,
                    peak_channel=peak_ch_name,
                    peak_amplitude=peak_amp,
                    n_channels_involved=n_involved,
                    dominant_freq_hz=dom_freq,
                ))
            i = j
        else:
            i += 1

    durs = [b.duration_s for b in bursts]
    longest = sorted(bursts, key=lambda b: -b.duration_s)[:10]

    return BurstResult(
        n_bursts=len(bursts),
        total_burst_seconds=float(sum(durs)),
        median_duration_s=float(np.median(durs)) if durs else 0.0,
        max_duration_s=float(max(durs)) if durs else 0.0,
        n_bursts_5s_or_longer=int(sum(1 for d in durs if d >= 5)),
        n_bursts_10s_or_longer=int(sum(1 for d in durs if d >= 10)),
        longest_bursts=longest,
        primary_channel=target_channel,
        search_window_hours=(end_epoch - start_epoch) * epoch_seconds / 3600,
    )


def summarize_bursts(result: BurstResult) -> dict:
    return {
        "primary_channel": result.primary_channel,
        "search_window_hours": round(result.search_window_hours, 2),
        "n_bursts": result.n_bursts,
        "total_burst_seconds": round(result.total_burst_seconds, 1),
        "median_duration_s": round(result.median_duration_s, 1),
        "max_duration_s": round(result.max_duration_s, 1),
        "n_bursts_5s_or_longer": result.n_bursts_5s_or_longer,
        "n_bursts_10s_or_longer": result.n_bursts_10s_or_longer,
        "longest_bursts": [
            {
                "start_s": round(b.start_s, 1),
                "duration_s": round(b.duration_s, 1),
                "peak_channel": b.peak_channel,
                "n_channels_involved": b.n_channels_involved,
                "dominant_freq_hz": round(b.dominant_freq_hz, 1),
            }
            for b in result.longest_bursts
        ],
    }
