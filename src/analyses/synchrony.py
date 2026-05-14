"""Bilateral synchrony and spread analysis.

For each detected spike on the primary channel, ask: which other channels
fire within ±50 ms? The pattern of co-firing tells us:

- **Focal**         — only 1–2 nearby channels (e.g. Cz + Pz only)
- **Regional**      — 3–5 adjacent channels on the same side
- **Bilateral synchronous** — homologous channels on left and right fire
                              simultaneously (e.g. C3 + C4)
- **Bilateral asynchronous** — both sides involved but offset in time
                                (suggests propagation, not generalization)
- **Generalized**   — most channels fire together (true generalized SW)

These categories have direct clinical implications:
- Generalized SW with bilateral synchrony → generalized epilepsy
- Bilateral asynchronous → multifocal pattern
- Strictly focal → can be benign focal epilepsy
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

from ..readers.base import EEGRecording


@dataclass
class SynchronyResult:
    primary_channel: str
    n_events_analyzed: int
    # Counts per pattern category
    focal_count: int               # 1-2 channels involved
    regional_count: int            # 3-5 channels, same hemisphere
    bilateral_sync_count: int      # homologous L+R within ±50ms
    bilateral_async_count: int     # both sides but delayed
    generalized_count: int         # ≥10 channels
    # Percentages
    focal_pct: float
    regional_pct: float
    bilateral_sync_pct: float
    bilateral_async_pct: float
    generalized_pct: float
    # Dominant pattern
    dominant_pattern: str
    # Median number of channels co-firing per event
    median_channels_per_event: float


# Channel pairs for bilateral-synchrony check (left, right homologue)
HOMOLOGOUS_PAIRS = [
    ("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8"),
    ("C3", "C4"), ("T3", "T4"), ("T5", "T6"),
    ("P3", "P4"), ("O1", "O2"),
]

LEFT_HEMI = {"Fp1", "F3", "F7", "C3", "T3", "T5", "P3", "O1"}
RIGHT_HEMI = {"Fp2", "F4", "F8", "C4", "T4", "T6", "P4", "O2"}
MIDLINE = {"Fz", "Cz", "Pz"}


def compute_synchrony(
    rec: EEGRecording,
    start_epoch: int,
    end_epoch: int,
    epoch_seconds: float = 30.0,
    primary_channel: str = "Pz",
    detection_bandpass: tuple[float, float] = (5.0, 25.0),
    mad_multiplier: float = 6.0,
    synchrony_window_ms: float = 50.0,
    max_events_to_analyze: int = 200,
) -> SynchronyResult:
    """Analyze the spread / synchrony pattern of detected spikes.

    Parameters
    ----------
    rec : EEGRecording
    start_epoch, end_epoch : int
    primary_channel : str
        Channel used as the trigger (typically the highest-burden channel).
    synchrony_window_ms : float
        Half-window for co-firing detection (default ±50 ms around primary peak).
    max_events_to_analyze : int
        Cap for performance — analyses N evenly-spaced events.
    """
    ch_idx = rec.channel_index(primary_channel)
    if ch_idx is None:
        for fb in ("Pz", "Cz", "C3", "C4"):
            ch_idx = rec.channel_index(fb)
            if ch_idx is not None:
                primary_channel = fb
                break
    if ch_idx is None:
        raise ValueError("No suitable primary channel for synchrony analysis.")

    sos = butter(4, list(detection_bandpass), btype="band", fs=rec.sfreq, output="sos")
    min_dist = max(1, int(0.08 * rec.sfreq))
    win_samples = int(synchrony_window_ms / 1000.0 * rec.sfreq)

    eeg_idx = rec.eeg_channel_indices
    eeg_names = [rec.channel_names[c] for c in eeg_idx]

    # Build continuous trace of all EEG channels in the window
    segs_multi: list[np.ndarray] = []
    for _, d in rec.iter_epochs(epoch_seconds=epoch_seconds,
                                start=start_epoch, end=end_epoch):
        segs_multi.append(d[eeg_idx])
    if not segs_multi:
        raise ValueError("No data in window.")
    multi = np.concatenate(segs_multi, axis=1)

    # Filter every channel for spike detection
    filtered = np.zeros_like(multi, dtype=np.float32)
    for j in range(multi.shape[0]):
        filtered[j] = sosfiltfilt(sos, multi[j])

    # Detect spikes on the primary channel (per-epoch local MAD)
    primary_idx_in_eeg = eeg_idx.index(ch_idx)
    primary_filt = filtered[primary_idx_in_eeg]

    samples_per_epoch = int(epoch_seconds * rec.sfreq)
    all_peaks: list[int] = []
    n_epochs_window = multi.shape[1] // samples_per_epoch
    for ep_i in range(n_epochs_window):
        s = ep_i * samples_per_epoch
        e = s + samples_per_epoch
        seg = primary_filt[s:e]
        mad = np.median(np.abs(seg - np.median(seg)))
        rms = float(np.sqrt(np.mean(seg ** 2)))
        threshold = max(mad_multiplier * mad, 3.0 * rms)
        if threshold <= 0 or not np.isfinite(threshold):
            continue
        local_peaks, _ = find_peaks(np.abs(seg), height=threshold, distance=min_dist)
        all_peaks.extend((local_peaks + s).tolist())

    if not all_peaks:
        return SynchronyResult(
            primary_channel=primary_channel, n_events_analyzed=0,
            focal_count=0, regional_count=0, bilateral_sync_count=0,
            bilateral_async_count=0, generalized_count=0,
            focal_pct=0, regional_pct=0, bilateral_sync_pct=0,
            bilateral_async_pct=0, generalized_pct=0,
            dominant_pattern="no_events",
            median_channels_per_event=0.0,
        )

    # Subsample events for performance
    if len(all_peaks) > max_events_to_analyze:
        step = len(all_peaks) // max_events_to_analyze
        all_peaks = all_peaks[::step][:max_events_to_analyze]

    # Per-channel co-firing thresholds (using global MAD per channel — adequate
    # for binary "is this channel involved" decision)
    channel_thresholds = np.zeros(filtered.shape[0])
    for j in range(filtered.shape[0]):
        mad_j = np.median(np.abs(filtered[j] - np.median(filtered[j])))
        channel_thresholds[j] = 4.0 * mad_j  # slightly more permissive than primary

    # Categorize each event
    focal = regional = bilat_sync = bilat_async = generalized = 0
    channels_per_event: list[int] = []

    for p in all_peaks:
        lo = max(0, p - win_samples)
        hi = min(filtered.shape[1], p + win_samples + 1)
        # For each channel, was there a peak in this window?
        involved: list[str] = []
        for j, name in enumerate(eeg_names):
            seg = np.abs(filtered[j, lo:hi])
            if seg.size == 0:
                continue
            if seg.max() > channel_thresholds[j]:
                involved.append(name)
        channels_per_event.append(len(involved))

        n_inv = len(involved)
        left = sum(1 for c in involved if c in LEFT_HEMI)
        right = sum(1 for c in involved if c in RIGHT_HEMI)
        # Bilateral-pair detection
        pair_count = sum(
            1 for L, R in HOMOLOGOUS_PAIRS if L in involved and R in involved
        )

        if n_inv >= 10:
            generalized += 1
        elif n_inv <= 2:
            focal += 1
        elif pair_count >= 2 and abs(left - right) <= 2:
            bilat_sync += 1
        elif left > 0 and right > 0 and pair_count == 0:
            bilat_async += 1
        else:
            regional += 1

    total = focal + regional + bilat_sync + bilat_async + generalized
    if total == 0:
        return SynchronyResult(
            primary_channel=primary_channel, n_events_analyzed=0,
            focal_count=0, regional_count=0, bilateral_sync_count=0,
            bilateral_async_count=0, generalized_count=0,
            focal_pct=0, regional_pct=0, bilateral_sync_pct=0,
            bilateral_async_pct=0, generalized_pct=0,
            dominant_pattern="no_events",
            median_channels_per_event=0.0,
        )

    def _pct(n): return 100 * n / total

    cats = {
        "focal": focal, "regional": regional,
        "bilateral_synchronous": bilat_sync,
        "bilateral_asynchronous": bilat_async,
        "generalized": generalized,
    }
    dominant = max(cats, key=cats.get)

    return SynchronyResult(
        primary_channel=primary_channel,
        n_events_analyzed=total,
        focal_count=focal, regional_count=regional,
        bilateral_sync_count=bilat_sync,
        bilateral_async_count=bilat_async,
        generalized_count=generalized,
        focal_pct=float(_pct(focal)),
        regional_pct=float(_pct(regional)),
        bilateral_sync_pct=float(_pct(bilat_sync)),
        bilateral_async_pct=float(_pct(bilat_async)),
        generalized_pct=float(_pct(generalized)),
        dominant_pattern=dominant,
        median_channels_per_event=float(np.median(channels_per_event)),
    )


def summarize_synchrony(result: SynchronyResult) -> dict:
    return {
        "primary_channel": result.primary_channel,
        "n_events_analyzed": result.n_events_analyzed,
        "focal_pct": round(result.focal_pct, 1),
        "regional_pct": round(result.regional_pct, 1),
        "bilateral_synchronous_pct": round(result.bilateral_sync_pct, 1),
        "bilateral_asynchronous_pct": round(result.bilateral_async_pct, 1),
        "generalized_pct": round(result.generalized_pct, 1),
        "dominant_pattern": result.dominant_pattern,
        "median_channels_per_event": round(result.median_channels_per_event, 1),
    }
