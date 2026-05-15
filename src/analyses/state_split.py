"""Wake vs sleep spike-rate split + activation factor.

Standard clinical EEG reports separate spike rates by state because the
clinical implications differ entirely:

- High wake spike rate → consider seizure liability, daytime cognitive impact
- High sleep spike rate with normal wake → sleep-activated pattern (CSWS)
- Activation factor = sleep_rate / wake_rate is a single number that
  clinicians use to decide whether sleep activation is present.

A factor of ~1 means no sleep activation. Factor ≥ 3 is significant sleep
activation. Factor ≥ 10 is dramatic sleep activation (typical of CSWS).

NOTE on activation_label cuts (none<1.5, mild<3, moderate<10, strong≥10):
these specific bin edges are this tool's reporting convention, NOT a
published clinical threshold. The CSWS criterion proper is N3 SWI ≥ 85%
(Tassinari 1971). The activation factor is descriptive context, not
diagnostic — treat the labels as a readability aid, not as a
literature-derived classification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

from ..readers.base import EEGRecording
from .sleep_stages import SleepStageResult


@dataclass
class StateSplitResult:
    channel: str
    wake_rate_per_min: float
    nrem_rate_per_min: float
    rem_rate_per_min: float
    activation_factor: float       # nrem / wake
    wake_minutes: float
    nrem_minutes: float
    rem_minutes: float
    n_wake_spikes: int
    n_nrem_spikes: int
    n_rem_spikes: int
    activation_label: str          # 'none' | 'mild' | 'moderate' | 'strong'


def compute_state_split(
    rec: EEGRecording,
    sleep_stages: SleepStageResult,
    target_channel: str = "Pz",
    epoch_seconds: float = 30.0,
    detection_bandpass: tuple[float, float] = (5.0, 25.0),
    mad_multiplier: float = 6.0,
) -> StateSplitResult:
    """Compute spike rates per state (wake / NREM / REM) and activation factor.

    Uses the same per-epoch local-MAD detector as morphology.py and swi.py
    so all spike-rate numbers across the report are mutually consistent.
    """
    ch_idx = rec.channel_index(target_channel)
    if ch_idx is None:
        for fb in ("Pz", "Cz", "C3", "C4", "Fz"):
            ch_idx = rec.channel_index(fb)
            if ch_idx is not None:
                target_channel = fb
                break
    if ch_idx is None:
        raise ValueError("No suitable channel for state-split analysis.")

    sos = butter(4, list(detection_bandpass), btype="band", fs=rec.sfreq, output="sos")
    min_dist = max(1, int(0.08 * rec.sfreq))

    labels = sleep_stages.epoch_labels

    n_wake = n_nrem = n_rem = 0       # spike counts
    m_wake = m_nrem = m_rem = 0.0     # minutes per state

    for ep_idx, d in rec.iter_epochs(epoch_seconds=epoch_seconds):
        if ep_idx >= len(labels):
            break
        stage = labels[ep_idx]
        signal = d[ch_idx]
        filtered = sosfiltfilt(sos, signal)
        centered = filtered - np.median(filtered)
        local_mad = np.median(np.abs(centered))
        local_rms = float(np.sqrt(np.mean(filtered ** 2)))
        threshold = max(mad_multiplier * local_mad, 3.0 * local_rms)
        if threshold <= 0 or not np.isfinite(threshold):
            n_peaks = 0
        else:
            peaks, _ = find_peaks(np.abs(filtered), height=threshold, distance=min_dist)
            n_peaks = len(peaks)

        epoch_min = epoch_seconds / 60.0
        if stage == "W":
            n_wake += n_peaks
            m_wake += epoch_min
        elif stage in ("N1", "N2", "N3"):
            n_nrem += n_peaks
            m_nrem += epoch_min
        elif stage == "REM":
            n_rem += n_peaks
            m_rem += epoch_min

    wake_rate = n_wake / m_wake if m_wake > 0 else 0.0
    nrem_rate = n_nrem / m_nrem if m_nrem > 0 else 0.0
    rem_rate = n_rem / m_rem if m_rem > 0 else 0.0

    # Activation factor with floor on wake_rate to avoid division by ~zero
    if wake_rate < 0.1:
        # If wake is essentially silent, define factor as NREM rate (cap)
        activation = nrem_rate
    else:
        activation = nrem_rate / wake_rate

    if activation < 1.5:
        label = "none"
    elif activation < 3.0:
        label = "mild"
    elif activation < 10.0:
        label = "moderate"
    else:
        label = "strong"

    return StateSplitResult(
        channel=target_channel,
        wake_rate_per_min=float(wake_rate),
        nrem_rate_per_min=float(nrem_rate),
        rem_rate_per_min=float(rem_rate),
        activation_factor=float(activation),
        wake_minutes=float(m_wake),
        nrem_minutes=float(m_nrem),
        rem_minutes=float(m_rem),
        n_wake_spikes=int(n_wake),
        n_nrem_spikes=int(n_nrem),
        n_rem_spikes=int(n_rem),
        activation_label=label,
    )


def summarize_state_split(result: StateSplitResult) -> dict:
    return {
        "channel": result.channel,
        "wake_rate_per_min": round(result.wake_rate_per_min, 1),
        "nrem_rate_per_min": round(result.nrem_rate_per_min, 1),
        "rem_rate_per_min": round(result.rem_rate_per_min, 1),
        "activation_factor": round(result.activation_factor, 1),
        "activation_label": result.activation_label,
        "wake_minutes": round(result.wake_minutes, 1),
        "nrem_minutes": round(result.nrem_minutes, 1),
        "rem_minutes": round(result.rem_minutes, 1),
        "n_wake_spikes": result.n_wake_spikes,
        "n_nrem_spikes": result.n_nrem_spikes,
        "n_rem_spikes": result.n_rem_spikes,
    }
