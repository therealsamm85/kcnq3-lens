"""Formal Spike-Wave Index (SWI) per sleep stage.

SWI is the percentage of sleep time occupied by continuous spike-wave
activity. It is THE clinical number used to diagnose Continuous Spike-Wave
in Sleep (CSWS) / Electrical Status Epilepticus during Sleep (ESES).

Diagnostic threshold (Tassinari): **SWI ≥ 85% during NREM sleep** indicates
CSWS / ESES. Some centers use ≥ 50% as a less-strict threshold.

This module:
1. Takes per-epoch sleep stages (from sleep_stages.compute_sleep_stages)
2. Takes per-epoch spike detections (from a re-run of morphology's per-epoch
   local-MAD detector)
3. Within each stage, computes the fraction of epoch time covered by
   continuous spike-wave bursts (≥1 spike/s sustained for ≥3 s)

The "continuous" criterion is what distinguishes SWI from raw spike count.
A child with isolated spikes scattered through sleep has 0% SWI; a child
with sustained spike-wave runs covering most of NREM has ≥85% SWI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

from ..readers.base import EEGRecording
from .sleep_stages import SleepStageResult


@dataclass
class SWIResult:
    channel: str
    swi_per_stage: dict[str, float]      # 'N1' / 'N2' / 'N3' / 'REM' / 'W' → % SWI
    swi_nrem_combined: float              # weighted across N1+N2+N3
    swi_n3_only: float                    # the most clinically relevant
    csws_criterion_met: bool              # True if N3 SWI ≥ 85%
    csws_threshold_pct: float
    minutes_analyzed_per_stage: dict[str, float]
    n_epochs_with_continuous_sw: int
    epoch_seconds: float


def compute_swi(
    rec: EEGRecording,
    sleep_stages: SleepStageResult,
    target_channel: str = "Pz",
    epoch_seconds: float = 30.0,
    detection_bandpass: tuple[float, float] = (5.0, 25.0),
    mad_multiplier: float = 6.0,
    min_sw_burst_s: float = 3.0,
    min_spikes_per_burst_per_sec: float = 1.0,
    csws_threshold_pct: float = 85.0,
) -> SWIResult:
    """Compute SWI per sleep stage.

    Parameters
    ----------
    rec : EEGRecording
    sleep_stages : SleepStageResult
        Per-epoch stage labels from compute_sleep_stages().
    target_channel : str
    detection_bandpass : (low, high)
        Same band used by the v0.3 morphology detector.
    mad_multiplier : float
        Same threshold scheme as morphology.py.
    min_sw_burst_s : float
        Minimum sustained-burst duration to count toward SWI.
    min_spikes_per_burst_per_sec : float
        Minimum spike density within the burst window.
    csws_threshold_pct : float
        Default 85% — Tassinari criterion for CSWS/ESES.
    """
    ch_idx = rec.channel_index(target_channel)
    if ch_idx is None:
        for fb in ("Pz", "Cz", "C3", "C4", "Fz"):
            ch_idx = rec.channel_index(fb)
            if ch_idx is not None:
                target_channel = fb
                break
    if ch_idx is None:
        raise ValueError("No suitable channel for SWI analysis.")

    sos = butter(4, list(detection_bandpass), btype="band", fs=rec.sfreq, output="sos")
    samples_per_epoch = int(epoch_seconds * rec.sfreq)
    min_dist = max(1, int(0.08 * rec.sfreq))
    min_burst_samples = int(min_sw_burst_s * rec.sfreq)

    # For each stage, count epochs covered by sustained SW activity
    stage_epoch_count: dict[str, int] = {s: 0 for s in ("W", "N1", "N2", "N3", "REM")}
    stage_sw_seconds: dict[str, float] = {s: 0.0 for s in ("W", "N1", "N2", "N3", "REM")}
    epochs_with_continuous_sw = 0

    labels = sleep_stages.epoch_labels
    n_labels = len(labels)

    for ep_idx, d in rec.iter_epochs(epoch_seconds=epoch_seconds):
        if ep_idx >= n_labels:
            break
        stage = labels[ep_idx]
        if stage not in stage_epoch_count:
            stage = "W"
        stage_epoch_count[stage] += 1

        signal = d[ch_idx]
        filtered = sosfiltfilt(sos, signal)
        centered = filtered - np.median(filtered)
        local_mad = np.median(np.abs(centered))
        local_rms = float(np.sqrt(np.mean(filtered ** 2)))
        threshold = max(mad_multiplier * local_mad, 3.0 * local_rms)
        if threshold <= 0 or not np.isfinite(threshold):
            continue

        # Detect spike peaks in this epoch
        peaks, _ = find_peaks(np.abs(filtered), height=threshold, distance=min_dist)
        if len(peaks) < 2:
            continue

        # Convert peak times to inter-peak intervals; find sustained runs
        # where the local rate is ≥ min_spikes_per_burst_per_sec
        sw_samples = _continuous_sw_coverage(
            peaks, samples_per_epoch, rec.sfreq,
            min_burst_samples=min_burst_samples,
            min_rate_hz=min_spikes_per_burst_per_sec,
        )
        if sw_samples > 0:
            sw_seconds = sw_samples / rec.sfreq
            stage_sw_seconds[stage] += sw_seconds
            if sw_samples >= min_burst_samples:
                epochs_with_continuous_sw += 1

    # Compute SWI per stage = total SW seconds / total stage seconds × 100
    swi_per_stage: dict[str, float] = {}
    minutes_per_stage: dict[str, float] = {}
    for stage in stage_epoch_count:
        total_s = stage_epoch_count[stage] * epoch_seconds
        minutes_per_stage[stage] = total_s / 60
        if total_s > 0:
            swi_per_stage[stage] = 100 * stage_sw_seconds[stage] / total_s
        else:
            swi_per_stage[stage] = 0.0

    # NREM-combined SWI (weighted by stage duration)
    nrem_total_s = (stage_epoch_count["N1"] + stage_epoch_count["N2"]
                    + stage_epoch_count["N3"]) * epoch_seconds
    nrem_sw_s = (stage_sw_seconds["N1"] + stage_sw_seconds["N2"]
                 + stage_sw_seconds["N3"])
    swi_nrem = 100 * nrem_sw_s / nrem_total_s if nrem_total_s > 0 else 0.0
    swi_n3 = swi_per_stage["N3"]
    csws_met = swi_n3 >= csws_threshold_pct

    return SWIResult(
        channel=target_channel,
        swi_per_stage={k: float(v) for k, v in swi_per_stage.items()},
        swi_nrem_combined=float(swi_nrem),
        swi_n3_only=float(swi_n3),
        csws_criterion_met=bool(csws_met),
        csws_threshold_pct=csws_threshold_pct,
        minutes_analyzed_per_stage=minutes_per_stage,
        n_epochs_with_continuous_sw=epochs_with_continuous_sw,
        epoch_seconds=epoch_seconds,
    )


def _continuous_sw_coverage(
    peaks: np.ndarray,
    n_samples: int,
    sfreq: float,
    min_burst_samples: int,
    min_rate_hz: float,
) -> int:
    """Return total samples within the epoch that are part of a sustained
    spike-wave burst (≥ min_rate_hz spikes/s, sustained ≥ min_burst_samples).
    """
    if len(peaks) < 2:
        return 0
    max_interval_samples = int(sfreq / min_rate_hz)
    # Cluster peaks into runs where consecutive peaks are < max_interval
    cluster_start = peaks[0]
    last_peak = peaks[0]
    coverage = 0
    for p in peaks[1:]:
        if p - last_peak <= max_interval_samples:
            last_peak = p
        else:
            run_len = last_peak - cluster_start
            if run_len >= min_burst_samples:
                coverage += run_len
            cluster_start = p
            last_peak = p
    # final cluster
    run_len = last_peak - cluster_start
    if run_len >= min_burst_samples:
        coverage += run_len
    return int(coverage)


def summarize_swi(result: SWIResult) -> dict:
    return {
        "channel": result.channel,
        "swi_per_stage_pct": {
            k: round(v, 1) for k, v in result.swi_per_stage.items()
        },
        "swi_nrem_combined_pct": round(result.swi_nrem_combined, 1),
        "swi_n3_only_pct": round(result.swi_n3_only, 1),
        "csws_criterion_met": result.csws_criterion_met,
        "csws_threshold_pct": result.csws_threshold_pct,
        "minutes_per_stage": {
            k: round(v, 1) for k, v in result.minutes_analyzed_per_stage.items()
        },
        "n_epochs_with_continuous_sw": result.n_epochs_with_continuous_sw,
    }
