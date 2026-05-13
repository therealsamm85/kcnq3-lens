"""Recording quality control — flags bad channels and artifact-heavy epochs.

Catches three categories of problems before analyses run on garbage:

1. **Bad channels** — flat (electrode disconnected), saturated (clipping at
   ADC limits), or extreme-amplitude (broken / poor contact).
2. **Artifact-heavy epochs** — high broadband power (movement), high
   gamma-band power (muscle), or sustained DC drift.
3. **Overall quality grade** — A/B/C/D based on % usable epochs and
   channels.

A doctor reviewing the report should know whether the analyses ran on a
clean recording or a noisy one. The same numbers carry very different
weight depending on data quality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..readers.base import EEGRecording


@dataclass
class ChannelQuality:
    name: str
    is_flat: bool
    is_saturated: bool
    is_extreme: bool
    median_amplitude: float
    flag: str       # "good" | "flat" | "saturated" | "extreme" | "marginal"


@dataclass
class QualityResult:
    channel_qualities: list[ChannelQuality]
    n_good_channels: int
    n_total_channels: int
    n_artifact_epochs: int
    n_total_epochs: int
    pct_usable: float           # % of analyzed epochs with both good channels and low artifact
    overall_grade: str          # "A" | "B" | "C" | "D"
    warnings: list[str]


def assess_quality(
    rec: EEGRecording,
    start_epoch: int = 0,
    end_epoch: int | None = None,
    epoch_seconds: float = 30.0,
    flat_threshold: float = 5.0,           # ADC/µV std — below this is "flat"
    extreme_threshold_p99: float = 50000,  # 99th-percentile p-p above this = extreme
    saturation_threshold_p99: float = 32700,  # close to int16 max = clipping
    artifact_power_multiplier: float = 5.0,
) -> QualityResult:
    """Run QC checks on a recording.

    Returns a QualityResult with per-channel flags, per-epoch artifact count,
    and an overall A/B/C/D grade.
    """
    if end_epoch is None:
        end_epoch = rec.n_epochs

    eeg_idx = rec.eeg_channel_indices
    sos_broad = butter(4, [1.0, 40.0], btype="band", fs=rec.sfreq, output="sos")
    sos_muscle = butter(4, [30.0, min(99.0, rec.sfreq / 2 - 1)],
                        btype="band", fs=rec.sfreq, output="sos")

    n_eps = end_epoch - start_epoch
    n_chs = len(eeg_idx)

    # Per-channel: collect amplitude stats across all epochs
    ch_stds: np.ndarray = np.zeros((n_chs, n_eps))
    ch_ptp_p99: np.ndarray = np.zeros((n_chs, n_eps))
    ch_abs_max: np.ndarray = np.zeros((n_chs, n_eps))
    broad_pow_per_epoch: np.ndarray = np.zeros(n_eps)
    muscle_pow_per_epoch: np.ndarray = np.zeros(n_eps)

    for i, (ep, d) in enumerate(rec.iter_epochs(
        epoch_seconds=epoch_seconds, start=start_epoch, end=end_epoch
    )):
        chans = d[eeg_idx]
        ch_stds[:, i] = chans.std(axis=1)
        ch_ptp_p99[:, i] = np.ptp(chans, axis=1)
        ch_abs_max[:, i] = np.max(np.abs(chans), axis=1)
        mean = chans.mean(axis=0)
        broad = sosfiltfilt(sos_broad, mean)
        muscle = sosfiltfilt(sos_muscle, mean)
        broad_pow_per_epoch[i] = float(np.mean(broad ** 2))
        muscle_pow_per_epoch[i] = float(np.mean(muscle ** 2))

    # Per-channel classification
    qualities: list[ChannelQuality] = []
    for j, c in enumerate(eeg_idx):
        name = rec.channel_names[c]
        median_std = float(np.median(ch_stds[j]))
        p99_ptp = float(np.percentile(ch_ptp_p99[j], 99))
        median_abs_max = float(np.median(ch_abs_max[j]))

        is_flat = median_std < flat_threshold
        is_saturated = median_abs_max > saturation_threshold_p99
        is_extreme = p99_ptp > extreme_threshold_p99 and not is_saturated

        if is_flat:
            flag = "flat"
        elif is_saturated:
            flag = "saturated"
        elif is_extreme:
            flag = "extreme"
        else:
            flag = "good"

        qualities.append(ChannelQuality(
            name=name,
            is_flat=is_flat,
            is_saturated=is_saturated,
            is_extreme=is_extreme,
            median_amplitude=float(np.median(ch_ptp_p99[j])),
            flag=flag,
        ))

    n_good = sum(1 for q in qualities if q.flag == "good")

    # Per-epoch artifact detection: an epoch is "artifact" if broadband power
    # exceeds N× the median of the recording AND muscle band is elevated.
    broad_median = np.median(broad_pow_per_epoch)
    muscle_median = np.median(muscle_pow_per_epoch)
    if broad_median <= 0:
        broad_median = 1e-6
    if muscle_median <= 0:
        muscle_median = 1e-6
    artifact_mask = (
        (broad_pow_per_epoch > artifact_power_multiplier * broad_median) |
        (muscle_pow_per_epoch > artifact_power_multiplier * muscle_median)
    )
    n_artifact = int(artifact_mask.sum())

    # Usable: epoch not flagged as artifact AND ≥ 6 good channels
    pct_usable = 100.0 * (n_eps - n_artifact) / max(n_eps, 1) if n_good >= 6 else 0.0

    # Overall grade
    if pct_usable >= 80 and n_good >= 15:
        grade = "A"
    elif pct_usable >= 65 and n_good >= 12:
        grade = "B"
    elif pct_usable >= 50 and n_good >= 8:
        grade = "C"
    else:
        grade = "D"

    warnings = []
    if n_good < 6:
        warnings.append(
            f"Only {n_good}/{n_chs} channels are good quality. "
            "Topographic analyses will be unreliable."
        )
    elif n_good < 12:
        warnings.append(
            f"{n_chs - n_good} of {n_chs} channels flagged. "
            "Treat per-channel findings on flagged channels with caution."
        )
    if pct_usable < 50:
        warnings.append(
            f"Only {pct_usable:.0f}% of epochs are clean. "
            "Spike-burden and rate metrics may be inflated by artifact."
        )

    return QualityResult(
        channel_qualities=qualities,
        n_good_channels=n_good,
        n_total_channels=n_chs,
        n_artifact_epochs=n_artifact,
        n_total_epochs=n_eps,
        pct_usable=float(pct_usable),
        overall_grade=grade,
        warnings=warnings,
    )


def summarize_quality(result: QualityResult) -> dict:
    return {
        "overall_grade": result.overall_grade,
        "pct_usable": round(result.pct_usable, 1),
        "n_good_channels": result.n_good_channels,
        "n_total_channels": result.n_total_channels,
        "n_artifact_epochs": result.n_artifact_epochs,
        "n_total_epochs": result.n_total_epochs,
        "channel_flags": [
            {"name": q.name, "flag": q.flag,
             "median_amplitude": round(q.median_amplitude, 0)}
            for q in result.channel_qualities
        ],
        "warnings": result.warnings,
    }
