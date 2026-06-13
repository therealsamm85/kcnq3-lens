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
    flag: str       # "good" | "flat" | "saturated" | "extreme" | "noisy" | "uncorrelated"
    # v0.18.6 (PREP-style relative detection):
    is_noisy_outlier: bool = False  # amplitude >> the across-channel median
    is_uncorrelated: bool = False   # low mean correlation with other channels
    mean_abs_corr: float = 1.0      # mean |Pearson r| vs all other EEG channels


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
    flat_threshold: float = 1.5,           # µV std — below this is "flat" (dead/unplugged)
    extreme_threshold_median_ptp: float = 1500.0,  # SUSTAINED median p-p (µV) = extreme
    saturation_threshold_p99: float = 3150.0,  # near NK ±3200 µV physical max = clipping
    artifact_power_multiplier: float = 5.0,
    noisy_outlier_multiplier: float = 4.0,  # amplitude > N× across-channel median = noisy
    uncorrelated_threshold: float = 0.40,   # MAX |r| with any other ch below this = uncorrelated
    corr_sample_epochs: int = 60,
) -> QualityResult:
    """Run QC checks on a recording.

    Returns a QualityResult with per-channel flags, per-epoch artifact count,
    and an overall A/B/C/D grade.

    v0.18.6: thresholds are now in µV (the reader path delivers µV after the
    0.18.4 calibration fix; the previous ADC-unit thresholds of 32700/50000
    never fired on µV data and silently disabled saturation/extreme
    detection). Adds PREP-style relative detection: a channel whose amplitude
    is N× the across-channel median is flagged "noisy", and a channel whose
    mean absolute correlation with the others is very low is flagged
    "uncorrelated" — this is what catches a noisy reference channel (e.g. the
    FA06301E Pz that sat at ~950 µV in wake while neighbours were ~50 µV) that
    absolute thresholds miss.
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

    # v0.18.6: accumulate an inter-channel correlation matrix on a sampled
    # subset of epochs (full 24 h would be wasteful). Each sampled epoch
    # contributes its channel×channel correlation; we average across samples.
    corr_accum = np.zeros((n_chs, n_chs))
    corr_count = 0
    _sample_stride = max(1, n_eps // max(1, corr_sample_epochs))

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

        if n_chs >= 3 and (i % _sample_stride == 0):
            # Correlate on band-passed data (1-40 Hz): raw channels carry
            # per-channel DC offset / slow drift that depresses correlation
            # even for good neighbours and would over-flag awake EEG.
            filt = sosfiltfilt(sos_broad, chans, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                cmat = np.corrcoef(filt)
            if np.all(np.isfinite(cmat)):
                corr_accum += np.abs(cmat)
                corr_count += 1

    # Correlation of each channel vs all others (exclude self-diagonal).
    # We flag on the MAX correlation with any other channel (PREP-style): a
    # good channel correlates highly with at least one neighbour, a dead or
    # junk-reference channel correlates with none. Mean correlation is kept
    # only for reporting — it is naturally low (~0.2-0.4) even for good EEG
    # with focal/referential activity, so it is a poor flag basis.
    if corr_count > 0 and n_chs >= 3:
        mean_corr_mat = corr_accum / corr_count
        np.fill_diagonal(mean_corr_mat, np.nan)
        ch_mean_corr = np.nanmean(mean_corr_mat, axis=1)
        ch_max_corr = np.nanmax(mean_corr_mat, axis=1)
    else:
        ch_mean_corr = np.ones(n_chs)
        ch_max_corr = np.ones(n_chs)

    # Across-channel median amplitude, for relative-outlier detection. Use the
    # median per-channel std (robust to occasional artifact epochs).
    per_ch_median_std = np.median(ch_stds, axis=1)
    cohort_median_std = float(np.median(per_ch_median_std)) or 1e-9

    # Per-channel classification
    qualities: list[ChannelQuality] = []
    for j, c in enumerate(eeg_idx):
        name = rec.channel_names[c]
        median_std = float(per_ch_median_std[j])
        median_ptp = float(np.median(ch_ptp_p99[j]))
        median_abs_max = float(np.median(ch_abs_max[j]))
        mean_corr = float(ch_mean_corr[j]) if np.isfinite(ch_mean_corr[j]) else 1.0
        max_corr = float(ch_max_corr[j]) if np.isfinite(ch_max_corr[j]) else 1.0

        is_flat = median_std < flat_threshold
        is_saturated = median_abs_max > saturation_threshold_p99
        # Extreme = SUSTAINED high amplitude (median p-p), not a single
        # transient (a p99 measure flagged clean channels on one artifact epoch).
        is_extreme = median_ptp > extreme_threshold_median_ptp and not is_saturated
        is_noisy_outlier = (
            not is_flat
            and median_std > noisy_outlier_multiplier * cohort_median_std
        )
        # Uncorrelated: best correlation with ANY other channel is low. A flat
        # channel is trivially uncorrelated, so only flag non-flat channels.
        is_uncorrelated = (
            not is_flat and n_chs >= 6 and max_corr < uncorrelated_threshold
        )

        # Priority: dead > clipping > extreme amplitude > noisy outlier >
        # uncorrelated > good. (Most actionable defect wins the label.)
        if is_flat:
            flag = "flat"
        elif is_saturated:
            flag = "saturated"
        elif is_extreme:
            flag = "extreme"
        elif is_noisy_outlier:
            flag = "noisy"
        elif is_uncorrelated:
            flag = "uncorrelated"
        else:
            flag = "good"

        qualities.append(ChannelQuality(
            name=name,
            is_flat=is_flat,
            is_saturated=is_saturated,
            is_extreme=is_extreme,
            median_amplitude=float(np.median(ch_ptp_p99[j])),
            flag=flag,
            is_noisy_outlier=is_noisy_outlier,
            is_uncorrelated=is_uncorrelated,
            mean_abs_corr=round(mean_corr, 3),
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
    # v0.18.6: surface noisy/uncorrelated channels explicitly — these are the
    # ones that absolute thresholds miss but that corrupt averaged metrics
    # (PDR posterior average, topography). Analyses should avoid them.
    _noisy = [q.name for q in qualities if q.flag == "noisy"]
    _uncorr = [q.name for q in qualities if q.flag == "uncorrelated"]
    if _noisy:
        warnings.append(
            f"Channel(s) {', '.join(_noisy)} have abnormally high amplitude "
            "vs the rest (possible reference/electrode problem) — exclude from "
            "averaged metrics."
        )
    if _uncorr:
        warnings.append(
            f"Channel(s) {', '.join(_uncorr)} are poorly correlated with the "
            "rest (likely bad contact or a reference channel) — interpret with "
            "caution."
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
             "median_amplitude": round(q.median_amplitude, 0),
             "mean_abs_corr": q.mean_abs_corr}
            for q in result.channel_qualities
        ],
        "bad_channels": [
            q.name for q in result.channel_qualities if q.flag != "good"
        ],
        "warnings": result.warnings,
    }
