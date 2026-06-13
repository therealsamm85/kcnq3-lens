"""Autoreject-inspired per-channel epoch rejection.

Why this exists
---------------
quality.py grades an epoch artifact-y using a SINGLE global power threshold
(broadband power > 5× the recording median). That misses the common case
where one channel has a big transient while the rest are clean — the epoch is
fine for most channels but garbage for one.

This module estimates a rejection threshold PER CHANNEL from that channel's
own peak-to-peak distribution (median + k·MAD, robust to the very artifacts
we are trying to find), then builds a per-epoch × per-channel bad-mask. An
epoch is rejected only when a meaningful fraction of channels are bad in it,
so a single noisy channel doesn't throw away otherwise-clean epochs.

This is the spirit of Jas et al.'s Autoreject (data-driven, channel-specific
thresholds) without the cross-validation machinery or the MNE Epochs
dependency, so it runs on the lazy EEGRecording epoch interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..readers.base import EEGRecording


@dataclass
class RejectionResult:
    n_epochs: int
    n_channels: int
    channel_names: list[str]
    per_channel_threshold_uv: dict[str, float]
    n_rejected_epochs: int
    pct_rejected_epochs: float
    clean_epoch_indices: list[int]
    rejected_epoch_indices: list[int] = field(default_factory=list)
    # mask[i, j] True == channel j is bad in epoch i (row order = epochs scanned)
    _epoch_order: list[int] = field(default_factory=list, repr=False)


def compute_rejection(
    rec: EEGRecording,
    start_epoch: int = 0,
    end_epoch: int | None = None,
    epoch_seconds: float = 30.0,
    mad_k: float = 5.0,                 # threshold = median + k·MAD per channel
    min_abs_threshold_uv: float = 150.0,  # never reject below this (real EEG varies)
    bad_channel_fraction: float = 0.25,   # epoch rejected if > this frac of ch bad
) -> RejectionResult:
    """Estimate per-channel rejection thresholds and flag bad epochs.

    One pass collects each channel's per-epoch peak-to-peak; thresholds are
    then median + mad_k·MAD per channel (floored at min_abs_threshold_uv so a
    very quiet channel doesn't get an implausibly tight threshold). An epoch is
    rejected when more than bad_channel_fraction of channels exceed their own
    threshold.
    """
    if end_epoch is None:
        end_epoch = rec.n_epochs

    eeg_idx = rec.eeg_channel_indices
    names = [rec.channel_names[i] for i in eeg_idx]
    n_chs = len(eeg_idx)

    # Pass 1: per-epoch peak-to-peak per channel.
    ptp_rows: list[np.ndarray] = []
    epoch_order: list[int] = []
    for ep, d in rec.iter_epochs(
        epoch_seconds=epoch_seconds, start=start_epoch, end=end_epoch
    ):
        chans = d[eeg_idx]
        ptp_rows.append(np.ptp(chans, axis=1).astype(np.float64))
        epoch_order.append(ep)

    if not ptp_rows:
        return RejectionResult(
            n_epochs=0, n_channels=n_chs, channel_names=names,
            per_channel_threshold_uv={}, n_rejected_epochs=0,
            pct_rejected_epochs=0.0, clean_epoch_indices=[],
        )

    ptp = np.vstack(ptp_rows)  # shape (n_eps, n_chs)
    n_eps = ptp.shape[0]

    # Robust per-channel threshold: median + k·MAD (MAD scaled to σ-units).
    med = np.median(ptp, axis=0)
    mad = np.median(np.abs(ptp - med), axis=0) * 1.4826  # → ~σ for normal data
    thresh = med + mad_k * mad
    thresh = np.maximum(thresh, min_abs_threshold_uv)

    bad_mask = ptp > thresh[np.newaxis, :]            # (n_eps, n_chs)
    frac_bad = bad_mask.mean(axis=1)                  # per-epoch fraction bad
    rejected = frac_bad > bad_channel_fraction

    rejected_eps = [epoch_order[i] for i in range(n_eps) if rejected[i]]
    clean_eps = [epoch_order[i] for i in range(n_eps) if not rejected[i]]

    return RejectionResult(
        n_epochs=n_eps,
        n_channels=n_chs,
        channel_names=names,
        per_channel_threshold_uv={
            names[j]: round(float(thresh[j]), 1) for j in range(n_chs)
        },
        n_rejected_epochs=int(rejected.sum()),
        pct_rejected_epochs=round(100.0 * float(rejected.sum()) / n_eps, 1),
        clean_epoch_indices=clean_eps,
        rejected_epoch_indices=rejected_eps,
        _epoch_order=epoch_order,
    )


def summarize_rejection(result: RejectionResult) -> dict:
    return {
        "n_epochs": result.n_epochs,
        "n_rejected_epochs": result.n_rejected_epochs,
        "pct_rejected_epochs": result.pct_rejected_epochs,
        "per_channel_threshold_uv": result.per_channel_threshold_uv,
        # clean/rejected index lists omitted from summary (can be long).
    }
