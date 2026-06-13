"""Functional connectivity — debiased weighted Phase-Lag Index (wPLI).

Why this exists
---------------
The unified picture in this case is a thalamocortical dysrhythmia (severely
slow, near-absent alpha, near-absent spindles). Connectivity quantifies how
coordinated the cortex is, band by band. We use the debiased wPLI (Vinck et
al. 2011) rather than coherence because wPLI discounts zero-lag interactions
and is therefore robust to volume conduction / a shared reference — the
biggest confound for scalp EEG connectivity.

wPLI is computed from the imaginary part of the cross-spectrum, accumulated
over epochs, so it runs on the lazy EEGRecording interface and on multi-hour
recordings.

This is descriptive. No normative pediatric wPLI cohort exists, so the module
reports values without a "normal/abnormal" call — comparison is intra-patient
(over time, or before/after an intervention) only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..readers.base import EEGRecording

_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


@dataclass
class ConnectivityResult:
    channels: list[str]
    bands: list[str]
    n_epochs_used: int
    # mean wPLI across all channel pairs, per band
    mean_wpli_by_band: dict[str, float]
    # full pairwise matrix per band (list-of-lists, symmetric, diag 0)
    matrices_by_band: dict[str, list[list[float]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def compute_connectivity(
    rec: EEGRecording,
    start_epoch: int = 0,
    end_epoch: int | None = None,
    epoch_seconds: float = 30.0,
    bands: dict[str, tuple[float, float]] | None = None,
    max_epochs: int = 400,
) -> ConnectivityResult:
    """Compute debiased wPLI between all EEG-channel pairs, per band.

    Up to max_epochs epochs (evenly sampled) contribute to the estimate; for a
    24 h recording that is plenty and keeps it fast.
    """
    if end_epoch is None:
        end_epoch = rec.n_epochs
    bands = bands or _BANDS

    eeg_idx = rec.eeg_channel_indices
    names = [rec.channel_names[i] for i in eeg_idx]
    n_ch = len(eeg_idx)
    sf = rec.sfreq

    if n_ch < 2:
        return ConnectivityResult(
            channels=names, bands=list(bands), n_epochs_used=0,
            mean_wpli_by_band={b: 0.0 for b in bands},
            notes=["fewer than 2 EEG channels — connectivity unavailable"],
        )

    total = end_epoch - start_epoch
    stride = max(1, total // max_epochs)

    # Per-band accumulators of the imaginary cross-spectrum across epochs.
    # wPLI_debiased = (|sum Im|^2 - sum Im^2) / (sum|Im|^2 - sum Im^2)
    sum_im: dict[str, np.ndarray] = {b: np.zeros((n_ch, n_ch)) for b in bands}
    sum_abs_im: dict[str, np.ndarray] = {b: np.zeros((n_ch, n_ch)) for b in bands}
    sum_im_sq: dict[str, np.ndarray] = {b: np.zeros((n_ch, n_ch)) for b in bands}
    n_used = 0

    # Precompute FFT frequency bins for the epoch length.
    nperseg = int(epoch_seconds * sf)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sf)
    band_bins = {
        b: np.where((freqs >= lo) & (freqs < hi))[0] for b, (lo, hi) in bands.items()
    }
    win = np.hanning(nperseg)

    for k, (ep, d) in enumerate(rec.iter_epochs(
        epoch_seconds=epoch_seconds, start=start_epoch, end=end_epoch
    )):
        if k % stride != 0:
            continue
        chans = d[eeg_idx]
        if chans.shape[1] < nperseg:
            continue
        x = chans[:, :nperseg] * win[np.newaxis, :]
        # FFT per channel → (n_ch, n_freq)
        X = np.fft.rfft(x, axis=1)
        n_used += 1
        for b, bins in band_bins.items():
            if bins.size == 0:
                continue
            Xb = X[:, bins]                       # (n_ch, n_bins)
            # Cross-spectrum per pair per bin: S_ij(f) = X_i(f)·conj(X_j(f)).
            # v0.18.18: accumulate each (epoch, frequency-bin) as a SEPARATE
            # observation of Im(S), instead of band-averaging Im per epoch
            # before accumulating. The earlier mean-over-bins collapsed the
            # debiased-wPLI observation count to n_epochs, leaving a large
            # positive noise floor on truly-uncorrelated signals (~0.07 at 5-20
            # epochs). Summing over bins as well pools n_epochs × n_bins
            # observations, so the debiasing (|ΣIm|²−ΣIm²)/(Σ|Im|²−ΣIm²)
            # converges to ~0 for uncorrelated input while still detecting
            # genuine phase-lagged coupling.
            imf = np.imag(np.einsum("if,jf->ijf", Xb, np.conj(Xb)))  # (n_ch,n_ch,n_bins)
            sum_im[b] += imf.sum(axis=2)
            sum_abs_im[b] += np.abs(imf).sum(axis=2)
            sum_im_sq[b] += (imf ** 2).sum(axis=2)

    mean_wpli: dict[str, float] = {}
    matrices: dict[str, list[list[float]]] = {}
    for b in bands:
        num = sum_im[b] ** 2 - sum_im_sq[b]
        den = sum_abs_im[b] ** 2 - sum_im_sq[b]
        with np.errstate(invalid="ignore", divide="ignore"):
            wpli = np.where(den > 0, num / den, 0.0)
        wpli = np.clip(wpli, 0.0, 1.0)
        np.fill_diagonal(wpli, 0.0)
        matrices[b] = np.round(wpli, 4).tolist()
        # mean over the upper triangle (unique pairs)
        iu = np.triu_indices(n_ch, k=1)
        mean_wpli[b] = round(float(np.mean(wpli[iu])), 4) if iu[0].size else 0.0

    notes = []
    if n_used < 5:
        notes.append(
            f"only {n_used} epochs contributed — wPLI estimate is unstable."
        )

    return ConnectivityResult(
        channels=names,
        bands=list(bands),
        n_epochs_used=n_used,
        mean_wpli_by_band=mean_wpli,
        matrices_by_band=matrices,
        notes=notes,
    )


def summarize_connectivity(result: ConnectivityResult) -> dict:
    return {
        "channels": result.channels,
        "n_epochs_used": result.n_epochs_used,
        "mean_wpli_by_band": result.mean_wpli_by_band,
        "notes": result.notes,
        # full matrices omitted from the compact summary (available on the
        # result object for callers that want the per-pair detail).
    }
