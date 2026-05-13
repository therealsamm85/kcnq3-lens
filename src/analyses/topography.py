"""Per-channel spike topography via robust kurtosis.

Kurtosis measures the "peakedness" of a signal distribution. Epileptiform spikes
add sharp transients to the EEG, which inflate kurtosis far above the baseline
~3 of normal background activity. By computing kurtosis per channel per epoch,
we get a robust topographic map of where epileptiform activity is concentrated.

This metric is particularly useful for finding the spike focus in multi-regional
patterns where standard reading might say "diffuse" without quantifying which
regions dominate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis as _kurtosis

from ..readers.base import EEGRecording


@dataclass
class TopographyResult:
    channel_names: list[str]
    median_kurtosis: list[float]
    p90_kurtosis: list[float]
    fraction_high_kurtosis: list[float]  # fraction of epochs with kurtosis > 10
    top_channels: list[tuple[str, float]]  # (name, median) ranked descending
    epoch_seconds: float
    n_epochs: int


def compute_topography(
    rec: EEGRecording,
    start_epoch: int = 0,
    end_epoch: int | None = None,
    epoch_seconds: float = 30.0,
    bandpass: tuple[float, float] = (1.0, 35.0),
) -> TopographyResult:
    """Compute per-channel kurtosis topography.

    Parameters
    ----------
    rec : EEGRecording
    start_epoch, end_epoch : int
        Range of epochs to analyze. By default, the whole recording.
    epoch_seconds : float
        Epoch length (default 30 s).
    bandpass : (low, high)
        Filter passband to isolate spike-relevant activity.

    Returns
    -------
    TopographyResult with per-channel medians, p90, and "high-spike" fractions.
    """
    if end_epoch is None:
        end_epoch = rec.n_epochs

    sos = butter(4, list(bandpass), btype="band", fs=rec.sfreq, output="sos")
    eeg_idx = rec.eeg_channel_indices
    n_eps = end_epoch - start_epoch
    ch_kurt = np.zeros((len(eeg_idx), n_eps))

    for i, (ep, d) in enumerate(
        rec.iter_epochs(epoch_seconds=epoch_seconds, start=start_epoch, end=end_epoch)
    ):
        for j, c in enumerate(eeg_idx):
            x = sosfiltfilt(sos, d[c])
            ch_kurt[j, i] = _kurtosis(x, fisher=False)

    medians = np.median(ch_kurt, axis=1)
    p90s = np.percentile(ch_kurt, 90, axis=1)
    frac_high = np.mean(ch_kurt > 10, axis=1)

    names = [rec.channel_names[c] for c in eeg_idx]
    ranked = sorted(zip(names, medians), key=lambda x: -x[1])

    return TopographyResult(
        channel_names=names,
        median_kurtosis=medians.tolist(),
        p90_kurtosis=p90s.tolist(),
        fraction_high_kurtosis=frac_high.tolist(),
        top_channels=ranked,
        epoch_seconds=epoch_seconds,
        n_epochs=n_eps,
    )


def summarize_topography(result: TopographyResult, top_n: int = 5) -> dict:
    """Return a JSON-serializable summary suitable for reports and AI interpretation."""
    top = result.top_channels[:top_n]
    return {
        "epochs_analyzed": result.n_epochs,
        "top_channels": [{"name": n, "median_kurtosis": round(k, 2)} for n, k in top],
        "all_channels": [
            {
                "name": name,
                "median": round(med, 2),
                "p90": round(p90, 2),
                "pct_high_kurtosis": round(100 * frac, 1),
            }
            for name, med, p90, frac in zip(
                result.channel_names,
                result.median_kurtosis,
                result.p90_kurtosis,
                result.fraction_high_kurtosis,
            )
        ],
    }
