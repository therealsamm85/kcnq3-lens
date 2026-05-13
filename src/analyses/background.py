"""Background EEG power and posterior dominant rhythm analysis.

The posterior dominant rhythm (PDR) is the awake-state alpha rhythm seen over
occipital electrodes. Its peak frequency reflects cortical maturation:
- Age 4-5:   8-9 Hz
- Age 6-7:   9-10 Hz
- Adults:    9-11 Hz

A PDR below age-expected (e.g., 4-6 Hz at age 5) indicates background slowing,
which is a marker of cortical dysfunction or immaturity. In KCNQ3-spectrum
patients this slowing reflects the channelopathy's effect on overall
thalamocortical function — distinct from epileptiform spikes.

Delta/alpha ratio (DAR) is another sensitive marker: > 1 in alert wake is
abnormal slowing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from ..readers.base import EEGRecording


@dataclass
class BackgroundResult:
    channels_used: list[str]
    n_epochs: int
    delta_pct: float
    theta_pct: float
    alpha_pct: float
    beta_pct: float
    delta_alpha_ratio: float
    posterior_dominant_rhythm_hz: float
    age_normative_pdr: tuple[float, float] | None
    interpretation: str  # "severely_slow", "mildly_slow", "age_appropriate"


_PDR_AGE_NORMS = {
    3: (7.0, 8.0),
    4: (7.5, 8.5),
    5: (8.0, 9.0),
    6: (8.5, 9.5),
    7: (9.0, 10.0),
    8: (9.0, 10.5),
    10: (9.5, 11.0),
    15: (9.5, 11.5),
    18: (9.5, 11.5),
}


def _pdr_normative(age: float | None) -> tuple[float, float] | None:
    if age is None:
        return None
    ages = sorted(_PDR_AGE_NORMS.keys())
    closest = min(ages, key=lambda a: abs(a - age))
    return _PDR_AGE_NORMS[closest]


def compute_background_power(
    rec: EEGRecording,
    wake_epoch_indices: list[int] | None = None,
    epoch_seconds: float = 30.0,
    posterior_channels: list[str] = ("O1", "O2", "P3", "P4", "Pz"),
    age_years: float | None = None,
    delta_artifact_threshold: float | None = None,
) -> BackgroundResult:
    """Quantify background EEG power and posterior dominant rhythm.

    Parameters
    ----------
    rec : EEGRecording
    wake_epoch_indices : list[int], optional
        Specific epochs to analyze. If None, uses first and last 10% of recording.
    epoch_seconds : float
    posterior_channels : iterable of str
        Channel names averaged for the PDR estimate.
    age_years : float, optional
    delta_artifact_threshold : float, optional
        If provided, skips epochs where delta RMS exceeds this (motion artifact).
    """
    sfreq = rec.sfreq
    sos_hp = butter(4, 0.5, btype="high", fs=sfreq, output="sos")
    sos_lp = butter(4, 40.0, btype="low", fs=sfreq, output="sos")

    # Resolve posterior channel indices that actually exist in this recording
    pc_idx = []
    pc_names = []
    for ch in posterior_channels:
        i = rec.channel_index(ch)
        if i is not None:
            pc_idx.append(i)
            pc_names.append(ch)
    if not pc_idx:
        raise ValueError("No posterior channels found in recording.")

    # Default epoch selection: first 10% + last 10% of recording (skip middle = sleep window)
    if wake_epoch_indices is None:
        n_ep = rec.n_epochs
        first = list(range(int(n_ep * 0.05), int(n_ep * 0.15)))
        last = list(range(int(n_ep * 0.85), int(n_ep * 0.95)))
        wake_epoch_indices = first + last

    delta_pow, theta_pow, alpha_pow, beta_pow = [], [], [], []
    posterior_traces = []

    for ep in wake_epoch_indices:
        d = rec.read_epoch(ep, epoch_seconds)
        if d is None:
            continue
        # Average across posterior channels (still one trace per epoch)
        post = d[pc_idx].mean(axis=0)
        post = sosfiltfilt(sos_hp, post)
        post = sosfiltfilt(sos_lp, post)

        # Optional artifact rejection on delta amplitude
        if delta_artifact_threshold is not None:
            delta_rms = float(np.sqrt(np.mean(post ** 2)))
            if delta_rms > delta_artifact_threshold:
                continue

        f, P = welch(post, fs=sfreq, nperseg=int(sfreq * 4))
        delta_pow.append(P[(f >= 1) & (f < 4)].sum())
        theta_pow.append(P[(f >= 4) & (f < 8)].sum())
        alpha_pow.append(P[(f >= 8) & (f < 13)].sum())
        beta_pow.append(P[(f >= 13) & (f < 30)].sum())
        posterior_traces.append(post)

    if not delta_pow:
        raise ValueError("No valid wake epochs found.")

    d_pow = np.array(delta_pow)
    t_pow = np.array(theta_pow)
    a_pow = np.array(alpha_pow)
    b_pow = np.array(beta_pow)
    total = d_pow + t_pow + a_pow + b_pow

    delta_pct = 100 * d_pow.mean() / total.mean()
    theta_pct = 100 * t_pow.mean() / total.mean()
    alpha_pct = 100 * a_pow.mean() / total.mean()
    beta_pct = 100 * b_pow.mean() / total.mean()
    dar = float(d_pow.mean() / a_pow.mean()) if a_pow.mean() > 0 else float("inf")

    # PDR estimate from concatenated posterior trace
    cat = np.concatenate(posterior_traces)
    f, P = welch(cat, fs=sfreq, nperseg=int(sfreq * 8))
    pdr_band = (f >= 4) & (f <= 13)
    pdr_hz = float(f[pdr_band][np.argmax(P[pdr_band])])

    norm = _pdr_normative(age_years)
    if norm is None:
        interp = "no_age_provided"
    elif pdr_hz < norm[0] - 2:
        interp = "severely_slow"
    elif pdr_hz < norm[0]:
        interp = "mildly_slow"
    else:
        interp = "age_appropriate"

    return BackgroundResult(
        channels_used=pc_names,
        n_epochs=len(d_pow),
        delta_pct=float(delta_pct),
        theta_pct=float(theta_pct),
        alpha_pct=float(alpha_pct),
        beta_pct=float(beta_pct),
        delta_alpha_ratio=dar,
        posterior_dominant_rhythm_hz=pdr_hz,
        age_normative_pdr=norm,
        interpretation=interp,
    )


def summarize_background(result: BackgroundResult) -> dict:
    return {
        "channels_used": result.channels_used,
        "n_epochs": result.n_epochs,
        "delta_pct": round(result.delta_pct, 1),
        "theta_pct": round(result.theta_pct, 1),
        "alpha_pct": round(result.alpha_pct, 1),
        "beta_pct": round(result.beta_pct, 1),
        "delta_alpha_ratio": round(result.delta_alpha_ratio, 2),
        "posterior_dominant_rhythm_hz": round(result.posterior_dominant_rhythm_hz, 1),
        "age_normative_pdr": result.age_normative_pdr,
        "interpretation": result.interpretation,
    }
