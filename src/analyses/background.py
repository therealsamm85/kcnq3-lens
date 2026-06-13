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

_DISCLAIMER = (
    "PDR norm interpretation ('age_appropriate'/'mildly_slow'/'severely_slow') "
    "uses _PDR_AGE_NORMS table values that are a TOOL CONVENTION based on "
    "common textbook ranges (Niedermeyer 2005, Hagne 1968). The lower bounds "
    "are conservative — tighter than some sources allow. The 'severely_slow' "
    "label fires at >2 Hz below the lower bound; use this only as a flag for "
    "discussion with the clinician, not as a diagnostic statement."
)

_DISCLAIMER_ZSCORE = (
    "PDR z-score uses age-normative center from _PDR_AGE_NORMS and assumes "
    "SD=1 Hz (TOOL CONVENTION, not a validated population parameter — no "
    "published pediatric population SD exists; the z-score is therefore a "
    "within-tool ranking, not a comparison to a normative cohort). "
    "Asymmetry index uses alpha-band power (8-13 Hz) on O1+P3 vs O2+P4. "
    "Threshold for 'marked_asymmetric' is |AI| > 0.20 (tool convention)."
)


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
    # v0.16.0: quantitative PDR z-score and posterior asymmetry
    pdr_z_score: float | None = None
    pdr_asymmetry_index: float | None = None   # (LH - RH) / (LH + RH), -1..1
    posterior_lh_power: float | None = None    # avg alpha power O1+P3
    posterior_rh_power: float | None = None    # avg alpha power O2+P4
    asymmetry_interpretation: str = "not_computed"  # symmetric/lh_dominant/rh_dominant/marked_asymmetric
    # v0.18.19: aperiodic(1/f)-corrected PDR. The raw posterior_dominant_rhythm_hz
    # is the argmax of RAW power over 4-13 Hz, which a steep 1/f background biases
    # toward the low (4 Hz) edge — so a "severely slow" PDR can be partly a
    # peak-picking artifact. This is the frequency of the largest spectral bump
    # ABOVE the fitted 1/f trend (None if no genuine peak rises above it).
    pdr_aperiodic_corrected_hz: float | None = None
    aperiodic_slope: float | None = None         # log-log 1/f slope (2-30 Hz)
    pdr_method_divergence_hz: float | None = None  # |raw − corrected|; large = caution


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


_LH_ASYMMETRY_CHANNELS = ("O1", "P3")
_RH_ASYMMETRY_CHANNELS = ("O2", "P4")
_PDR_ZSCORE_SD = 1.0  # TOOL CONVENTION: assumed SD in Hz. No validated pediatric
                      # population SD exists; the resulting z-score is a within-tool
                      # ranking metric, not a normative-cohort comparison.
_ASYMMETRY_MARKED_THRESHOLD = 0.20


def _compute_pdr_z_score(pdr_hz: float, norm: tuple[float, float] | None) -> float | None:
    """Z-score of PDR vs age-normative center. SD=1 Hz (TOOL CONVENTION)."""
    if norm is None:
        return None
    norm_center = (norm[0] + norm[1]) / 2.0
    return float((pdr_hz - norm_center) / _PDR_ZSCORE_SD)


def _compute_alpha_power(
    rec: EEGRecording,
    wake_epoch_indices: list[int],
    epoch_seconds: float,
    channels: tuple[str, ...],
) -> float | None:
    """Average alpha-band (8-13 Hz) power across named channels."""
    ch_indices = []
    for ch in channels:
        idx = rec.channel_index(ch)
        if idx is not None:
            ch_indices.append(idx)
    if not ch_indices:
        return None

    from scipy.signal import butter, sosfiltfilt
    sos_hp = butter(4, 0.5, btype="high", fs=rec.sfreq, output="sos")
    sos_lp = butter(4, 40.0, btype="low", fs=rec.sfreq, output="sos")

    alpha_pows = []
    for ep in wake_epoch_indices:
        d = rec.read_epoch(ep, epoch_seconds)
        if d is None:
            continue
        ch_pows = []
        for ch_idx in ch_indices:
            if ch_idx >= d.shape[0]:
                continue
            sig = d[ch_idx].astype(float)
            sig = sosfiltfilt(sos_hp, sig)
            sig = sosfiltfilt(sos_lp, sig)
            f, P = welch(sig, fs=rec.sfreq, nperseg=int(rec.sfreq * 4))
            ch_pows.append(P[(f >= 8) & (f < 13)].sum())
        if ch_pows:
            alpha_pows.append(float(np.mean(ch_pows)))
    if not alpha_pows:
        return None
    return float(np.mean(alpha_pows))


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

    Notes
    -----
    Delta band: [0.5, 4.0) Hz — AASM convention (Niedermeyer 2005). The 0.5 Hz
    lower bound harmonises with sleep_stages.py fallback staging. Previously
    this module used [1, 4) Hz, which differed from the sleep module.
    """
    sfreq = rec.sfreq
    sos_hp = butter(4, 0.5, btype="high", fs=sfreq, output="sos")
    sos_lp = butter(4, 40.0, btype="low", fs=sfreq, output="sos")

    # Resolve posterior channel indices that actually exist in this recording.
    # v0.18.5: exclude present-but-dead channels (flat/unplugged) from the
    # posterior average — a 0 µV channel only dilutes the averaged trace the
    # PDR is read from. (If every posterior channel looks flat, keep them all
    # rather than fail — the recording may be genuinely low-amplitude.)
    pc_idx = []
    pc_names = []
    dead_idx = []
    dead_names = []
    for ch in posterior_channels:
        i = rec.channel_index(ch)
        if i is None:
            continue
        if not hasattr(rec, "is_channel_live") or rec.is_channel_live(i):
            pc_idx.append(i)
            pc_names.append(ch)
        else:
            dead_idx.append(i)
            dead_names.append(ch)
    if not pc_idx and dead_idx:
        pc_idx, pc_names = dead_idx, dead_names  # all flat — fall back to all
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
        # D1: delta = [0.5, 4.0) Hz — harmonised with sleep_stages.py (AASM/Niedermeyer).
        # Previously [1, 4) Hz which disagreed with the fallback staging band.
        delta_pow.append(P[(f >= 0.5) & (f < 4)].sum())
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

    # v0.18.19: aperiodic(1/f)-corrected PDR. Fit the 1/f trend (log-log linear
    # over 2-30 Hz), whiten the 4-13 Hz band by it, and take the largest bump
    # above the trend — but only if it is a genuine peak (≥1.5× the whitened
    # median), else report None (no rhythm rises above the 1/f background).
    pdr_corrected: float | None = None
    aperiodic_slope: float | None = None
    pdr_divergence: float | None = None
    try:
        fit_mask = (f >= 2) & (f <= 30) & (P > 0)
        if fit_mask.sum() >= 5:
            coef = np.polyfit(np.log(f[fit_mask]), np.log(P[fit_mask]), 1)
            aperiodic_slope = float(round(coef[0], 2))
            fa = f[pdr_band]
            trend = np.exp(np.polyval(coef, np.log(fa)))
            whit = P[pdr_band] / trend
            if whit.size and np.isfinite(whit).all():
                med = float(np.median(whit))
                if med > 0 and whit.max() >= 1.5 * med:
                    pdr_corrected = float(fa[np.argmax(whit)])
                    pdr_divergence = round(abs(pdr_hz - pdr_corrected), 1)
    except Exception:
        pass

    norm = _pdr_normative(age_years)
    if norm is None:
        interp = "no_age_provided"
    elif pdr_hz < norm[0] - 2:
        interp = "severely_slow"
    elif pdr_hz < norm[0]:
        interp = "mildly_slow"
    else:
        interp = "age_appropriate"

    # --- v0.16.0: PDR z-score and posterior asymmetry ---
    pdr_z = _compute_pdr_z_score(pdr_hz, norm)

    lh_power = _compute_alpha_power(
        rec, wake_epoch_indices, epoch_seconds, _LH_ASYMMETRY_CHANNELS
    )
    rh_power = _compute_alpha_power(
        rec, wake_epoch_indices, epoch_seconds, _RH_ASYMMETRY_CHANNELS
    )
    if lh_power is not None and rh_power is not None and (lh_power + rh_power) > 0:
        ai = (lh_power - rh_power) / (lh_power + rh_power)
        ai = float(ai)
    else:
        ai = None

    if ai is None:
        asym_interp = "not_computed"
    elif abs(ai) > _ASYMMETRY_MARKED_THRESHOLD:
        asym_interp = "marked_asymmetric"
    elif ai > 0.05:
        asym_interp = "lh_dominant"
    elif ai < -0.05:
        asym_interp = "rh_dominant"
    else:
        asym_interp = "symmetric"

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
        pdr_z_score=pdr_z,
        pdr_asymmetry_index=ai,
        posterior_lh_power=lh_power,
        posterior_rh_power=rh_power,
        asymmetry_interpretation=asym_interp,
        pdr_aperiodic_corrected_hz=pdr_corrected,
        aperiodic_slope=aperiodic_slope,
        pdr_method_divergence_hz=pdr_divergence,
    )


def summarize_background(result: BackgroundResult) -> dict:
    out: dict = {
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
        "disclaimer": _DISCLAIMER,
    }
    # v0.16.0 fields
    if result.pdr_z_score is not None:
        out["pdr_z_score"] = round(result.pdr_z_score, 2)
    if result.pdr_asymmetry_index is not None:
        out["pdr_asymmetry_index"] = round(result.pdr_asymmetry_index, 3)
    if result.posterior_lh_power is not None:
        out["posterior_lh_power"] = result.posterior_lh_power
    if result.posterior_rh_power is not None:
        out["posterior_rh_power"] = result.posterior_rh_power
    out["asymmetry_interpretation"] = result.asymmetry_interpretation
    if result.pdr_z_score is not None:
        out["disclaimer_zscore"] = _DISCLAIMER_ZSCORE
    # v0.18.19: aperiodic-corrected PDR + divergence caveat.
    if result.aperiodic_slope is not None:
        out["aperiodic_slope"] = result.aperiodic_slope
    if result.pdr_aperiodic_corrected_hz is not None:
        out["pdr_aperiodic_corrected_hz"] = round(
            result.pdr_aperiodic_corrected_hz, 1)
        out["pdr_method_divergence_hz"] = result.pdr_method_divergence_hz
        if (result.pdr_method_divergence_hz or 0) >= 1.0:
            out["pdr_caveat"] = (
                "The raw PDR is the argmax of raw posterior power, which a steep "
                f"1/f background biases low. A 1/f-corrected peak sits at "
                f"{result.pdr_aperiodic_corrected_hz:.1f} Hz "
                f"({result.pdr_method_divergence_hz:.1f} Hz higher) — so the "
                "'slow' raw PDR may partly be a peak-picking artifact. A human "
                "EEG reader should confirm the true posterior dominant rhythm."
            )
    else:
        out["pdr_aperiodic_corrected_hz"] = None
        out["pdr_caveat"] = (
            "No posterior rhythm rises clearly above the 1/f background in the "
            "4–13 Hz band — consistent with a genuinely attenuated/absent PDR "
            "(not merely a slow one). Human confirmation recommended."
        )
    return out
