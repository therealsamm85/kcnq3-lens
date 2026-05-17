"""Aperiodic EEG exponent (1/f slope) analysis.

The aperiodic (1/f) component of the EEG power spectrum reflects the balance
between excitation and inhibition (E/I balance) in the underlying neural
circuitry. A steeper negative slope (higher exponent χ) indicates greater
inhibitory tone; a flatter slope (lower χ) reflects relative hyperexcitability.

Scientific basis
----------------
- Donoghue T et al. (2020) Parameterizing neural power spectra into periodic
  and aperiodic components. Nature Neuroscience 23:1655–1665. PMID 33230329.
  (fooof / specparam method)
- Gao R et al. (2017) Inferring synaptic excitation/inhibition balance from
  field potentials. NeuroImage 158:70–78. doi:10.1016/j.neuroimage.2017.06.078
- Donoghue T et al. (2024) Aperiodic Activity Indexes Neural Hyperexcitability.
  eNeuro 2024. https://doi.org/10.1523/ENEURO.0175-24.2024
- Panzeri S et al. (2024) Aperiodic neural activity as a biomarker for epilepsy.
  Brain Communications 6:fcae231. https://doi.org/10.1093/braincomms/fcae231

KCNQ3-GoF hypothesis: gain-of-function KCNQ3 variants increase M-current,
reducing neuronal excitability → expected higher χ (steeper slope) compared
to typical controls. This is the OPPOSITE direction of classical epilepsy
hyperexcitability (which shows flatter slopes), consistent with the
channelopathy's complex gain-of-function phenotype.

Pediatric reference values (TOOL CONVENTION)
--------------------------------------------
The values below are adapted from Donoghue 2020 (adult) and the limited
pediatric literature (Cellier D et al. 2021, Donoghue T 2020 supplementary).
Pediatric norms are insufficiently established — treat these as ROUGH GUIDES
only. Pre/post within-patient comparison is far more informative than absolute
interpretation against these normative values.

  Wake:  mean=1.25, SD=0.30  (range 1.0–1.5 in controls)
  N2:    mean=1.75, SD=0.30  (range 1.5–2.0 in controls)
  N3:    mean=2.00, SD=0.30  (range 1.5–2.5 in controls)

Reference: Donoghue 2020 PMID 33230329 (adult); Cellier 2021
(doi:10.1016/j.neuroimage.2021.118141) for pediatric developmental trends.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import welch, find_peaks

from ..readers.base import EEGRecording

# ─── Pediatric reference values (TOOL CONVENTION) ────────────────────────────
# Source: Donoghue 2020 (adult) + limited pediatric data. SD assumed = 0.30
# for all states. Document this assumption prominently.
_PEDIATRIC_NORMS: dict[str, tuple[float, float]] = {
    # (mean, sd)
    "wake": (1.25, 0.30),
    "n2":   (1.75, 0.30),
    "n3":   (2.00, 0.30),
}

_DISCLAIMER = (
    "DISCLAIMER: Pediatric aperiodic exponent reference values are based on "
    "ADULT literature (Donoghue 2020, PMID 33230329) and limited developmental "
    "data. The SD=0.30 assumption is a TOOL CONVENTION, not a validated "
    "pediatric norm. Pre/post within-patient comparison is more informative "
    "than interpretation against absolute normative z-scores. "
    "Reference: Donoghue 2020 Nature Neuroscience 23:1655-1665; "
    "eNeuro 2024 aperiodic hyperexcitability paper; "
    "Brain Communications 2024 fcae231."
)

# Fit range and peak exclusion zone (in Hz)
_FIT_RANGE_HZ = (2.0, 30.0)
_PEAK_EXCLUSION_HZ = (3.5, 14.0)  # pediatric theta + alpha range

# Outlier filter bounds for χ
_CHI_MIN = 0.1
_CHI_MAX = 5.0

# Stage labels (internal)
_STAGE_LABELS = {"wake", "n1", "n2", "n3", "rem"}


@dataclass
class AperiodicResult:
    """Per-channel, per-stage aperiodic exponent results.

    Fields
    ------
    chi_by_channel : dict[str, dict[str, float]]
        channel_name -> {sleep_stage -> median_chi}
    chi_by_state_summary : dict[str, dict]
        sleep_stage -> {median, p25, p75, n_epochs, n_channels}
    pediatric_norm_z_scores : dict[str, float]
        sleep_stage -> z-score vs pediatric reference (TOOL CONVENTION)
    pediatric_norm_reference : dict[str, tuple[float, float]]
        sleep_stage -> (mean, sd) from literature
    fit_range_hz : tuple[float, float]
    excluded_peak_range_hz : tuple[float, float]
    method : str
        "specparam" if specparam library available, else "log_log_regression"
    notes : list[str]
    """

    chi_by_channel: dict[str, dict[str, float]]
    chi_by_state_summary: dict[str, dict]
    pediatric_norm_z_scores: dict[str, float]
    pediatric_norm_reference: dict[str, tuple[float, float]]
    fit_range_hz: tuple[float, float]
    excluded_peak_range_hz: tuple[float, float]
    method: str
    notes: list[str] = field(default_factory=list)


def _fit_aperiodic_loglog(
    freqs: np.ndarray,
    psd: np.ndarray,
    fit_range: tuple[float, float],
    peak_exclusion: tuple[float, float],
) -> float | None:
    """Fit log-log linear regression to PSD, excluding peak range.

    Returns the aperiodic exponent χ (positive = steeper = more inhibitory).
    Returns None if fit quality is too poor (too few points).
    """
    lo, hi = fit_range
    plo, phi = peak_exclusion

    # Select fit frequencies, excluding peak band
    mask = (freqs >= lo) & (freqs <= hi) & ~((freqs >= plo) & (freqs <= phi))
    f_fit = freqs[mask]
    p_fit = psd[mask]

    if len(f_fit) < 5:
        return None

    # Guard against zero/negative PSD (log undefined)
    if np.any(p_fit <= 0):
        p_fit = np.maximum(p_fit, 1e-30)

    log_f = np.log10(f_fit)
    log_p = np.log10(p_fit)

    # Linear regression: log(P) = b - χ * log(f)
    # Fit slope; negate to get positive χ for steeper (more negative) slopes
    coeffs = np.polyfit(log_f, log_p, 1)
    chi = -coeffs[0]  # negate because PSD falls with frequency
    return float(chi)


def _fit_aperiodic_specparam(
    freqs: np.ndarray,
    psd: np.ndarray,
    fit_range: tuple[float, float],
    peak_exclusion: tuple[float, float],
) -> tuple[float | None, str]:
    """Fit using specparam (formerly fooof) if available.

    Returns (chi, method_string).
    """
    try:
        from specparam import SpectralModel  # type: ignore
        fm = SpectralModel(
            aperiodic_mode="fixed",
            peak_width_limits=[0.5, 12.0],
            max_n_peaks=6,
            min_peak_height=0.05,
            verbose=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fm.fit(freqs, psd, freq_range=list(fit_range))
        if fm.has_model:
            chi = float(fm.aperiodic_params_[1])  # exponent
            return chi, "specparam"
    except Exception:
        pass
    # Fall back to log-log regression
    chi = _fit_aperiodic_loglog(freqs, psd, fit_range, peak_exclusion)
    return chi, "log_log_regression"


def compute_aperiodic_exponent(
    rec: EEGRecording,
    sleep_stages: object | None = None,
    epoch_seconds: float = 30.0,
    channels: list[str] | None = None,
    fit_range_hz: tuple[float, float] = _FIT_RANGE_HZ,
    peak_exclusion_hz: tuple[float, float] = _PEAK_EXCLUSION_HZ,
    method: str = "auto",
) -> AperiodicResult:
    """Compute aperiodic (1/f) exponent per channel per sleep stage.

    Parameters
    ----------
    rec : EEGRecording
    sleep_stages : SleepStageResult, optional
        If provided, epochs are classified by sleep stage. Otherwise all
        epochs are labelled "wake".
    epoch_seconds : float
        Epoch length (default 30 s = AASM standard epoch).
    channels : list[str], optional
        Subset of channel names to use. Default: all EEG channels up to 32.
    fit_range_hz : tuple[float, float]
        Frequency range for 1/f fit.
    peak_exclusion_hz : tuple[float, float]
        Sub-range excluded from fit to avoid oscillatory peaks biasing slope.
    method : str
        "auto" → try specparam, fall back to log_log_regression.
        "log_log_regression" → always use regression fallback.
        "specparam" → require specparam (raises ImportError if absent).

    Returns
    -------
    AperiodicResult
    """
    notes: list[str] = []

    # --- Resolve channels ---
    if channels is None:
        # Use EEG channels, cap at 32 for compute time
        use_indices = rec.eeg_channel_indices[:32]
        use_names = [rec.channel_names[i] for i in use_indices]
    else:
        use_indices = []
        use_names = []
        for ch in channels:
            idx = rec.channel_index(ch)
            if idx is not None:
                use_indices.append(idx)
                use_names.append(ch)
        if not use_indices:
            raise ValueError(f"None of the requested channels found: {channels}")

    # --- Determine method ---
    actual_method = "log_log_regression"
    if method in ("auto", "specparam"):
        try:
            import specparam  # noqa: F401
            actual_method = "specparam"
        except ImportError:
            if method == "specparam":
                raise ImportError(
                    "specparam library not available. Install with: pip install specparam"
                )
            notes.append("specparam not available; using log_log_regression fallback")
            actual_method = "log_log_regression"

    # --- Resolve sleep stage labels per epoch ---
    stage_map: dict[int, str] = {}  # epoch_idx -> stage label
    if sleep_stages is not None and hasattr(sleep_stages, "stage_per_epoch"):
        for ep_idx, label in enumerate(sleep_stages.stage_per_epoch):
            stage_map[ep_idx] = label.lower() if label else "unknown"
    else:
        # No staging available: treat all as wake
        for ep_idx in range(rec.n_epochs):
            stage_map[ep_idx] = "wake"
        if sleep_stages is None:
            notes.append("No sleep staging provided; all epochs labelled as 'wake'")

    # --- Compute PSD and fit per channel per epoch ---
    # Accumulate: {channel_name: {stage: [chi_values]}}
    chi_accum: dict[str, dict[str, list[float]]] = {
        ch: {s: [] for s in ("wake", "n1", "n2", "n3", "rem")}
        for ch in use_names
    }

    n_epochs = rec.n_epochs
    if n_epochs == 0:
        raise ValueError("Recording has no complete epochs.")

    nperseg = int(rec.sfreq * 4)  # 4-second Welch segments
    nfft = max(nperseg, 256)

    for ep_idx in range(n_epochs):
        data = rec.read_epoch(ep_idx, epoch_seconds)
        if data is None:
            continue

        stage = stage_map.get(ep_idx, "unknown")
        if stage not in ("wake", "n1", "n2", "n3", "rem"):
            continue  # skip unknown/undefined epochs

        for ch_name, ch_idx in zip(use_names, use_indices):
            if ch_idx >= data.shape[0]:
                continue
            sig = data[ch_idx].astype(float)

            # Skip flat/NaN/all-zero epochs
            if not np.isfinite(sig).all() or np.std(sig) < 1e-10:
                continue

            freqs, psd = welch(sig, fs=rec.sfreq, nperseg=nperseg, nfft=nfft)

            if actual_method == "specparam":
                chi, _ = _fit_aperiodic_specparam(
                    freqs, psd, fit_range_hz, peak_exclusion_hz
                )
            else:
                chi = _fit_aperiodic_loglog(freqs, psd, fit_range_hz, peak_exclusion_hz)

            if chi is None:
                continue
            # Outlier filter
            if not (_CHI_MIN <= chi <= _CHI_MAX):
                continue

            chi_accum[ch_name][stage].append(chi)

    # --- Summarize per channel ---
    chi_by_channel: dict[str, dict[str, float]] = {}
    for ch_name in use_names:
        ch_result: dict[str, float] = {}
        for stage in ("wake", "n1", "n2", "n3", "rem"):
            vals = chi_accum[ch_name][stage]
            if vals:
                ch_result[stage] = float(np.median(vals))
        if ch_result:
            chi_by_channel[ch_name] = ch_result

    # --- Cross-channel summary per state ---
    chi_by_state_summary: dict[str, dict] = {}
    for stage in ("wake", "n1", "n2", "n3", "rem"):
        all_vals = []
        n_ch = 0
        for ch_name in use_names:
            vals = chi_accum[ch_name][stage]
            if vals:
                all_vals.extend(vals)
                n_ch += 1
        if all_vals:
            arr = np.array(all_vals)
            chi_by_state_summary[stage] = {
                "median": float(np.median(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "n_epochs": len(all_vals),
                "n_channels": n_ch,
            }

    # --- Pediatric norm z-scores ---
    z_scores: dict[str, float] = {}
    for stage_key, (norm_mean, norm_sd) in _PEDIATRIC_NORMS.items():
        if stage_key in chi_by_state_summary:
            obs = chi_by_state_summary[stage_key]["median"]
            z_scores[stage_key] = float((obs - norm_mean) / norm_sd)

    notes.append(_DISCLAIMER)

    return AperiodicResult(
        chi_by_channel=chi_by_channel,
        chi_by_state_summary=chi_by_state_summary,
        pediatric_norm_z_scores=z_scores,
        pediatric_norm_reference={k: v for k, v in _PEDIATRIC_NORMS.items()},
        fit_range_hz=fit_range_hz,
        excluded_peak_range_hz=peak_exclusion_hz,
        method=actual_method,
        notes=notes,
    )


def summarize_aperiodic(result: AperiodicResult) -> dict:
    """Return a JSON-serializable summary dict."""
    return {
        "method": result.method,
        "fit_range_hz": list(result.fit_range_hz),
        "excluded_peak_range_hz": list(result.excluded_peak_range_hz),
        "chi_by_state": result.chi_by_state_summary,
        "pediatric_norm_z_scores": result.pediatric_norm_z_scores,
        "pediatric_norm_reference": {
            k: list(v) for k, v in result.pediatric_norm_reference.items()
        },
        "n_channels_with_data": len(result.chi_by_channel),
        "disclaimer": result.notes[-1] if result.notes else "",
    }
