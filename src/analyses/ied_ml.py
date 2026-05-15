"""Automated IED (interictal epileptiform discharge) detection — v0.13.3.

Three-mode architecture, with honest framing about what is and isn't ML:

1. ``external_spikenet`` — user supplies a SpikeNet-compatible PyTorch
   weights file. We lazy-import torch, hash the weights for provenance,
   and surface the non-commercial-research license. Inference itself is
   STUBBED — implementing real SpikeNet inference without the upstream
   repo's exact architecture/preprocessing would silently mismatch the
   model. We raise NotImplementedError with a clear pointer to
   https://github.com/bdsp-core/SpikeNet so the user knows what to
   provide. The runner catches the error and falls back to the heuristic.

2. ``ensemble_heuristic`` — production default. Rule-based ensemble over
   morphology events: (R1) epileptiform morphology, (R2) HF transient
   burst at the spike peak, (R3) focal topography. Per-event confidence
   is high/medium/low based on 3/2/1 rules passing. **NOT ML.**

3. ``unavailable`` — no morphology events AND no usable weights.

Pediatric drift
---------------
SpikeNet was trained on adult EEG (Jing et al. 2020, JAMA Neurol,
PMID 32049322). Reported sensitivity ~0.85 in adults but lower in
children, with higher false-positive rates on benign Rolandic
(centrotemporal) spikes. For age_years < 12 we flag drift_warning and
slightly relax confidence so we don't drop legitimate pediatric
morphologies that adult-trained features may underweight.

Centrotemporal/diphasic spikes are flagged ``likely_rolandic_benign``
but NOT dropped — the clinical signal is preserved; the flag is
informational.

WARNING — events array is internal only. It is never copied into a
registry submission (license-friction with SpikeNet probabilities;
pattern-leak risk regardless). See ``summarize_ied_ml``.

References
----------
Jing J et al. 2020  PMID 32049322  SpikeNet: ResNet IED classifier (adult)
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..readers.base import EEGRecording
from .sleep_stages import SleepStageResult


_DISCLAIMER_EXTERNAL = (
    "RESEARCH USE ONLY — SpikeNet was trained on adult EEG (Jing 2020). "
    "Sensitivity in pediatric recordings is lower; benign centrotemporal "
    "(Rolandic) spikes may be over-called. Non-commercial-research license. "
    "NOTE: external SpikeNet path is STUBBED — real inference requires "
    "user-supplied implementation."
)

_DISCLAIMER_ENSEMBLE = (
    "RESEARCH METRIC — Rule-based ensemble, NOT machine learning. "
    "Combines morphology, HF-burst, and topographic-focality heuristics. "
    "No validated pediatric normative thresholds for IED rate. "
    "Centrotemporal/diphasic spikes flagged as likely_rolandic_benign "
    "but not dropped."
)

_ALLOWED_METHODS = frozenset({
    "external_spikenet", "ensemble_heuristic", "unavailable",
})

# Channels considered "centrotemporal" for Rolandic-spike flagging.
# Covers 10-20 + 10-10 montages used in pediatric BCECTS recordings.
_CENTROTEMPORAL_CHANNELS = frozenset({
    # 10-20 standard
    "C3", "C4", "T3", "T4",
    # 10-10 equivalents
    "T7", "T8", "C5", "C6",
    # Adjacent involvement
    "CP3", "CP4", "CP5", "CP6",
    "FC3", "FC4",
    # Parietal nearby
    "P3", "P4",
})
# Keep backward-compat alias
_CENTROTEMPORAL_CHS = _CENTROTEMPORAL_CHANNELS

# Focal topography: <=4 channels involved at >=50% of primary-channel peak
# amplitude. Absolute-count convention chosen for clinical interpretability
# (focal vs regional vs generalized). Was 3 — bumped to 4 to handle
# adjacent-pair spread.
_FOCAL_MAX_CHANNELS = 4


# ─── Result dataclass ─────────────────────────────────────────────────────────


@dataclass
class IEDDetectionResult:
    method: str                          # see _ALLOWED_METHODS
    available: bool
    unavailable_reason: str
    model_version: str | None
    model_license: str | None
    n_ied_candidates: int
    rate_per_minute: float
    per_channel_rates: dict
    confidence_distribution: dict        # {"high": n, "medium": n, "low": n}
    age_appropriateness_flag: str        # "ok" | "drift_warning" | "untested"
    agreement_with_morphology_pct: float
    n_likely_rolandic_benign: int
    nrem_rate_per_min: float | None = None
    disclaimer: str = ""
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)  # internal only


# ─── Method selection ────────────────────────────────────────────────────────


def _select_method(
    weights_path: str | None,
    morphology_events: list[dict] | None,
) -> str:
    """Pick which detection mode to run.

    Note: even if morphology_events is an empty list, we return
    "ensemble_heuristic" — the run will simply produce zero candidates.
    "unavailable" is reserved for the case where there is no usable
    event source at all.
    """
    if weights_path is not None and Path(weights_path).exists():
        try:
            import torch  # noqa: F401 — lazy import probe
            return "external_spikenet"
        except ImportError:
            if morphology_events is not None:
                return "ensemble_heuristic"
            return "unavailable"
    if morphology_events is not None:
        return "ensemble_heuristic"
    return "unavailable"


# ─── Age handling ────────────────────────────────────────────────────────────


def _age_flag(age_years: float | int | None) -> str:
    if age_years is None:
        return "untested"
    try:
        a = float(age_years)
    except (TypeError, ValueError):
        return "untested"
    if not math.isfinite(a) or a < 0:
        return "untested"
    if a < 12.0:
        return "drift_warning"
    return "ok"


# ─── Signal extraction helpers ───────────────────────────────────────────────


def _pick_channel(rec: EEGRecording, preferred: str = "Cz") -> tuple[int | None, str]:
    """Find a usable EEG channel index + canonical name."""
    ch_idx = rec.channel_index(preferred)
    if ch_idx is not None:
        return ch_idx, rec.channel_names[ch_idx]
    for fb in ("Cz", "C3", "C4", "Fz", "Pz"):
        ch_idx = rec.channel_index(fb)
        if ch_idx is not None:
            return ch_idx, rec.channel_names[ch_idx]
    if getattr(rec, "eeg_channel_indices", None):
        ch_idx = rec.eeg_channel_indices[0]
        return ch_idx, rec.channel_names[ch_idx]
    return None, preferred


def _gather_signal(rec: EEGRecording, ch_idx: int) -> np.ndarray:
    """Concatenate channel data across 30-s epochs."""
    segments = []
    for _, d in rec.iter_epochs(epoch_seconds=30.0):
        segments.append(d[ch_idx])
    if not segments:
        return np.zeros(0, dtype=np.float64)
    sig = np.concatenate(segments).astype(np.float64)
    # NaN/Inf cleanup (same convention as coupling.py)
    if not np.all(np.isfinite(sig)):
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    return sig


def _gather_all_channels(rec: EEGRecording) -> tuple[np.ndarray, list[int], list[str]]:
    """Return (n_ch, n_samp) array, eeg-channel indices, and names."""
    eeg_idx = list(getattr(rec, "eeg_channel_indices", None) or [])
    if not eeg_idx:
        eeg_idx = list(range(len(rec.channel_names)))
    segments: list[np.ndarray] = []
    for _, d in rec.iter_epochs(epoch_seconds=30.0):
        segments.append(d[eeg_idx, :])
    if not segments:
        return np.zeros((len(eeg_idx), 0)), eeg_idx, [rec.channel_names[i] for i in eeg_idx]
    sig = np.concatenate(segments, axis=1).astype(np.float64)
    if not np.all(np.isfinite(sig)):
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    names = [rec.channel_names[i] for i in eeg_idx]
    return sig, eeg_idx, names


# ─── Ensemble rules ──────────────────────────────────────────────────────────


def _hf_burst_ratio(signal: np.ndarray, sfreq: float, peak_sample: int) -> float:
    """Compute fraction of power in 30–70 Hz vs 1–100 Hz, in a ±50 ms window
    around peak_sample. Returns 0.0 on degenerate input."""
    half = int(round(0.05 * sfreq))
    lo = max(0, peak_sample - half)
    hi = min(len(signal), peak_sample + half + 1)
    seg = signal[lo:hi]
    if len(seg) < 8:
        return 0.0
    # Remove DC
    seg = seg - float(np.mean(seg))
    if not np.any(seg):
        return 0.0
    fft = np.fft.rfft(seg)
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / sfreq)
    psd = np.abs(fft) ** 2
    hf_mask = (freqs >= 30.0) & (freqs <= 70.0)
    broad_mask = (freqs >= 1.0) & (freqs <= 100.0)
    total = float(np.sum(psd[broad_mask]))
    if total <= 0.0 or not math.isfinite(total):
        return 0.0
    hf = float(np.sum(psd[hf_mask]))
    ratio = hf / total
    if not math.isfinite(ratio):
        return 0.0
    return ratio


def _classify_morphology(
    signal: np.ndarray,
    sfreq: float,
    peak_sample: int,
) -> tuple[str, bool]:
    """Heuristically classify an event by its broadband shape.

    Returns (category, has_aftercoming_slow_wave).

    Categories:
      "complex_spike_wave" — slow wave (>200 ms FWHM) after sharp peak
      "simple_spike"       — <70 ms FWHM
      "sharp_wave"         — 70–200 ms FWHM
      "unclassified"       — degenerate
    """
    # Window ±400 ms around the peak (enough to see aftercoming slow wave)
    half = int(round(0.4 * sfreq))
    lo = max(0, peak_sample - half)
    hi = min(len(signal), peak_sample + half + 1)
    seg = signal[lo:hi]
    rel_p = peak_sample - lo
    if len(seg) < 8 or rel_p < 0 or rel_p >= len(seg):
        return "unclassified", False

    # FWHM around peak (mirrors morphology.py logic)
    seg_d = seg - float(np.median(seg))
    pk = abs(seg_d[rel_p])
    if pk <= 0 or not math.isfinite(pk):
        return "unclassified", False
    half_amp = pk * 0.5
    l = rel_p
    while l > 0 and abs(seg_d[l]) > half_amp:
        l -= 1
    r = rel_p
    while r < len(seg_d) - 1 and abs(seg_d[r]) > half_amp:
        r += 1
    fwhm_ms = (r - l) * 1000.0 / sfreq

    # Aftercoming slow wave: look 80–400 ms AFTER the peak for a slow
    # opposite-polarity excursion. We check whether there's a sustained
    # excursion past the peak baseline.
    after_lo = rel_p + int(round(0.08 * sfreq))
    after_hi = min(len(seg_d), rel_p + int(round(0.4 * sfreq)))
    has_sw = False
    if after_hi > after_lo + 4:
        after = seg_d[after_lo:after_hi]
        peak_sign = float(np.sign(seg_d[rel_p]))
        # Look for opposite-sign excursion of magnitude >= 25% of peak
        opp = -peak_sign * after
        if np.max(opp) >= 0.25 * pk:
            has_sw = True

    if fwhm_ms >= 200.0:
        return "complex_spike_wave", True
    if fwhm_ms < 70.0:
        return "simple_spike", has_sw
    return "sharp_wave", has_sw


def _topography_at_peak(
    multi_signal: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    peak_time_s: float,
    primary_channel: str,
) -> tuple[list[str], int]:
    """Return list of channels involved (those whose ±50 ms amplitude is
    >= 50% of the primary channel's peak) and the count.

    For a 1-channel recording, returns ([primary_channel], 1).
    """
    if multi_signal.size == 0:
        return [primary_channel], 1
    n_ch, n_samp = multi_signal.shape
    peak_sample = int(round(peak_time_s * sfreq))
    if peak_sample < 0 or peak_sample >= n_samp:
        return [primary_channel], 1
    half = max(1, int(round(0.05 * sfreq)))
    lo = max(0, peak_sample - half)
    hi = min(n_samp, peak_sample + half + 1)
    # Per-channel max |amplitude| in window
    amps = np.max(np.abs(multi_signal[:, lo:hi]), axis=1)
    primary_idx = None
    for i, nm in enumerate(ch_names):
        if nm.upper() == primary_channel.upper():
            primary_idx = i
            break
    if primary_idx is None:
        primary_amp = float(np.max(amps))
    else:
        primary_amp = float(amps[primary_idx])
    if primary_amp <= 0 or not math.isfinite(primary_amp):
        return [primary_channel], 1
    involved = [
        ch_names[i] for i in range(n_ch)
        if amps[i] >= 0.5 * primary_amp
    ]
    if not involved:
        involved = [primary_channel]
    return involved, len(involved)


def _is_centrotemporal_dominant(involved_channels: list[str]) -> bool:
    """Are the dominant involved channels centrotemporal?"""
    if not involved_channels:
        return False
    ct_hits = sum(
        1 for c in involved_channels if c.upper() in _CENTROTEMPORAL_CHS
    )
    return ct_hits >= 1 and ct_hits >= len(involved_channels) / 2.0


# ─── Typed stub exception for SpikeNet provenance (C2) ───────────────────────


class _SpikeNetStubError(NotImplementedError):
    """Stub-error with provenance attributes. Caller catches this and uses
    .model_version, .model_license directly (no string parsing)."""

    def __init__(self, msg: str, model_version: str, model_license: str):
        super().__init__(msg)
        self.model_version = model_version
        self.model_license = model_license


# ─── External SpikeNet path (stubbed inference) ──────────────────────────────


def _run_spikenet(
    rec: EEGRecording,
    weights_path: str,
    age_years: float | None,
) -> tuple[str, str]:
    """Compute weights provenance and surface the license.

    Intentionally STUBBED at the inference step: we cannot ship a faithful
    SpikeNet forward pass without the upstream repo's exact architecture
    and preprocessing. We raise NotImplementedError so the runner falls
    back to ensemble_heuristic. Returns (model_version, model_license)
    *before* raising so the caller has provenance even on failure.

    See https://github.com/bdsp-core/SpikeNet
    """
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "torch not installed; cannot run external_spikenet"
        ) from e

    p = Path(weights_path)
    if not p.exists():
        raise RuntimeError(f"weights file not found: {weights_path}")

    # Hash the first 1 KiB for fast provenance.
    # Streamed open avoids loading a 50–200 MB checkpoint fully into RAM.
    # NOTE: provenance label only — NOT collision-resistant against
    # adversarial weights.
    with p.open("rb") as fh:
        head = fh.read(1024)
    weights_hash = hashlib.sha256(head).hexdigest()[:16]
    model_version = f"sha256:{weights_hash}"
    model_license = "non-commercial-research (SpikeNet, Jing 2020)"

    # Age-aware threshold is part of the real inference loop; surface
    # it for transparency even though we don't run forward yet.
    _threshold = 0.7 if (age_years is not None and float(age_years) < 12.0) else 0.5  # noqa: F841

    raise _SpikeNetStubError(
        "External SpikeNet inference requires a user-supplied PyTorch "
        "model + weights compatible with bdsp-core/SpikeNet's ResNet "
        "architecture and 19-channel, 1-s, 128 Hz windowed input "
        "convention. See https://github.com/bdsp-core/SpikeNet. "
        "external SpikeNet path is STUBBED — real inference requires "
        "user-supplied implementation.",
        model_version,
        model_license,
    )


# ─── Ensemble heuristic ──────────────────────────────────────────────────────


def _run_ensemble(
    rec: EEGRecording,
    morphology_events: list[dict],
    age_years: float | None,
    sleep_stages: SleepStageResult | None,
) -> tuple[list[dict], list[str]]:
    """Apply R1/R2/R3 rules per morphology event.

    Returns (kept_events, warnings). Each event dict has:
      time_s, category, confidence, rules_passed (3-tuple),
      hf_burst_ratio, channels_involved, channel_count,
      likely_rolandic_benign.
    Events with 0/3 rules are skipped entirely.
    """
    warnings: list[str] = []
    if not morphology_events:
        return [], warnings

    sfreq = rec.sfreq
    if not math.isfinite(sfreq) or sfreq <= 0:
        warnings.append("invalid_sfreq")
        return [], warnings

    primary_idx, primary_name = _pick_channel(rec, preferred="Cz")
    if primary_idx is None:
        warnings.append("no_eeg_channel")
        return [], warnings

    primary_signal = _gather_signal(rec, primary_idx)
    if primary_signal.size == 0:
        warnings.append("no_signal_data")
        return [], warnings

    multi_signal, _eeg_idx, ch_names = _gather_all_channels(rec)

    # Pediatric drift: relax the morphology rule slightly (don't require
    # aftercoming-slow-wave evidence for simple spikes) to avoid
    # underweighting pediatric simple-spike morphologies the adult
    # heuristic would otherwise miss.
    pediatric = age_years is not None and (
        isinstance(age_years, (int, float))
        and math.isfinite(float(age_years))
        and float(age_years) < 12.0
    )

    kept: list[dict] = []
    for ev in morphology_events:
        t = ev.get("time_s")
        if t is None:
            continue
        try:
            t_f = float(t)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(t_f) or t_f < 0:
            continue

        peak_sample = int(round(t_f * sfreq))
        if peak_sample < 0 or peak_sample >= len(primary_signal):
            continue

        # R1: epileptiform morphology
        category, has_sw = _classify_morphology(primary_signal, sfreq, peak_sample)
        if category == "complex_spike_wave":
            r1 = 1
        elif category == "simple_spike" and (has_sw or pediatric):
            r1 = 1
        elif category == "sharp_wave" and has_sw:
            r1 = 1
        else:
            r1 = 0

        # R2: HF burst
        hf_ratio = _hf_burst_ratio(primary_signal, sfreq, peak_sample)
        r2 = 1 if hf_ratio >= 0.15 else 0

        # R3: focal topography
        involved, ch_count = _topography_at_peak(
            multi_signal, ch_names, sfreq, t_f, primary_name,
        )
        r3 = 1 if 0 < ch_count <= _FOCAL_MAX_CHANNELS else 0

        rules_passed = (r1, r2, r3)
        total = r1 + r2 + r3
        if total == 0:
            continue

        # Record rules_passed count BEFORE pediatric confidence promotion.
        # This is used by the agreement metric (C1): pediatric promotion is a
        # display decision, not a match decision — we don't want it to inflate
        # the agreement percentage.
        rules_passed_pre_pediatric = total

        if total == 3:
            confidence = "high"
        elif total == 2:
            confidence = "medium"
        else:
            confidence = "low"
            # In pediatric mode, promote low → medium so we don't drop
            # legitimate pediatric morphologies. (Drift handling.)
            if pediatric:
                confidence = "medium"

        # Rolandic flag is informational for any age — BCECTS peaks at 7-10
        # but can extend to 14, and retrospective lookback in young adults is
        # also valid. The age_appropriateness_flag (drift_warning) is the
        # age-specific signal; this flag is purely topographic/morphologic.
        likely_rolandic = (
            _is_centrotemporal_dominant(involved)
            and category in ("simple_spike", "sharp_wave")
        )

        kept.append({
            "time_s": t_f,
            "category": category,
            "confidence": confidence,
            "rules_passed": list(rules_passed),
            "rules_passed_pre_pediatric": rules_passed_pre_pediatric,
            "hf_burst_ratio": round(hf_ratio, 4),
            "channels_involved": involved,
            "channel_count": ch_count,
            "likely_rolandic_benign": bool(likely_rolandic),
        })

    return kept, warnings


def _per_channel_rates(
    kept: list[dict],
    duration_min: float,
) -> dict:
    if duration_min <= 0:
        return {}
    counts: dict[str, int] = {}
    for ev in kept:
        for ch in ev.get("channels_involved", []):
            counts[ch] = counts.get(ch, 0) + 1
    return {ch: round(n / duration_min, 3) for ch, n in counts.items()}


def _nrem_rate(
    kept: list[dict],
    sleep_stages: SleepStageResult | None,
) -> float | None:
    if sleep_stages is None:
        return None
    nrem_labels = {"N2", "N3"}
    epoch_s = sleep_stages.epoch_seconds
    nrem_indices = [
        i for i, lbl in enumerate(sleep_stages.epoch_labels)
        if lbl in nrem_labels
    ]
    if not nrem_indices:
        return 0.0
    nrem_windows = [
        (i * epoch_s, (i + 1) * epoch_s) for i in nrem_indices
    ]
    nrem_min = (len(nrem_indices) * epoch_s) / 60.0
    if nrem_min <= 0:
        return 0.0
    in_nrem = 0
    for ev in kept:
        t = ev["time_s"]
        if any(s <= t < e for s, e in nrem_windows):
            in_nrem += 1
    return round(in_nrem / nrem_min, 3)


# ─── Public API ──────────────────────────────────────────────────────────────


def compute_ied_ml(
    rec: EEGRecording,
    sleep_stages: SleepStageResult | None = None,
    morphology_events: list[dict] | None = None,
    weights_path: str | None = None,
    method: str = "auto",
    age_years: float | None = None,
) -> IEDDetectionResult:
    """Detect IEDs via the chosen method (auto / external_spikenet /
    ensemble_heuristic / unavailable).

    For ``method="auto"`` the choice follows ``_select_method``:
      - external_spikenet (if torch + weights present),
      - ensemble_heuristic (if morphology_events provided),
      - unavailable otherwise.

    If external_spikenet raises (currently always: stubbed), we fall
    back to ensemble_heuristic + a warning.
    """
    warnings: list[str] = []
    notes: list[str] = []

    age_flag = _age_flag(age_years)

    # Method resolution
    if method == "auto":
        chosen = _select_method(weights_path, morphology_events)
    elif method in _ALLOWED_METHODS:
        chosen = method
    else:
        warnings.append(f"unknown_method:{method}_falling_back_to_auto")
        chosen = _select_method(weights_path, morphology_events)

    model_version: str | None = None
    model_license: str | None = None

    # ── external_spikenet ──
    if chosen == "external_spikenet":
        try:
            model_version, model_license = _run_spikenet(
                rec, weights_path, age_years,
            )
            # Should never reach here while inference is stubbed
            notes.append("spikenet_inference_returned_unexpectedly")
            chosen = "ensemble_heuristic"
        except _SpikeNetStubError as e:
            warnings.append(f"spikenet_stub:{type(e).__name__}")
            # Provenance attributes are set directly on the typed exception —
            # no string parsing needed.
            model_version = e.model_version
            model_license = e.model_license
            notes.append(
                "external SpikeNet path is STUBBED — real inference requires "
                "user-supplied implementation"
            )
            # Fall back to ensemble if morphology events are available
            if morphology_events is not None:
                chosen = "ensemble_heuristic"
            else:
                chosen = "unavailable"
                warnings.append("spikenet_unavailable_no_morphology_fallback")
        except Exception as e:
            warnings.append(f"spikenet_error:{type(e).__name__}:{e}")
            if morphology_events is not None:
                chosen = "ensemble_heuristic"
            else:
                chosen = "unavailable"

    # ── unavailable ──
    if chosen == "unavailable":
        reason = (
            "no_morphology_events_and_no_weights"
            if morphology_events is None
            else "weights_missing_and_no_events"
        )
        return IEDDetectionResult(
            method="unavailable",
            available=False,
            unavailable_reason=reason,
            model_version=model_version,
            model_license=model_license,
            n_ied_candidates=0,
            rate_per_minute=0.0,
            per_channel_rates={},
            confidence_distribution={"high": 0, "medium": 0, "low": 0},
            age_appropriateness_flag=age_flag,
            agreement_with_morphology_pct=0.0,
            n_likely_rolandic_benign=0,
            nrem_rate_per_min=None,
            disclaimer="",
            warnings=warnings,
            notes=notes,
            events=[],
        )

    # ── ensemble_heuristic (the production default) ──
    if chosen == "ensemble_heuristic":
        if model_license is None:
            model_license = "rule-based"
        # model_version stays as whatever (possibly None or SpikeNet hash)
        events, ens_warnings = _run_ensemble(
            rec, morphology_events or [], age_years, sleep_stages,
        )
        warnings.extend(ens_warnings)

        duration_min = float(rec.duration_s) / 60.0 if rec.duration_s > 0 else 0.0
        n = len(events)
        rate = round(n / duration_min, 3) if duration_min > 0 else 0.0

        conf_dist = {"high": 0, "medium": 0, "low": 0}
        n_rolandic = 0
        for ev in events:
            c = ev.get("confidence", "low")
            if c in conf_dist:
                conf_dist[c] += 1
            if ev.get("likely_rolandic_benign"):
                n_rolandic += 1

        per_ch = _per_channel_rates(events, duration_min)

        # Agreement metric (C1): fraction of processable morphology events that
        # resulted in an ensemble event with rules_passed_pre_pediatric >= 2.
        # Denominator uses only processable events (valid finite time_s in range)
        # to avoid penalizing for malformed inputs.
        # Pediatric promotion (low→medium confidence) is a display decision only
        # and must NOT inflate the agreement percentage.
        n_morph_processable = sum(
            1 for e in (morphology_events or [])
            if isinstance(e.get("time_s"), (int, float))
            and math.isfinite(float(e["time_s"]))
            and 0 <= float(e["time_s"]) < rec.duration_s
        )
        n_matches = sum(
            1 for ev in events
            if ev.get("rules_passed_pre_pediatric", 0) >= 2
        )
        agreement = round(100.0 * n_matches / max(n_morph_processable, 1), 1)

        nrem_rate = _nrem_rate(events, sleep_stages)
        notes.append(_DISCLAIMER_ENSEMBLE)
        if age_flag == "drift_warning":
            notes.append("pediatric_drift_relaxed_thresholds_applied")

        return IEDDetectionResult(
            method="ensemble_heuristic",
            available=True,
            unavailable_reason="",
            model_version=model_version,
            model_license=model_license,
            n_ied_candidates=n,
            rate_per_minute=rate,
            per_channel_rates=per_ch,
            confidence_distribution=conf_dist,
            age_appropriateness_flag=age_flag,
            agreement_with_morphology_pct=agreement,
            n_likely_rolandic_benign=n_rolandic,
            nrem_rate_per_min=nrem_rate,
            disclaimer=_DISCLAIMER_ENSEMBLE,
            warnings=warnings,
            notes=notes,
            events=events,
        )

    # Shouldn't reach here
    return IEDDetectionResult(
        method="unavailable",
        available=False,
        unavailable_reason=f"unexpected_method:{chosen}",
        model_version=model_version,
        model_license=model_license,
        n_ied_candidates=0,
        rate_per_minute=0.0,
        per_channel_rates={},
        confidence_distribution={"high": 0, "medium": 0, "low": 0},
        age_appropriateness_flag=age_flag,
        agreement_with_morphology_pct=0.0,
        n_likely_rolandic_benign=0,
        nrem_rate_per_min=None,
        disclaimer="",
        warnings=warnings,
        notes=notes,
        events=[],
    )


# ─── Summary ─────────────────────────────────────────────────────────────────


def summarize_ied_ml(result: IEDDetectionResult) -> dict:
    """JSON-safe summary, without the events array."""
    def _sf(x) -> float | None:
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    out: dict = {
        "available": bool(result.available),
        "method": result.method,
        "unavailable_reason": result.unavailable_reason,
        "model_version": result.model_version,
        "model_license": result.model_license,
        "n_ied_candidates": int(result.n_ied_candidates),
        "rate_per_minute": _sf(result.rate_per_minute) or 0.0,
        "per_channel_rates": dict(result.per_channel_rates or {}),
        "confidence_distribution": dict(result.confidence_distribution or {}),
        "age_appropriateness_flag": result.age_appropriateness_flag,
        "agreement_with_morphology_pct": _sf(result.agreement_with_morphology_pct) or 0.0,
        "n_likely_rolandic_benign": int(result.n_likely_rolandic_benign),
        "nrem_rate_per_min": _sf(result.nrem_rate_per_min),
        "disclaimer": result.disclaimer or "",
        "warnings": list(result.warnings or []),
        "notes": list(result.notes or []),
    }
    # events intentionally NOT exported
    return out
