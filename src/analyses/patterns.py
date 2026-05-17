"""Pattern Recognition Module — v0.17.0.

Consolidates clinically interpreted patterns:

A1) Spike topography classification (centro_temporal_BCECTS, multifocal,
    generalized, frontal_dominant, occipital_dominant, lateralized_*)
A2) Spike morphology sub-classification (simple / sharp / spike-wave / polyspike)
A3) Sleep activation classification (wake_dominant_atypical → strong_activation_eses_risk)
A4) Pediatric normal-pattern filter (hypnagogic hypersync, PSWY, mu-rhythm, etc.)
A5) Centro-temporal spike co-occurrence with sleep state

All functions accept already-computed summary dicts (from runner.run_all_analyses)
rather than raw EEGRecording objects, keeping this layer stateless and fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ─── Region membership ────────────────────────────────────────────────────────

_CENTRO_TEMPORAL_CHANNELS: frozenset[str] = frozenset({
    "C3", "C4", "T3", "T4", "T5", "T6", "CP3", "CP4", "CP5", "CP6",
    # Alternative 10-10 labels
    "T7", "T8", "P7", "P8",
})

_FRONTAL_CHANNELS: frozenset[str] = frozenset({
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
})

_PARIETAL_OCCIPITAL_CHANNELS: frozenset[str] = frozenset({
    "P3", "P4", "Pz", "O1", "O2",
})

# Left hemisphere channels (for asymmetry)
_LEFT_CHANNELS: frozenset[str] = frozenset({
    "Fp1", "F3", "F7", "C3", "T3", "T5", "P3", "O1",
    "CP3", "CP5", "T7", "P7",
})

# Right hemisphere channels
_RIGHT_CHANNELS: frozenset[str] = frozenset({
    "Fp2", "F4", "F8", "C4", "T4", "T6", "P4", "O2",
    "CP4", "CP6", "T8", "P8",
})

# Homologous pairs for asymmetry index
_HOMOLOGOUS_PAIRS: list[tuple[str, str]] = [
    ("Fp1", "Fp2"),
    ("F3", "F4"),
    ("F7", "F8"),
    ("C3", "C4"),
    ("T3", "T4"),
    ("T5", "T6"),
    ("P3", "P4"),
    ("O1", "O2"),
    ("CP3", "CP4"),
    ("CP5", "CP6"),
    ("T7", "T8"),
    ("P7", "P8"),
]


# ─── A1) Spike Topography Classification ──────────────────────────────────────

@dataclass
class SpikeTopographyResult:
    """Classified spike topography from per-channel kurtosis."""

    top_5_channels: list[tuple[str, float]]  # (channel, kurtosis score)
    pattern_type: str  # see _TOPO_TYPES
    regional_rates: dict[str, float]  # region → mean kurtosis
    asymmetry_index: float  # (R - L) / (R + L), range [-1, 1]
    lateralization_significant: bool  # |asymmetry_index| > 0.10
    classification_confidence: str  # "high" / "medium" / "low"
    notes: list[str] = field(default_factory=list)


_TOPO_TYPES = frozenset({
    "centro_temporal_BCECTS",
    "multifocal",
    "generalized",
    "frontal_dominant",
    "occipital_dominant",
    "lateralized_right",
    "lateralized_left",
    "indeterminate",
})


def classify_spike_topography(
    topography_summary: dict[str, Any],
) -> SpikeTopographyResult:
    """Classify spike topography from a summarize_topography() output dict.

    Parameters
    ----------
    topography_summary : dict
        Output of summarize_topography() — must contain "all_channels" and
        "top_channels" lists.

    Returns
    -------
    SpikeTopographyResult with pattern_type and supporting metrics.
    """
    all_ch = topography_summary.get("all_channels", [])
    top_ch = topography_summary.get("top_channels", [])

    notes: list[str] = []

    if not all_ch:
        return SpikeTopographyResult(
            top_5_channels=[],
            pattern_type="indeterminate",
            regional_rates={},
            asymmetry_index=0.0,
            lateralization_significant=False,
            classification_confidence="low",
            notes=["no channel data available"],
        )

    # Build lookup: channel_name -> median_kurtosis
    ch_kurt: dict[str, float] = {
        c["name"]: float(c.get("median", 0.0))
        for c in all_ch
        if isinstance(c.get("name"), str)
    }

    # Top-5 channels
    sorted_ch = sorted(ch_kurt.items(), key=lambda x: -x[1])
    top_5 = sorted_ch[:5]

    # Regional means
    ct_vals = [v for k, v in ch_kurt.items() if k in _CENTRO_TEMPORAL_CHANNELS]
    fr_vals = [v for k, v in ch_kurt.items() if k in _FRONTAL_CHANNELS]
    po_vals = [v for k, v in ch_kurt.items() if k in _PARIETAL_OCCIPITAL_CHANNELS]

    ct_mean = float(np.mean(ct_vals)) if ct_vals else 0.0
    fr_mean = float(np.mean(fr_vals)) if fr_vals else 0.0
    po_mean = float(np.mean(po_vals)) if po_vals else 0.0

    regional_rates = {
        "centro_temporal": round(ct_mean, 3),
        "frontal": round(fr_mean, 3),
        "parietal_occipital": round(po_mean, 3),
    }

    # Asymmetry index: (R - L) / (R + L) using matched pairs
    lh_vals, rh_vals = [], []
    for l_ch, r_ch in _HOMOLOGOUS_PAIRS:
        if l_ch in ch_kurt and r_ch in ch_kurt:
            lh_vals.append(ch_kurt[l_ch])
            rh_vals.append(ch_kurt[r_ch])

    if lh_vals and rh_vals:
        lh_avg = float(np.mean(lh_vals))
        rh_avg = float(np.mean(rh_vals))
        denom = lh_avg + rh_avg
        asymmetry_index = (rh_avg - lh_avg) / denom if denom > 0 else 0.0
    else:
        asymmetry_index = 0.0
        notes.append("insufficient paired channels for asymmetry index")

    lateralization_significant = abs(asymmetry_index) > 0.10

    # Count top-5 by region
    top5_names = {c for c, _ in top_5}
    n_top5_ct = len(top5_names & _CENTRO_TEMPORAL_CHANNELS)
    n_top5_fr = len(top5_names & _FRONTAL_CHANNELS)
    n_top5_po = len(top5_names & _PARIETAL_OCCIPITAL_CHANNELS)

    # Count distinct "hotspot regions" (a region counts if ≥1 top-5 channel)
    n_hotspot_regions = (
        (1 if n_top5_ct > 0 else 0)
        + (1 if n_top5_fr > 0 else 0)
        + (1 if n_top5_po > 0 else 0)
    )

    # --- Classification logic ---
    # Compute generalized ratio BEFORE deciding — but don't classify as
    # generalized if there are ≥3 distinct hotspot regions (that's multifocal).
    if len(sorted_ch) >= 3:
        kvals = [v for _, v in sorted_ch]
        max_k = kvals[0]
        med_k = float(np.median(kvals))
        ratio = max_k / med_k if med_k > 0 else 0.0
    else:
        ratio = 0.0
        max_k = 0.0

    # Multifocal: ≥3 independent hotspot regions in top-5
    # Check this BEFORE generalized — multifocal has multiple high-kurtosis
    # regions but they are distinct foci, not uniformly elevated background.
    if n_hotspot_regions >= 3:
        pattern_type = "multifocal"
        confidence = "high" if len(top_5) >= 5 else "medium"
        notes.append(
            f"multifocal: {n_top5_ct} CT, {n_top5_fr} frontal, "
            f"{n_top5_po} parietal-occipital in top-5"
        )

    # Generalized: uniform elevation, ratio max/median < 2 (and NOT multifocal)
    elif ratio < 2.0 and max_k > 3.0:
        pattern_type = "generalized"
        confidence = "medium"

    # BCECTS: centro-temporal dominant AND ≥3 of top-5 in CT region
    # AND ct_mean >> other regions (>1.5×)
    elif (
        n_top5_ct >= 3
        and ct_mean > 1.5 * fr_mean
        and ct_mean > 1.5 * po_mean
    ):
        pattern_type = "centro_temporal_BCECTS"
        confidence = "high"
        notes.append(
            f"CT dominant: ct_mean={ct_mean:.2f}, "
            f"fr_mean={fr_mean:.2f}, po_mean={po_mean:.2f}"
        )

    # Frontal dominant
    elif n_top5_fr >= 3 and fr_mean >= ct_mean and fr_mean >= po_mean:
        pattern_type = "frontal_dominant"
        confidence = "medium"

    # Occipital/parietal dominant
    elif n_top5_po >= 3 and po_mean >= ct_mean and po_mean >= fr_mean:
        pattern_type = "occipital_dominant"
        confidence = "medium"

    # Lateralized: significant asymmetry AND not multifocal
    elif lateralization_significant:
        pattern_type = "lateralized_right" if asymmetry_index > 0 else "lateralized_left"
        confidence = "medium" if abs(asymmetry_index) > 0.20 else "low"

    # Multifocal fallback: 2 hotspot regions
    elif n_hotspot_regions == 2:
        pattern_type = "multifocal"
        confidence = "low"

    else:
        pattern_type = "indeterminate"
        confidence = "low"
        notes.append("insufficient signal or channel coverage")

    return SpikeTopographyResult(
        top_5_channels=top_5,
        pattern_type=pattern_type,
        regional_rates=regional_rates,
        asymmetry_index=round(asymmetry_index, 4),
        lateralization_significant=lateralization_significant,
        classification_confidence=confidence,
        notes=notes,
    )


# ─── A2) Spike Morphology Sub-classification ──────────────────────────────────

@dataclass
class SpikeMorphologySubtypes:
    """Finer sub-classification of detected spikes."""

    n_total: int
    n_spike_short: int        # < 70 ms (classic simple spike)
    n_sharp: int              # 70–200 ms (sharp wave)
    n_sharp_wave_complex: int  # ≥ 200 ms (spike-wave complex)
    n_polyspike: int          # inferred from polyspike_fraction
    pct_polyspike: float      # fraction × 100 — encephalopathy marker
    interpretation: str       # "benign_focal" | "epileptiform_spectrum" | "encephalopathic"
    notes: list[str] = field(default_factory=list)


def classify_spike_morphology_subtypes(
    morphology_summary: dict[str, Any],
) -> SpikeMorphologySubtypes:
    """Classify morphology subtypes from a summarize_morphology() output dict.

    Parameters
    ----------
    morphology_summary : dict
        Output of summarize_morphology() — must contain pct_simple_spikes,
        pct_sharp_waves, pct_complex_spike_wave, polyspike_fraction, n_events.

    Returns
    -------
    SpikeMorphologySubtypes
    """
    notes: list[str] = []
    n_total = int(morphology_summary.get("n_events", 0))

    pct_simple = float(morphology_summary.get("pct_simple_spikes", 0.0))
    pct_sharp = float(morphology_summary.get("pct_sharp_waves", 0.0))
    pct_complex = float(morphology_summary.get("pct_complex_spike_wave", 0.0))
    poly_frac = float(morphology_summary.get("polyspike_fraction", 0.0))

    # Infer counts from percentages
    n_short = int(round(n_total * pct_simple / 100.0))
    n_sharp = int(round(n_total * pct_sharp / 100.0))
    n_complex = int(round(n_total * pct_complex / 100.0))
    n_poly = int(round(n_total * poly_frac / 100.0))
    pct_poly = poly_frac  # already a percentage (renamed for clarity in report)

    # Interpretation
    if pct_poly >= 20.0 or pct_complex >= 50.0:
        interpretation = "encephalopathic"
        notes.append(
            f"polyspike {pct_poly:.0f}% and/or complex spike-wave "
            f"{pct_complex:.0f}% — encephalopathic burden"
        )
    elif pct_simple > 60.0:
        interpretation = "benign_focal"
        notes.append(f"predominantly simple spikes ({pct_simple:.0f}%) — benign focal pattern")
    else:
        interpretation = "epileptiform_spectrum"
        notes.append("mixed morphology — epileptiform spectrum")

    if n_total == 0:
        interpretation = "benign_focal"
        notes = ["no events detected"]

    return SpikeMorphologySubtypes(
        n_total=n_total,
        n_spike_short=n_short,
        n_sharp=n_sharp,
        n_sharp_wave_complex=n_complex,
        n_polyspike=n_poly,
        pct_polyspike=round(pct_poly, 1),
        interpretation=interpretation,
        notes=notes,
    )


# ─── A3) Sleep Activation Classification ──────────────────────────────────────

@dataclass
class SleepActivationResult:
    """Sleep-activation classification with CSWS risk composite score."""

    wake_rate_per_min: float
    nrem_rate_per_min: float
    rem_rate_per_min: float
    activation_ratio: float  # nrem / wake (or 0 if wake_rate < 0.1)
    classification: str  # see _SLEEP_ACT_TYPES
    csws_risk_score: float  # 0–1 composite
    notes: list[str] = field(default_factory=list)


_SLEEP_ACT_TYPES = frozenset({
    "wake_dominant_atypical",
    "no_activation",
    "mild_activation",
    "moderate_activation",
    "strong_activation_eses_risk",
    "indeterminate",
})


def classify_sleep_activation(
    state_split_summary: dict[str, Any],
    swi_summary: dict[str, Any] | None = None,
) -> SleepActivationResult:
    """Classify sleep activation from state_split and optional SWI summaries.

    Parameters
    ----------
    state_split_summary : dict
        Output of summarize_state_split() — contains wake_rate_per_min,
        nrem_rate_per_min, rem_rate_per_min, activation_factor, activation_label.
    swi_summary : dict, optional
        Output of summarize_swi() — used to compute CSWS risk composite.

    Returns
    -------
    SleepActivationResult
    """
    notes: list[str] = []

    wake_rate = float(state_split_summary.get("wake_rate_per_min", 0.0))
    nrem_rate = float(state_split_summary.get("nrem_rate_per_min", 0.0))
    rem_rate = float(state_split_summary.get("rem_rate_per_min", 0.0))

    # Compute activation_ratio
    if wake_rate < 0.1:
        activation_ratio = 0.0
        notes.append("wake_rate < 0.1/min — indeterminate ratio")
    else:
        activation_ratio = nrem_rate / wake_rate

    # Classification thresholds
    if wake_rate < 0.1:
        classification = "indeterminate"
    elif activation_ratio < 0.7:
        classification = "wake_dominant_atypical"
        notes.append(
            f"activation ratio {activation_ratio:.2f} < 0.7 — "
            "sleep activation INVERS, atypisch günstig für KCNQ3-Spektrum"
        )
    elif activation_ratio < 1.5:
        classification = "no_activation"
    elif activation_ratio < 3.0:
        classification = "mild_activation"
    elif activation_ratio < 10.0:
        classification = "moderate_activation"
    else:
        classification = "strong_activation_eses_risk"
        notes.append("activation ratio > 10× — ESES risk zone")

    # CSWS risk composite (0–1)
    # Components:
    #   1. Activation ratio contribution (0–0.5)
    #   2. N3 SWI contribution (0–0.3)
    #   3. NREM combined SWI contribution (0–0.2)
    risk = 0.0

    # Component 1: activation ratio
    # 0 → 0 risk, ≥10 → 0.5 risk (linear up to 10)
    risk += min(0.5, max(0.0, activation_ratio / 20.0))

    if swi_summary and isinstance(swi_summary, dict):
        swi_per_stage = swi_summary.get("swi_per_stage_pct", {})
        n3_swi = float(swi_per_stage.get("N3", 0.0))
        nrem_combined = float(swi_summary.get("swi_nrem_combined_pct", 0.0))
        csws_met = bool(swi_summary.get("csws_criterion_met", False))

        # Component 2: N3 SWI (0–85 → 0–0.3, capped)
        risk += min(0.3, n3_swi / 85.0 * 0.3)

        # Component 3: NREM combined SWI (0–50 → 0–0.2)
        risk += min(0.2, nrem_combined / 50.0 * 0.2)

        if csws_met:
            risk = max(risk, 0.85)
            notes.append("CSWS Tassinari criterion met — risk clamped to ≥0.85")
    else:
        notes.append("no SWI data — CSWS risk components 2+3 unavailable")

    csws_risk_score = round(min(1.0, max(0.0, risk)), 3)

    return SleepActivationResult(
        wake_rate_per_min=round(wake_rate, 2),
        nrem_rate_per_min=round(nrem_rate, 2),
        rem_rate_per_min=round(rem_rate, 2),
        activation_ratio=round(activation_ratio, 3),
        classification=classification,
        csws_risk_score=csws_risk_score,
        notes=notes,
    )


# ─── A4) Pediatric Normal-Pattern Filter ──────────────────────────────────────

@dataclass
class PediatricNormalFilterResult:
    """Counts of age-normal EEG patterns that may have been filtered."""

    patterns_detected: dict[str, int]  # pattern_name → count (estimated)
    total_filtered: int
    notes: list[str] = field(default_factory=list)


_PEDIATRIC_NORMAL_PATTERNS = (
    "hypnagogic_hypersynchrony",      # 3–5 Hz, high amplitude, sleep-onset
    "posterior_slow_waves_of_youth",  # PSWY: 2.5–4.5 Hz occipital
    "mu_rhythm",                      # 8–13 Hz central, arch-shaped
    "lambda_waves",                   # occipital, saccade-triggered, wake
    "14_and_6_hz_positive_spikes",    # posterior-temporal, sleep
    "rmtd",                           # Rhythmic Mid-Temporal Theta of Drowsiness
)


def estimate_pediatric_normal_patterns(
    findings: dict[str, Any],
    age_years: float | None = None,
) -> PediatricNormalFilterResult:
    """Estimate which pediatric normal patterns may be present.

    This is a heuristic filter based on available summary data. True
    waveform-level classification is outside the current pipeline scope.
    Returns estimated counts based on indirect indicators.

    Parameters
    ----------
    findings : dict
        Full findings dict from run_all_analyses().
    age_years : float, optional
        Patient age — some patterns are age-dependent.
    """
    notes: list[str] = []
    patterns: dict[str, int] = {p: 0 for p in _PEDIATRIC_NORMAL_PATTERNS}

    # Hypnagogic hypersynchrony: common in children < 5, detectable as
    # high-amplitude 3–5 Hz at sleep onset. Indirect indicator: bursts at
    # sleep onset window.
    arch = findings.get("sleep_architecture") or {}
    onset_min = arch.get("sleep_onset_minute")
    if onset_min is not None and age_years is not None and age_years < 6:
        patterns["hypnagogic_hypersynchrony"] = 1
        notes.append(
            f"hypnagogic hypersynchrony likely at age {age_years:.1f}y; "
            "marked as detected (waveform verification recommended)"
        )

    # PSWY: common age 3–15, occipital
    if age_years is not None and 3.0 <= age_years <= 15.0:
        # Check if occipital channels are prominent in background
        bg = findings.get("background") or {}
        pdr = bg.get("posterior_dominant_rhythm_hz")
        if pdr is not None and 2.5 <= float(pdr) <= 4.5:
            patterns["posterior_slow_waves_of_youth"] = 1
            notes.append(
                f"PDR {pdr:.1f} Hz in PSWY range (2.5–4.5 Hz) for age {age_years:.1f}y"
            )

    # Mu rhythm: 8–13 Hz central — not directly distinguishable without
    # reactivity testing; flag as possible if central channels are prominent
    topo = findings.get("topography") or {}
    top_chs = [c.get("name", "") for c in topo.get("top_channels", [])]
    if any(c in ("C3", "C4", "Cz") for c in top_chs[:5]):
        patterns["mu_rhythm"] = 1
        notes.append("central channel prominence — mu rhythm possible (needs reactivity test)")

    # 14&6 Hz positive spikes: posterior-temporal, sleep, age 4–20
    if age_years is not None and 4.0 <= age_years <= 20.0:
        patterns["14_and_6_hz_positive_spikes"] = 1
        notes.append("14&6 Hz positive spikes possible at this age (NREM, posterior-temporal)")

    # RMTD: temporal theta of drowsiness — common in adults, less so in children
    if age_years is None or age_years > 12:
        patterns["rmtd"] = 1
        notes.append("RMTD possible (mid-temporal theta of drowsiness)")

    total_filtered = sum(patterns.values())

    return PediatricNormalFilterResult(
        patterns_detected=patterns,
        total_filtered=total_filtered,
        notes=notes,
    )


# ─── A5) Centro-temporal Spike Co-occurrence with Sleep ───────────────────────

@dataclass
class CTSleepCooccurrenceResult:
    """Centro-temporal spike co-occurrence with sleep state."""

    ct_wake_rate: float       # CT-spike rate during wake
    ct_n2_rate: float         # CT-spike rate during N2
    ct_n3_rate: float         # CT-spike rate during N3
    ct_nrem_rate: float       # combined NREM rate
    rolandic_sleep_activated: bool  # N2/N3 rate > 2× wake AND CT-dominant
    notes: list[str] = field(default_factory=list)


def classify_ct_sleep_cooccurrence(
    topography_result: SpikeTopographyResult,
    sleep_activation_result: SleepActivationResult,
    state_split_summary: dict[str, Any] | None = None,
) -> CTSleepCooccurrenceResult:
    """Assess centro-temporal spike co-occurrence with sleep.

    Uses the topography classification (to confirm CT dominance) plus
    the sleep-activation result to compute the rolandic_sleep_activated flag.

    Parameters
    ----------
    topography_result : SpikeTopographyResult
    sleep_activation_result : SleepActivationResult
    state_split_summary : dict, optional
        Full state_split summary dict for raw rates.
    """
    notes: list[str] = []

    # Use state-split rates as proxy for CT-channel rates (pipeline computes
    # on the single-channel detector; per-channel-per-state is not currently
    # available in the summary layer)
    wake_rate = sleep_activation_result.wake_rate_per_min
    nrem_rate = sleep_activation_result.nrem_rate_per_min
    # N2/N3 split not available separately in state_split; use nrem as combined
    n2_rate = nrem_rate  # approximation
    n3_rate = nrem_rate  # approximation
    notes.append("N2/N3 rates approximated from combined NREM (no per-stage split available)")

    # rolandic_sleep_activated criterion:
    # nrem > 2× wake AND CT-dominant topography
    ct_dominant = topography_result.pattern_type == "centro_temporal_BCECTS"
    sleep_activated = nrem_rate > 2.0 * wake_rate if wake_rate > 0.1 else False
    rolandic_sleep_activated = ct_dominant and sleep_activated

    if ct_dominant and not sleep_activated:
        notes.append(
            "CT-dominant topography but no sleep activation "
            f"(nrem/wake = {nrem_rate:.1f}/{wake_rate:.1f})"
        )
    if not ct_dominant:
        notes.append(
            f"topography is '{topography_result.pattern_type}' — "
            "rolandic classification not applicable"
        )

    return CTSleepCooccurrenceResult(
        ct_wake_rate=round(wake_rate, 2),
        ct_n2_rate=round(n2_rate, 2),
        ct_n3_rate=round(n3_rate, 2),
        ct_nrem_rate=round(nrem_rate, 2),
        rolandic_sleep_activated=rolandic_sleep_activated,
        notes=notes,
    )


# ─── Top-level entry point ────────────────────────────────────────────────────

@dataclass
class PatternRecognitionResult:
    """All pattern-recognition outputs bundled."""

    topography: SpikeTopographyResult
    morphology_subtypes: SpikeMorphologySubtypes
    sleep_activation: SleepActivationResult
    pediatric_filter: PediatricNormalFilterResult
    ct_sleep_cooccurrence: CTSleepCooccurrenceResult


def run_pattern_recognition(
    findings: dict[str, Any],
    age_years: float | None = None,
) -> PatternRecognitionResult:
    """Run all pattern-recognition modules on a findings dict.

    Parameters
    ----------
    findings : dict
        Output of run_all_analyses().
    age_years : float, optional

    Returns
    -------
    PatternRecognitionResult
    """
    topo_sum = findings.get("topography") or {}
    morph_sum = findings.get("morphology") or {}
    state_sum = findings.get("state_split") or {}
    swi_sum = findings.get("swi")

    topo_result = classify_spike_topography(topo_sum)
    morph_result = classify_spike_morphology_subtypes(morph_sum)
    sleep_result = classify_sleep_activation(state_sum, swi_sum)
    pedi_result = estimate_pediatric_normal_patterns(findings, age_years)
    ct_result = classify_ct_sleep_cooccurrence(topo_result, sleep_result, state_sum)

    return PatternRecognitionResult(
        topography=topo_result,
        morphology_subtypes=morph_result,
        sleep_activation=sleep_result,
        pediatric_filter=pedi_result,
        ct_sleep_cooccurrence=ct_result,
    )


# ─── Summarize helpers ────────────────────────────────────────────────────────

def summarize_pattern_recognition(result: PatternRecognitionResult) -> dict:
    """Return a JSON-serializable summary of all pattern-recognition results."""
    topo = result.topography
    morph = result.morphology_subtypes
    sleep = result.sleep_activation
    pedi = result.pediatric_filter
    ct = result.ct_sleep_cooccurrence

    return {
        "spike_topography_pattern": topo.pattern_type,
        "topography": {
            "pattern_type": topo.pattern_type,
            "top_5_channels": [
                {"name": n, "kurtosis": round(k, 3)} for n, k in topo.top_5_channels
            ],
            "regional_rates": topo.regional_rates,
            "asymmetry_index": topo.asymmetry_index,
            "lateralization_significant": topo.lateralization_significant,
            "classification_confidence": topo.classification_confidence,
            "notes": topo.notes,
        },
        "morphology_subtypes": {
            "n_total": morph.n_total,
            "n_spike_short": morph.n_spike_short,
            "n_sharp": morph.n_sharp,
            "n_sharp_wave_complex": morph.n_sharp_wave_complex,
            "n_polyspike": morph.n_polyspike,
            "pct_polyspike": morph.pct_polyspike,
            "interpretation": morph.interpretation,
            "notes": morph.notes,
        },
        "sleep_activation": {
            "wake_rate_per_min": sleep.wake_rate_per_min,
            "nrem_rate_per_min": sleep.nrem_rate_per_min,
            "rem_rate_per_min": sleep.rem_rate_per_min,
            "activation_ratio": sleep.activation_ratio,
            "classification": sleep.classification,
            "csws_risk_score": sleep.csws_risk_score,
            "notes": sleep.notes,
        },
        "sleep_activation_classification": sleep.classification,
        "csws_risk_score": sleep.csws_risk_score,
        "pediatric_normal_filter": {
            "patterns_detected": pedi.patterns_detected,
            "total_filtered": pedi.total_filtered,
            "notes": pedi.notes,
        },
        "ct_sleep_cooccurrence": {
            "ct_wake_rate": ct.ct_wake_rate,
            "ct_n2_rate": ct.ct_n2_rate,
            "ct_n3_rate": ct.ct_n3_rate,
            "ct_nrem_rate": ct.ct_nrem_rate,
            "rolandic_sleep_activated": ct.rolandic_sleep_activated,
            "notes": ct.notes,
        },
    }
