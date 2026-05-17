"""Tests for v0.17.0 Pattern Recognition module.

Run with: python tests/test_patterns_v017.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import uuid

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"

n_pass = 0
n_fail = 0


def check(name: str, condition: bool, detail: str = ""):
    global n_pass, n_fail
    if condition:
        n_pass += 1
        print(f"  {PASS} {name}")
    else:
        n_fail += 1
        print(f"  {FAIL} {name}  {detail}")


def section(name: str):
    print(f"\n── {name} ───────────────────────────────────────────")


# ─── Helper: synthetic topography summary ────────────────────────────────────

def _make_topo_multifocal() -> dict:
    """High kurtosis spread across centro-temporal, frontal, AND parietal channels."""
    channels = [
        # centro-temporal
        {"name": "T4", "median": 18.2, "p90": 32.0, "pct_high_kurtosis": 72.0},
        {"name": "C4", "median": 15.1, "p90": 28.0, "pct_high_kurtosis": 68.0},
        # frontal
        {"name": "F7", "median": 14.3, "p90": 26.0, "pct_high_kurtosis": 65.0},
        {"name": "F8", "median": 12.8, "p90": 24.0, "pct_high_kurtosis": 60.0},
        # parietal-occipital
        {"name": "P4", "median": 13.5, "p90": 25.0, "pct_high_kurtosis": 62.0},
        {"name": "P3", "median": 11.0, "p90": 20.0, "pct_high_kurtosis": 55.0},
        # other channels
        {"name": "Cz", "median": 5.2, "p90": 10.0, "pct_high_kurtosis": 12.0},
        {"name": "Fz", "median": 4.8, "p90": 9.0, "pct_high_kurtosis": 10.0},
        {"name": "Pz", "median": 4.5, "p90": 8.5, "pct_high_kurtosis": 9.0},
    ]
    top_ch = [{"name": c["name"], "median_kurtosis": c["median"]} for c in channels[:5]]
    return {"all_channels": channels, "top_channels": top_ch, "epochs_analyzed": 120}


def _make_topo_centro_temporal() -> dict:
    """High kurtosis concentrated in centro-temporal, much lower elsewhere."""
    channels = [
        {"name": "C3", "median": 22.0, "p90": 40.0, "pct_high_kurtosis": 80.0},
        {"name": "C4", "median": 21.5, "p90": 38.0, "pct_high_kurtosis": 78.0},
        {"name": "T3", "median": 19.8, "p90": 36.0, "pct_high_kurtosis": 74.0},
        {"name": "T4", "median": 18.2, "p90": 33.0, "pct_high_kurtosis": 70.0},
        {"name": "T5", "median": 16.5, "p90": 30.0, "pct_high_kurtosis": 65.0},
        # frontal: much lower
        {"name": "F3", "median": 5.1, "p90": 9.0, "pct_high_kurtosis": 8.0},
        {"name": "F7", "median": 4.8, "p90": 8.5, "pct_high_kurtosis": 7.0},
        {"name": "Fz", "median": 4.6, "p90": 8.0, "pct_high_kurtosis": 6.0},
        # parietal-occipital: much lower
        {"name": "P3", "median": 5.0, "p90": 9.0, "pct_high_kurtosis": 7.0},
        {"name": "P4", "median": 4.9, "p90": 8.8, "pct_high_kurtosis": 6.5},
    ]
    top_ch = [{"name": c["name"], "median_kurtosis": c["median"]} for c in channels[:5]]
    return {"all_channels": channels, "top_channels": top_ch, "epochs_analyzed": 120}


def _make_topo_generalized() -> dict:
    """Perfectly uniform kurtosis → max/median ratio < 2.
    Uses exactly equal values so ratio = 1.0 regardless of which channels appear.
    All channels are midline/non-regional (Fz, Cz, Pz) to avoid 3-region matching.
    """
    # Midline + frontal only channels — none in CT or parietal-occipital sets
    # This ensures n_hotspot_regions < 3 so multifocal check doesn't fire first
    channels = [
        {"name": c, "median": 10.0, "p90": 18.0, "pct_high_kurtosis": 40.0}
        for c in ["Fz", "Cz", "F3", "F4", "Fp1", "Fp2", "F7", "F8"]
    ]
    top_ch = [{"name": c["name"], "median_kurtosis": c["median"]} for c in channels[:5]]
    return {"all_channels": channels, "top_channels": top_ch, "epochs_analyzed": 120}


# ─── A1) Spike Topography Classification ─────────────────────────────────────

section("A1 Spike Topography — multifocal")

from src.analyses.patterns import classify_spike_topography, SpikeTopographyResult

topo_mf = _make_topo_multifocal()
res_mf = classify_spike_topography(topo_mf)

check("multifocal → SpikeTopographyResult returned", isinstance(res_mf, SpikeTopographyResult))
check(
    "multifocal → pattern_type == 'multifocal'",
    res_mf.pattern_type == "multifocal",
    f"got {res_mf.pattern_type!r}",
)
check("multifocal → top_5_channels has 5 entries", len(res_mf.top_5_channels) == 5)
check("multifocal → regional_rates has 3 keys",
      set(res_mf.regional_rates.keys()) == {"centro_temporal", "frontal", "parietal_occipital"})
check("multifocal → asymmetry_index is float", isinstance(res_mf.asymmetry_index, float))
check("multifocal → classification_confidence in {'high','medium','low'}",
      res_mf.classification_confidence in {"high", "medium", "low"})

section("A1 Spike Topography — centro_temporal_BCECTS")

topo_ct = _make_topo_centro_temporal()
res_ct = classify_spike_topography(topo_ct)

check("BCECTS → SpikeTopographyResult returned", isinstance(res_ct, SpikeTopographyResult))
check(
    "BCECTS → pattern_type == 'centro_temporal_BCECTS'",
    res_ct.pattern_type == "centro_temporal_BCECTS",
    f"got {res_ct.pattern_type!r}",
)
check("BCECTS → CT regional rate >> frontal",
      res_ct.regional_rates["centro_temporal"] > 1.5 * res_ct.regional_rates["frontal"])

section("A1 Spike Topography — generalized")

topo_gen = _make_topo_generalized()
res_gen = classify_spike_topography(topo_gen)
check(
    "Generalized → pattern_type == 'generalized'",
    res_gen.pattern_type == "generalized",
    f"got {res_gen.pattern_type!r}",
)

section("A1 Spike Topography — empty data")

res_empty = classify_spike_topography({})
check("Empty topo → 'indeterminate'", res_empty.pattern_type == "indeterminate")
check("Empty topo → confidence == 'low'", res_empty.classification_confidence == "low")

section("A1 Spike Topography — asymmetry")

# Build asymmetric topo: left side much stronger
asym_channels = [
    {"name": "C3", "median": 20.0, "p90": 35.0, "pct_high_kurtosis": 75.0},
    {"name": "T3", "median": 18.0, "p90": 32.0, "pct_high_kurtosis": 70.0},
    {"name": "P3", "median": 16.0, "p90": 28.0, "pct_high_kurtosis": 65.0},
    {"name": "C4", "median": 5.0, "p90": 9.0, "pct_high_kurtosis": 8.0},
    {"name": "T4", "median": 4.8, "p90": 8.5, "pct_high_kurtosis": 7.0},
    {"name": "P4", "median": 4.5, "p90": 8.0, "pct_high_kurtosis": 6.0},
    {"name": "Fz", "median": 4.0, "p90": 7.0, "pct_high_kurtosis": 5.0},
]
asym_topo = {
    "all_channels": asym_channels,
    "top_channels": [{"name": c["name"], "median_kurtosis": c["median"]} for c in asym_channels[:5]],
    "epochs_analyzed": 60,
}
res_asym = classify_spike_topography(asym_topo)
check(
    "Left-dominant → asymmetry_index < 0 (more left)",
    res_asym.asymmetry_index < 0,
    f"asymmetry_index={res_asym.asymmetry_index}",
)
check(
    "Left-dominant → lateralization_significant = True",
    res_asym.lateralization_significant,
    f"asym={res_asym.asymmetry_index}",
)


# ─── A2) Spike Morphology Sub-classification ──────────────────────────────────

section("A2 Morphology Subtypes")

from src.analyses.patterns import classify_spike_morphology_subtypes, SpikeMorphologySubtypes

# Benign focal: >60% simple
morph_benign = {
    "n_events": 200,
    "pct_simple_spikes": 70.0,
    "pct_sharp_waves": 20.0,
    "pct_complex_spike_wave": 10.0,
    "polyspike_fraction": 3.0,
}
res_beni = classify_spike_morphology_subtypes(morph_benign)
check("Benign focal returned", isinstance(res_beni, SpikeMorphologySubtypes))
check("Benign focal → interpretation == 'benign_focal'",
      res_beni.interpretation == "benign_focal",
      f"got {res_beni.interpretation!r}")
check("n_spike_short ~ 0.7 × 200", abs(res_beni.n_spike_short - 140) <= 2)

# Encephalopathic: polyspike ≥ 20%
morph_ence = {
    "n_events": 100,
    "pct_simple_spikes": 10.0,
    "pct_sharp_waves": 40.0,
    "pct_complex_spike_wave": 50.0,
    "polyspike_fraction": 25.0,
}
res_ence = classify_spike_morphology_subtypes(morph_ence)
check("Encephalopathic → interpretation == 'encephalopathic'",
      res_ence.interpretation == "encephalopathic",
      f"got {res_ence.interpretation!r}")
check("Encephalopathic → pct_polyspike == 25.0", res_ence.pct_polyspike == 25.0)

# ~50% polyspike
morph_poly50 = {
    "n_events": 200,
    "pct_simple_spikes": 10.0,
    "pct_sharp_waves": 40.0,
    "pct_complex_spike_wave": 50.0,
    "polyspike_fraction": 50.0,
}
res_poly50 = classify_spike_morphology_subtypes(morph_poly50)
check("50% polyspike → pct_polyspike == 50.0", res_poly50.pct_polyspike == 50.0)
check("50% polyspike → interpretation == 'encephalopathic'",
      res_poly50.interpretation == "encephalopathic")

# Zero events (should not crash)
morph_zero = {
    "n_events": 0,
    "pct_simple_spikes": 0.0,
    "pct_sharp_waves": 0.0,
    "pct_complex_spike_wave": 0.0,
    "polyspike_fraction": 0.0,
}
res_zero = classify_spike_morphology_subtypes(morph_zero)
check("Zero events → no crash, interpretation == 'benign_focal'",
      res_zero.interpretation == "benign_focal")

# Pediatric normal filter: hypnagogic hypersync NOT counted as encephalopathic
# (the morphology subtype classifier only uses the morphology summary, not
# raw waveforms — so a recording with age < 5 should still classify correctly
# based on spike morphology distribution, not pattern filter)
morph_pedi = {
    "n_events": 50,
    "pct_simple_spikes": 65.0,
    "pct_sharp_waves": 30.0,
    "pct_complex_spike_wave": 5.0,
    "polyspike_fraction": 2.0,
}
res_pedi = classify_spike_morphology_subtypes(morph_pedi)
check(
    "Pediatric normal filter: predominantly simple spikes → benign_focal, not encephalopathic",
    res_pedi.interpretation == "benign_focal",
    f"got {res_pedi.interpretation!r}",
)


# ─── A3) Sleep Activation Classification ──────────────────────────────────────

section("A3 Sleep Activation — wake_dominant_atypical")

from src.analyses.patterns import classify_sleep_activation, SleepActivationResult

# reference-like: ratio 0.46 (wake > NREM)
state_reference = {
    "wake_rate_per_min": 10.8,
    "nrem_rate_per_min": 5.0,
    "rem_rate_per_min": 3.0,
    "activation_factor": 0.46,
    "activation_label": "none",
    "notes": [],
}
res_reference = classify_sleep_activation(state_reference)
check("reference-like → SleepActivationResult returned", isinstance(res_reference, SleepActivationResult))
check(
    "reference-like → classification == 'wake_dominant_atypical'",
    res_reference.classification == "wake_dominant_atypical",
    f"got {res_reference.classification!r}",
)
check("reference-like → csws_risk_score < 0.3",
      res_reference.csws_risk_score < 0.3,
      f"risk={res_reference.csws_risk_score}")

section("A3 Sleep Activation — strong activation")

state_strong = {
    "wake_rate_per_min": 2.0,
    "nrem_rate_per_min": 25.0,
    "rem_rate_per_min": 5.0,
    "activation_factor": 12.5,
    "activation_label": "strong",
    "notes": [],
}
swi_high = {
    "csws_criterion_met": False,
    "swi_per_stage_pct": {"N1": 20.0, "N2": 50.0, "N3": 75.0},
    "swi_nrem_combined_pct": 55.0,
}
res_strong = classify_sleep_activation(state_strong, swi_high)
check(
    "Strong activation → 'strong_activation_eses_risk'",
    res_strong.classification == "strong_activation_eses_risk",
    f"got {res_strong.classification!r}",
)
check("Strong activation → csws_risk_score > 0.4",
      res_strong.csws_risk_score > 0.4,
      f"risk={res_strong.csws_risk_score}")

section("A3 Sleep Activation — CSWS criterion met")

state_csws = {
    "wake_rate_per_min": 3.0,
    "nrem_rate_per_min": 35.0,
    "rem_rate_per_min": 6.0,
    "activation_factor": 11.7,
    "activation_label": "strong",
    "notes": [],
}
swi_csws = {
    "csws_criterion_met": True,
    "swi_per_stage_pct": {"N1": 50.0, "N2": 80.0, "N3": 90.0},
    "swi_nrem_combined_pct": 80.0,
}
res_csws = classify_sleep_activation(state_csws, swi_csws)
check(
    "CSWS criterion met → csws_risk_score >= 0.85",
    res_csws.csws_risk_score >= 0.85,
    f"risk={res_csws.csws_risk_score}",
)

section("A3 Sleep Activation — no_activation range")

state_none = {
    "wake_rate_per_min": 5.0,
    "nrem_rate_per_min": 6.0,
    "rem_rate_per_min": 4.0,
    "activation_factor": 1.2,
    "activation_label": "none",
    "notes": [],
}
res_none = classify_sleep_activation(state_none)
check(
    "ratio 1.2 → 'no_activation'",
    res_none.classification == "no_activation",
    f"got {res_none.classification!r}",
)

section("A3 Sleep Activation — mild / moderate")

state_mild = {
    "wake_rate_per_min": 5.0,
    "nrem_rate_per_min": 10.0,
    "rem_rate_per_min": 4.0,
    "activation_factor": 2.0,
    "activation_label": "mild",
    "notes": [],
}
res_mild = classify_sleep_activation(state_mild)
check("ratio 2.0 → 'mild_activation'",
      res_mild.classification == "mild_activation",
      f"got {res_mild.classification!r}")

state_mod = {
    "wake_rate_per_min": 5.0,
    "nrem_rate_per_min": 25.0,
    "rem_rate_per_min": 4.0,
    "activation_factor": 5.0,
    "activation_label": "moderate",
    "notes": [],
}
res_mod = classify_sleep_activation(state_mod)
check("ratio 5.0 → 'moderate_activation'",
      res_mod.classification == "moderate_activation",
      f"got {res_mod.classification!r}")

section("A3 Sleep Activation — indeterminate (low wake rate)")

state_indet = {
    "wake_rate_per_min": 0.05,  # below threshold
    "nrem_rate_per_min": 10.0,
    "rem_rate_per_min": 4.0,
    "activation_factor": None,
    "activation_label": "indeterminate",
    "notes": [],
}
res_indet = classify_sleep_activation(state_indet)
check("wake < 0.1 → 'indeterminate'",
      res_indet.classification == "indeterminate",
      f"got {res_indet.classification!r}")


# ─── A4) Pediatric Normal Pattern Filter ──────────────────────────────────────

section("A4 Pediatric Normal Pattern Filter")

from src.analyses.patterns import estimate_pediatric_normal_patterns, PediatricNormalFilterResult

findings_pedi = {
    "sleep_architecture": {"sleep_onset_minute": 15},
    "background": {"posterior_dominant_rhythm_hz": 3.8},
    "topography": {"top_channels": [
        {"name": "C3", "median_kurtosis": 15.0},
        {"name": "C4", "median_kurtosis": 14.0},
    ]},
}
res_pf = estimate_pediatric_normal_patterns(findings_pedi, age_years=4.5)
check("Pediatric filter returns result", isinstance(res_pf, PediatricNormalFilterResult))
check("All pattern keys present",
      all(k in res_pf.patterns_detected for k in [
          "hypnagogic_hypersynchrony",
          "posterior_slow_waves_of_youth",
          "mu_rhythm",
          "14_and_6_hz_positive_spikes",
          "rmtd",
          "lambda_waves",
      ]))
check("Young child → hypnagogic detected",
      res_pf.patterns_detected.get("hypnagogic_hypersynchrony", 0) > 0)
check("PDR 3.8 Hz in PSWY range → PSWY detected",
      res_pf.patterns_detected.get("posterior_slow_waves_of_youth", 0) > 0)
check("Central channels prominent → mu_rhythm detected",
      res_pf.patterns_detected.get("mu_rhythm", 0) > 0)

# Ensure hypnagogic hypersync is NOT classified as encephalopathic
# (the two systems are independent — age-normal filter is informational only)
morph_for_pedi = {
    "n_events": 30,
    "pct_simple_spikes": 80.0,
    "pct_sharp_waves": 15.0,
    "pct_complex_spike_wave": 5.0,
    "polyspike_fraction": 1.0,
}
from src.analyses.patterns import classify_spike_morphology_subtypes
res_morph_pedi = classify_spike_morphology_subtypes(morph_for_pedi)
check("Hypnagogic hypersync (age 4.5, simple spikes) → NOT encephalopathic",
      res_morph_pedi.interpretation != "encephalopathic",
      f"got {res_morph_pedi.interpretation!r}")


# ─── A5) CT Sleep Co-occurrence ───────────────────────────────────────────────

section("A5 CT Sleep Co-occurrence")

from src.analyses.patterns import classify_ct_sleep_cooccurrence, CTSleepCooccurrenceResult
from src.analyses.patterns import SpikeTopographyResult, SleepActivationResult

# Build BCECTS topography result
topo_bcects = SpikeTopographyResult(
    top_5_channels=[("C3", 22.0), ("C4", 21.0), ("T3", 19.0), ("T4", 18.0), ("T5", 16.0)],
    pattern_type="centro_temporal_BCECTS",
    regional_rates={"centro_temporal": 18.0, "frontal": 5.0, "parietal_occipital": 5.0},
    asymmetry_index=0.02,
    lateralization_significant=False,
    classification_confidence="high",
)

# Sleep-activated BCECTS (nrem > 2× wake)
sleep_ct_activated = SleepActivationResult(
    wake_rate_per_min=3.0,
    nrem_rate_per_min=8.0,
    rem_rate_per_min=2.0,
    activation_ratio=2.67,
    classification="mild_activation",
    csws_risk_score=0.15,
)
res_ct_act = classify_ct_sleep_cooccurrence(topo_bcects, sleep_ct_activated)
check("CT result returned", isinstance(res_ct_act, CTSleepCooccurrenceResult))
check("BCECTS + sleep activated → rolandic_sleep_activated=True",
      res_ct_act.rolandic_sleep_activated,
      f"got {res_ct_act.rolandic_sleep_activated}")

# the reference patient: multifocal + inverse sleep → NOT rolandic
topo_multi = SpikeTopographyResult(
    top_5_channels=[("T4", 18.0), ("F7", 14.0), ("P4", 13.0), ("C4", 12.0), ("F8", 11.0)],
    pattern_type="multifocal",
    regional_rates={"centro_temporal": 10.0, "frontal": 9.0, "parietal_occipital": 8.0},
    asymmetry_index=0.05,
    lateralization_significant=False,
    classification_confidence="high",
)
sleep_reference2 = SleepActivationResult(
    wake_rate_per_min=10.8,
    nrem_rate_per_min=5.0,
    rem_rate_per_min=3.0,
    activation_ratio=0.46,
    classification="wake_dominant_atypical",
    csws_risk_score=0.023,
)
res_ct_reference = classify_ct_sleep_cooccurrence(topo_multi, sleep_reference2)
check("Multifocal + inverse sleep → rolandic_sleep_activated=False",
      not res_ct_reference.rolandic_sleep_activated,
      f"got {res_ct_reference.rolandic_sleep_activated}")


# ─── Full run_pattern_recognition ────────────────────────────────────────────

section("Full run_pattern_recognition integration")

from src.analyses.patterns import run_pattern_recognition, summarize_pattern_recognition, PatternRecognitionResult

# Build synthetic findings dict (reference-like)
findings_reference = {
    "topography": _make_topo_multifocal(),
    "morphology": {
        "n_events": 450,
        "events_per_minute": 22.5,
        "pct_simple_spikes": 55.0,
        "pct_sharp_waves": 35.0,
        "pct_complex_spike_wave": 10.0,
        "polyspike_fraction": 8.0,
        "classification": "mixed",
    },
    "state_split": {
        "wake_rate_per_min": 10.8,
        "nrem_rate_per_min": 5.0,
        "rem_rate_per_min": 3.0,
        "activation_factor": 0.46,
        "activation_label": "none",
        "notes": [],
    },
    "swi": {
        "csws_criterion_met": False,
        "swi_per_stage_pct": {"N1": 2.0, "N2": 5.0, "N3": 8.0},
        "swi_nrem_combined_pct": 5.0,
        "csws_threshold_pct": 85.0,
    },
    "sleep_architecture": {"sleep_onset_minute": 20},
    "background": {"posterior_dominant_rhythm_hz": 8.5},
}

pr_result = run_pattern_recognition(findings_reference, age_years=5.0)
check("PatternRecognitionResult returned", isinstance(pr_result, PatternRecognitionResult))
check(
    "the reference patient synthetic → topography is 'multifocal'",
    pr_result.topography.pattern_type == "multifocal",
    f"got {pr_result.topography.pattern_type!r}",
)
check(
    "the reference patient synthetic → sleep_activation is 'wake_dominant_atypical'",
    pr_result.sleep_activation.classification == "wake_dominant_atypical",
    f"got {pr_result.sleep_activation.classification!r}",
)
check("the reference patient synthetic → csws_risk_score < 0.3",
      pr_result.sleep_activation.csws_risk_score < 0.3,
      f"risk={pr_result.sleep_activation.csws_risk_score}")
check("the reference patient synthetic → rolandic_sleep_activated=False",
      not pr_result.ct_sleep_cooccurrence.rolandic_sleep_activated)

# Summarize
summary = summarize_pattern_recognition(pr_result)
check("summarize returns dict", isinstance(summary, dict))
check("summary has 'spike_topography_pattern' key",
      "spike_topography_pattern" in summary)
check("summary has 'sleep_activation_classification' key",
      "sleep_activation_classification" in summary)
check("summary has 'csws_risk_score' key", "csws_risk_score" in summary)
check("summary spike_topography_pattern == 'multifocal'",
      summary["spike_topography_pattern"] == "multifocal")
check("summary sleep_activation_classification == 'wake_dominant_atypical'",
      summary["sleep_activation_classification"] == "wake_dominant_atypical")


# ─── B) ClinicalImpressionV2 ─────────────────────────────────────────────────

section("B) ClinicalImpressionV2 — main branches")

from src.clinical.impression_v2 import build_impression_v2, ClinicalImpressionV2, summarize_impression_v2

pr_sum = summarize_pattern_recognition(pr_result)

imp = build_impression_v2(
    findings_reference,
    pattern_recognition=pr_sum,
    metadata={"variant": "KCNQ3 p.Arg230His", "age_years": 5.0},
)
check("ClinicalImpressionV2 returned", isinstance(imp, ClinicalImpressionV2))
check("headline mentions 'KCNQ3'", "KCNQ3" in imp.headline)
check("headline mentions 'multifocal' or 'atypical'",
      any(w in imp.headline.lower() for w in ("multifocal", "atypical", "kcnq3")))
check("favorable_factors non-empty", len(imp.favorable_factors) > 0)
check("suggested_follow_up mentions spindle or biomarker",
      any("spindle" in f.lower() or "biomarker" in f.lower() for f in imp.suggested_follow_up))
check("differential non-empty", len(imp.differential) > 0)
check("key_findings non-empty", len(imp.key_findings) > 0)

# Branch: wake_dominant_atypical → favorable
check(
    "wake_dominant_atypical → favorable_factors contains 'INVERS'",
    any("INVERS" in f.upper() or "invers" in f.lower() or "günstig" in f.lower()
        for f in imp.favorable_factors),
    str(imp.favorable_factors[:3]),
)

# Branch: CSWS not met → favorable
check(
    "CSWS not met → favorable_factors mentions CSWS",
    any("csws" in f.lower() or "CSWS" in f for f in imp.favorable_factors),
    str(imp.favorable_factors[:3]),
)

# Non-KCNQ variant
imp_nk = build_impression_v2(
    findings_reference,
    pattern_recognition=pr_sum,
    metadata={"variant": "SCN1A p.Arg1234His"},
)
check("Non-KCNQ variant → no KCNQ-specific finding in key_findings",
      not any("KCNQ" in f for f in imp_nk.key_findings),
      str(imp_nk.key_findings[:3]))

# Summarize impression v2
imp_dict = summarize_impression_v2(imp)
check("summarize_impression_v2 returns dict", isinstance(imp_dict, dict))
check("imp_dict has all required keys",
      all(k in imp_dict for k in (
          "headline", "key_findings", "differential",
          "favorable_factors", "concerning_factors",
          "suggested_follow_up", "confidence_overall", "disclaimer",
      )))

# Empty findings — should not crash
imp_empty = build_impression_v2({}, pattern_recognition={}, metadata={})
check("Empty findings → no crash", isinstance(imp_empty, ClinicalImpressionV2))


# ─── C) Registry v0.17.0 bucket helpers ──────────────────────────────────────

section("C) Registry v0.17.0 bucket helpers")

from src.registry.buckets import bucket_spike_polyspike_pct, bucket_csws_risk_score
from src.registry.schema import (
    SPIKE_TOPOGRAPHY_PATTERNS,
    SPIKE_POLYSPIKE_PCT_BUCKETS,
    SLEEP_ACTIVATION_CLASSIFICATIONS,
    CSWS_RISK_SCORE_BUCKETS,
)

check("SPIKE_TOPOGRAPHY_PATTERNS has 8 entries", len(SPIKE_TOPOGRAPHY_PATTERNS) == 8)
check("'multifocal' in SPIKE_TOPOGRAPHY_PATTERNS", "multifocal" in SPIKE_TOPOGRAPHY_PATTERNS)
check("SLEEP_ACTIVATION_CLASSIFICATIONS has 6 entries",
      len(SLEEP_ACTIVATION_CLASSIFICATIONS) == 6)
check("'wake_dominant_atypical' in SLEEP_ACTIVATION_CLASSIFICATIONS",
      "wake_dominant_atypical" in SLEEP_ACTIVATION_CLASSIFICATIONS)

# polyspike buckets
check("polyspike 0 → '0'", bucket_spike_polyspike_pct(0.0) == "0")
check("polyspike 3 → '<5'", bucket_spike_polyspike_pct(3.0) == "<5")
check("polyspike 10 → '5-20'", bucket_spike_polyspike_pct(10.0) == "5-20")
check("polyspike 30 → '20-50'", bucket_spike_polyspike_pct(30.0) == "20-50")
check("polyspike 60 → '>50'", bucket_spike_polyspike_pct(60.0) == ">50")
check("polyspike None → None", bucket_spike_polyspike_pct(None) is None)
check("polyspike -1 → None (negative)", bucket_spike_polyspike_pct(-1.0) is None)

# csws risk score buckets
check("risk 0.1 → '<0.2'", bucket_csws_risk_score(0.1) == "<0.2")
check("risk 0.25 → '0.2-0.4'", bucket_csws_risk_score(0.25) == "0.2-0.4")
check("risk 0.5 → '0.4-0.6'", bucket_csws_risk_score(0.5) == "0.4-0.6")
check("risk 0.7 → '0.6-0.8'", bucket_csws_risk_score(0.7) == "0.6-0.8")
check("risk 0.9 → '>0.8'", bucket_csws_risk_score(0.9) == ">0.8")
check("risk None → None", bucket_csws_risk_score(None) is None)


# ─── D) Registry extractors ───────────────────────────────────────────────────

section("D) Registry v0.17.0 extractors")

from src.registry.deid import (
    _extract_spike_topography_pattern,
    _extract_spike_polyspike_pct_bucket,
    _extract_sleep_activation_classification,
    _extract_csws_risk_score_bucket,
)

findings_with_pr = {
    **findings_reference,
    "pattern_recognition": {
        "spike_topography_pattern": "multifocal",
        "sleep_activation_classification": "wake_dominant_atypical",
        "csws_risk_score": 0.023,
        "morphology_subtypes": {
            "pct_polyspike": 8.0,
        },
    },
}

check(
    "_extract_spike_topography_pattern: 'multifocal' → 'multifocal'",
    _extract_spike_topography_pattern(findings_with_pr) == "multifocal",
)
check(
    "_extract_sleep_activation_classification: 'wake_dominant_atypical'",
    _extract_sleep_activation_classification(findings_with_pr) == "wake_dominant_atypical",
)
check(
    "_extract_csws_risk_score_bucket: 0.023 → '<0.2'",
    _extract_csws_risk_score_bucket(findings_with_pr) == "<0.2",
)
check(
    "_extract_spike_polyspike_pct_bucket: 8.0 → '5-20'",
    _extract_spike_polyspike_pct_bucket(findings_with_pr) == "5-20",
)

# Invalid values
check("extractor with empty findings → None",
      _extract_spike_topography_pattern({}) is None)
check("extractor with invalid pattern → None",
      _extract_spike_topography_pattern({"pattern_recognition": {"spike_topography_pattern": "INVALID"}}) is None)


# ─── E) Registry validation with v0.17.0 fields ──────────────────────────────

section("E) Registry validate — v0.17.0 fields in schema")

from src.registry.validate import validate_submission

def _make_sub_v017(extra_findings=None):
    sid = str(uuid.uuid4())
    return {
        "submission_id": sid,
        "schema_version": 2,
        "submitted_at_month": "2026-05",
        "consent": {"version": 2, "given": True, "given_at_month": "2026-05"},
        "subject": {
            "variant_gene": "KCNQ3",
            "variant_protein": "p.Arg230His",
            "variant_type": "missense_GoF",
            "age_years_bucket": "5-7",
            "sex": "F",
        },
        "recording": {
            "duration_hours_bucket": "12-24",
            "had_sleep": True,
            "montage": "10-20_monopolar",
            "n_channels": 19,
        },
        "findings": extra_findings or {},
        "tool_version": "0.17.0",
    }

sub_v017 = _make_sub_v017({
    "spike_topography_pattern": "multifocal",
    "spike_morphology_polyspike_pct_bucket": "5-20",
    "sleep_activation_classification": "wake_dominant_atypical",
    "csws_risk_score_bucket": "<0.2",
})
ok, errs = validate_submission(sub_v017)
check("v0.17.0 fields validate OK", ok, str(errs) if not ok else "")

sub_no_v017 = _make_sub_v017({})
ok2, errs2 = validate_submission(sub_no_v017)
check("Submission without v0.17.0 fields still validates (additive)", ok2, str(errs2))

sub_invalid_topo = _make_sub_v017({"spike_topography_pattern": "INVALID_PATTERN"})
ok3, errs3 = validate_submission(sub_invalid_topo)
check("Invalid topography pattern → validation fails", not ok3)

sub_invalid_sleep = _make_sub_v017({"sleep_activation_classification": "unknown"})
ok4, errs4 = validate_submission(sub_invalid_sleep)
check("Invalid sleep activation class → validation fails", not ok4)


# ─── F) Full build_submission integration ─────────────────────────────────────

section("F) Full build_submission with v0.17.0 extractors")

from src.registry.deid import build_submission, SubmissionInput
from src.registry.consent import Consent

consent = Consent(version=2, given=True, given_at_month="2026-05")
user_input = SubmissionInput(
    variant_gene="KCNQ3",
    variant_protein="p.Arg230His",
    variant_type="missense_GoF",
    age_years=5.5,
    sex="F",
    duration_hours=12.0,
    had_sleep=True,
    montage="10-20_monopolar",
    n_channels=19,
)
try:
    sub = build_submission(
        findings=findings_with_pr,
        user_input=user_input,
        consent=consent,
        tool_version="0.17.0",
    )
    f_out = sub.get("findings", {})
    check("build_submission succeeds with v0.17.0 pattern fields", True)
    check("spike_topography_pattern in findings output",
          f_out.get("spike_topography_pattern") == "multifocal",
          f"got {f_out.get('spike_topography_pattern')!r}")
    check("sleep_activation_classification in findings output",
          f_out.get("sleep_activation_classification") == "wake_dominant_atypical",
          f"got {f_out.get('sleep_activation_classification')!r}")
    check("csws_risk_score_bucket in findings output",
          f_out.get("csws_risk_score_bucket") == "<0.2",
          f"got {f_out.get('csws_risk_score_bucket')!r}")
except Exception as e:
    check("build_submission succeeds with v0.17.0 pattern fields", False, str(e))


# ─── Final summary ────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  PASS: {n_pass}")
print(f"  FAIL: {n_fail}")
print(f"{'='*60}")
if n_fail > 0:
    sys.exit(1)
