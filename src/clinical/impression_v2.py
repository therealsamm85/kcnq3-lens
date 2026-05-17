"""Enhanced structured clinical impression — v0.17.0.

Generates a machine-readable ClinicalImpressionV2 dataclass from the full
findings dict including v0.17.0 pattern-recognition outputs.

Key design principles:
- Deterministic rule-based logic (no LLM)
- All assertions are caveated: "consistent with" / "raises concern for"
- KCNQ3-specific logic gated on variant field
- Favourable findings explicitly called out (not just concerns)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClinicalImpressionV2:
    """Structured clinical impression built from pattern-recognition findings."""

    headline: str
    key_findings: list[str]
    differential: list[str]
    favorable_factors: list[str]
    concerning_factors: list[str]
    suggested_follow_up: list[str]
    confidence_overall: str  # "high" / "medium" / "low"
    disclaimer: str = (
        "Research-grade tool, not a medical device. "
        "All findings must be interpreted in full clinical context."
    )


# ─── Sands 2019 KCNQ3 cohort reference ────────────────────────────────────────
# Based on: Sands TT et al. (2019) KCNQ3 gain-of-function epilepsy
# Expected pattern in KCNQ3-GoF: multifocal IEDs, prominent sleep activation,
# CSWS spectrum, markedly reduced spindles.

_SANDS_2019_NOTES = {
    "citation": "Sands 2019 (PMID 31254974)",
    "typical_topography": "multifocal with centrotemporal emphasis",
    "csws_frequency": "CSWS/ESES present in ~60% of cohort",
    "spindle_typical": "markedly reduced spindle density",
    "activation_typical": "moderate–strong sleep activation",
}


def build_impression_v2(
    findings: dict[str, Any],
    pattern_recognition: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ClinicalImpressionV2:
    """Build a structured ClinicalImpressionV2 from findings.

    Parameters
    ----------
    findings : dict
        Output of run_all_analyses().
    pattern_recognition : dict, optional
        Output of summarize_pattern_recognition(). If None, derived fields
        from findings["pattern_recognition"] are attempted.
    metadata : dict, optional
        Contains 'variant', 'age_years', 'patient_label'.

    Returns
    -------
    ClinicalImpressionV2
    """
    meta = metadata or {}
    variant = meta.get("variant", "")
    age_years = meta.get("age_years")

    # Resolve pattern_recognition source
    pr = pattern_recognition
    if pr is None:
        pr = findings.get("pattern_recognition") or {}

    # Extract key values
    topo_type = pr.get("spike_topography_pattern", "")
    sleep_act_class = pr.get("sleep_activation_classification", "")
    csws_risk = float(pr.get("csws_risk_score", 0.0))

    sleep_pr = pr.get("sleep_activation", {})
    act_ratio = float(sleep_pr.get("activation_ratio", 0.0))

    morph_pr = pr.get("morphology_subtypes", {})
    pct_poly = float(morph_pr.get("pct_polyspike", 0.0))
    morph_interp = morph_pr.get("interpretation", "")

    topo_pr = pr.get("topography", {})
    top5 = topo_pr.get("top_5_channels", [])
    asym_idx = float(topo_pr.get("asymmetry_index", 0.0))

    # Existing findings
    swi = findings.get("swi") or {}
    spindles = findings.get("spindles") or {}
    state_split = findings.get("state_split") or {}
    bg = findings.get("background") or {}
    microstates = findings.get("microstates") or {}
    aperiodic = findings.get("aperiodic") or {}

    csws_met = bool(swi.get("csws_criterion_met", False))
    swi_n3 = float((swi.get("swi_per_stage_pct") or {}).get("N3", 0.0))
    spindle_density = float(spindles.get("density_per_minute", 0.0) or 0.0)
    spindle_interp = spindles.get("interpretation", "")
    spindle_norm = spindles.get("age_normative_range") or (0.8, 1.5)
    morph_rate = float((findings.get("morphology") or {}).get("events_per_minute", 0.0) or 0.0)

    # Microstate D coverage
    ms_cov = microstates.get("coverage_pct") or {}
    ms_d_pct = float(ms_cov.get("D", 0.0) or 0.0)

    # Aperiodic chi
    ap_chi_state = (aperiodic.get("chi_by_state") or {})
    ap_wake = ap_chi_state.get("wake") or {}
    ap_chi_wake = float(ap_wake.get("median", 0.0) or 0.0)

    is_kcnq = bool(variant and "KCNQ" in variant.upper())

    key_findings: list[str] = []
    favorable_factors: list[str] = []
    concerning_factors: list[str] = []
    differential: list[str] = []
    suggested_follow_up: list[str] = []

    # ── Key findings ───────────────────────────────────────────────────────────

    if topo_type and topo_type != "indeterminate":
        topo_label = topo_type.replace("_", " ").title()
        key_findings.append(f"Spike topography: {topo_label}")

    if morph_rate > 0:
        rate_label = f"{morph_rate:.1f}/min"
        key_findings.append(f"IED rate: {rate_label}")

    if topo_type == "multifocal" and morph_rate > 20 and is_kcnq:
        key_findings.append(
            f"Multifokale IEDs mit hoher Last ({morph_rate:.0f}/min) — "
            "KCNQ3-GoF-konsistent (Sands 2019)"
        )
    elif topo_type == "multifocal" and morph_rate > 20:
        key_findings.append(
            f"Multifokale IEDs mit hoher Last ({morph_rate:.0f}/min)"
        )

    if csws_met:
        key_findings.append(
            f"CSWS criterion MET (N3 SWI {swi_n3:.0f}% — Tassinari threshold ≥85%)"
        )
    elif swi_n3 > 0:
        key_findings.append(f"N3 spike-wave index: {swi_n3:.0f}% (criterion not met)")

    if sleep_act_class:
        act_label = sleep_act_class.replace("_", " ").title()
        key_findings.append(
            f"Sleep activation: {act_label} "
            f"(ratio {act_ratio:.2f})"
        )

    if spindle_interp == "below":
        key_findings.append(
            f"Sleep spindle density markedly reduced: {spindle_density:.2f}/min "
            f"(normative {spindle_norm[0]}–{spindle_norm[1]}/min)"
        )

    if pct_poly >= 20.0:
        key_findings.append(
            f"Polyspike burden: {pct_poly:.0f}% of events — encephalopathic marker"
        )

    # ── Favorable factors ──────────────────────────────────────────────────────

    if sleep_act_class == "wake_dominant_atypical":
        favorable_factors.append(
            f"Sleep-Aktivierung INVERS (ratio {act_ratio:.2f} < 0.7) — "
            "atypisch günstig für KCNQ3-Spektrum; ESES-Risiko niedrig"
        )

    if not csws_met:
        favorable_factors.append("CSWS Tassinari-Kriterium NICHT erfüllt")

    if csws_risk < 0.3:
        favorable_factors.append(
            f"CSWS-Risiko-Score niedrig ({csws_risk:.2f}/1.0)"
        )

    if sleep_act_class in ("no_activation", "mild_activation", "wake_dominant_atypical"):
        favorable_factors.append(
            "Keine signifikante Schlaf-Aktivierung der epileptiformen Aktivität"
        )

    if topo_type == "centro_temporal_BCECTS":
        favorable_factors.append(
            "Centro-temporale Topographie vereinbar mit BCECTS-Spektrum — "
            "günstige Langzeitprognose"
        )

    # ── Concerning factors ─────────────────────────────────────────────────────

    if spindle_interp == "below":
        concerning_factors.append(
            f"Dramatische Spindel-Reduktion ({spindle_density:.2f}/min) — "
            "KCNQ3-mechanistisches Korrelat; Gedächtniskonsolidierung beeinträchtigt"
        )

    if ms_d_pct > 40.0:
        concerning_factors.append(
            f"Microstate D-Dominanz ({ms_d_pct:.0f}%) — "
            "Aufmerksamkeits/Salienz-Netzwerk-Imbalanz"
        )

    if ap_chi_wake > 0 and ap_chi_wake < 1.5:
        concerning_factors.append(
            f"Aperiodic Exponent χ={ap_chi_wake:.2f} (Wake) — "
            "erhöhte kortikale Exzitabilität (KCNQ3-konsistent)"
        )

    if pct_poly >= 20.0:
        concerning_factors.append(
            f"Polyspike-Rate {pct_poly:.0f}% — encephalopathischer Marker"
        )

    if morph_rate > 30:
        concerning_factors.append(
            f"Hohe IED-Last ({morph_rate:.0f}/min) — "
            "kognitive Wirkung möglich (Tassinari 2005)"
        )

    if csws_met:
        concerning_factors.append(
            "CSWS-Kriterium erfüllt — Behandlungseskalation erwägen "
            "(Sulthiame, Levetiracetam)"
        )

    # ── KCNQ3-specific ─────────────────────────────────────────────────────────

    if is_kcnq:
        key_findings.append(
            f"KCNQ3-Spektrum ({variant}): Befundkonstellation wird gegen "
            "Sands 2019 Kohorte eingeordnet"
        )

        # Sands 2019 comparison
        if sleep_act_class == "wake_dominant_atypical":
            favorable_factors.append(
                f"Sands 2019: CSWS/ESES in ~60% der KCNQ3-GoF-Kohorte — "
                f"dieser Patient zeigt KEINEN Schlafaktivierungs-Phänotyp "
                f"(atypisch günstig)"
            )
        elif sleep_act_class == "strong_activation_eses_risk":
            concerning_factors.append(
                "Sands 2019: Schlafaktivierungsmuster passend zum KCNQ3-GoF-ESES-Phänotyp"
            )

    # ── Differential ──────────────────────────────────────────────────────────

    if topo_type == "centro_temporal_BCECTS":
        differential.append("BCECTS / SeLECTS (Self-limited epilepsy with centrotemporal spikes)")
        differential.append("KCNQ3-GoF mit atypisch günstiger Verteilung")

    if topo_type == "multifocal":
        differential.append("KCNQ3-Gain-of-Function Epilepsie (Sands 2019)")
        differential.append("Dravet-Spektrum (bei zusätzlichen klinischen Zeichen)")
        differential.append("Lenox-Gastaut-Spektrum (bei diffuser Verlangsamung)")

    if csws_met or sleep_act_class == "strong_activation_eses_risk":
        differential.append("CSWS / ESES (Continuous Spike-Wave during Slow Sleep)")
        differential.append("Atypische Absence-Epilepsie")

    if not differential:
        differential.append("Fokale Epilepsie unklarer Ätiologie")

    # ── Suggested follow-up ────────────────────────────────────────────────────

    suggested_follow_up.append("Kontroll-EEG in 3–6 Monaten mit Fokus auf Top-5-Biomarker")
    suggested_follow_up.append(
        "Top-5-Biomarker zu wiederholen: "
        "(1) Spindel-Dichte, (2) N3-SWI, (3) Aperiodic χ, "
        "(4) Microstate D Coverage, (5) Topographic Spike Rate"
    )

    if spindle_interp == "below":
        suggested_follow_up.append(
            "Spindel-Dichte als primären Behandlungseffekt-Marker verwenden"
        )

    if is_kcnq:
        suggested_follow_up.append(
            f"KCNQ3-Registry-Eintrag erwägen (Einwilligung erforderlich)"
        )

    if csws_risk > 0.5:
        suggested_follow_up.append(
            "Formelles Schlaf-EEG mit PSG-Staging für SWI-Berechnung empfohlen"
        )

    # ── Headline ──────────────────────────────────────────────────────────────

    headline = _build_headline(
        topo_type=topo_type,
        sleep_act_class=sleep_act_class,
        csws_met=csws_met,
        csws_risk=csws_risk,
        is_kcnq=is_kcnq,
        variant=variant,
        morph_rate=morph_rate,
    )

    # ── Overall confidence ────────────────────────────────────────────────────

    n_findings = len(key_findings)
    if n_findings >= 3 and topo_type not in ("indeterminate", ""):
        confidence = "high"
    elif n_findings >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    return ClinicalImpressionV2(
        headline=headline,
        key_findings=key_findings,
        differential=differential,
        favorable_factors=favorable_factors,
        concerning_factors=concerning_factors,
        suggested_follow_up=suggested_follow_up,
        confidence_overall=confidence,
    )


def _build_headline(
    *,
    topo_type: str,
    sleep_act_class: str,
    csws_met: bool,
    csws_risk: float,
    is_kcnq: bool,
    variant: str,
    morph_rate: float,
) -> str:
    """Construct a single-line clinical headline."""
    parts: list[str] = []

    if is_kcnq:
        parts.append(f"Atypical KCNQ3-spectrum ({variant})")
    else:
        parts.append("Epileptiform EEG pattern")

    if topo_type and topo_type not in ("indeterminate", ""):
        topo_label = topo_type.replace("_", " ").replace("BCECTS", "BCECTS-spectrum")
        parts.append(topo_label)

    if morph_rate > 20:
        parts.append(f"high IED burden ({morph_rate:.0f}/min)")

    if sleep_act_class and sleep_act_class != "indeterminate":
        if sleep_act_class == "wake_dominant_atypical":
            parts.append("favorable inverse sleep activation")
        elif sleep_act_class == "no_activation":
            parts.append("no sleep activation (favorable)")
        elif sleep_act_class == "strong_activation_eses_risk":
            parts.append("strong sleep activation — ESES risk")
        else:
            parts.append(sleep_act_class.replace("_", " "))

    if not csws_met and csws_risk < 0.3:
        parts.append("CSWS criterion NOT met (favorable)")

    return ": ".join(parts[:3]) if len(parts) > 1 else parts[0] if parts else "EEG pattern recognized"


def summarize_impression_v2(impression: ClinicalImpressionV2) -> dict:
    """Return JSON-serializable dict of the impression."""
    return {
        "headline": impression.headline,
        "key_findings": impression.key_findings,
        "differential": impression.differential,
        "favorable_factors": impression.favorable_factors,
        "concerning_factors": impression.concerning_factors,
        "suggested_follow_up": impression.suggested_follow_up,
        "confidence_overall": impression.confidence_overall,
        "disclaimer": impression.disclaimer,
    }
