"""Build a structured clinical Impression from findings.

Generates the one-paragraph clinical summary that goes at the TOP of a
neurologist-oriented report — the section a busy clinician reads first
before scanning the rest.

The impression is rule-based (deterministic), not LLM-generated:
- Pulls the top-3 most clinically relevant findings
- Phrases them in standard neurology vocabulary
- Includes a bottom-line interpretation if patterns matched
- Does NOT diagnose — uses "consistent with" / "compatible with" / "raises
  concern for" phrasing only

The structured order mirrors how clinicians naturally read a report:

1. IMPRESSION (one paragraph, top-of-page)
2. HEADLINE FINDINGS (bulleted, structured by category)
3. METHODS (small print at the back)
4. RECOMMENDATIONS / QUESTIONS for the family
"""

from __future__ import annotations

from typing import Any


def build_impression(findings: dict[str, Any],
                     patterns: list[dict] | None = None,
                     metadata: dict | None = None) -> str:
    """Build the one-paragraph clinical Impression.

    Returns plain text suitable for the top of the PDF report.
    """
    sentences: list[str] = []

    # 1. Quality preface (if grade is poor, flag it first)
    qc = findings.get("quality") or {}
    grade = qc.get("overall_grade", "")
    if grade in ("C", "D"):
        sentences.append(
            f"Recording quality is grade {grade} ({qc.get('pct_usable', 0):.0f}% "
            "usable epochs); interpretation should account for this."
        )

    # 2. Background characterization
    bg = findings.get("background") or {}
    bg_interp = bg.get("interpretation")
    pdr = bg.get("posterior_dominant_rhythm_hz")
    if bg_interp == "severely_slow":
        sentences.append(
            f"Background activity is severely slow "
            f"(posterior dominant rhythm {pdr:.1f} Hz)."
        )
    elif bg_interp == "mildly_slow":
        sentences.append(
            f"Background activity is mildly slow (PDR {pdr:.1f} Hz)."
        )

    # 3. Topographic summary
    topo = findings.get("topography") or {}
    top_chs = topo.get("top_channels", [])[:3]
    if top_chs:
        names = ", ".join(c.get("name", "") for c in top_chs)
        sentences.append(
            f"Multi-regional epileptiform activity with maximum involvement "
            f"of {names}."
        )

    # 4. SWI / CSWS
    swi = findings.get("swi") or {}
    if swi.get("csws_criterion_met"):
        sentences.append(
            f"Spike-wave index in N3 sleep is {swi.get('swi_n3_only_pct', 0):.0f}%, "
            "meeting the Tassinari criterion for CSWS / ESES."
        )
    elif swi.get("swi_nrem_combined_pct", 0) > 30:
        sentences.append(
            f"Substantial NREM spike-wave activity "
            f"(NREM SWI {swi.get('swi_nrem_combined_pct', 0):.0f}%, "
            f"N3 SWI {swi.get('swi_n3_only_pct', 0):.0f}%)."
        )

    # 5. State split
    state = findings.get("state_split") or {}
    label = state.get("activation_label", "")
    if label in ("moderate", "strong"):
        sentences.append(
            f"Sleep activation factor {state.get('activation_factor', 0):.1f}× "
            f"({label}; NREM rate {state.get('nrem_rate_per_min', 0):.1f}/min vs "
            f"wake rate {state.get('wake_rate_per_min', 0):.1f}/min)."
        )

    # 6. Sleep architecture concern
    sp = findings.get("spindles") or {}
    if sp.get("interpretation") == "below":
        sentences.append(
            f"Sleep spindle density at {sp.get('channel', 'Cz')} is markedly "
            f"reduced ({sp.get('density_per_minute', 0):.2f}/min vs age-typical "
            f"{sp.get('age_normative_range', (3, 5))[0]}–"
            f"{sp.get('age_normative_range', (3, 5))[1]}/min)."
        )

    # 7. Synchrony pattern
    syn = findings.get("synchrony") or {}
    dom = syn.get("dominant_pattern", "")
    if dom and dom != "no_events":
        readable = dom.replace("_", " ")
        sentences.append(f"Dominant spread pattern is {readable}.")

    # 8. Pattern match interpretation (carefully phrased)
    if patterns:
        strong_matches = [p for p in patterns if p.get("confidence_label") == "strong"]
        if strong_matches:
            names = ", ".join(p.get("name", "") for p in strong_matches[:2])
            sentences.append(f"Findings are consistent with: {names}.")

    # 9. Variant-aware closing line
    variant = (metadata or {}).get("variant")
    if variant and ("KCNQ" in variant.upper()):
        sentences.append(
            f"In the context of {variant}, the combination of multi-regional "
            "discharges, low spindle density, and slow background is "
            "characteristic of the spectrum disorder."
        )

    if not sentences:
        return ("EEG analysis completed. No prominent quantitative abnormalities "
                "detected at default thresholds.")

    return " ".join(sentences)


def build_recommendations(findings: dict[str, Any],
                          patterns: list[dict] | None = None) -> list[str]:
    """Build clinician-oriented recommendations + follow-up suggestions."""
    recs: list[str] = []

    # If CSWS criterion is met, very specific recommendation
    swi = findings.get("swi") or {}
    if swi.get("csws_criterion_met"):
        recs.append(
            "CSWS criterion met: consider sleep EEG review with formal "
            "stage scoring, and discuss treatment escalation given evidence "
            "of continuous spike-wave during sleep."
        )

    # Strong sleep activation
    state = findings.get("state_split") or {}
    if state.get("activation_label") == "strong":
        recs.append(
            "Strong sleep activation (>10× NREM:wake spike ratio): "
            "AEDs known to suppress sleep-activated discharges (sulthiame, "
            "levetiracetam, benzodiazepines) may be considered."
        )

    # Low spindle density
    sp = findings.get("spindles") or {}
    if sp.get("interpretation") == "below":
        recs.append(
            "Reduced sleep spindle density: consider that overnight "
            "memory consolidation may be impaired even if daytime EEG appears "
            "less concerning. Sleep-quality–focused interventions may help."
        )

    # Background slowing
    bg = findings.get("background") or {}
    if bg.get("interpretation") in ("severely_slow", "mildly_slow"):
        recs.append(
            "Background slowing: independent clinical confirmation of "
            "posterior dominant rhythm during awake-alert recording recommended."
        )

    # Pull pattern-specific questions
    if patterns:
        for p in patterns[:2]:  # top 2 patterns
            for q in (p.get("questions_for_doctor") or [])[:2]:
                if q not in recs:
                    recs.append(q)

    # Default if nothing applies
    if not recs:
        recs.append(
            "No specific clinical actions surfaced by the quantitative "
            "analysis. Continue routine clinical follow-up as planned."
        )

    return recs
