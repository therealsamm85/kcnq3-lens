"""Synthesize anatomy + patterns + cross-modal observations into a
structured family-readable narrative.

The narrative is generated from rule-based logic (deterministic, auditable),
not from an LLM. The optional LLM interpretation in the AI section still
exists; this module produces the rule-based structured output that can be
included in PDFs and shown in the UI without any external API call.
"""

from __future__ import annotations

from typing import Any

from .anatomical import analyze_topography, summarize_anatomy
from .patterns import match_patterns, summarize_patterns


def _cross_modal_observations(findings: dict) -> list[str]:
    """Look for combinations of findings that imply something more than each
    finding alone. Each observation is a complete plain-language sentence."""
    obs: list[str] = []

    spindles = findings.get("spindles") or {}
    bursts = findings.get("bursts") or {}
    background = findings.get("background") or {}
    morphology = findings.get("morphology") or {}
    time_of_night = findings.get("time_of_night") or {}
    topography = findings.get("topography") or {}

    # 1. Low spindles + high burst burden → memory consolidation impact
    if (spindles.get("interpretation") == "below"
            and bursts.get("n_bursts_5s_or_longer", 0) >= 5):
        obs.append(
            "**Low sleep spindles combined with sustained nighttime bursts** "
            "suggests that the brain's overnight memory-consolidation system "
            "is being disrupted. New learning (words, motor skills) during "
            "the day may not be retained as efficiently as in a typical "
            "child. This is a mechanism worth raising with the doctor — it "
            "explains why some children stagnate or regress when EEG looks "
            "active during sleep, even without clinical seizures."
        )

    # 2. Sleep-cycle peak — first NREM dominance
    peak_h = time_of_night.get("peak_bin_hours", 0)
    total_h = time_of_night.get("total_hours", 0)
    if 0.5 < peak_h < 4 and total_h > 4:
        obs.append(
            "**The peak in spike activity falls within the first NREM "
            "sleep cycle.** This is the classical sleep-activation signature "
            "(CSWS pattern), and it's the part of the night where deep "
            "slow-wave sleep is most concentrated. Treatments that suppress "
            "sleep-activated patterns target exactly this window."
        )

    # 3. Slow background + low spindles → developmental severity marker
    if (background.get("interpretation") in ("severely_slow", "mildly_slow")
            and spindles.get("interpretation") == "below"):
        obs.append(
            "**Background slowing combined with reduced spindles** points "
            "beyond focal epileptiform activity to a broader thalamocortical "
            "network disruption. This pattern often correlates with "
            "developmental delay severity and can persist even after spikes "
            "are suppressed by medication — addressing the underlying "
            "circuit health (sleep quality, mitochondrial support) becomes "
            "important alongside spike suppression."
        )

    # 4. Complex morphology + multi-regional topography
    pct_complex = morphology.get("pct_complex_spike_wave", 0)
    multi_regional = sum(
        1 for c in topography.get("all_channels", [])
        if c.get("median", 0) > 5
    ) >= 4
    if pct_complex >= 30 and multi_regional:
        obs.append(
            "**Complex spike-wave morphology spread across multiple regions** "
            "is more concerning than simple-spike single-focus patterns. The "
            "broader the spatial spread and the more complex the wave form, "
            "the more likely there are cognitive consequences that warrant "
            "active treatment — even in the absence of overt seizures."
        )

    # 5. Cz/Pz dominance + intact-comprehension speech profile cue
    top_chs = sorted(topography.get("all_channels", []),
                     key=lambda c: -c.get("median", 0))[:3]
    top_names = [c.get("name") for c in top_chs]
    if any(n in ("Cz", "Pz") for n in top_names):
        obs.append(
            "**The strongest activity sits over the midline central-parietal "
            "region (SMA / pre-SMA).** This brain area programs speech "
            "motor sequences — distinct from speech comprehension, which "
            "lives in left temporal regions. Children with this topography "
            "often understand fluently but struggle to produce speech (the "
            "profile of Childhood Apraxia of Speech). This is the single "
            "highest-leverage observation for therapy planning."
        )

    # 6. Quality caveat
    qc = findings.get("quality") or {}
    if qc.get("overall_grade") in ("C", "D"):
        obs.append(
            f"**Recording quality is grade {qc.get('overall_grade')}.** "
            "Some metrics may be affected by artifact. Treat the numerical "
            "findings as approximate and prioritize discussion of the "
            "qualitative patterns over exact rates."
        )

    return obs


def build_narrative(findings: dict) -> dict[str, Any]:
    """Build the full Insights output: anatomy + patterns + cross-modal.

    Returns a dict that can be serialized to JSON or rendered in the UI.
    """
    topo_findings = findings.get("topography") or {}
    anat = analyze_topography(topo_findings)
    patterns = match_patterns(findings)
    cross_modal = _cross_modal_observations(findings)

    return {
        "anatomy": summarize_anatomy(anat),
        "patterns": summarize_patterns(patterns),
        "cross_modal_observations": cross_modal,
    }
