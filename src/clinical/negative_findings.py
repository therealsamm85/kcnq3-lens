"""Negative findings — what we checked for and DIDN'T find.

In a clinical EEG report, the absence of certain patterns is as
informative as their presence. "No periodic discharges" rules out one
class of acute encephalopathy. "No generalized 3 Hz spike-wave" makes
classical childhood absence epilepsy unlikely. "No focal slowing" rules
out a localized cortical lesion.

This module enumerates the standard "looked-for but not present"
patterns and emits a structured list based on findings. Each item
is phrased neutrally and includes the threshold used.
"""

from __future__ import annotations

from typing import Any


def _get(findings: dict, *path: str) -> Any:
    cur = findings
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def build_negative_findings(findings: dict[str, Any]) -> list[str]:
    """Return list of human-readable 'looked for and not present' statements.

    Each statement explicitly names what was checked and the threshold used.
    """
    out: list[str] = []

    # 1. CSWS criterion not met (only mention if it was checked)
    swi = findings.get("swi") or {}
    if swi and not swi.get("csws_criterion_met"):
        n3 = swi.get("swi_per_stage_pct", {}).get("N3", 0)
        thr = swi.get("csws_threshold_pct", 85)
        out.append(
            f"No CSWS / ESES pattern: N3 spike-wave index is "
            f"{n3:.0f}% (criterion ≥ {thr:.0f}%)."
        )

    # 2. Generalized 3 Hz spike-wave absent
    syn = findings.get("synchrony") or {}
    morph = findings.get("morphology") or {}
    if syn.get("n_events_analyzed", 0) > 0:
        gen_pct = syn.get("generalized_pct", 0)
        complex_pct = morph.get("pct_complex_spike_wave", 0)
        if gen_pct < 20 and complex_pct < 30:
            out.append(
                "No predominant generalized spike-wave pattern: only "
                f"{gen_pct:.0f}% of events are generalized, with "
                f"{complex_pct:.0f}% complex spike-wave morphology — "
                "argues against classical generalized epilepsy syndromes."
            )

    # 3. Severe background slowing absent
    bg = findings.get("background") or {}
    if bg and bg.get("interpretation") == "age_appropriate":
        pdr = bg.get("posterior_dominant_rhythm_hz", 0)
        out.append(
            f"No background slowing: posterior dominant rhythm is "
            f"age-appropriate at {pdr:.1f} Hz."
        )

    # 4. Strong sleep activation absent
    state = findings.get("state_split") or {}
    label = state.get("activation_label", "")
    if state and label in ("none", "mild"):
        af = state.get("activation_factor", 0)
        out.append(
            f"No strong sleep activation: activation factor {af:.1f}× "
            "(threshold for strong activation is ≥ 10×)."
        )

    # 5. Sustained long bursts absent
    bursts = findings.get("bursts") or {}
    if bursts and bursts.get("n_bursts_10s_or_longer", 0) == 0:
        out.append(
            "No sustained rhythmic bursts ≥ 10 seconds detected."
        )

    # 6. Quality grade — flag if EVERYTHING looked OK
    qc = findings.get("quality") or {}
    if qc.get("overall_grade") == "A":
        if qc.get("n_good_channels", 0) >= 17:
            out.append(
                f"No widespread channel-quality problems: "
                f"{qc.get('n_good_channels', 0)} of "
                f"{qc.get('n_total_channels', 0)} channels are good "
                "quality, no quality warnings."
            )

    # 7. Spindle density in normal range
    sp = findings.get("spindles") or {}
    if sp.get("interpretation") == "in":
        density = sp.get("density_per_minute", 0)
        norm = sp.get("age_normative_range", (3, 5))
        out.append(
            f"No spindle reduction: spindle density is age-appropriate "
            f"at {density:.1f}/min (norm {norm[0]}-{norm[1]}/min)."
        )

    return out
