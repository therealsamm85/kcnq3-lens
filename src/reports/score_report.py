"""D2 — SCORE/IFCN-structured report mapping.  [BUILD]

The doctor PDF is bespoke. SCORE (Standardized Computer-based Organized Reporting
of EEG; IFCN/ILAE consensus) defines a structured vocabulary + section layout
clinicians expect. This maps the tool's existing findings onto SCORE-style
sections (background, sleep, interictal epileptiform, ictal, other quantitative,
impression) so a neurologist reads a familiar structure.

BUILD: the SCORE standard is open terminology but there is no open code library
to borrow — this is a transparent mapping from ``findings`` to a structured dict
+ Markdown. It re-words, it does not re-analyze; absent analyses are reported as
"not assessed" rather than invented.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_SECTION_ORDER = [
    "Background activity",
    "Sleep",
    "Interictal epileptiform activity",
    "Ictal findings",
    "Other quantitative findings",
]


@dataclass
class ScoreReport:
    sections: dict[str, list[str]] = field(default_factory=dict)
    impression: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _g(findings: dict, *path):
    cur = findings
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def build_score_report(findings: dict) -> ScoreReport:
    """Map findings onto SCORE/IFCN-structured sections."""
    sections: dict[str, list[str]] = {s: [] for s in _SECTION_ORDER}
    impression: list[str] = []

    # ── Background activity ────────────────────────────────────────────────
    bg = sections["Background activity"]
    pdr = _g(findings, "background", "posterior_dominant_rhythm_hz")
    if pdr is not None:
        line = f"Posterior dominant rhythm: {pdr:g} Hz"
        corr = _g(findings, "background", "pdr_aperiodic_corrected_hz")
        if corr is not None:
            line += f" (1/f-corrected {corr:g} Hz)"
        bg.append(line)
        impression.append(f"PDR {pdr:g} Hz")
    else:
        bg.append("Posterior dominant rhythm: not assessed")
    dar = _g(findings, "background", "delta_alpha_ratio")
    if dar is not None:
        bg.append(f"Delta/alpha ratio: {dar:g}")
    bg.append("Reactivity / symmetry: not assessed by this automated pipeline")

    # ── Sleep ──────────────────────────────────────────────────────────────
    sl = sections["Sleep"]
    eff = _g(findings, "sleep_stages", "sleep_efficiency_pct")
    spin = _g(findings, "spindles", "density_per_minute")
    if eff is not None:
        sl.append(f"Sleep efficiency: {eff:g}%")
    if spin is not None:
        sl.append(f"Spindle density: {spin:g}/min")
    if not sl:
        sl.append("No sleep recorded / not assessed")

    # ── Interictal epileptiform ────────────────────────────────────────────
    ie = sections["Interictal epileptiform activity"]
    spike_rate = _g(findings, "morphology", "events_per_minute")
    if spike_rate is not None:
        ch = _g(findings, "morphology", "channel")
        ie.append(f"Spike rate: {spike_rate:g}/min" + (f" (max {ch})" if ch else ""))
        impression.append(f"interictal spike burden {spike_rate:g}/min")
    sharp = _g(findings, "sharp_spikes", "events_per_minute")
    if sharp is not None:
        ie.append(f"Broadband sharp-spike rate: {sharp:g}/min")
    swi = _g(findings, "swi", "swi_n3_only_pct")
    if swi is not None:
        csws = _g(findings, "swi", "csws_criterion_met")
        tag = ("CSWS criterion MET" if csws is True
               else "below CSWS threshold" if csws is False else "not evaluable")
        ie.append(f"Spike-wave index (N3): {swi:g}% — {tag}")
        impression.append(f"SWI(N3) {swi:g}%")
    spread = _g(findings, "spike_average", "field_spread")
    if spread:
        ie.append(f"Averaged spike field: {spread}")
    if not ie:
        ie.append("No interictal epileptiform activity quantified")

    # ── Ictal ──────────────────────────────────────────────────────────────
    ic = sections["Ictal findings"]
    n_cand = _g(findings, "ictal", "n_candidates")
    if n_cand is not None:
        if n_cand > 0:
            ic.append(f"{n_cand} candidate electrographic seizure(s) flagged "
                      "for human review (screening only — not confirmed).")
            impression.append(f"{n_cand} ictal candidate(s) to review")
        else:
            ic.append("No electrographic seizures flagged by the screener.")
    else:
        ic.append("Ictal screening not run")

    # ── Other quantitative ─────────────────────────────────────────────────
    oq = sections["Other quantitative findings"]
    hfo = _g(findings, "hfo_ripples", "rate_per_min")
    if hfo is not None:
        oq.append(f"HFO/ripple rate: {hfo:g}/min")
    spk_hfo = _g(findings, "hfo_classify", "n_spike_coupled")
    if spk_hfo is not None:
        oq.append(f"Spike-coupled HFOs (spkHFO): {spk_hfo}")
    wpli = _g(findings, "connectivity", "mean_wpli_by_band")
    if isinstance(wpli, dict) and wpli:
        alpha = wpli.get("alpha")
        if alpha is not None:
            oq.append(f"Mean alpha wPLI: {alpha:g}")
    ent = _g(findings, "entropy", "metrics")
    if isinstance(ent, dict) and ent.get("sample_entropy") is not None:
        oq.append(f"Sample entropy: {ent['sample_entropy']:g}")
    if not oq:
        oq.append("No additional quantitative findings")

    notes = [
        "SCORE/IFCN-structured presentation auto-generated from quantitative "
        "findings. It re-words the tool's output into a familiar structure — it "
        "is NOT a substitute for a clinician's SCORE report, and reactivity, "
        "semiology and visual morphology assessment are not automated.",
    ]
    if not impression:
        impression.append("Insufficient quantitative findings for an impression.")
    return ScoreReport(sections=sections, impression=impression, notes=notes)


def render_score_markdown(report: ScoreReport) -> str:
    lines = ["# EEG report (SCORE/IFCN-structured)", ""]
    for sec in _SECTION_ORDER:
        lines.append(f"## {sec}")
        for item in report.sections.get(sec, []):
            lines.append(f"- {item}")
        lines.append("")
    lines.append("## Impression")
    for imp in report.impression:
        lines.append(f"- {imp}")
    lines.append("")
    for n in report.notes:
        lines.append(f"> {n}")
    return "\n".join(lines)
