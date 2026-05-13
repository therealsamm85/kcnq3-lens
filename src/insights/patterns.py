"""Clinical pattern matching — surfaces "this combination of findings is
consistent with known clinical pattern X" with explicit confidence.

These are NEVER diagnoses. They are pattern-recognition prompts for a
conversation with a doctor. Each pattern has:

- A name (the syndrome / spectrum)
- A list of criteria the findings must satisfy, each scored 0–1
- A confidence threshold below which we don't show the pattern at all
- A textual explanation of what it means and what to ask the doctor

Multiple patterns can match simultaneously. The output is ranked by
confidence, but we never tell the user "your child has X."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class PatternCriterion:
    name: str
    description: str
    check: Callable[[dict], bool]
    weight: float = 1.0
    required: bool = False  # if True, pattern is suppressed when this criterion fails


@dataclass
class PatternMatch:
    pattern_id: str
    pattern_name: str
    confidence: float           # 0.0 – 1.0
    confidence_label: str       # "weak" | "moderate" | "strong"
    criteria_met: list[str]
    criteria_unmet: list[str]
    explanation: str
    questions_for_doctor: list[str]


# ─── Pattern definitions ────────────────────────────────────────────────────

def _get(findings: dict, *path: str) -> Any:
    """Safely access nested dict; returns None if any key missing."""
    cur = findings
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


PATTERNS = [
    {
        "id": "kcnq_spectrum",
        "name": "KCNQ-spectrum / multi-regional sleep-activated pattern",
        "explanation": (
            "Multi-regional epileptiform activity that peaks during sleep, "
            "with reduced sleep spindles and slow background. This combination "
            "is characteristic of KCNQ-channel-related developmental and "
            "epileptic disorders (especially KCNQ3, also KCNQ2). The "
            "channelopathy disrupts both the cortical excitability (causing "
            "the spikes) and the thalamocortical sleep machinery (causing "
            "the low spindle density and slow background)."
        ),
        "criteria": [
            PatternCriterion(
                "multi_regional",
                "Activity spread across multiple non-adjacent regions",
                lambda f: sum(
                    1 for c in _get(f, "topography", "all_channels") or []
                    if c.get("median", 0) > 5.5
                ) >= 4,
                required=True,  # gate: must have multi-regional activity
            ),
            PatternCriterion(
                "low_spindle",
                "Sleep spindle density below age-typical range",
                lambda f: _get(f, "spindles", "interpretation") == "below",
            ),
            PatternCriterion(
                "slow_background",
                "Background rhythm slower than age-typical",
                lambda f: _get(f, "background", "interpretation") in
                          ("severely_slow", "mildly_slow"),
            ),
            PatternCriterion(
                "sustained_bursts",
                "Multiple sustained rhythmic bursts ≥ 5 seconds",
                lambda f: (_get(f, "bursts", "n_bursts_5s_or_longer") or 0) >= 5,
            ),
        ],
        "questions": [
            "Was genetic testing done for KCNQ2 / KCNQ3 variants?",
            "If a variant is known, is it loss-of-function or gain-of-function? "
            "(This determines whether channel-opener or channel-blocker drugs "
            "are appropriate.)",
            "Should the family consider contacting RIKEE (rikee.org) for "
            "variant-specific outcome data?",
            "Would a sleep-quality–focused therapeutic strategy (sleep "
            "hygiene, spindle-augmenting interventions) help offset the "
            "memory-consolidation impact?",
        ],
    },

    {
        "id": "csws_ses",
        "name": "CSWS / ESES spectrum (continuous spike-wave during sleep)",
        "explanation": (
            "Continuous Spike-Wave during Sleep (CSWS) or Electrical Status "
            "Epilepticus during Sleep (ESES) is defined by near-continuous "
            "spike-wave activity occupying ≥85% of slow-wave sleep, often "
            "associated with cognitive regression. Even when the formal SWI "
            "criterion is not fully met, sustained bursts, complex spike-wave "
            "morphology, and prominent sleep activation suggest the spectrum "
            "and warrant urgent treatment attention."
        ),
        "criteria": [
            PatternCriterion(
                "sustained_long_bursts",
                "Sustained rhythmic bursts ≥10 s present",
                lambda f: (_get(f, "bursts", "n_bursts_10s_or_longer") or 0) >= 3,
                required=True,  # gate: CSWS needs sustained bursts
            ),
            PatternCriterion(
                "complex_morphology",
                "Predominantly complex spike-wave morphology",
                lambda f: (_get(f, "morphology", "pct_complex_spike_wave") or 0) >= 30,
            ),
            PatternCriterion(
                "sleep_activation",
                "Spike burden peaks during sleep",
                lambda f: (_get(f, "time_of_night", "peak_count_per_min") or 0) > 20,
            ),
            PatternCriterion(
                "low_spindle",
                "Spindle density below age-typical (sleep architecture disrupted)",
                lambda f: _get(f, "spindles", "interpretation") == "below",
            ),
        ],
        "questions": [
            "Has formal SWI been calculated specifically over slow-wave sleep "
            "(N3), not just whole-night average?",
            "Are antiepileptic drugs known to suppress sleep-activated patterns "
            "(sulthiame, levetiracetam, benzodiazepines) being considered?",
            "Is cognitive testing scheduled to track regression over time?",
            "Should steroids / IVIG be discussed for CSWS-spectrum cases not "
            "responding to first-line AEDs?",
        ],
    },

    {
        "id": "bects_like",
        "name": "BECTS / Rolandic spectrum (centro-temporal pattern)",
        "explanation": (
            "Benign Epilepsy with Centro-Temporal Spikes (BECTS / Rolandic) "
            "shows simple spike morphology with predominant activity over "
            "central and temporal regions, typically in school-age children. "
            "The classical form has good prognosis with spontaneous remission "
            "by puberty. Atypical variants and overlap with CSWS exist."
        ),
        "criteria": [
            PatternCriterion(
                "centrotemporal_focus",
                "Top activity over central or temporal regions",
                lambda f: any(
                    c.get("name") in ("C3", "C4", "T3", "T4", "T5", "T6") and c.get("median", 0) > 5.5
                    for c in (_get(f, "topography", "all_channels") or [])[:5]
                ),
                required=True,  # gate: BECTS needs centro-temporal focus, not nothing
            ),
            PatternCriterion(
                "simple_morphology",
                "Predominantly simple spike morphology",
                lambda f: (_get(f, "morphology", "pct_simple_spikes") or 0) >= 50,
            ),
            PatternCriterion(
                "limited_complex",
                "Complex spike-wave morphology not dominant",
                lambda f: (_get(f, "morphology", "pct_complex_spike_wave") or 0) < 30
                          and (_get(f, "morphology", "pct_simple_spikes") or 0) > 0,
                # also require non-zero simple spikes — prevents pass on missing morphology
            ),
        ],
        "questions": [
            "Does the EEG pattern match classical Rolandic or an atypical variant?",
            "Is the family history consistent with benign focal epilepsy of childhood?",
            "Has the spike rate been compared between awake and sleep recordings?",
        ],
    },

    {
        "id": "speech_motor_pattern",
        "name": "Speech-motor / SMA-region predominance",
        "explanation": (
            "Epileptiform activity concentrated over the supplementary motor "
            "area (Cz/Pz midline). This region programs the motor sequencing "
            "for speech production. When the activity is focal to this "
            "region, the clinical picture often shows preserved language "
            "comprehension with disproportionately impaired speech "
            "production — a profile compatible with Childhood Apraxia of "
            "Speech (CAS)."
        ),
        "criteria": [
            PatternCriterion(
                "midline_central_focus",
                "Highest activity on midline central (Cz/Pz) channels",
                lambda f: any(
                    c.get("name") in ("Cz", "Pz") and c.get("median", 0) > 5.5
                    for c in (_get(f, "topography", "all_channels") or [])[:3]
                ),
                required=True,  # gate: this pattern is defined by Cz/Pz dominance
            ),
            PatternCriterion(
                "language_areas_quieter",
                "Left temporal language areas relatively spared",
                lambda f: all(
                    c.get("median", 0) < 6
                    for c in (_get(f, "topography", "all_channels") or [])
                    if c.get("name") in ("T3", "T5", "F7")
                ) and len([
                    c for c in (_get(f, "topography", "all_channels") or [])
                    if c.get("name") in ("T3", "T5", "F7")
                ]) > 0,  # require these channels to actually be present
            ),
        ],
        "questions": [
            "Has a formal Childhood Apraxia of Speech (CAS) evaluation been done?",
            "Is PROMPT (Prompts for Restructuring Oral Muscular Phonetic Targets) "
            "speech therapy available in the area? Standard speech therapy is "
            "not designed to remediate motor-speech-planning deficits.",
            "Are concurrent fine-motor and tongue-protrusion difficulties present? "
            "(They share the same network.)",
        ],
    },
]


# ─── Public API ──────────────────────────────────────────────────────────────

def _confidence_label(score: float) -> str:
    if score >= 0.75:
        return "strong"
    if score >= 0.5:
        return "moderate"
    if score >= 0.3:
        return "weak"
    return "below_threshold"


def match_patterns(findings: dict) -> list[PatternMatch]:
    """Score every defined pattern against the findings.

    Patterns are scored only if all their `required` criteria are met
    (these are gating criteria — without them, the pattern doesn't apply
    regardless of how many supporting criteria match). Among remaining
    patterns, matches with confidence ≥ 0.3 are returned, sorted descending.

    The gating mechanism prevents false-positive matches when findings
    are partially missing or when a child's EEG is normal (where some
    "less than" criteria like `pct_complex < 30` would otherwise trivially
    pass on missing data).
    """
    matches: list[PatternMatch] = []
    for pattern in PATTERNS:
        criteria = pattern["criteria"]
        if not criteria:
            continue

        # Pre-check all `required` (gating) criteria. If any fails, skip.
        gating_passed = True
        for crit in criteria:
            if not crit.required:
                continue
            try:
                ok = bool(crit.check(findings))
            except Exception:
                ok = False
            if not ok:
                gating_passed = False
                break
        if not gating_passed:
            continue

        met = []
        unmet = []
        total_weight = 0.0
        score = 0.0
        for crit in criteria:
            total_weight += crit.weight
            try:
                ok = bool(crit.check(findings))
            except Exception:
                ok = False
            if ok:
                score += crit.weight
                met.append(crit.description)
            else:
                unmet.append(crit.description)
        confidence = score / total_weight if total_weight > 0 else 0.0
        label = _confidence_label(confidence)
        if label == "below_threshold":
            continue
        matches.append(PatternMatch(
            pattern_id=pattern["id"],
            pattern_name=pattern["name"],
            confidence=confidence,
            confidence_label=label,
            criteria_met=met,
            criteria_unmet=unmet,
            explanation=pattern["explanation"],
            questions_for_doctor=pattern["questions"],
        ))

    matches.sort(key=lambda m: -m.confidence)
    return matches


def summarize_patterns(matches: list[PatternMatch]) -> list[dict]:
    return [
        {
            "id": m.pattern_id,
            "name": m.pattern_name,
            "confidence": round(m.confidence, 2),
            "confidence_label": m.confidence_label,
            "criteria_met": m.criteria_met,
            "criteria_unmet": m.criteria_unmet,
            "explanation": m.explanation,
            "questions_for_doctor": m.questions_for_doctor,
        }
        for m in matches
    ]
