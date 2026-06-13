"""Treatment-response dashboard — did an intervention move the biomarkers?

Why this exists
---------------
A clinician changing a child's medication wants one question answered: did the
EEG get better after the change? This module anchors every stored EEG biomarker
to the medication-change events recorded in the development diary, and for each
intervention reports the before→after change of each biomarker, with a clinical
direction (improved / worsened / no clear change) derived from the shared
polarity map.

What it deliberately does NOT do
--------------------------------
It does not claim a treatment *caused* a change. Each comparison is a single
recording before vs a single recording after — it cannot separate a drug effect
from normal maturation, a different sleep state on the two nights, or ordinary
measurement variability. Every intervention carries those caveats explicitly,
and maturation-confounded metrics (PDR, spindles) are flagged. This is a
structured way to *look*, not a significance test.

It reads only already-stored findings (StoredEntry) and diary entries — no EEG
is re-read here, so it is cheap and runs anywhere the longitudinal DB is loaded.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field, asdict

from .storage import StoredEntry
from .diary import DiaryEntry
from .trends import get_metric_series
from .time_align import parse_date, split_before_after, days_between
from .metric_polarity import direction_label, polarity_of, CONFOUNDED_BY_MATURATION

# The biomarkers worth foregrounding for this case. Override via `metrics=`.
HEADLINE_METRICS: list[str] = [
    "spike_rate_per_min",
    "pdr_hz",
    "swi_n3_pct",
    "delta_alpha_ratio",
    "spindle_density_per_min",
    "sleep_efficiency_pct",
]


@dataclass
class MetricChange:
    metric: str
    baseline: float | None
    baseline_date: str | None
    followup: float | None
    followup_date: str | None
    delta: float | None
    pct_change: float | None
    direction: str                 # improved | worsened | no_clear_change | ambiguous | not_evaluable
    gap_days: int | None
    maturation_confounded: bool


@dataclass
class InterventionEffect:
    date: str
    description: str
    metric_changes: list[MetricChange] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class TreatmentResponse:
    interventions: list[InterventionEffect] = field(default_factory=list)
    n_recordings: int = 0
    n_interventions: int = 0
    notes: list[str] = field(default_factory=list)


def compute_treatment_response(
    entries: list[StoredEntry],
    diary: list[DiaryEntry],
    metrics: list[str] | None = None,
    change_tolerance_pct: float = 10.0,
    min_gap_days: int = 14,
) -> TreatmentResponse:
    """Build before/after biomarker changes for every medication-change event.

    Parameters
    ----------
    entries : stored recordings (with findings) over time.
    diary : diary entries; those with a non-empty ``medication_change`` define
        the interventions.
    metrics : biomarker labels (trends.METRICS keys). Defaults to HEADLINE_METRICS.
    change_tolerance_pct : a relative dead-band — a |percent change| below this
        is reported as "no_clear_change" rather than improved/worsened, so noise
        isn't dressed up as a treatment effect.
    min_gap_days : interventions whose nearest before/after recordings are closer
        together than this get a low-separation caveat.
    """
    metrics = metrics or HEADLINE_METRICS
    resp = TreatmentResponse(
        n_recordings=len(entries),
        n_interventions=0,
    )

    # Pre-build each metric's dated series once (reused across interventions).
    metric_series: dict[str, list[tuple[datetime.date, float]]] = {}
    for m in metrics:
        dates, vals = get_metric_series(entries, m)
        pairs: list[tuple[datetime.date, float]] = []
        for ds, v in zip(dates, vals):
            d = parse_date(ds)
            fv = float(v)
            # Drop non-finite findings (NaN/inf): get_metric_series only filters
            # None, so a garbage value would otherwise reach direction_label and
            # be rendered as a false "worsened" — a wrong clinical direction.
            if d is not None and math.isfinite(fv):
                pairs.append((d, fv))
        metric_series[m] = pairs

    # Collect medication-change interventions from the diary.
    interventions: list[tuple[datetime.date, str]] = []
    for e in diary:
        change = (e.medication_change or "").strip()
        d = parse_date(e.date)
        if change and d is not None:
            interventions.append((d, change))
    interventions.sort(key=lambda t: t[0])
    resp.n_interventions = len(interventions)

    if not interventions:
        resp.notes.append(
            "No medication changes recorded in the diary — nothing to anchor a "
            "treatment-response comparison to. Add a diary entry with a "
            "medication change to populate this dashboard."
        )
        return resp

    for ev_date, desc in interventions:
        eff = InterventionEffect(date=ev_date.isoformat(), description=desc)
        any_evaluable = False
        for m in metrics:
            before, after = split_before_after(metric_series.get(m, []), ev_date)
            if before is None or after is None:
                eff.metric_changes.append(MetricChange(
                    metric=m, baseline=None, baseline_date=None,
                    followup=None, followup_date=None, delta=None,
                    pct_change=None, direction="not_evaluable", gap_days=None,
                    maturation_confounded=m in CONFOUNDED_BY_MATURATION,
                ))
                continue
            any_evaluable = True
            (bd, bv), (ad, av) = before, after
            delta = av - bv
            pct = (100.0 * delta / abs(bv)) if bv != 0 else None
            # Clinical direction, with a relative dead-band.
            raw_dir = direction_label(m, delta, tol=0.0)
            if (raw_dir in ("improved", "worsened") and pct is not None
                    and abs(pct) < change_tolerance_pct):
                direction = "no_clear_change"
            else:
                direction = raw_dir
            gap = days_between(bd, ad)
            eff.metric_changes.append(MetricChange(
                metric=m,
                baseline=round(bv, 3), baseline_date=bd.isoformat(),
                followup=round(av, 3), followup_date=ad.isoformat(),
                delta=round(delta, 3),
                pct_change=(round(pct, 1) if pct is not None else None),
                direction=direction, gap_days=gap,
                maturation_confounded=m in CONFOUNDED_BY_MATURATION,
            ))
            if gap < min_gap_days:
                eff.notes.append(
                    f"{m}: only {gap} days separate the before/after recordings "
                    f"— low separation, treat the change as provisional."
                )

        if not any_evaluable:
            eff.notes.append(
                "No biomarker had both a before and an after recording around "
                "this date — record an EEG on both sides of a medication change "
                "to evaluate it."
            )
        resp.interventions.append(eff)

    resp.notes.append(
        "Each row is one recording before vs one recording after — it cannot "
        "separate a treatment effect from maturation, a different sleep state, "
        "or measurement variability. Metrics flagged 'maturation-confounded' "
        "(PDR, spindles) also rise with normal development. Use this to look "
        "for signals to confirm clinically, not as proof of a drug's effect."
    )
    return resp


def summarize_treatment_response(resp: TreatmentResponse) -> dict:
    """JSON-serializable summary."""
    return {
        "n_recordings": resp.n_recordings,
        "n_interventions": resp.n_interventions,
        "interventions": [
            {
                "date": iv.date,
                "description": iv.description,
                "metric_changes": [asdict(mc) for mc in iv.metric_changes],
                "notes": iv.notes,
            }
            for iv in resp.interventions
        ],
        "notes": resp.notes,
    }


def render_treatment_response_md(resp: TreatmentResponse) -> str:
    """Human-readable Markdown for the family / clinician."""
    lines: list[str] = ["# Treatment-response dashboard", ""]
    lines.append(
        f"_{resp.n_recordings} recording(s), {resp.n_interventions} medication "
        f"change(s) on record._"
    )
    lines.append("")
    if not resp.interventions:
        for n in resp.notes:
            lines.append(f"> {n}")
        return "\n".join(lines)

    arrows = {
        "improved": "✅ improved", "worsened": "⚠️ worsened",
        "no_clear_change": "→ no clear change", "ambiguous": "· (direction n/a)",
        "not_evaluable": "— no before/after",
    }
    for iv in resp.interventions:
        lines.append(f"## {iv.date} — {iv.description}")
        lines.append("")
        lines.append("| Biomarker | Before | After | Change | Direction |")
        lines.append("|---|---|---|---|---|")
        for mc in iv.metric_changes:
            label = mc.metric + (" *" if mc.maturation_confounded else "")
            if mc.direction == "not_evaluable":
                lines.append(f"| {label} | — | — | — | {arrows['not_evaluable']} |")
                continue
            change = (f"{mc.delta:+g}"
                      + (f" ({mc.pct_change:+g}%)" if mc.pct_change is not None else ""))
            before = f"{mc.baseline:g} ({mc.baseline_date})"
            after = f"{mc.followup:g} ({mc.followup_date})"
            lines.append(
                f"| {label} | {before} | {after} | {change} | "
                f"{arrows.get(mc.direction, mc.direction)} |"
            )
        lines.append("")
        for n in iv.notes:
            lines.append(f"> {n}")
        if any(mc.maturation_confounded for mc in iv.metric_changes):
            lines.append("> \\* also rises with normal maturation — not a clean treatment readout.")
        lines.append("")
    lines.append("---")
    for n in resp.notes:
        lines.append(f"> {n}")
    return "\n".join(lines)
